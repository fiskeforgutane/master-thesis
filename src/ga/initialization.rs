use std::{cell::RefCell, rc::Rc, sync::Arc};

use float_ord::FloatOrd;
use log::trace;
use rand::Rng;

use crate::{
    models::quantity::sparse,
    problem::{Node, Problem, Vessel},
    solution::{routing::RoutingSolution, Visit},
};

pub trait Initialization {
    type Out;
    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<sparse::QuantityLp>>) -> Self::Out;
}

impl<F, O> Initialization for F
where
    F: Fn(Arc<Problem>, Rc<RefCell<sparse::QuantityLp>>) -> O,
{
    type Out = O;
    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<sparse::QuantityLp>>) -> Self::Out {
        self(problem, quantities)
    }
}

/// Greedily construct an individual visit by visit, at each step choosing the visit resulting in the best solution,
/// as evaluated by the quantity LP. Evaluation is skipped with probability `blink_rate`, to allow for some stochasticity / randomness
#[derive(Debug, Clone, Copy)]
pub struct GreedyWithBlinks {
    blink_rate: f64,
}

impl GreedyWithBlinks {
    pub fn new(blink_rate: f64) -> Self {
        GreedyWithBlinks { blink_rate }
    }
}

impl GreedyWithBlinks {
    /// Return the violation and loss of the insertion, if it incurs no warp.
    pub fn evaluate(
        vessel: usize,
        insertion: Visit,
        solution: &mut RoutingSolution,
        baseline_warp: usize,
    ) -> Option<(usize, FloatOrd<f64>, FloatOrd<f64>)> {
        // Insert the visit
        {
            let mut solution = solution.mutate();
            let mut plan = solution[vessel].mutate();
            plan.push(insertion);
        }

        // We will only evaluate those that have no warp, as an optimization
        let warp = solution.warp();
        let obj = match warp > baseline_warp {
            false => Some((
                warp,
                FloatOrd(solution.violation()),
                FloatOrd(solution.cost() - solution.revenue()),
            )),
            true => None,
        };

        // Undo the insertion
        {
            let mut solution = solution.mutate();
            let mut plan = solution[vessel].mutate();
            let inserted = plan.iter().position(|&v| v == insertion);
            plan.remove(inserted.unwrap());
        }

        // Return the relevant objectives
        obj
    }

    /// Choose among the possible (usize, Visit)s using greedy with blinks.
    pub fn choose_inc_obj<I>(
        &self,
        solution: &mut RoutingSolution,
        candidates: I,
    ) -> Option<((usize, Visit), (usize, FloatOrd<f64>, FloatOrd<f64>))>
    where
        I: Iterator<Item = (usize, Visit)>,
    {
        // Return the minimum (sans blinking).
        // Note that we will filter out occupied time slots, just in case.
        let baseline_warp = solution.warp();
        candidates
            .filter_map(|(v, visit)| {
                let blink = rand::thread_rng().gen_bool(self.blink_rate);
                let occupied = solution[v]
                    .binary_search_by_key(&visit.time, |v| v.time)
                    .is_ok();
                let bad = (usize::MAX, FloatOrd(f64::INFINITY), FloatOrd(f64::INFINITY));

                let mut cost = || match blink {
                    true => Some(bad),
                    false => Some(Self::evaluate(v, visit, solution, baseline_warp)?),
                };

                match occupied {
                    true => None,
                    false => Some(((v, visit), cost()?)),
                }
            })
            .min_by_key(|(_, cost)| *cost)
    }

    pub fn choose<I>(&self, solution: &mut RoutingSolution, candidates: I) -> Option<(usize, Visit)>
    where
        I: Iterator<Item = (usize, Visit)>,
    {
        self.choose_inc_obj(solution, candidates).map(|t| t.0)
    }

    // Repeatedly insert at the best position according to greedy with blinks while improvements are > epsilon.
    pub fn converge(
        &self,
        solution: &mut RoutingSolution,
        epsilon: (f64, f64),
        candidates: Vec<(usize, Visit)>,
    ) {
        let mut best = (
            solution.warp(),
            FloatOrd(solution.violation()),
            FloatOrd(solution.cost() - solution.revenue()),
        );

        trace!("start converge.");
        while let Some((idx, obj)) = self.choose_inc_obj(solution, candidates.iter().cloned()) {
            let dv = best.1 .0 - obj.1 .0;
            let dl = best.2 .0 - obj.2 .0;
            trace!(
                "\tconverge: z = {:?}, dv = {}, dl = {}, idx = {:?}",
                obj,
                dv,
                dl,
                idx
            );

            if (dv, dl) <= epsilon || obj.0 > best.0 {
                trace!("converge done.");
                return;
            }

            best = obj;
            let mut solution = solution.mutate();
            let mut plan = solution[idx.0].mutate();
            plan.push(idx.1);
        }
    }
}

impl Initialization for GreedyWithBlinks {
    type Out = RoutingSolution;

    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<sparse::QuantityLp>>) -> Self::Out {
        let t = problem.timesteps();
        // We start with an empty solution
        let mut solution = RoutingSolution::new_with_model(
            problem.clone(),
            vec![Vec::new(); problem.count::<Vessel>()],
            quantities.clone(),
        );

        let mut best = (usize::MAX, FloatOrd(f64::INFINITY), FloatOrd(f64::INFINITY));

        loop {
            let s = &solution;
            let baseline_warp = solution.warp();
            let candidates = problem
                .indices::<Vessel>()
                .flat_map(|v| {
                    let vessel = &problem.vessels()[v];
                    let available = vessel.available_from();
                    let plan = &s[v];
                    problem.indices::<Node>().flat_map(move |n| {
                        (available + 1..t)
                            .filter(|&t| plan.iter().all(|v| v.time != t))
                            .map(move |t| (v, n, t))
                    })
                })
                .collect::<Vec<_>>();

            let (v, node, time, cost) = candidates
                .into_iter()
                .map(|(v, node, time)| {
                    let bad = (usize::MAX, FloatOrd(f64::INFINITY), FloatOrd(f64::INFINITY));
                    match rand::thread_rng().gen_bool(self.blink_rate) {
                        true => (v, node, time, bad),
                        false => {
                            let cost = Self::evaluate(
                                v,
                                Visit { node, time },
                                &mut solution,
                                baseline_warp,
                            );
                            (v, node, time, cost.unwrap_or(bad))
                        }
                    }
                })
                .min_by_key(|t| t.3)
                .unwrap();

            if cost >= best {
                return solution;
            }

            // Do the insertion
            let mut solution = solution.mutate();
            let mut plan = solution[v].mutate();
            plan.push(Visit { node, time });
            best = cost;
        }
    }
}
