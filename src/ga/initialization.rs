use std::{
    cell::RefCell,
    iter::once,
    rc::Rc,
    sync::{Arc, Mutex},
};

use float_ord::FloatOrd;
use log::{debug, info, trace};
use rand::Rng;

use crate::{
    models::quantity::QuantityLp,
    problem::{
        Node,
        NodeType::{Consumption, Production},
        Problem, Vessel,
    },
    solution::{
        routing::{Evaluation, RoutingSolution},
        Visit,
    },
};

use super::Fitness;

pub trait Initialization {
    type Out;
    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<QuantityLp>>) -> Self::Out;
}

/* impl<F, O> Initialization for F
where
    F: Fn(Arc<Problem>, Rc<RefCell<QuantityLp>>) -> O,
{
    type Out = O;
    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<QuantityLp>>) -> Self::Out {
        self(problem, quantities)
    }
}
 */
impl Initialization for Arc<Mutex<dyn Initialization<Out = RoutingSolution> + Send>> {
    type Out = RoutingSolution;

    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<QuantityLp>>) -> Self::Out {
        let inner = (*self).lock().unwrap();
        inner.new(problem, quantities)
    }
}

#[derive(Clone)]
pub struct FromPopulation {
    population: Arc<Mutex<Vec<Vec<Vec<Visit>>>>>,
}

impl FromPopulation {
    pub fn new(population: Vec<RoutingSolution>) -> Self {
        Self {
            population: Arc::new(Mutex::new(population.iter().map(|s| s.to_vec()).collect())),
        }
    }
}

impl Initialization for FromPopulation {
    type Out = RoutingSolution;

    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<QuantityLp>>) -> Self::Out {
        let mut population = self.population.lock().unwrap();
        let solution = population.pop().expect("should not be empty");
        drop(population);
        RoutingSolution::new_with_model(problem, solution, quantities)
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

    pub fn insert_best(
        &self,
        solution: &mut RoutingSolution,
        epsilon: (f64, f64),
        candidates: &Vec<(usize, Visit)>,
        best: (usize, FloatOrd<f64>, FloatOrd<f64>),
    ) -> Option<((usize, Visit), (usize, FloatOrd<f64>, FloatOrd<f64>))> {
        // choose the best among the candidates

        self.choose_inc_obj(solution, candidates.into_iter().cloned())
            .and_then(|(idx, obj)| {
                let dv = best.1 .0 - obj.1 .0;
                let dl = best.2 .0 - obj.2 .0;

                if (dv, dl) <= epsilon || obj.0 > best.0 {
                    trace!("Iterative converge done");
                    return None;
                }

                let mutator = &mut solution.mutate();
                mutator[idx.0].mutate().push(idx.1);
                Some((idx, obj))
            })
    }
}

impl Initialization for GreedyWithBlinks {
    type Out = RoutingSolution;

    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<QuantityLp>>) -> Self::Out {
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

pub struct StartPopulation {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Improvement {
    pub warp: isize,
    pub approx_berth_violation: isize,
    pub violation: FloatOrd<f64>,
    pub loss: FloatOrd<f64>,
}

impl Improvement {
    pub fn between(incumbent: Evaluation, candidate: Evaluation) -> Improvement {
        Improvement {
            warp: incumbent.warp as isize - candidate.warp as isize,
            approx_berth_violation: incumbent.approx_berth_violation as isize
                - candidate.approx_berth_violation as isize,
            violation: FloatOrd(incumbent.violation - candidate.violation),
            loss: FloatOrd(
                (incumbent.cost + incumbent.spot_cost - incumbent.revenue)
                    - (candidate.cost + candidate.spot_cost - candidate.revenue),
            ),
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// Greedy with blinks and lookahead
pub struct Greedy {
    /// The blink rate between insertion ranges
    pub inter_blink_rate: f64,
    /// The blink rate within an insertion range
    pub intra_blink_rate: f64,
    /// The number of lookaheads to do
    pub lookahead: usize,
    /// We will only evaluate the first `range_max` within each continuous insertion range
    pub range_max: usize,
    /// The epsilon used to abort the construction
    pub epsilon: Improvement,
}

#[derive(Debug, Clone, Copy)]
pub struct Insertion {
    vessel: usize,
    visit: Visit,
}

impl Greedy {
    /// Insert `insertion` into `solution`
    fn insert(insertion: Insertion, solution: &mut RoutingSolution) {
        let mut solution = solution.mutate();
        solution[insertion.vessel].mutate().push(insertion.visit);
    }

    /// Removes `insertion` from `solution`
    fn remove(insertion: Insertion, solution: &mut RoutingSolution) {
        let mut solution = solution.mutate();
        let mut plan = solution[insertion.vessel].mutate();
        let inserted = plan.iter().position(|&v| v == insertion.visit);

        if let Some(x) = inserted {
            plan.remove(x);
        }
    }

    /// Evaluate `insertion` in `solution` using a lookahead of `lookahead` on the plan of the `insertion.vessel`
    /// That is: if there's a lookahead, we will also try all the possible insertions on `insertion.vessel` *after* `insertion` has been performed
    pub fn evaluate(
        &self,
        insertion: Insertion,
        solution: &mut RoutingSolution,
        lookahead: usize,
    ) -> (FloatOrd<f64>, Evaluation) {
        // Do the insertion.
        Self::insert(insertion, solution);

        let timesteps = solution.problem().timesteps();
        let p = solution.problem().products();
        let v = insertion.vessel;
        let t = insertion.visit.time;
        let n = insertion.visit.node;
        let kind = solution.problem().nodes()[n].r#type();
        let capacity = solution.problem().nodes()[n].capacity().clone();

        let quantities = solution.quantities();

        let mut weighty = 0.0;

        for p in 0..p {
            let delivered = (t..timesteps)
                .map(|t| {
                    quantities
                        .model
                        .get_obj_attr(grb::attr::X, &quantities.vars.x[t][n][v][p])
                        .unwrap()
                })
                .sum::<f64>();

            let inventory = quantities
                .model
                .get_obj_attr(grb::attr::X, &quantities.vars.s[t][n][p])
                .unwrap();

            let distance_from_breach = match kind {
                Consumption => inventory,
                Production => capacity[p] - inventory,
            };

            weighty += delivered * (capacity[p] - distance_from_breach.max(0.0)) / capacity[p];
        }

        drop(quantities);

        let mut evaluation = (FloatOrd(-weighty), solution.evaluation());

        // See if we're able to get a better result by performing the look-ahead
        if lookahead > 0 {
            let lookaheads = solution
                .available()
                .nth(insertion.vessel)
                .expect("exists by constrction")
                .1
                .collect::<Vec<_>>();

            for (node, range) in lookaheads {
                for time in range
                    .filter(|&t| t > insertion.visit.time)
                    .take(self.range_max)
                {
                    evaluation = evaluation.min(self.evaluate(
                        Insertion {
                            vessel: insertion.vessel,
                            visit: Visit { time, node },
                        },
                        solution,
                        lookahead - 1,
                    ))
                }
            }
        }

        // Undo the insertion.
        Self::remove(insertion, solution);

        evaluation
    }
}

impl Initialization for Greedy {
    type Out = RoutingSolution;

    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<QuantityLp>>) -> Self::Out {
        let t = problem.timesteps();
        // We start with an empty solution
        let mut solution = RoutingSolution::new_with_model(
            problem.clone(),
            vec![Vec::new(); problem.count::<Vessel>()],
            quantities.clone(),
        );

        let mut incumbent = solution.evaluation();

        loop {
            info!("Incumbent: {incumbent:?}");
            info!(
                "Incumbent obj = {}",
                incumbent.cost + incumbent.spot_cost - incumbent.revenue
            );
            debug!("Routes: {:?}", solution.to_vec());
            let mut candidates = solution
                .available()
                .map(|(v, it)| (v, it.collect::<Vec<_>>()))
                .collect::<Vec<_>>();

            let mut best: Option<(FloatOrd<f64>, Evaluation)> = None;
            let mut insertion: Option<Insertion> = None;

            let (t0, vessel) = solution
                .iter()
                .enumerate()
                .map(|(v, plan)| (plan.last().unwrap().time, v))
                .min()
                .unwrap();

            let (vessel, it) = candidates.swap_remove(vessel);
            for (node, range) in it.into_iter() {
                debug!("v = {vessel}, n = {node}, t = {range:?}");
                // Do some random blinks.
                if rand::thread_rng().gen_bool(self.inter_blink_rate) {
                    continue;
                }

                // Find the best evaluation
                let (evaluation, time) = match range
                    .filter(|&t| t > t0)
                    .take(self.range_max)
                    .map(|time| {
                        let visit = Visit { node, time };
                        let insertion = Insertion { vessel, visit };

                        match rand::thread_rng().gen_bool(self.intra_blink_rate) {
                            true => ((FloatOrd(f64::INFINITY), Evaluation::bad()), time),
                            false => (
                                self.evaluate(insertion, &mut solution, self.lookahead),
                                time,
                            ),
                        }
                    })
                    .min()
                {
                    Some(x) => x,
                    None => continue,
                };

                // Update the best insertion
                if evaluation <= best.unwrap_or((FloatOrd(f64::INFINITY), Evaluation::bad())) {
                    insertion = Some(Insertion {
                        vessel,
                        visit: Visit { node, time },
                    });
                    best = Some(evaluation);
                }
            }

            let (candidate, insertion) = match (best, insertion) {
                (Some(best), Some(insertion)) => (best, insertion),
                _ => return solution,
            };

            info!("Choose {insertion:?}");

            if Improvement::between(incumbent, candidate.1) <= self.epsilon {
                debug!("Improvement below epsilon");
                return solution;
            }

            // Note: we can not use `candidate` here, in case lookahead is used.
            Self::insert(insertion, &mut solution);
            incumbent = solution.evaluation();
        }
    }
}
