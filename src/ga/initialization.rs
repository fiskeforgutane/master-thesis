use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

use float_ord::FloatOrd;
use log::{debug, trace};
use rand::Rng;

use crate::{
    models::quantity::QuantityLp,
    problem::{Node, Problem, Vessel},
    solution::{
        routing::{Evaluation, Improvement, RoutingSolution},
        Visit,
    },
};

use super::{fitness::Weighted, Fitness};

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
        fitness: &dyn Fitness,
    ) -> Option<f64> {
        // Insert the visit
        {
            let mut solution = solution.mutate();
            let mut plan = solution[vessel].mutate();
            plan.push(insertion);
        }

        // We will only evaluate those that have no warp, as an optimization
        let warp = solution.warp();
        let obj = match warp > baseline_warp {
            false => Some(fitness.of(solution.problem(), solution)),
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
        fitness: &dyn Fitness,
    ) -> Option<((usize, Visit), f64)>
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

                let mut cost = || match blink {
                    true => Some(f64::INFINITY),
                    false => Some(Self::evaluate(v, visit, solution, baseline_warp, fitness)?),
                };

                match occupied {
                    true => {
                        debug!("None should be occupied?");
                        None
                    }
                    false => Some(((v, visit), cost()?)),
                }
            })
            .min_by_key(|(_, cost)| FloatOrd(*cost))
    }

    pub fn choose<I>(
        &self,
        solution: &mut RoutingSolution,
        candidates: I,
        fitness: &dyn Fitness,
    ) -> Option<(usize, Visit)>
    where
        I: Iterator<Item = (usize, Visit)>,
    {
        self.choose_inc_obj(solution, candidates, fitness)
            .map(|t| t.0)
    }

    // Repeatedly insert at the best position according to greedy with blinks while improvements are > epsilon.
    pub fn converge(
        &self,
        solution: &mut RoutingSolution,
        epsilon: f64,
        candidates: Vec<(usize, Visit)>,
        fitness: &dyn Fitness,
    ) {
        let mut best = fitness.of(solution.problem(), solution);

        trace!("start converge.");
        while let Some((idx, obj)) =
            self.choose_inc_obj(solution, candidates.iter().cloned(), fitness)
        {
            if obj - best < epsilon {
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
        epsilon: f64,
        candidates: &Vec<(usize, Visit)>,
        best: f64,
        fitness: &dyn Fitness,
    ) -> Option<((usize, Visit), f64)> {
        // choose the best among the candidates

        self.choose_inc_obj(solution, candidates.into_iter().cloned(), fitness)
            .and_then(|(idx, obj)| {
                //let improvement = Improvement::between(best, obj);
                debug!("Best insertion is {idx:?} with obj {obj:?}");
                if best - obj <= epsilon {
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

        let mut best = f64::INFINITY;
        let fitness: &dyn Fitness = &Weighted {
            warp: 1e8,
            violation: 1e4,
            revenue: -1.0,
            cost: 1.0,
            approx_berth_violation: 1e8,
            spot: 1.0,
            offset: problem.max_revenue() + 1.0,
        };

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
                .map(
                    |(v, node, time)| match rand::thread_rng().gen_bool(self.blink_rate) {
                        true => (v, node, time, f64::INFINITY),
                        false => {
                            let cost = Self::evaluate(
                                v,
                                Visit { node, time },
                                &mut solution,
                                baseline_warp,
                                fitness,
                            );
                            (v, node, time, cost.unwrap_or(f64::INFINITY))
                        }
                    },
                )
                .min_by_key(|t| FloatOrd(t.3))
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

pub struct Empty;

impl Initialization for Empty {
    type Out = RoutingSolution;

    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<QuantityLp>>) -> Self::Out {
        let vessels = problem.count::<Vessel>();
        RoutingSolution::new_with_model(problem, vec![Vec::new(); vessels], quantities)
    }
}
