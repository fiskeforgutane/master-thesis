use std::{cell::RefCell, collections::HashSet, rc::Rc, sync::Arc};

use float_ord::FloatOrd;
use rand::{prelude::StdRng, Rng};

use crate::{
    models::quantity::QuantityLp,
    problem::{Node, Problem, Timestep, Vessel},
    solution::{routing::RoutingSolution, Visit},
};

pub trait Initialization {
    type Out;
    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<QuantityLp>>) -> Self::Out;
}

impl<F, O> Initialization for F
where
    F: Fn(Arc<Problem>, Rc<RefCell<QuantityLp>>) -> O,
{
    type Out = O;
    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<QuantityLp>>) -> Self::Out {
        self(problem, quantities)
    }
}

/// Greedily construct an individual visit by visit, at each step choosing the visit resulting in the best solution,
/// as evaluated by the quantity LP. Evaluation is skipped with probability `blink_rate`, to allow for some stochasticity / randomness
pub struct GreedyWithBlinks {
    blink_rate: f64,
}

impl GreedyWithBlinks {
    fn evaluate(
        vessel: usize,
        insertion: Visit,
        solution: &mut RoutingSolution,
    ) -> (usize, FloatOrd<f64>, FloatOrd<f64>) {
        // Insert the visit
        {
            let mut solution = solution.mutate();
            let mut plan = solution[vessel].mutate();
            plan.push(insertion);
        }
        // Evaluate the solution
        let warp = solution.warp();
        let violation = solution.violation();
        let loss = solution.cost() - solution.revenue();
        // Undo the insertion
        {
            let mut solution = solution.mutate();
            let mut plan = solution[vessel].mutate();
            let inserted = plan.iter().position(|&v| v == insertion);
            plan.remove(inserted.unwrap());
        }

        // Return the relevant objectives
        (warp, FloatOrd(violation), FloatOrd(loss))
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

            let (v, node, time, _, cost) = candidates
                .into_iter()
                .map(|(v, node, time)| {
                    let blink = rand::thread_rng().gen_bool(self.blink_rate);
                    let cost = Self::evaluate(v, Visit { node, time }, &mut solution);
                    (v, node, time, blink, cost)
                })
                .min_by_key(|t| (t.3, t.4))
                .unwrap();

            if cost >= best {
                return solution;
            }

            // Do the insertion
            let mut solution = solution.mutate();
            let mut plan = solution[v].mutate();
            plan.push(Visit { node, time });
        }
    }
}
