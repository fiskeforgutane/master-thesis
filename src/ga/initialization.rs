use std::{
    cell::RefCell,
    collections::HashMap,
    rc::Rc,
    sync::{Arc, Mutex},
};

use float_ord::FloatOrd;
use log::{debug, trace};
use rand::{
    prelude::{IteratorRandom, SliceRandom},
    Rng,
};

use crate::{
    models::quantity::QuantityLp,
    problem::{Node, Problem, Vessel},
    solution::{routing::RoutingSolution, Visit},
};

use super::{fitness::Weighted, Fitness};

use crate::quants::{self, Order};

/// A trait for solution initialization. This can be used to construct initial solutions for the GA.
pub trait Initialization {
    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<QuantityLp>>) -> RoutingSolution;
}

impl Initialization for Arc<Mutex<dyn Initialization + Send>> {
    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<QuantityLp>>) -> RoutingSolution {
        let inner = (*self).lock().unwrap();
        inner.new(problem, quantities)
    }
}

/// A "warm-starting" initialization. An individual will be popped
/// from the population and returned each time `Initialization::new` is called
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
    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<QuantityLp>>) -> RoutingSolution {
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

    /// Choose the optimal insertion according to `GreedyWithBlinks`
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

    /// Insert the visit that is best according to `GreedyWithBlinks`,
    /// returning the visit that was inserted and the objective of the new solution
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
    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<QuantityLp>>) -> RoutingSolution {
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
            travel_empty: 1e5,
            travel_at_cap: 1e5,
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

/// An initialization that simply initializes each solution as "empty", i.e. with no visits.
pub struct Empty;

impl Initialization for Empty {
    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<QuantityLp>>) -> RoutingSolution {
        let vessels = problem.count::<Vessel>();
        RoutingSolution::new_with_model(problem, vec![Vec::new(); vessels], quantities)
    }
}

#[derive(Clone, Copy)]
pub struct InitRoutingSolution;

impl Initialization for InitRoutingSolution {
    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<QuantityLp>>) -> RoutingSolution {
        let routes = Self::routes(&problem).unwrap();
        RoutingSolution::new_with_model(problem, routes, quantities)
    }
}

impl InitRoutingSolution {
    pub fn routes(problem: &Problem) -> Result<Vec<Vec<Visit>>, Box<dyn std::error::Error>> {
        // Generate the initiale orders using the transportation model and sort by closing time
        let mut initial_orders: Vec<Order> = quants::initial_orders(problem)?;
        initial_orders.sort_by_key(|o| o.close());
        // Retrieve the vessels from the problem
        let vessels = problem.vessels();

        let mut rng = rand::thread_rng();

        // Initialize the chromosome with a visit for each vessel at its origin with the visit time set to
        // the time period it becomes available
        let mut chromosome: Vec<Vec<Visit>> = (0..vessels.len())
            .map(|v| {
                vec![Visit::new(problem, vessels[v].origin(), vessels[v].available_from()).unwrap()]
            })
            .collect();

        // Hashmap used for selecting the vessel to handle the orders. The avail_from hashmap contains information
        // about the vessels whereabouts and at what time they become available for serving a new order
        let mut avail_from = vessels
            .iter()
            .map(|vessel| (vessel.index(), (vessel.origin(), vessel.available_from())))
            .collect::<HashMap<_, _>>();

        // Serve all orders in chronological order
        for order in &initial_orders {
            // Select a random serve time between the opening and closing of the order's serving window
            // let mut serve_time = rng.gen_range(order.open()..(order.close() + 1));

            // Idea: alternate between production sites and consumption sites as far as this is possible
            let order_node_type = problem.nodes().get(order.node()).unwrap().r#type();

            // We want the vessels that are currently on the opposite node type of the current order, that
            // are available before the closing of the order
            let available_vessels: Vec<&Vessel> = vessels
                .iter()
                .filter(|v| {
                    let (node, available) = avail_from[&v.index()];
                    problem.nodes()[node].r#type() != order_node_type && available < order.close()
                })
                .collect::<Vec<_>>();

            // If some vessels fullfils the abovementioned criteria
            if let Some(vessel) = available_vessels.choose(&mut rng) {
                let chosen = vessel.index();
                let serve_time = rng.gen_range(avail_from[&chosen].1 + 1..=order.close());
                let visit = Visit::new(problem, order.node(), serve_time).unwrap();

                avail_from.insert(chosen, (order.node(), serve_time + 1));
                chromosome[chosen].push(visit);
            } else {
                // If no vessels fullfils the above criteria, the next step depends on the node type
                match order_node_type {
                    crate::problem::NodeType::Consumption => {
                        // We want to ensure that the chosen vessel isn't currently at the same node as the order node
                        // and we also want to ensure that the vessel becomes available before the order time window
                        // closes
                        let chosen_vessel = vessels
                            .iter()
                            .filter(|v| {
                                (avail_from[&v.index()].0 != order.node())
                                    && (avail_from[&v.index()].1 < order.close())
                            })
                            .choose(&mut rng);

                        // If a vessel is available, use it, otherwise skip the order
                        match chosen_vessel {
                            Some(v) => {
                                let serve_time =
                                    rng.gen_range(avail_from[&v.index()].1 + 1..=order.close());

                                avail_from.insert(v.index(), (order.node(), serve_time + 1));

                                chromosome
                                    .get_mut(v.index())
                                    .unwrap()
                                    .push(Visit::new(problem, order.node(), serve_time).unwrap());
                            }
                            None => (),
                        }
                    }
                    // If two consecutive productions node is unavoidable, skip the order
                    crate::problem::NodeType::Production => (),
                }
            }
        }
        Ok(chromosome)
    }
}
