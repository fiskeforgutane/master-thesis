use rand::{
    prelude::{IteratorRandom, SliceRandom},
    Rng,
};

use std::{cell::RefCell, collections::HashMap, rc::Rc, sync::Arc};

use crate::{
    models::quantity::QuantityLp,
    problem::{Problem, Vessel},
    quants::{self, Order},
    solution::{routing::RoutingSolution, Visit},
};

use super::initialization::Initialization;

#[derive(Clone, Copy)]
pub struct InitRoutingSolution;

impl Initialization for InitRoutingSolution {
    type Out = RoutingSolution;

    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<QuantityLp>>) -> Self::Out {
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
