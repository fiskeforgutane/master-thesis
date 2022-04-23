use pyo3::pyclass;
use rand::{prelude::IteratorRandom, Rng};
use std::{collections::HashMap, sync::Arc};

use crate::{
    problem::{Problem, VesselIndex},
    quants::{self, Order},
    solution::{routing::RoutingSolution, Visit},
};

use super::initialization::Initialization;

#[pyclass]
#[derive(Debug, Clone)]
pub struct Chromosome {
    /// The population consists of a set of chromosomes (to be implemented)
    chromosome: Vec<Vec<Visit>>,
}

pub struct Init;

impl Initialization for Init {
    type Out = Chromosome;

    fn new(&self, problem: Arc<Problem>) -> Self::Out {
        Chromosome::new(&problem).unwrap()
    }
}

pub struct InitRoutingSolution;

impl Initialization for InitRoutingSolution {
    type Out = RoutingSolution;

    fn new(&self, problem: Arc<Problem>) -> Self::Out {
        let routes = Chromosome::new(&problem).unwrap().chromosome;
        RoutingSolution::new(problem, routes)
    }
}

impl Chromosome {
    pub fn new(problem: &Problem) -> Result<Chromosome, Box<dyn std::error::Error>> {
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
        
        // Binary indicator indicating if the nodes are a consumption or production node
        let node_types: Vec<u64> = problem
            .nodes()
            .iter()
            .map(|n| match n.r#type() {
                crate::problem::NodeType::Consumption => 1,
                crate::problem::NodeType::Production => 0,
            })
            .collect();

        // Serve all orders in chronological order
        for order in &initial_orders {
            // Select a random serve time between the opening and closing of the order's serving window
            let serve_time = rng.gen_range(order.open()..(order.close() + 1));

            // To avoid multiple consequtive production site visits, if the current order is to a production site, all
            // vessels currently located at a production node are filtered
            let possible_vessel_ids: Vec<VesselIndex> = match problem.nodes().get(order.node()).unwrap().r#type() {
                crate::problem::NodeType::Consumption => {
                    problem.vessels()
                        .iter()
                        .map(|v| v.index())
                        .collect()
                },
                crate::problem::NodeType::Production => {
                    problem.vessels()
                        .iter()
                        .filter(|v| *node_types.get(chromosome.get(v.index()).unwrap().last().unwrap().node).unwrap() == 1)
                        .map(|v| v.index())
                        .collect()
                },
            };


            let first_choice = vessels
                .iter()
                .filter(|v| {
                    (avail_from[&v.index()].1
                        + problem.travel_time(avail_from[&v.index()].0, order.node(), *v)
                        <= serve_time)
                        && ({
                            chromosome.get(v.index()).unwrap().last().unwrap().node != order.node()
                        })
                        && ({
                            possible_vessel_ids.contains(&v.index())
                        })
                })
                .choose(&mut rng);

            let chosen = match first_choice {
                Some(x) => Some(x),
                None => vessels
                    .iter()
                    .filter(|v| {
                        (avail_from[&v.index()].1 < serve_time)
                            && (chromosome[v.index()]
                                .iter()
                                .all(|visit| visit.time != serve_time))
                    })
                    .choose(&mut rng),
            };

            match chosen {
                Some(x) => {
                    chromosome
                        .get_mut(x.index())
                        .unwrap()
                        .push(Visit::new(problem, order.node(), serve_time).unwrap());

                    avail_from.insert(x.index(), (order.node(), serve_time + 1));
                }
                None => continue,
            }
        }

        Ok(Self { chromosome })
    }
    
    pub fn get_chromosome(&self) -> &Vec<Vec<Visit>> {
        &self.chromosome
    }
}
