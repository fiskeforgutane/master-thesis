use pyo3::pyclass;
use rand::{prelude::{IteratorRandom, SliceRandom}, Rng};
use serde::ser;
use std::{collections::HashMap, sync::Arc};

use crate::{
    problem::{Problem, VesselIndex, Vessel},
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

#[derive(Clone, Copy)]
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
            // let mut serve_time = rng.gen_range(order.open()..(order.close() + 1));

            // Idea: alternate between production sites and consumption sites as far as this is possible
            let order_node_type = problem.nodes().get(order.node()).unwrap().r#type();

            // We want the vessels that are currently on the opposite node type of the current order, that 
            // are available before the closing of the order
            let available_vessels: Vec<&Vessel> = vessels.iter().filter(
                |v| {
                    (problem.nodes().get(avail_from[&v.index()].0).unwrap().r#type() != order_node_type)
                    &&
                    (avail_from[&v.index()].1 < order.close())
                }
            ).collect::<Vec<_>>();

            // If some vessels fullfils the abovementioned criteria
            if available_vessels.len() > 0 {
                let chosen = available_vessels.choose(&mut rng).unwrap().index();

                let serve_time = rng.gen_range(avail_from[&chosen].1 + 1..(order.close() + 1));

                avail_from.insert(chosen, (order.node(), serve_time + 1));

                chromosome
                    .get_mut(chosen)
                    .unwrap()
                    .push(Visit::new(problem, order.node(), serve_time).unwrap());
            }
            else {
                // If no vessels fullfils the above criteria, the next step depends on the node type
                match order_node_type {
                    crate::problem::NodeType::Consumption => {
                        // We want to ensure that the chosen vessel isn't currently at the same node as the order node
                        // and we also want to ensure that the vessel becomes available before the order time window 
                        // closes
                        let chosen_vessel = vessels.iter().filter(
                            |v| {
                                (avail_from[&v.index()].0 != order.node())
                                &&
                                (avail_from[&v.index()].1 < order.close())
                            }
                        ).choose(&mut rng);
                        
                        // If a vessel is available, use it, otherwise skip the order
                        match chosen_vessel {
                            Some(v) => {
                                let serve_time = rng.gen_range(avail_from[&v.index()].1 + 1..(order.close() + 1));

                                avail_from.insert(v.index(), (order.node(), serve_time + 1));

                                chromosome
                                    .get_mut(v.index())
                                    .unwrap()
                                    .push(Visit::new(problem, order.node(), serve_time).unwrap());                                
                            },
                            None => (),
                        }
                    },
                    // If two consecutive productions node is unavoidable, skip the order
                    crate::problem::NodeType::Production => (),
                }
            }
        }
            
        Ok(Self { chromosome })
    }

    pub fn get_chromosome(&self) -> &Vec<Vec<Visit>> {
        &self.chromosome
    }
}



/* 

************** RANKING BASED ***************************
    //println!("Serve time: {}", serve_time);
    //println!("Vessels available from: {:?}", avail_from);

    /*
    Idea: rank the vessels based on a set of factors.
    Factors:
        1. If the last visit of the vessel was to the same node
        2. If both the current order and the last serving of the vessel were to production nodes
        3. If the vessel isn't serving another order at or after the serve time
        4. If the vessel can make it to the order in time for the serving time
    */

    let mut scores = vessels.iter().map(
        |v| {
            let mut score = 0; 

            if serve_time <= v.available_from() + 1 {
                score -= 64;
            }

            if serve_time <= avail_from[&v.index()].1 + 1 {
                score -= 32;
            }

            if chromosome.get(v.index()).unwrap().last().unwrap().node != order.node() {
                score += 8;
            }
            
            match problem.nodes().get(chromosome.get(v.index()).unwrap().last().unwrap().node).unwrap().r#type() {
                crate::problem::NodeType::Consumption => score += 4,
                crate::problem::NodeType::Production =>
                    match problem.nodes().get(order.node()).unwrap().r#type() {
                        crate::problem::NodeType::Consumption => score += 4,
                        crate::problem::NodeType::Production => ()
                    },
            }

            if avail_from[&v.index()].1 < serve_time {
                score += 2;
            }

            if avail_from[&v.index()].1 + problem.travel_time(avail_from[&v.index()].0, order.node(), v) <= serve_time {
                score += 1;
            }

            (v.index(), score)
        }
    ).collect::<Vec<_>>();

    scores.sort_by_key(|k| k.1);
    scores.reverse();
    let high_score = scores[0].1;

    let chosen = scores.iter().filter(
        |v| v.1 == high_score
    ).choose(&mut rng).unwrap().0;

    //println!("Scores: {:?}    :::    Chosen: {:?}", scores, chosen);

    if high_score < 0 {
        serve_time = std::cmp::min(problem.timesteps(), avail_from[&chosen].1 + problem.travel_time(avail_from[&chosen].0, order.node(), &vessels[chosen]) + 1);
        avail_from.insert(chosen, (order.node(), serve_time));
    }
    else {
        avail_from.insert(chosen, (order.node(), serve_time + 1));
    }

    chromosome
        .get_mut(chosen)
        .unwrap()
        .push(Visit::new(problem, order.node(), serve_time).unwrap());
}



************* FILTERING BASED **********************

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

*/