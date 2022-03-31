use log::{trace, info};
use pyo3::pyclass;
use rand::{
    prelude::{IteratorRandom, SliceRandom},
    Rng,
};
use std::{collections::HashMap, sync::Arc};

use crate::{
    problem::Problem,
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
        trace!("Starting new initialization.");
        let routes = Chromosome::new(&problem).unwrap().chromosome;
        RoutingSolution::new(problem, routes)
    }
}

impl Chromosome {
    pub fn new(problem: &Problem) -> Result<Chromosome, Box<dyn std::error::Error>> {
        trace!("Starting on chromosomes!");
        let initial_orders: Vec<Order> = quants::initial_orders(problem)?;
        let vessels = problem.vessels();
        let mut rng = rand::thread_rng();

        let mut chromosome = std::iter::repeat(vec![])
            .take(vessels.len())
            .collect::<Vec<Vec<Visit>>>();
        
        trace!("Chromosome: {:?}", chromosome);

        let mut avail_from = problem
            .vessels()
            .iter()
            .map(|vessel| (vessel.index(), (vessel.origin(), vessel.available_from())))
            .collect::<HashMap<_, _>>();
        
        trace!("Available from: {:?}", avail_from);

        for node in problem.nodes() {
            trace!("Node id: {}     Initial inventory: {:?}   Consumption rate: {:?}", node.index(), node.initial_inventory(), node.inventory_changes()[0]);
        }

        for order in &initial_orders {
            trace!("Order: {:?}", order);
            let serve_time = rng.gen_range(order.open()..order.close());

            trace!("Serve time: {:?}", serve_time);

            let first_choice = vessels
                .iter()
                .filter(|v| {
                    avail_from[&v.index()].1
                        + problem.travel_time(avail_from[&v.index()].0, order.node(), *v)
                        <= serve_time
                })
                .choose(&mut rng);
            
            trace!("First choice {:?}", first_choice);

            let chosen = first_choice.unwrap_or_else(|| vessels.choose(&mut rng).unwrap());

            chromosome
                .get_mut(chosen.index())
                .unwrap()
                .push(Visit::new(problem, order.node(), serve_time).unwrap());

            avail_from.insert(chosen.index(), (order.node(), serve_time + 1));

            trace!("Updated chromosome: {:?}", chromosome);
        }

        Ok(Self { chromosome })
    }

    pub fn get_chromosome(&self) -> &Vec<Vec<Visit>> {
        &self.chromosome
    }
}
