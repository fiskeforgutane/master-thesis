use std::collections::HashMap;
use rand::{Rng, prelude::{IteratorRandom, SliceRandom}};

use crate::{
    problem::Problem,
    quants::{self, Order},
    solution::Visit,
};

pub struct Population {
    /// The population consists of a set of chromosomes (to be implemented)
    population: Vec<Vec<Vec<Visit>>>,
}

impl Population {
    pub fn new(
        problem: &Problem,
        population_size: usize,
    ) -> Result<Population, Box<dyn std::error::Error>> {
        let initial_orders: Vec<Order> = quants::initial_orders(problem)?;
        let vessels = problem.vessels();
        let mut rng = rand::thread_rng();

        let mut population: Vec<Vec<Vec<Visit>>> = Vec::new();

        for _ in 0..population_size {
            let mut chromosome = std::iter::repeat(vec![])
                .take(vessels.len())
                .collect::<Vec<Vec<Visit>>>();

            let mut avail_from = problem
                .vessels()
                .iter()
                .map(|vessel| (vessel.index(), (vessel.origin(), vessel.available_from())))
                .collect::<HashMap<_, _>>();
            
            for order in &initial_orders {
                let serve_time = rng.gen_range(order.open()..order.close());

                let first_choice = vessels
                    .iter()
                    .filter(|v| 
                        avail_from[&v.index()].1 + problem.travel_time(avail_from[&v.index()].0, order.node(), *v) <= serve_time)
                        .choose(&mut rng);
                
                let chosen = first_choice.unwrap_or_else(|| vessels.choose(&mut rng).unwrap());
                                
            }
        }

        todo!()
    }

    fn assign_vessel(order: &Order) {
        todo!()
    }
}
