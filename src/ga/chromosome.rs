use std::collections::HashMap;
use rand::{Rng, prelude::{IteratorRandom, SliceRandom}};

use crate::{
    problem::Problem,
    quants::{self, Order},
    solution::Visit,
};

use super::initialization::Initialization;

#[derive(Debug, Clone)]
pub struct Chromosome {
    /// The population consists of a set of chromosomes (to be implemented)
    chromosome: Vec<Vec<Visit>>,
}

pub struct Init;

impl Initialization for Init {
    type Out = Chromosome;

    fn new(&self, problem: &Problem) -> Self::Out {
        Chromosome::new(problem).unwrap()
    }
}

impl Chromosome {
    
    fn new(
        problem: &Problem,
    ) -> Result<Chromosome, Box<dyn std::error::Error>> {
        let initial_orders: Vec<Order> = quants::initial_orders(problem)?;
        let vessels = problem.vessels();
        let mut rng = rand::thread_rng();

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
            
            chromosome.get_mut(chosen.index()).unwrap().push(Visit::new(problem, order.node(), serve_time).unwrap());
            
            avail_from.insert(chosen.index(), (order.node(), serve_time + 1));
        }
        
        Ok(Self { chromosome })
    }

    pub fn get_chromosome(&self) -> &Vec<Vec<Visit>> {
        &self.chromosome
    }


}
