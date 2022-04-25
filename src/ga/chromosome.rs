use pyo3::pyclass;
use rand::{prelude::IteratorRandom, Rng};
use std::{cell::RefCell, collections::HashMap, rc::Rc, sync::Arc};

use crate::{
    models::quantity::QuantityLp,
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

    fn new(&self, problem: Arc<Problem>, _: Rc<RefCell<QuantityLp>>) -> Self::Out {
        Chromosome::new(&problem).unwrap()
    }
}

#[derive(Clone, Copy)]
pub struct InitRoutingSolution;

impl Initialization for InitRoutingSolution {
    type Out = RoutingSolution;

    fn new(&self, problem: Arc<Problem>, quantities: Rc<RefCell<QuantityLp>>) -> Self::Out {
        let routes = Chromosome::new(&problem).unwrap().chromosome;
        RoutingSolution::new_with_model(problem, routes, quantities)
    }
}

impl Chromosome {
    pub fn new(problem: &Problem) -> Result<Chromosome, Box<dyn std::error::Error>> {
        let mut initial_orders: Vec<Order> = quants::initial_orders(problem)?;
        initial_orders.sort_by_key(|o| o.close());
        let vessels = problem.vessels();
        let mut rng = rand::thread_rng();

        let mut chromosome: Vec<Vec<Visit>> = (0..vessels.len())
            .map(|v| {
                vec![Visit::new(problem, vessels[v].origin(), vessels[v].available_from()).unwrap()]
            })
            .collect();

        //let mut chromosome = std::iter::repeat(vec![])
        //    .take(vessels.len())
        //    .collect::<Vec<Vec<Visit>>>();

        let mut avail_from = vessels
            .iter()
            .map(|vessel| (vessel.index(), (vessel.origin(), vessel.available_from())))
            .collect::<HashMap<_, _>>();

        for order in &initial_orders {
            let serve_time = rng.gen_range(order.open()..(order.close() + 1));

            let first_choice = vessels
                .iter()
                .filter(|v| {
                    (avail_from[&v.index()].1
                        + problem.travel_time(avail_from[&v.index()].0, order.node(), *v)
                        <= serve_time)
                        && ({
                            chromosome.get(v.index()).unwrap().last().unwrap().node != order.node()
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
