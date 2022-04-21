use rand::{self, SeedableRng};
use std::sync::Arc;
pub mod ga;
pub mod models;
pub mod problem;
pub mod quants;
pub mod solution;
pub mod utils;

use crate::ga::{
    chromosome::InitRoutingSolution,
    fitness::{self},
    mutations::{self, TwoOptMode},
    parent_selection,
    recombinations::PIX,
    survival_selection, Fitness, GeneticAlgorithm, Stochastic,
};

use crate::problem::Problem;

pub fn run_ga(path: &str, epochs: usize) {
    let file = std::fs::File::open(path).unwrap();
    let mut reader = std::io::BufReader::new(file);

    let problem: Problem = serde_json::from_reader(reader).unwrap();
    let problem = Arc::new(problem);

    let mut ga = GeneticAlgorithm::new(
        problem.clone(),
        200, // population size
        200, // child count
        InitRoutingSolution,
        parent_selection::Tournament::new(3).unwrap(),
        Stochastic::new(0.10, PIX),
        chain!(
            Stochastic::new(0.05, mutations::AddRandom::new()),
            Stochastic::new(0.05, mutations::RemoveRandom::new()),
            // Stochastic::new(0.05, mutations::BestMove::new()),
            Stochastic::new(0.05, mutations::Bounce::new(3, mutations::BounceMode::All)),
            Stochastic::new(
                0.05,
                mutations::Bounce::new(3, mutations::BounceMode::Random)
            ),
            /*Stochastic::new(
                0.05,
                mutations::DistanceReduction::new(mutations::DistanceReductionMode::All)
            ),
            Stochastic::new(
                0.05,
                mutations::DistanceReduction::new(mutations::DistanceReductionMode::Random)
            ),*/
            Stochastic::new(0.05, mutations::InterSwap),
            Stochastic::new(0.05, mutations::IntraSwap),
            Stochastic::new(0.05, mutations::RedCost::red_cost_mutation(10)),
            //Stochastic::new(0.05, mutations::RedCost::red_cost_local_search(10)),
            /*Stochastic::new(0.05, mutations::TimeSetter::new(0.5).unwrap()),*/
            Stochastic::new(0.05, mutations::Twerk::everybody()),
            Stochastic::new(0.05, mutations::Twerk::some_random_person()),
            Stochastic::new(0.05, mutations::TwoOpt::new(TwoOptMode::IntraRandom)),
            Stochastic::new(
                0.00,
                mutations::TwoOpt::new(TwoOptMode::LocalSerach(100, 1e-3))
            ) /*Stochastic::new(0.05, mutations::VesselSwap::new())*/
        ),
        survival_selection::Elite(3, survival_selection::Proportionate(|x| 1.0 / (1.0 + x))),
        fitness::Weighted {
            warp: 1e8,
            violation: 1e4,
            revenue: -1.0,
            cost: 1.0,
        },
    );

    let fitness = fitness::Weighted {
        warp: 1e8,
        violation: 1e4,
        revenue: -1.0,
        cost: 1.0,
    };

    for _ in 0..epochs {
        ga.epoch();
        let best = ga.best_individual();
        println!(
            "F = {}. warp = {}, violation = {}, revenue = {}, cost = {}",
            fitness.of(&problem, best),
            best.warp(),
            best.violation(),
            best.revenue(),
            best.cost()
        );
    }
}

pub fn main() {
    println!("Hello world!");
    run_ga("mirplib/t60/LR2_22_DR3_333_VC4_V17a.json", 100000)
}
