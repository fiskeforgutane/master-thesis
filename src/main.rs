use float_ord::FloatOrd;
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
use crate::solution::Visit;

pub fn run_ga(path: &str, epochs: usize) {
    let file = std::fs::File::open(path).unwrap();
    let mut reader = std::io::BufReader::new(file);

    let problem: Problem = serde_json::from_reader(reader).unwrap();
    let problem = Arc::new(problem);

    let mut ga = GeneticAlgorithm::new(
        problem.clone(),
        100, // population size
        100, // child count
        InitRoutingSolution,
        parent_selection::Tournament::new(3).unwrap(),
        Stochastic::new(0.10, PIX),
        chain!(
            Stochastic::new(0.03, mutations::AddRandom::new()),
            Stochastic::new(0.03, mutations::RemoveRandom::new()),
            // Stochastic::new(0.05, mutations::BestMove::new()),
            /*Stochastic::new(
                0.05,
                mutations::DistanceReduction::new(mutations::DistanceReductionMode::All)
            ),
            Stochastic::new(
                0.05,
                mutations::DistanceReduction::new(mutations::DistanceReductionMode::Random)
            ),*/
            Stochastic::new(0.03, mutations::InterSwap),
            Stochastic::new(0.03, mutations::IntraSwap),
            Stochastic::new(0.03, mutations::RedCost::red_cost_mutation(10)),
            //Stochastic::new(0.05, mutations::RedCost::red_cost_local_search(10)),
            // Stochastic::new(0.05, mutations::TimeSetter::new(0.5).unwrap()),
            Stochastic::new(0.03, mutations::Twerk::everybody()),
            Stochastic::new(0.03, mutations::Twerk::some_random_person()),
            Stochastic::new(0.03, mutations::TwoOpt::new(TwoOptMode::IntraRandom)),
            Stochastic::new(
                0.00,
                mutations::TwoOpt::new(TwoOptMode::LocalSerach(100, 1e-3))
            ), /*Stochastic::new(0.05, mutations::VesselSwap::new())*/
            Stochastic::new(0.03, mutations::TimeSetter::new(0.4).unwrap()),
            Stochastic::new(0.03, mutations::TimeSetter::new(0.0).unwrap()), // Stochastic::new(0.05, mutations::AddSmart)
            Stochastic::new(0.03, mutations::Bounce::new(3, mutations::BounceMode::All)),
            Stochastic::new(
                0.03,
                mutations::Bounce::new(3, mutations::BounceMode::Random)
            ),
            Stochastic::new(0.03, mutations::AddSmart)
        ),
        survival_selection::Elite(1, survival_selection::Proportionate(|x| 1.0 / (1.0 + x))),
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

    for i in 0..epochs {
        ga.epoch();
        let best = ga.best_individual();
        let worst_fitness = ga
            .population
            .iter()
            .map(|solution| FloatOrd(fitness.of(&problem, solution)))
            .max()
            .unwrap();

        println!(
            "F = {}. warp = {}, violation = {}, revenue = {}, cost = {}; (worst fitness = {})",
            fitness.of(&problem, best),
            best.warp(),
            best.violation(),
            best.revenue(),
            best.cost(),
            worst_fitness.0
        );

        if i % 100 == 0 {
            let folder = path.replace("/", "-").replace(".json", "");
            let _ = std::fs::create_dir_all(&format!("solutions/{}", folder));
            let file = std::fs::File::create(&format!("solutions/{}/{}.json", folder, i)).unwrap();

            let visits: Vec<&[Visit]> = best.iter().map(|plan| &plan[..]).collect();
            serde_json::to_writer(file, &visits).expect("writing failed");
        }
    }
}

pub fn main() {
    println!("Hello world!");
    run_ga(
        "/Users/sjurwold/Documents/NTNU/TIO4905/master/master-playground/mirplib-rs/t60/LR1_1_DR1_4_VC3_V8a.json",
        100000,
    )
}
