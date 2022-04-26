use float_ord::FloatOrd;
use rand;
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
    mutations::{
        AddRandom, AddSmart, Bounce, BounceMode, InterSwap, IntraSwap, RedCost, RemoveRandom,
        TimeSetter, Twerk, TwoOpt, TwoOptMode,
    },
    parent_selection,
    recombinations::PIX,
    survival_selection, Fitness, GeneticAlgorithm, Stochastic,
};

use crate::problem::Problem;
use crate::solution::routing::RoutingSolution;
use crate::solution::Visit;

pub fn run_ga(path: &str, epochs: usize) {
    let file = std::fs::File::open(path).unwrap();
    let reader = std::io::BufReader::new(file);

    let problem: Problem = serde_json::from_reader(reader).unwrap();
    let problem = Arc::new(problem);

    let config = ga::Config {
        problem: problem.clone(),
        population_size: 100,
        child_count: 100,
        parent_selection: parent_selection::Tournament::new(3).unwrap(),
        recombination: Stochastic::new(0.10, PIX),
        mutation: chain!(
            Stochastic::new(0.03, AddRandom::new()),
            Stochastic::new(0.03, RemoveRandom::new()),
            Stochastic::new(0.03, InterSwap),
            Stochastic::new(0.03, IntraSwap),
            Stochastic::new(0.03, RedCost::red_cost_mutation(10)),
            Stochastic::new(0.03, Twerk::everybody()),
            Stochastic::new(0.03, Twerk::some_random_person()),
            Stochastic::new(0.03, TwoOpt::new(TwoOptMode::IntraRandom)),
            Stochastic::new(0.03, TimeSetter::new(0.4).unwrap()),
            Stochastic::new(0.03, TimeSetter::new(0.0).unwrap()), // Stochastic::new(0.05, mutations::AddSmart)
            Stochastic::new(0.03, Bounce::new(3, BounceMode::All)),
            Stochastic::new(0.03, Bounce::new(3, BounceMode::Random)),
            Stochastic::new(0.03, AddSmart)
        ),
        selection: survival_selection::Elite(
            1,
            survival_selection::Proportionate(|x| 1.0 / (1.0 + x)),
        ),
        fitness: fitness::Weighted {
            warp: 1e8,
            violation: 1e4,
            revenue: -1.0,
            cost: 1.0,
        },
    };

    let mut ga = GeneticAlgorithm::new(InitRoutingSolution, config);

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
            "Iteration: {}, F = {}. warp = {}, violation = {}, revenue = {}, cost = {}; (worst fitness = {})",
            i+1,
            fitness.of(&problem, best),
            best.warp(),
            best.violation(),
            best.revenue(),
            best.cost(),
            worst_fitness.0
        );

        if i % 100 == 0 {
            //let folder = "solutions"; path.replace("/", "-").replace(".json", "");
            //let _ = std::fs::create_dir_all(&format!("solutions", folder));
            let file = std::fs::File::create(&format!("solutions-{}.json", i)).unwrap();

            let visits: Vec<&[Visit]> = best.iter().map(|plan| &plan[..]).collect();
            serde_json::to_writer(file, &visits).expect("writing failed");
        }
    }
}

pub fn run_island_ga(path: &str, epochs: usize) {
    let file = std::fs::File::open(path).unwrap();
    let reader = std::io::BufReader::new(file);

    let problem: Problem = serde_json::from_reader(reader).unwrap();
    let problem = Arc::new(problem);
    let closure_problem = problem.clone();

    let config = move || ga::Config {
        problem: closure_problem.clone(),
        population_size: 100,
        child_count: 100,
        parent_selection: parent_selection::Tournament::new(3).unwrap(),
        recombination: Stochastic::new(0.10, PIX),
        mutation: chain!(
            Stochastic::new(0.03, AddRandom::new()),
            Stochastic::new(0.03, RemoveRandom::new()),
            Stochastic::new(0.03, InterSwap),
            Stochastic::new(0.03, IntraSwap),
            Stochastic::new(0.03, RedCost::red_cost_mutation(10)),
            Stochastic::new(0.03, Twerk::everybody()),
            Stochastic::new(0.03, Twerk::some_random_person()),
            Stochastic::new(0.03, TwoOpt::new(TwoOptMode::IntraRandom)),
            Stochastic::new(0.03, TimeSetter::new(0.4).unwrap()),
            Stochastic::new(0.03, TimeSetter::new(0.0).unwrap()), // Stochastic::new(0.05, mutations::AddSmart)
            Stochastic::new(0.03, Bounce::new(3, BounceMode::All)),
            Stochastic::new(0.03, Bounce::new(3, BounceMode::Random)),
            Stochastic::new(0.03, AddSmart)
        ),
        selection: survival_selection::Elite(
            1,
            survival_selection::Proportionate(|x| 1.0 / (1.0 + x)),
        ),
        fitness: fitness::Weighted {
            warp: 1e8,
            violation: 1e4,
            revenue: -1.0,
            cost: 1.0,
        },
    };

    let islands = std::thread::available_parallelism().unwrap().get();
    let mut ga = ga::islands::IslandGA::new(InitRoutingSolution, config, 8);

    let fitness = fitness::Weighted {
        warp: 1e8,
        violation: 1e4,
        revenue: -1.0,
        cost: 1.0,
    };

    let mut last_migration = 0;
    let mut last_save = 0;

    loop {
        let epochs = ga.epochs();
        let best = RoutingSolution::new(
            problem.clone(),
            ga.best()
                .0
                .iter()
                .map(|plan| plan.iter().cloned().collect())
                .collect(),
        );

        if epochs - last_migration > 500 {
            print!("Migrating...");
            ga.migrate(5);
            println!(" DONE");
            last_migration = epochs;
        }

        println!(
            "{:>010}: F = {}. warp = {}, violation = {}, revenue = {}, cost = {}; (worst fitness = N/A)",
            epochs,
            fitness.of(&problem, &best),
            best.warp(),
            best.violation(),
            best.revenue(),
            best.cost(),
            //worst_fitness.0
        );

        if epochs - last_save > 0 {
            let _ = std::fs::create_dir_all(&format!("solutions/"));
            let file = std::fs::File::create(&format!("solutions/{}.json", epochs)).unwrap();

            let visits: Vec<&[Visit]> = best.iter().map(|plan| &plan[..]).collect();
            serde_json::to_writer(file, &visits).expect("writing failed");
        }

        std::thread::sleep(std::time::Duration::from_millis(10_000));
    }
}

pub fn main() {
    println!("Hello world!");
    run_island_ga(
        "C:\\Users\\akselbor\\master-playground\\mirplib-rs\\t180\\LR1_1_DR1_3_VC1_V7a.json",
        100000,
    )
}
