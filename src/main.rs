use float_ord::FloatOrd;
use rand;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
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
            "F = {}. warp = {}, violation = {}, revenue = {}, cost = {}; (worst fitness = {})",
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

pub fn run_island_ga(path: &Path, mut output: PathBuf, termination: Termination) {
    let file = std::fs::File::open(path).unwrap();
    let reader = std::io::BufReader::new(file);
    let problem: Problem = serde_json::from_reader(reader).unwrap();
    let problem = Arc::new(problem);
    let closure_problem = problem.clone();

    let fitness = fitness::Weighted {
        warp: 1e8,
        violation: 1e4,
        revenue: -1.0,
        cost: 1.0,
    };

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
        fitness,
    };

    let mut ga = ga::islands::IslandGA::new(InitRoutingSolution, config, 8);

    let mut last_migration = 0;

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

        output.push(&format!("{}.json", epochs));
        let file = std::fs::File::create(&output).unwrap();
        output.pop();

        let visits: Vec<&[Visit]> = best.iter().map(|plan| &plan[..]).collect();
        serde_json::to_writer(file, &visits).expect("writing failed");

        // Whether we should terminate now
        let terminate = match termination {
            Termination::NoViolation => best.violation() <= 1e-3,
            Termination::Never => false,
            Termination::Epochs(x) => epochs > x,
        };

        if terminate {
            return;
        }

        std::thread::sleep(std::time::Duration::from_millis(10_000));
    }
}

pub enum Termination {
    Epochs(u64),
    NoViolation,
    Never,
}

pub fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let path = std::path::Path::new(&args[1]);

    println!("Problem path: {}", path);

    let problem_name = path.file_stem().unwrap().to_str().unwrap();
    let directory = path.parent().unwrap();
    let timesteps = directory.file_stem().unwrap().to_str().unwrap();

    // The output directory is ./solutions/TIME/PROBLEM/,
    let mut out = directory.to_path_buf();
    out.push("solutions");
    out.push(timesteps);
    out.push(problem_name);

    // Create the output directory
    std::fs::create_dir_all(&out).expect("failed to create out dir");

    // Run the GA.
    run_island_ga(path, out, Termination::NoViolation);
}
