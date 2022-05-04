use chrono::Local;
use env_logger::Builder;
use float_ord::FloatOrd;

use itertools::Itertools;
use log::{info, LevelFilter};
use rand;
use serde::Serialize;
use std::fs::OpenOptions;
use std::io::Write;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

pub mod ga;
pub mod models;
pub mod problem;
pub mod quants;
pub mod rolling_horizon;
pub mod solution;
pub mod utils;

use crate::ga::{
    chromosome::InitRoutingSolution,
    fitness::{self},
    mutations::{
        rr, AddRandom, AddSmart, Bounce, BounceMode, InterSwap, IntraSwap, RedCost, RemoveRandom,
        TimeSetter, Twerk, TwoOpt, TwoOptMode,
    },
    parent_selection,
    recombinations::PIX,
    survival_selection, Fitness, GeneticAlgorithm, Stochastic,
};
use crate::rolling_horizon::rolling_horizon::RollingHorizon;

use crate::problem::Problem;
use crate::solution::routing::RoutingSolution;
use crate::solution::Visit;

#[derive(Serialize)]
struct Config {
    pub population: usize,
    pub children: usize,
    pub add_random: f64,
    pub remove_random: f64,
    pub inter_swap: f64,
    pub intra_swap: f64,
    pub red_cost: (f64, usize),
    pub twerk_all: f64,
    pub twerk_some: f64,
    pub two_opt_intra: f64,
    pub time_setter: (f64, f64),
    pub bounce_all: (f64, usize),
    pub bounce_some: (f64, usize),
    pub add_smart: f64,
    pub rr_period: (f64, f64, f64, usize, usize),
    pub rr_vessel: (f64, f64, f64, usize),
    pub rr_sisr: (f64, rr::sisr::Config),
    pub pix: f64,
    pub threads: usize,
    pub migrate_every: u64,
}

static CONF: Config = Config {
    population: 3,
    children: 3,
    pix: 0.10,
    threads: 8,
    add_random: 0.03,
    remove_random: 0.03,
    inter_swap: 0.03,
    intra_swap: 0.03,
    red_cost: (0.03, 10),
    twerk_all: 0.03,
    twerk_some: 0.03,
    two_opt_intra: 0.03,
    time_setter: (0.04, 0.4),
    bounce_all: (0.03, 3),
    bounce_some: (0.03, 3),
    add_smart: 0.03,
    rr_period: (0.01, 0.1, 0.5, 15, 3),
    rr_vessel: (0.01, 0.1, 0.75, 3),
    rr_sisr: (
        0.01,
        rr::sisr::Config {
            average_removal: 2,
            max_cardinality: 5,
            alpha: 0.0,
            blink_rate: 0.1,
            first_n: 5,
            epsilon: (0.9, 10.0),
        },
    ),
    migrate_every: 500,
};

pub fn read_problem(path: &Path) -> Arc<Problem> {
    let file = std::fs::File::open(path).unwrap();
    let reader = std::io::BufReader::new(file);
    let problem: Problem = serde_json::from_reader(reader).unwrap();
    Arc::new(problem)
}

pub fn run_island_on(
    problem: Arc<Problem>,
    mut output: PathBuf,
    termination: Termination,
    write: bool,
) -> RoutingSolution {
    let closure_problem = problem.clone();
    let fitness = fitness::Weighted {
        warp: 1e8,
        violation: 1e4,
        revenue: -1.0,
        cost: 1.0,
    };

    let config = move || ga::Config {
        problem: closure_problem.clone(),
        population_size: CONF.population,
        child_count: CONF.children,
        parent_selection: parent_selection::Tournament::new(3).unwrap(),
        recombination: Stochastic::new(CONF.pix, PIX),
        mutation: chain!(
            Stochastic::new(CONF.add_random, AddRandom::new()),
            Stochastic::new(CONF.remove_random, RemoveRandom::new()),
            Stochastic::new(CONF.inter_swap, InterSwap),
            Stochastic::new(CONF.intra_swap, IntraSwap),
            Stochastic::new(CONF.red_cost.0, RedCost::red_cost_mutation(CONF.red_cost.1)),
            Stochastic::new(CONF.twerk_all, Twerk::everybody()),
            Stochastic::new(CONF.twerk_some, Twerk::some_random_person()),
            Stochastic::new(CONF.two_opt_intra, TwoOpt::new(TwoOptMode::IntraRandom)),
            Stochastic::new(
                CONF.time_setter.0,
                TimeSetter::new(CONF.time_setter.1).unwrap()
            ),
            Stochastic::new(
                CONF.bounce_all.0,
                Bounce::new(CONF.bounce_all.1, BounceMode::All)
            ),
            Stochastic::new(
                CONF.bounce_some.0,
                Bounce::new(CONF.bounce_some.1, BounceMode::Random)
            ),
            Stochastic::new(CONF.add_smart, AddSmart),
            Stochastic::new(
                CONF.rr_period.0,
                rr::Period::new(
                    CONF.rr_period.1,
                    CONF.rr_period.2,
                    CONF.rr_period.3,
                    CONF.rr_period.4
                )
            ),
            Stochastic::new(
                CONF.rr_vessel.0,
                rr::Vessel::new(CONF.rr_vessel.1, CONF.rr_vessel.2, CONF.rr_vessel.3)
            ),
            Stochastic::new(
                CONF.rr_sisr.0,
                rr::sisr::SlackInductionByStringRemoval::new(CONF.rr_sisr.1)
            )
        ),
        selection: survival_selection::Elite(
            1,
            survival_selection::Proportionate(|x| 1.0 / (1.0 + x)),
        ),
        fitness,
    };

    let mut ga = ga::islands::IslandGA::new(InitRoutingSolution, config, CONF.threads);
    if write {
        output.push("config.json");
        let file = std::fs::File::create(&output).unwrap();
        output.pop();
        serde_json::to_writer(file, &CONF).expect("writing failed");
    }

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

        if epochs - last_migration > CONF.migrate_every {
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

        if write {
            output.push(&format!("{}.json", epochs));
            let file = std::fs::File::create(&output).unwrap();
            output.pop();

            let visits = best.to_vec();
            serde_json::to_writer(file, &visits).expect("writing failed");
        }

        // Whether we should terminate now
        let terminate = match termination {
            Termination::NoViolation => best.violation() <= 1e-3,
            Termination::Never => false,
            Termination::Epochs(x) => epochs > x,
        };

        if terminate {
            return best;
        }

        std::thread::sleep(std::time::Duration::from_millis(10_000));
    }
}

pub fn run_island_ga(path: &Path, output: PathBuf, termination: Termination, write: bool) {
    let problem = read_problem(path);

    let start = std::time::Instant::now();
    run_island_on(problem, output, termination, write);
    let end = std::time::Instant::now();

    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("time.txt")
        .expect("failed to create times.txt");

    writeln!(file, "{}: {} ms", path.display(), (end - start).as_millis());
}

pub fn run_unfixed_rolling_horizon(
    path: &Path,
    mut output: PathBuf,
    termination: Termination,
    subproblem_size: usize,
    step_length: usize,
) {
    let main_problem = read_problem(path);
    let main_closure_problem = main_problem.clone();

    let num_subproblems = f64::ceil(
        ((*main_problem).timesteps() as f64 - subproblem_size as f64) / step_length as f64 + 1.0,
    ) as usize;

    let rh = RollingHorizon::new(main_problem);

    let mut period = 0..subproblem_size;
    let mut solutions = Vec::new();
    for i in 0..num_subproblems {
        println!("Solving subproblem in range: {:?}", period);

        let sub_problem = rh
            .slice_problem(period)
            .expect("Failed to create subproblem");

        let sub_problem = Arc::new(sub_problem);

        let best = run_island_on(sub_problem, output.clone(), termination.clone(), false);

        period = 0..(subproblem_size + (i + 1) * step_length).min(main_closure_problem.timesteps());

        solutions.push(best);
    }

    // convert all solutions into one
    let mut routes = (0..main_closure_problem.vessels().len())
        .map(|_| Vec::new())
        .collect::<Vec<_>>();
    for (i, solution) in solutions.iter().enumerate() {
        let range = i * step_length..(subproblem_size + (i + 1) * step_length);
        solution.iter().enumerate().for_each(|(v, plan)| {
            plan.iter().for_each(|visit| {
                if range.contains(&visit.time) {
                    routes[v].push(visit.clone());
                }
            })
        })
    }
    let final_sol = solutions.iter().last().unwrap();

    output.push(&format!("final_rh.json"));
    let file = std::fs::File::create(&output).unwrap();
    output.pop();

    let visits = final_sol.to_vec();
    serde_json::to_writer(file, &visits).expect("writing failed");
}

pub fn run_rolling_horizon(
    path: &Path,
    mut output: PathBuf,
    termination: Termination,
    subproblem_size: usize,
    step_length: usize,
) {
    let main_problem = read_problem(path);
    let main_closure_problem = main_problem.clone();

    let num_subproblems = f64::ceil(
        ((*main_problem).timesteps() as f64 - subproblem_size as f64) / step_length as f64 + 1.0,
    ) as usize;

    let rh = RollingHorizon::new(main_problem);

    let mut initial_loads = main_closure_problem
        .vessels()
        .iter()
        .map(|v| v.initial_inventory().clone())
        .collect();
    let mut origins = main_closure_problem
        .vessels()
        .iter()
        .map(|v| v.origin())
        .collect();
    let mut available_from = main_closure_problem
        .vessels()
        .iter()
        .map(|v| v.available_from())
        .collect();
    let mut initial_inventory = main_closure_problem
        .nodes()
        .iter()
        .map(|n| n.initial_inventory().clone())
        .collect();
    let mut period = 0..subproblem_size;

    let mut solutions = Vec::new();
    for i in 0..num_subproblems {
        println!("Solving subproblem in range: {:?}", period);

        let sub_problem = rh
            .create_subproblem(
                initial_loads,
                origins,
                available_from,
                initial_inventory,
                period,
            )
            .expect("Failed to create subproblem");

        let sub_problem = Arc::new(sub_problem);
        let closure_problem = sub_problem.clone();

        let best = run_island_on(sub_problem, output.clone(), termination.clone(), false);

        origins = (0..closure_problem.vessels().len())
            .map(|v| best.next_position(v, step_length - 1).0)
            .collect();

        available_from = (0..closure_problem.vessels().len())
            .map(|v| step_length.max(best.next_position(v, step_length).1) - step_length)
            .collect();

        initial_loads = (0..closure_problem.vessels().len())
            .map(|v| best.load_at(v, step_length + available_from[v]))
            .collect();

        initial_inventory = (0..closure_problem.nodes().len())
            .map(|n| best.inventory_at(n, step_length))
            .collect();
        period = (i + 1) * step_length
            ..(subproblem_size + (i + 1) * step_length).min(main_closure_problem.timesteps());

        solutions.push(best);
    }

    // convert all solutions into one
    let mut routes = (0..main_closure_problem.vessels().len())
        .map(|_| Vec::new())
        .collect::<Vec<_>>();
    for (i, solution) in solutions.iter().enumerate() {
        let range = i * step_length..(subproblem_size + (i + 1) * step_length);
        solution.iter().enumerate().for_each(|(v, plan)| {
            plan.iter().for_each(|visit| {
                if range.contains(&visit.time) {
                    routes[v].push(visit.clone());
                }
            })
        })
    }
    routes.iter().for_each(|r| {
        r.iter().dedup_by(|x, y| x.node == y.node);
    });

    let final_sol = RoutingSolution::new(main_closure_problem, routes);

    output.push(&format!("final_rh.json"));
    let file = std::fs::File::create(&output).unwrap();
    output.pop();

    let visits: Vec<&[Visit]> = final_sol.iter().map(|plan| &plan[..]).collect();
    serde_json::to_writer(file, &visits).expect("writing failed");
}

#[derive(Clone)]
pub enum Termination {
    Epochs(u64),
    NoViolation,
    Never,
}

pub fn main() {
    //env_logger::init();

    /*     Builder::new()
        .format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] - {}",
                Local::now().format("%Y-%m-%dT%H:%M:%S"),
                record.level(),
                record.args()
            )
        })
        .filter(None, LevelFilter::Trace)
        .init();
    info!("test"); */

    let args = std::env::args().collect::<Vec<_>>();
    let path = std::path::Path::new(&args[1]);

    println!("Problem path: {:?}", path);

    let problem_name = path.file_stem().unwrap().to_str().unwrap();
    let directory = path.parent().unwrap();
    let timesteps = directory.file_stem().unwrap().to_str().unwrap();

    // The output directory is ./solutions/TIME/PROBLEM/,
    let mut out = std::env::current_dir().expect("unable to fetch current dir");
    out.push("solutions");
    out.push(timesteps);
    out.push(problem_name);

    // Create the output directory
    std::fs::create_dir_all(&out).expect("failed to create out dir");

    // Run the GA.
    run_rolling_horizon(path, out, Termination::NoViolation, 30, 5);
    //run_island_ga(path, out, Termination::NoViolation, true);
}
