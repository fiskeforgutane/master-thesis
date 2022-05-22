use chrono::Local;
use clap::{Parser, Subcommand};
use env_logger::Builder;

use itertools::Itertools;
use log::{info, LevelFilter};
use uuid::Uuid;

use std::io::Write;
use std::iter::once;
use std::sync::Mutex;
use std::{path::PathBuf, sync::Arc};

pub mod ga;
pub mod models;
pub mod parse;
pub mod problem;
pub mod quants;
pub mod rolling_horizon;
pub mod solution;
pub mod termination;
pub mod utils;

use crate::ga::{
    fitness::Weighted,
    initialization::{Empty, FromPopulation, Initialization},
    parent_selection::Tournament,
    recombinations::PIX,
    survival_selection::{Elite, Proportionate},
    Stochastic,
};
use crate::rolling_horizon::rolling_horizon::RollingHorizon;

use crate::ga::Fitness;
use crate::models::exact_model::model::ExactModelSolver;
use crate::parse::*;
use crate::problem::Problem;
use crate::solution::routing::RoutingSolution;
use crate::solution::Visit;
use crate::termination::Termination;

pub fn weighted(problem: &Problem) -> Weighted {
    Weighted {
        warp: 1e8,
        violation: 1e4,
        revenue: -1.0,
        cost: 1.0,
        approx_berth_violation: 1e8,
        spot: 1.0,
        travel_empty: 1e5,
        travel_at_cap: 1e5,
        offset: problem.max_revenue() + 1.0,
    }
}

pub fn run_island_on<I: Initialization + Clone + Send + 'static>(
    problem: Arc<Problem>,
    _output: PathBuf,
    init: I,
    mut termination: Termination,
    config: &Config,
) -> (RoutingSolution, Vec<Vec<Vec<Vec<Visit>>>>) {
    let mut fitness = Weighted {
        warp: 0.0,
        violation: 0.0,
        revenue: -1.0,
        cost: 1.0,
        approx_berth_violation: 0.0,
        spot: 1.0,
        offset: problem.max_revenue() + 1.0,
        travel_empty: 0.0,
        travel_at_cap: 0.0,
    };

    let closure_fitness = fitness.clone();
    let closure_problem = problem.clone();
    let population = config.population;
    let children = config.children;
    let tournament = config.tournament;
    let mutation = format!("{}", config.mutation);
    let threads = config.threads;

    let mut ga = ga::islands::IslandGA::new(
        init,
        move || ga::Config {
            problem: closure_problem.clone(),
            population_size: population,
            child_count: children,
            parent_selection: Tournament::new(tournament).unwrap(),
            recombination: Stochastic::new(0.1, PIX),
            mutation: Box::<dyn RPNMutation>::try_from(mutation.as_str()).unwrap(),
            selection: Elite(1, Proportionate(|x| 1.0 / (1.0 + x))),
            fitness: closure_fitness.clone(),
        },
        threads,
    );

    let mut last_migration = 0;
    let start = std::time::Instant::now();
    loop {
        let epochs = ga.epochs();
        let best = ga.best();

        if epochs - last_migration > config.migrate_every {
            info!("Migrating...");
            ga.migrate(1);
            info!(" DONE");
            last_migration = epochs;
        }

        info!(
            "{:>010}: F = {}. warp = {}, apx berth = {}, violation = {}, spot: {}, obj = {}, revenue = {}, cost = {}, travel empty: {}, travel at cap: {}",
            epochs,
            fitness.of(&problem, &best),
            best.warp(),
            best.approx_berth_violation(),
            best.violation(),
            best.spot_cost(),
            best.cost() + best.spot_cost() - best.revenue(),
            best.revenue(),
            best.cost(),
            best.travel_empty(),
            best.travel_at_cap(),

        );

        // Update the weights
        let elapsed = (std::time::Instant::now() - start).as_secs_f64();
        let d = elapsed.min(config.full_penalty_after) / config.full_penalty_after;
        fitness.approx_berth_violation = d * config.approx_berth_violation;
        fitness.warp = d * config.warp;
        fitness.violation = d * config.violation;
        fitness.travel_at_cap = d * config.travel_at_cap;
        fitness.travel_empty = d * config.travel_empty;
        // Update the fitness functions of the islands
        ga.set_fitness(fitness.clone());

        if termination.should_terminate(epochs, &best, fitness.of(&problem, &best)) {
            return (best, ga.populations());
        }

        std::thread::sleep(std::time::Duration::from_millis(config.loop_delay));
    }
}

pub fn run_exact_model(problem: &Problem, termination: Termination) {
    let timeout = match termination {
        Termination::Timeout(_, y) => y.as_secs() as f64,
        t => panic!("termination criterion for exact model must be a timeout, not {t:?}"),
    };

    let _ = ExactModelSolver::solve(&problem, timeout);
}

pub fn run_unfixed_rolling_horizon(
    problem: Arc<Problem>,
    mut out: PathBuf,
    termination: Termination,
    config: Config,
) {
    let rh = RollingHorizon::new(problem.clone());
    let uuid = config.uuid.hyphenated().to_string();
    // Default to the normal termination
    let checkpoint_termination = match &config.checkpoint_termination {
        Some(x) => x.clone(),
        None => termination.clone(),
    };

    let mut init: Arc<Mutex<dyn Initialization + Send>> = Arc::new(Mutex::new(Empty));

    // The "normal" ends, i.e. (start, start + step, ...)
    let start = config.subproblem_size;
    let step = config.step_length;
    let normal = (start..problem.timesteps()).step_by(step);
    // Our `ends` will consist of both the normal ones and the checkpoints, plus the full problem size
    let ends = normal
        .chain(config.checkpoints.iter().cloned())
        .chain(once(problem.timesteps()))
        .sorted()
        .dedup()
        .collect::<Vec<_>>();

    info!("Checkpoints: {:?}", config.checkpoints);
    info!("Checkpoint termination: {checkpoint_termination}");
    info!("RH: {ends:?}");

    for end in ends {
        let period = 0..end;
        info!("Solving subproblem in range: {:?}", period);

        let sub = Arc::new(rh.slice_problem(period).unwrap());

        // We use a different termination criterion for the checkpoints
        let term = match config.checkpoints.contains(&end) {
            true => checkpoint_termination.clone(),
            false => termination.clone(),
        };

        let (best, pop) = run_island_on(sub.clone(), out.clone(), init, term, &config);

        init = Arc::new(Mutex::new(FromPopulation::new(
            pop.into_iter()
                .flatten()
                .map(move |routes| RoutingSolution::new(sub.clone(), routes))
                .collect(),
        )));

        // Write the best at the end of the solve of the subproblem
        out.push(&format!("{uuid}-{end}.json"));
        let file = std::fs::File::create(&out).unwrap();
        serde_json::to_writer(file, &best.to_vec()).expect("writing failed");
        out.pop();
    }
}

#[derive(Parser)]
#[clap(author = "Fiskefôrgutane", about = "CLI for MIRP solver")]
struct Args {
    /// Path to problem specification
    #[clap(short, long, parse(from_str = read_problem), value_name = "FILE")]
    problem: ProblemFromFile,
    /// What logging level to enable.
    #[clap(
        short,
        long,
        value_name = "off | trace | debug | info | warn | error",
        default_value = "off"
    )]
    log: String,

    /// The termination criteria used
    #[clap(
        short,
        long,
        value_name = "never | <count> epochs | <secs> timeout | no-violation | <secs> no-improvement  | '<term> <term> |' | '<term> <term> &'",
        default_value = "600 no-improvement",
        parse(try_from_str = Termination::try_from)
    )]
    termination: Termination,

    /// Subcommands
    #[clap(subcommand)]
    commands: Commands,
}

pub struct Config {
    pub subproblem_size: usize,
    pub step_length: usize,
    pub checkpoint_termination: Option<Termination>,
    pub checkpoints: Vec<usize>,
    pub population: usize,
    pub children: usize,
    pub tournament: usize,
    pub mutation: Box<dyn RPNMutation>,
    pub migrate_every: u64,
    pub loop_delay: u64,
    pub threads: usize,
    pub full_penalty_after: f64,
    pub uuid: Uuid,
    pub approx_berth_violation: f64,
    pub warp: f64,
    pub violation: f64,
    pub travel_at_cap: f64,
    pub travel_empty: f64,
}

#[derive(Subcommand)]
enum Commands {
    RollingHorizon {
        /// The mutation used
        #[clap(
        short, long,
        default_value = "std",
        parse(try_from_str = Box::<dyn RPNMutation>::try_from)
        )]
        mutation: Box<dyn RPNMutation>,
        /// The waiting time between each iteration of the print loop, in ms
        #[clap(short, long, default_value_t = 5000)]
        loop_delay: u64,
        #[clap(long, default_value_t = 30)]
        subproblem_size: usize,
        #[clap(long, default_value_t = 5)]
        step_length: usize,
        #[clap(long, parse(try_from_str = Termination::try_from))]
        checkpoint_termination: Option<Termination>,
        #[clap(long)]
        checkpoints: Vec<usize>,
        #[clap(long, default_value_t = 3)]
        population: usize,
        #[clap(long, default_value_t = 3)]
        children: usize,
        #[clap(long, default_value_t = 2)]
        tournament: usize,
        #[clap(long, default_value_t = 100)]
        migrate_every: u64,
        #[clap(long, default_value_t = -2)]
        threads: i64,
        #[clap(long, default_value_t = -1.0)]
        full_penalty_after: f64,
        #[clap(long, default_value_t = 1e8)]
        approx_berth_violation: f64,
        #[clap(long, default_value_t = 1e8)]
        warp: f64,
        #[clap(long, default_value_t = 1e4)]
        violation: f64,
        #[clap(long, default_value_t = 1e4)]
        travel_at_cap: f64,
        #[clap(long, default_value_t = 1e4)]
        travel_empty: f64,
    },
    Exact,
}

fn enable_logger(level: LevelFilter) {
    Builder::new()
        .format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] - {}",
                Local::now().format("%Y-%m-%dT%H:%M:%S%.3f"),
                record.level(),
                record.args()
            )
        })
        .filter(None, level)
        .init();
}

pub fn main() {
    // Parse command line arguments
    let Args {
        problem,
        log,
        termination,
        commands,
    } = Args::parse();

    let ProblemFromFile {
        name,
        timesteps,
        problem,
    } = problem;

    // Create a UUID identifying this run
    // This is used to bind together the log entry and the solution
    let uuid = Uuid::new_v4();

    // Convert the log level to a LevelFilter
    let level = match log.as_str() {
        "debug" => LevelFilter::Debug,
        "trace" => LevelFilter::Trace,
        "info" => LevelFilter::Info,
        "warn" => LevelFilter::Warn,
        "off" => LevelFilter::Off,
        "error" => LevelFilter::Error,
        _ => LevelFilter::Off,
    };

    // Enable logging at the specified level.
    enable_logger(level);

    info!("-------- NEW RUN --------");
    info!("{:?}", std::env::args());
    info!("uuid: {}", uuid.hyphenated().to_string());
    info!("logger level: {level:?}");
    info!("problem: {name}");
    info!("timesteps: {timesteps}");
    info!("termination: {termination}");

    // The output directory is ./solutions/TIME/PROBLEM/,
    let mut out = std::env::current_dir().expect("unable to fetch current dir");
    out.push("solutions");
    out.push(timesteps);
    out.push(name);

    // Create the output directory
    std::fs::create_dir_all(&out).expect("failed to create out dir");

    // Run the GA.
    match commands {
        Commands::RollingHorizon {
            subproblem_size,
            step_length,
            checkpoint_termination,
            checkpoints,
            population,
            children,
            tournament,
            migrate_every,
            mutation,
            loop_delay,
            threads,
            full_penalty_after,
            approx_berth_violation,
            warp,
            violation,
            travel_at_cap,
            travel_empty,
        } => {
            let threads = match threads {
                -2 => std::thread::available_parallelism().unwrap().get() / 2,
                -1 => std::thread::available_parallelism().unwrap().get(),
                _ => threads.max(1) as usize,
            };

            info!("mutation:    {mutation}");
            info!("threads:     {threads}");

            let config = Config {
                subproblem_size,
                step_length,
                checkpoint_termination,
                checkpoints,
                population,
                children,
                tournament,
                mutation,
                migrate_every,
                loop_delay,
                threads,
                full_penalty_after,
                uuid,
                approx_berth_violation,
                warp,
                violation,
                travel_at_cap,
                travel_empty,
            };
            run_unfixed_rolling_horizon(Arc::new(problem), out, termination.clone(), config);
        }
        Commands::Exact => run_exact_model(&problem, termination),
    };
}
