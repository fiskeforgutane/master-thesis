use chrono::Local;
use clap::{Parser, Subcommand};
use env_logger::Builder;

use itertools::Itertools;
use log::{info, LevelFilter};

use std::io::Write;
use std::iter::once;
use std::sync::atomic::Ordering;
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
    fitness::{AtomicF64, AtomicWeighted, Weighted},
    initialization::{Empty, FromPopulation, Initialization},
    mutations::{
        rr, AddRandom, AddSmart, Bounce, BounceMode, Dedup, DedupPolicy, InterSwap, IntraSwap,
        RedCost, RemoveRandom, ReplaceNode, SwapStar, TimeSetter, Twerk, TwoOpt, TwoOptMode,
    },
    parent_selection::Tournament,
    recombinations::PIX,
    survival_selection::{Elite, Proportionate},
    Stochastic,
};
use crate::rolling_horizon::rolling_horizon::RollingHorizon;

use crate::ga::{Fitness, Mutation, ParentSelection, Recombination, SurvivalSelection};
use crate::models::exact_model::model::ExactModelSolver;
use crate::parse::*;
use crate::problem::Problem;
use crate::solution::routing::{Improvement, RoutingSolution};
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
        offset: problem.max_revenue() + 1.0,
    }
}

pub fn ga_config(
    problem: Arc<Problem>,
) -> ga::Config<
    impl ParentSelection,
    impl Recombination,
    impl Mutation,
    impl SurvivalSelection,
    impl Fitness,
> {
    let fitness = weighted(&problem);
    ga::Config {
        problem,
        population_size: 3,
        child_count: 3,
        parent_selection: Tournament::new(2).unwrap(),
        recombination: Stochastic::new(0.01, PIX),
        mutation: chain!(
            Stochastic::new(0.01, AddRandom::new()),
            Stochastic::new(0.01, RemoveRandom::new()),
            Stochastic::new(0.01, InterSwap),
            Stochastic::new(0.01, IntraSwap),
            Stochastic::new(0.01, RedCost::red_cost_mutation(10)),
            Stochastic::new(0.01, Twerk::everybody()),
            Stochastic::new(0.01, Twerk::some_random_person()),
            Stochastic::new(0.01, TwoOpt::new(TwoOptMode::IntraRandom)),
            Stochastic::new(0.01, TimeSetter::new(0.5).unwrap()),
            Stochastic::new(0.01, ReplaceNode::new(0.1)),
            Stochastic::new(0.01, SwapStar),
            Dedup(DedupPolicy::KeepFirst),
            Stochastic::new(0.01, Bounce::new(3, BounceMode::All)),
            Stochastic::new(0.01, Bounce::new(3, BounceMode::Random)),
            Stochastic::new(0.01, AddSmart),
            Stochastic::new(0.01, rr::Period::new(0.1, 0.8, 15, 3)),
            Stochastic::new(0.01, rr::Vessel::new(0.1, 0.8, 3)),
            Stochastic::new(
                0.01,
                rr::sisr::SlackInductionByStringRemoval::new(rr::sisr::Config {
                    average_removal: 2,
                    max_cardinality: 5,
                    alpha: 0.0,
                    blink_rate: 0.1,
                    first_n: 5,
                    epsilon: Improvement {
                        warp: 0,
                        approx_berth_violation: 0,
                        violation: 0.9,
                        loss: 10.0,
                    },
                })
            )
        ),
        selection: Elite(1, Proportionate(|x| 1.0 / (1.0 + x))),
        fitness,
    }
}

pub fn run_island_on<I: Initialization<Out = RoutingSolution> + Clone + Send + 'static>(
    problem: Arc<Problem>,
    _output: PathBuf,
    init: I,
    mut termination: Termination,
    config: &Config,
) -> (RoutingSolution, Vec<Vec<Vec<Vec<Visit>>>>) {
    let fitness = Arc::new(AtomicWeighted {
        warp: AtomicF64::new(0.0),
        violation: AtomicF64::new(0.0),
        revenue: AtomicF64::new(-1.0),
        cost: AtomicF64::new(1.0),
        approx_berth_violation: AtomicF64::new(0.0),
        spot: AtomicF64::new(1.0),
        offset: AtomicF64::new(problem.max_revenue() + 1.0),
    });

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
        let best = RoutingSolution::new(problem.clone(), ga.best().0.clone());

        if epochs - last_migration > config.migrate_every {
            info!("Migrating...");
            ga.migrate(1);
            info!(" DONE");
            last_migration = epochs;
        }

        info!(
            "{:>010}: F = {}. warp = {}, apx berth = {}, violation = {}, obj = {}, revenue = {}, cost = {}",
            epochs,
            fitness.of(&problem, &best),
            best.warp(),
            best.approx_berth_violation(),
            best.violation(),
            best.cost() + best.spot_cost() - best.revenue(),
            best.revenue(),
            best.cost(),
        );

        // Update the weights
        let elapsed = (std::time::Instant::now() - start).as_secs_f64();
        let d = elapsed.min(config.full_penalty_after) / config.full_penalty_after;
        fitness
            .approx_berth_violation
            .store(d * 1e8, Ordering::Relaxed);
        fitness.warp.store(d * 1e8, Ordering::Relaxed);
        fitness.violation.store(d * 1e4, Ordering::Relaxed);

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
    // Default to the normal termination
    let checkpoint_termination = match &config.checkpoint_termination {
        Some(x) => x.clone(),
        None => termination.clone(),
    };

    let mut init: Arc<Mutex<dyn Initialization<Out = RoutingSolution> + Send>> =
        Arc::new(Mutex::new(Empty));

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
        out.push(&format!("{end}.json"));
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
        #[clap(long, default_value_t = 0)]
        full_penalty_after: f64,
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
            };
            run_unfixed_rolling_horizon(Arc::new(problem), out, termination.clone(), config);
        }
        Commands::Exact => run_exact_model(&problem, termination),
    };
}
