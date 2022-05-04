use chrono::Local;
use clap::Parser;
use env_logger::Builder;

use log::LevelFilter;
use serde::{Deserialize, Serialize};
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
    survival_selection, Fitness, Stochastic,
};

use crate::problem::Problem;
use crate::solution::routing::RoutingSolution;
use crate::solution::Visit;

#[derive(Serialize, Deserialize)]
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

impl Default for Config {
    fn default() -> Self {
        Config {
            population: 100,
            children: 100,
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
        }
    }
}

fn run_island_ga(path: &Path, mut output: PathBuf, mut termination: Termination, conf: Config) {
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
        population_size: conf.population,
        child_count: conf.children,
        parent_selection: parent_selection::Tournament::new(3).unwrap(),
        recombination: Stochastic::new(conf.pix, PIX),
        mutation: chain!(
            Stochastic::new(conf.add_random, AddRandom::new()),
            Stochastic::new(conf.remove_random, RemoveRandom::new()),
            Stochastic::new(conf.inter_swap, InterSwap),
            Stochastic::new(conf.intra_swap, IntraSwap),
            Stochastic::new(conf.red_cost.0, RedCost::red_cost_mutation(conf.red_cost.1)),
            Stochastic::new(conf.twerk_all, Twerk::everybody()),
            Stochastic::new(conf.twerk_some, Twerk::some_random_person()),
            Stochastic::new(conf.two_opt_intra, TwoOpt::new(TwoOptMode::IntraRandom)),
            Stochastic::new(
                conf.time_setter.0,
                TimeSetter::new(conf.time_setter.1).unwrap()
            ),
            Stochastic::new(
                conf.bounce_all.0,
                Bounce::new(conf.bounce_all.1, BounceMode::All)
            ),
            Stochastic::new(
                conf.bounce_some.0,
                Bounce::new(conf.bounce_some.1, BounceMode::Random)
            ),
            Stochastic::new(conf.add_smart, AddSmart),
            Stochastic::new(
                conf.rr_period.0,
                rr::Period::new(
                    conf.rr_period.1,
                    conf.rr_period.2,
                    conf.rr_period.3,
                    conf.rr_period.4
                )
            ),
            Stochastic::new(
                conf.rr_vessel.0,
                rr::Vessel::new(conf.rr_vessel.1, conf.rr_vessel.2, conf.rr_vessel.3)
            ),
            Stochastic::new(
                conf.rr_sisr.0,
                rr::sisr::SlackInductionByStringRemoval::new(conf.rr_sisr.1)
            )
        ),
        selection: survival_selection::Elite(
            1,
            survival_selection::Proportionate(|x| 1.0 / (1.0 + x)),
        ),
        fitness,
    };

    let mut ga = ga::islands::IslandGA::new(InitRoutingSolution, config, conf.threads);

    output.push("config.json");
    let file = std::fs::File::create(&output).unwrap();
    output.pop();
    serde_json::to_writer(file, &conf).expect("writing failed");

    let mut last_migration = 0;
    let start = std::time::Instant::now();
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

        if epochs - last_migration > conf.migrate_every {
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
        if termination.should_terminate(epochs, &best, fitness.of(&problem, &best)) {
            break;
        }

        std::thread::sleep(std::time::Duration::from_millis(10_000));
    }
    let end = std::time::Instant::now();

    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("time.txt")
        .expect("failed to create times.txt");

    writeln!(file, "{}: {} ms", path.display(), (end - start).as_millis())
        .expect("writing 'time' failed");
}

#[derive(Debug)]
pub enum Termination {
    /// Terminate after a given number of epochs
    Epochs(u64),
    /// Terminate upon finding a solution with no violation
    NoViolation,
    /// Terminate if there has been no improvement for the given amount of time
    NoImprovement(std::time::Instant, std::time::Duration, f64),
    /// Maximum running time from `Instant`
    Timeout(std::time::Instant, std::time::Duration),
    /// Run forever
    Never,
    /// Terminate if either of the two termination criteria
    /// tells it to terminate
    Any(Box<Termination>, Box<Termination>),
    /// Terminate when both of the criteria tells it to terminate
    All(Box<Termination>, Box<Termination>),
}

impl Termination {
    pub fn should_terminate(
        &mut self,
        epoch: u64,
        solution: &RoutingSolution,
        fitness: f64,
    ) -> bool {
        match self {
            Termination::Timeout(from, duration) => (std::time::Instant::now() - *from) > *duration,
            Termination::Epochs(e) => *e > epoch,
            Termination::NoViolation => solution.warp() == 0 && solution.violation() < 1e-3,
            Termination::Never => false,
            Termination::Any(one, two) => {
                one.should_terminate(epoch, solution, fitness)
                    || two.should_terminate(epoch, solution, fitness)
            }
            Termination::All(one, two) => {
                one.should_terminate(epoch, solution, fitness)
                    && two.should_terminate(epoch, solution, fitness)
            }
            Termination::NoImprovement(last, duration, best) => {
                // Replace best fitness and reset time of last improvement if the current solution is better than the incumbent.
                if fitness < *best {
                    *best = fitness;
                    *last = std::time::Instant::now();
                }

                (std::time::Instant::now() - *last) > *duration
            }
        }
    }
}

#[derive(Parser, Debug)]
#[clap(author = "Fiskefôrgutane", about = "CLI for MIRP solver")]
struct Args {
    /// Sets the configuration file used.
    #[clap(short, long, parse(from_os_str), value_name = "FILE")]
    config: Option<PathBuf>,
    /// Path to problem specification
    #[clap(short, long, parse(from_os_str), value_name = "FILE")]
    problem: PathBuf,
    /// What logging level to enable
    log: Option<String>,
    /// The timeout used when the solution is stuck. The elapsed time will be reset each
    /// time a better solution if found
    #[clap(short, long, default_value_t = 3600)]
    stuck_timeout: u64,
}

fn enable_logger(level: LevelFilter) {
    env_logger::init();

    Builder::new()
        .format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] - {}",
                Local::now().format("%Y-%m-%dT%H:%M:%S"),
                record.level(),
                record.args()
            )
        })
        .filter(None, level)
        .init();
}

pub fn main() {
    // Parse command line arguments
    let args = Args::parse();

    // Convert the log level to a LevelFilter
    let level = match args.log.as_deref() {
        Some("debug") => LevelFilter::Debug,
        Some("trace") => LevelFilter::Trace,
        Some("info") => LevelFilter::Info,
        Some("warn") => LevelFilter::Warn,
        Some("off") => LevelFilter::Off,
        Some("error") => LevelFilter::Error,
        _ => LevelFilter::Off,
    };

    // Enable logging at the specified level.
    enable_logger(level);
    println!("Logger level: {level:?}");

    let config = args
        .config
        .map(|path| {
            serde_json::from_reader::<_, Config>(
                std::fs::File::open(path).expect("failed to open config file"),
            )
            .expect("invalid config file")
        })
        .unwrap_or_default();

    println!("Problem path: {:?}", args.problem.as_path());

    let path = &args.problem;
    let problem_name = path.file_stem().unwrap().to_str().unwrap();
    let directory = path.parent().unwrap();
    let timesteps = directory.file_stem().unwrap().to_str().unwrap();

    // The termination criteria
    let termination = Termination::NoImprovement(
        std::time::Instant::now(),
        std::time::Duration::from_secs(args.stuck_timeout),
        std::f64::INFINITY,
    );

    // The output directory is ./solutions/TIME/PROBLEM/,
    let mut out = std::env::current_dir().expect("unable to fetch current dir");
    out.push("solutions");
    out.push(timesteps);
    out.push(problem_name);

    // Create the output directory
    std::fs::create_dir_all(&out).expect("failed to create out dir");

    // Run the GA.
    run_island_ga(path, out, termination, config);
}
