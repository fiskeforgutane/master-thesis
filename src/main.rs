use chrono::Local;
use clap::{Parser, Subcommand};
use env_logger::Builder;

use itertools::Itertools;
use log::LevelFilter;

use derive_more::{Display, Error};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::iter::once;
use std::sync::Mutex;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

pub mod ga;
pub mod models;
pub mod parse;
pub mod problem;
pub mod quants;
pub mod rolling_horizon;
pub mod solution;
pub mod utils;

use crate::ga::{
    chromosome::InitRoutingSolution,
    fitness::{self, Weighted},
    initialization::{FromPopulation, Initialization},
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
use crate::solution::routing::RoutingSolution;
use crate::solution::Visit;

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
                    epsilon: (0.9, 10.0),
                })
            )
        ),
        selection: Elite(1, Proportionate(|x| 1.0 / (1.0 + x))),
        fitness,
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Config {
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

pub fn read_problem(path: &Path) -> Arc<Problem> {
    let file = std::fs::File::open(path).unwrap();
    let reader = std::io::BufReader::new(file);
    let problem: Problem = serde_json::from_reader(reader).unwrap();
    Arc::new(problem)
}

pub fn run_island_on<I: Initialization<Out = RoutingSolution> + Clone + Send + 'static>(
    problem: Arc<Problem>,
    mut output: PathBuf,
    init: I,
    mut termination: Termination,
    conf: Config,
    write: bool,
) -> (RoutingSolution, Vec<Vec<Vec<Vec<Visit>>>>) {
    let closure_problem = problem.clone();
    let fitness = fitness::Weighted {
        warp: 1e8,
        violation: 1e4,
        revenue: -1.0,
        cost: 1.0,
        approx_berth_violation: 1e8,
        spot: 1.0,
        offset: problem.max_revenue() + 1.0,
    };

    let config = move || ga::Config {
        problem: closure_problem.clone(),
        population_size: conf.population,
        child_count: conf.children,
        parent_selection: Tournament::new(3).unwrap(),
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
        selection: Elite(1, Proportionate(|x| 1.0 / (1.0 + x))),
        fitness,
    };

    let mut ga = ga::islands::IslandGA::new(init, config, conf.threads);
    if write {
        output.push("config.json");
        let file = std::fs::File::create(&output).unwrap();
        output.pop();
        serde_json::to_writer(file, &conf).expect("writing failed");
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

        if epochs - last_migration > conf.migrate_every {
            print!("Migrating...");
            ga.migrate(5);
            println!(" DONE");
            last_migration = epochs;
        }

        println!(
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

        if write {
            output.push(&format!("{}.json", epochs));
            let file = std::fs::File::create(&output).unwrap();
            output.pop();

            let visits = best.to_vec();
            serde_json::to_writer(file, &visits).expect("writing failed");
        }

        if termination.should_terminate(epochs, &best, fitness.of(&problem, &best)) {
            return (best, ga.populations());
        }

        std::thread::sleep(std::time::Duration::from_millis(10_000));
    }
}

pub fn run_exact_model(path: &Path, mut _output: PathBuf, termination: Termination) {
    assert!(
        matches!(termination, Termination::Timeout { .. }),
        "termination criterion for exact model must be a timeout, not {:?}",
        termination
    );
    let timeout = match termination {
        Termination::Timeout(_, y) => Some(y.as_secs() as f64),
        _ => None,
    }
    .unwrap();
    let file = std::fs::File::open(path).unwrap();
    let reader = std::io::BufReader::new(file);
    let problem: Problem = serde_json::from_reader(reader).unwrap();
    let problem = Arc::new(problem);

    let _ = ExactModelSolver::solve(&problem, timeout);
}

pub fn run_island_ga<I: Initialization<Out = RoutingSolution> + Clone + Send + 'static>(
    path: &Path,
    output: PathBuf,
    termination: Termination,
    init: I,
    conf: Config,
    write: bool,
) {
    let problem = read_problem(path);

    let start = std::time::Instant::now();
    run_island_on(problem, output, init, termination, conf.clone(), write);

    let end = std::time::Instant::now();

    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("time.txt")
        .expect("failed to create times.txt");

    writeln!(file, "{}: {} ms", path.display(), (end - start).as_millis())
        .expect("writing 'time' failed");
}

pub fn run_unfixed_rolling_horizon(
    path: &Path,
    mut out: PathBuf,
    termination: Termination,
    subproblem_size: usize,
    step_length: usize,
    conf: Config,
    write: bool,
    checkpoints: Vec<usize>,
    checkpoint_termination: Termination,
) {
    println!("Checkpoints: {checkpoints:?}");
    println!("Checkpoint termination: {checkpoint_termination}");
    let problem = read_problem(path);
    let rh = RollingHorizon::new(problem.clone());

    let mut init: Arc<Mutex<dyn Initialization<Out = RoutingSolution> + Send>> =
        Arc::new(Mutex::new(InitRoutingSolution));

    // The "normal" ends, i.e. (start, start + step, ...)
    let normal = (subproblem_size..problem.timesteps()).step_by(step_length);
    // Our `ends` will consist of both the normal ones and the checkpoints, plus the full problem size
    let ends = normal
        .chain(checkpoints.iter().cloned())
        .chain(once(problem.timesteps()))
        .sorted()
        .dedup()
        .collect::<Vec<_>>();

    println!("RH: {ends:?}");

    for end in ends {
        let period = 0..end;
        println!("Solving subproblem in range: {:?}", period);

        let sub = Arc::new(rh.slice_problem(period).unwrap());

        // We use a different termination criterion for the checkpoints
        let term = match checkpoints.contains(&end) {
            true => checkpoint_termination.clone(),
            false => termination.clone(),
        };

        let (best, pop) = run_island_on(sub.clone(), out.clone(), init, term, conf.clone(), false);

        init = Arc::new(Mutex::new(FromPopulation::new(
            pop.into_iter()
                .flatten()
                .map(move |routes| RoutingSolution::new(sub.clone(), routes))
                .collect(),
        )));

        // Write the best at the end of the solve of the subproblem
        if write {
            out.push(&format!("{end}.json"));
            let file = std::fs::File::create(&out).unwrap();
            serde_json::to_writer(file, &best.to_vec()).expect("writing failed");
            out.pop();
        }
    }
}

pub fn run_rolling_horizon(
    path: &Path,
    mut output: PathBuf,
    termination: Termination,
    subproblem_size: usize,
    step_length: usize,
    conf: Config,
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

        let (best, _) = run_island_on(
            sub_problem,
            output.clone(),
            InitRoutingSolution,
            termination.clone(),
            conf.clone(),
            false,
        );

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

#[derive(Clone, Debug)]
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

impl std::fmt::Display for Termination {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Termination::Epochs(e) => write!(f, "{e} epochs"),
            Termination::NoViolation => write!(f, "no-violation"),
            Termination::NoImprovement(_, dur, _) => write!(f, "{} no-improvement", dur.as_secs()),
            Termination::Timeout(_, dur) => write!(f, "{} timeout", dur.as_secs()),
            Termination::Never => write!(f, "never"),
            Termination::Any(lhs, rhs) => write!(f, "({lhs}) | ({rhs})"),
            Termination::All(lhs, rhs) => write!(f, "({lhs}) & ({rhs})"),
        }
    }
}

#[derive(Parser)]
#[clap(author = "Fiskefôrgutane", about = "CLI for MIRP solver")]
struct Args {
    /// Sets the configuration file used.
    #[clap(short, long, parse(from_os_str), value_name = "FILE")]
    config: Option<PathBuf>,
    /// Path to problem specification
    #[clap(short, long, parse(from_os_str), value_name = "FILE")]
    problem: PathBuf,
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
    /// The mutation used
    #[clap(
        short, long,
        default_value = "std",
        parse(try_from_str = Box::<dyn RPNMutation>::try_from)
    )]
    mutation: Box<dyn RPNMutation>,

    /// Subcommands
    #[clap(subcommand)]
    commands: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Normal {
        #[clap(long)]
        write: bool,
    },
    RollingHorizon {
        #[clap(long, default_value_t = 30)]
        subproblem_size: usize,
        #[clap(long, default_value_t = 5)]
        step_length: usize,
        #[clap(long)]
        write: bool,
        #[clap(long, parse(try_from_str = Termination::try_from))]
        checkpoint_termination: Option<Termination>,
        #[clap(long)]
        checkpoints: Option<Vec<usize>>,
    },
    Exact,
}

fn enable_logger(level: LevelFilter) {
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

#[derive(Debug, Display)]
pub enum ParseError {
    ExpectedInt,
    ExpectedTerm,
    UnconsumedTokens,
    EmptyStack,
    UnrecognizedToken(String),
}

impl std::error::Error for ParseError {}

impl<'s> std::convert::TryFrom<&'s str> for Termination {
    type Error = ParseError;

    fn try_from(value: &'s str) -> Result<Self, Self::Error> {
        let tokens = value.split_ascii_whitespace();

        enum Arg {
            Int(u64),
            Term(Box<Termination>),
        }

        let mut stack = Vec::new();

        let int = |s: &mut Vec<Arg>| match s.pop() {
            Some(Arg::Int(x)) => Ok(x),
            Some(Arg::Term(_)) => Err(ParseError::ExpectedInt),
            None => Err(ParseError::EmptyStack),
        };

        let term = |s: &mut Vec<Arg>| match s.pop() {
            Some(Arg::Term(x)) => Ok(x),
            Some(Arg::Int(_)) => Err(ParseError::ExpectedTerm),
            None => Err(ParseError::EmptyStack),
        };

        for token in tokens {
            let new = match token {
                "never" => Arg::Term(Box::new(Termination::Never)),
                "epochs" => Arg::Term(Box::new(Termination::Epochs(int(&mut stack)?))),
                "timeout" => Arg::Term(Box::new(Termination::Timeout(
                    std::time::Instant::now(),
                    std::time::Duration::from_secs(int(&mut stack)?),
                ))),
                "no-violation" => Arg::Term(Box::new(Termination::NoViolation)),
                "no-improvement" => Arg::Term(Box::new(Termination::NoImprovement(
                    std::time::Instant::now(),
                    std::time::Duration::from_secs(int(&mut stack)?),
                    std::f64::INFINITY,
                ))),
                "|" => Arg::Term(Box::new(Termination::Any(
                    term(&mut stack)?,
                    term(&mut stack)?,
                ))),
                "&" => Arg::Term(Box::new(Termination::All(
                    term(&mut stack)?,
                    term(&mut stack)?,
                ))),
                x => match x.parse::<u64>() {
                    Ok(num) => Arg::Int(num),
                    Err(_) => return Err(ParseError::UnrecognizedToken(x.to_string())),
                },
            };

            stack.push(new);
        }

        term(&mut stack).map(|x| *x)
    }
}

pub fn main() {
    // Parse command line arguments
    let args = Args::parse();

    // Convert the log level to a LevelFilter
    let level = match args.log.as_str() {
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
    let termination = args.termination;
    let mutation = args.mutation;
    println!("Termination: {termination}");
    println!("Mutation:    {mutation}");
    panic!("yo");

    // The output directory is ./solutions/TIME/PROBLEM/,
    let mut out = std::env::current_dir().expect("unable to fetch current dir");
    out.push("solutions");
    out.push(timesteps);
    out.push(problem_name);

    // Create the output directory
    std::fs::create_dir_all(&out).expect("failed to create out dir");

    // Run the GA.
    match args.commands {
        Commands::Normal { write } => {
            run_island_ga(path, out, termination, InitRoutingSolution, config, write)
        }
        Commands::RollingHorizon {
            subproblem_size,
            step_length,
            write,
            checkpoint_termination,
            checkpoints,
        } => run_unfixed_rolling_horizon(
            path,
            out,
            termination.clone(),
            subproblem_size,
            step_length,
            config,
            write,
            checkpoints.unwrap_or_default(),
            checkpoint_termination.unwrap_or(termination),
        ),
        Commands::Exact => run_exact_model(path, out, termination),
    };
}
