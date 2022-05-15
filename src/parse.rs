use std::{
    fmt::{write, Display},
    num::{ParseFloatError, ParseIntError},
    path::Path,
};

use crate::{
    chain,
    ga::{
        self,
        mutations::{
            rr::{
                self,
                sisr::{self, SlackInductionByStringRemoval},
                Period, Vessel,
            },
            AddRandom, AddSmart, Bounce, BounceMode, Dedup, DedupPolicy, InterSwap, IntraSwap,
            RedCost, RemoveRandom, ReplaceNode, SwapStar, TimeSetter, Twerk, TwoOpt, TwoOptMode,
        },
        Chain, Mutation, Nop, Stochastic,
    },
    problem::Problem,
    solution::routing::Improvement,
    termination::Termination,
};
use derive_more::Display;
use log::debug;

#[derive(Debug, Display)]
pub enum ParseMutationError {
    ExpectedInt,
    ExpectedFloat,
    ExpectedMutation,
    EmptyStack,
    ParseFloatError(ParseFloatError),
    ParseIntError(ParseIntError),
}

impl std::error::Error for ParseMutationError {}

pub trait RPNMutation: Mutation + Display {}
impl<M> RPNMutation for M where M: Mutation + Display {}

/// This is a horrible work-around.
impl Clone for Box<dyn RPNMutation> {
    fn clone(&self) -> Self {
        Box::<dyn RPNMutation>::try_from(format!("{}", self).as_str()).unwrap()
    }
}

// The "standard" mutation
pub fn std_mutation() -> Box<dyn RPNMutation> {
    Box::new(chain!(
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
        // Stochastic::new(0.01, SwapStar), <- crashes
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
    ))
}

impl<'s> TryFrom<&'s str> for Box<dyn RPNMutation> {
    type Error = ParseMutationError;

    fn try_from(value: &'s str) -> Result<Self, Self::Error> {
        use ParseMutationError::*;
        let tokens = value.split_ascii_whitespace();

        enum Arg {
            Int(i64),
            Float(f64),
            Mut(Box<dyn RPNMutation>),
        }

        let mut stack: Vec<Arg> = Vec::new();

        let int = |s: &mut Vec<Arg>| match s.pop() {
            Some(Arg::Int(x)) => Ok(x),
            None => Err(EmptyStack),
            Some(_) => Err(ExpectedInt),
        };

        let float = |s: &mut Vec<Arg>| match s.pop() {
            Some(Arg::Float(x)) => Ok(x),
            Some(Arg::Int(x)) => Ok(x as f64),
            Some(_) => Err(ExpectedFloat),
            None => Err(EmptyStack),
        };

        let mutation = |s: &mut Vec<Arg>| match s.pop() {
            Some(Arg::Mut(x)) => Ok(x),
            None => Err(EmptyStack),
            Some(_) => Err(ExpectedMutation),
        };

        for token in tokens {
            debug!("token = {token}");
            let new = match token {
                "nop" => Arg::Mut(Box::new(Nop)),
                "std" => Arg::Mut(std_mutation()),
                "add-random" => Arg::Mut(Box::new(AddRandom::new())),
                "remove-random" => Arg::Mut(Box::new(RemoveRandom::new())),
                "interswap" => Arg::Mut(Box::new(InterSwap)),
                "intraswap" => Arg::Mut(Box::new(IntraSwap)),
                "redcost" => Arg::Mut(Box::new(RedCost::red_cost_mutation(
                    int(&mut stack)?.try_into().unwrap(),
                ))),
                "twerk" => Arg::Mut(Box::new(Twerk::everybody())),
                "2opt" => Arg::Mut(Box::new(TwoOpt::new(TwoOptMode::IntraRandom))),
                "time-setter" => Arg::Mut(Box::new(TimeSetter::new(float(&mut stack)?).unwrap())),
                "replace-node" => Arg::Mut(Box::new(ReplaceNode::new(float(&mut stack)?))),
                "swap*" => Arg::Mut(Box::new(SwapStar)),
                "dedup" => Arg::Mut(Box::new(Dedup(DedupPolicy::KeepFirst))),
                "bounce" => Arg::Mut(Box::new(Bounce::new(
                    int(&mut stack)?.try_into().unwrap(),
                    BounceMode::All,
                ))),
                "add-smart" => Arg::Mut(Box::new(AddSmart)),
                "rr-period" => Arg::Mut(Box::new(Period::new(
                    float(&mut stack)?,
                    float(&mut stack)?,
                    int(&mut stack)?.try_into().unwrap(),
                    int(&mut stack)?.try_into().unwrap(),
                ))),
                "rr-vessel" => Arg::Mut(Box::new(Vessel::new(
                    float(&mut stack)?,
                    float(&mut stack)?,
                    int(&mut stack)?.try_into().unwrap(),
                ))),
                "sisr" => Arg::Mut(Box::new(SlackInductionByStringRemoval::new(sisr::Config {
                    average_removal: int(&mut stack)?.try_into().unwrap(),
                    max_cardinality: int(&mut stack)?.try_into().unwrap(),
                    alpha: float(&mut stack)?,
                    blink_rate: float(&mut stack)?,
                    first_n: int(&mut stack)?.try_into().unwrap(),
                    epsilon: Improvement {
                        warp: int(&mut stack)?.try_into().unwrap(),
                        approx_berth_violation: int(&mut stack)?.try_into().unwrap(),
                        violation: float(&mut stack)?,
                        loss: float(&mut stack)?,
                    },
                }))),
                "?" => Arg::Mut(Box::new(Stochastic::new(
                    float(&mut stack)?,
                    mutation(&mut stack)?,
                ))),
                x => match x.contains('.') {
                    true => Arg::Float(x.parse().map_err(|e| ParseFloatError(e))?),
                    false => Arg::Int(x.parse().map_err(|e| ParseIntError(e))?),
                },
            };

            // At this point, we will chain them all together
            stack.push(new);
        }

        // Can't wait for try_reduce to hit stable...
        let mut it = stack.into_iter();
        let mut mutation = match it.next() {
            Some(Arg::Mut(m)) => m,
            _ => return Err(ExpectedMutation),
        };

        for arg in it {
            match arg {
                Arg::Mut(m) => mutation = Box::new(Chain(mutation, m)),
                _ => return Err(ExpectedMutation),
            }
        }

        Ok(mutation)
    }
}

#[derive(Debug, Display)]
pub enum ParseTerminationError {
    ExpectedInt,
    ExpectedTerm,
    UnconsumedTokens,
    EmptyStack,
    UnrecognizedToken(String),
}

impl std::error::Error for ParseTerminationError {}

impl<'s> std::convert::TryFrom<&'s str> for Termination {
    type Error = ParseTerminationError;

    fn try_from(value: &'s str) -> Result<Self, Self::Error> {
        use ParseTerminationError::*;
        let tokens = value.split_ascii_whitespace();

        enum Arg {
            Int(u64),
            Term(Box<Termination>),
        }

        let mut stack = Vec::new();

        let int = |s: &mut Vec<Arg>| match s.pop() {
            Some(Arg::Int(x)) => Ok(x),
            Some(Arg::Term(_)) => Err(ExpectedInt),
            None => Err(EmptyStack),
        };

        let term = |s: &mut Vec<Arg>| match s.pop() {
            Some(Arg::Term(x)) => Ok(x),
            Some(Arg::Int(_)) => Err(ExpectedTerm),
            None => Err(EmptyStack),
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
                    Err(_) => return Err(UnrecognizedToken(x.to_string())),
                },
            };

            stack.push(new);
        }

        term(&mut stack).map(|x| *x)
    }
}

pub struct ProblemFromFile {
    pub name: String,
    pub timesteps: String,
    pub problem: Problem,
}

pub fn read_problem(path: &str) -> ProblemFromFile {
    let path = Path::new(path);
    let problem_name = path.file_stem().unwrap().to_str().unwrap();
    let directory = path.parent().unwrap();
    let timesteps = directory.file_stem().unwrap().to_str().unwrap();

    let file = std::fs::File::open(path).unwrap();
    let reader = std::io::BufReader::new(file);
    let problem: Problem = serde_json::from_reader(reader).unwrap();

    ProblemFromFile {
        name: problem_name.to_string(),
        timesteps: timesteps.to_string(),
        problem,
    }
}
