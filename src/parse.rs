use std::{
    fmt::{write, Display},
    num::{ParseFloatError, ParseIntError},
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
};
use derive_more::Display;

#[derive(Debug, Display)]
pub enum ParseError {
    ExpectedInt,
    ExpectedFloat,
    ExpectedMutation,
    EmptyStack,
    ParseFloatError(ParseFloatError),
    ParseIntError(ParseIntError),
}

impl std::error::Error for ParseError {}

pub trait RPNMutation: Mutation + Display {}
impl<M> RPNMutation for M where M: Mutation + Display {}

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
    ))
}

impl<'s> TryFrom<&'s str> for Box<dyn RPNMutation> {
    type Error = ParseError;

    fn try_from(value: &'s str) -> Result<Self, Self::Error> {
        let tokens = value.split_ascii_whitespace();

        enum Arg {
            Int(usize),
            Float(f64),
            Mut(Box<dyn RPNMutation>),
        }

        let mut stack: Vec<Arg> = Vec::new();

        let int = |s: &mut Vec<Arg>| match s.pop() {
            Some(Arg::Int(x)) => Ok(x),
            None => Err(ParseError::EmptyStack),
            Some(_) => Err(ParseError::ExpectedInt),
        };

        let float = |s: &mut Vec<Arg>| match s.pop() {
            Some(Arg::Float(x)) => Ok(x),
            None => Err(ParseError::EmptyStack),
            Some(_) => Err(ParseError::ExpectedFloat),
        };

        let mutation = |s: &mut Vec<Arg>| match s.pop() {
            Some(Arg::Mut(x)) => Ok(x),
            None => Err(ParseError::EmptyStack),
            Some(_) => Err(ParseError::ExpectedMutation),
        };

        for token in tokens {
            let new = match token {
                "nop" => Arg::Mut(Box::new(Nop)),
                "std" => Arg::Mut(std_mutation()),
                "add-random" => Arg::Mut(Box::new(AddRandom::new())),
                "remove-random" => Arg::Mut(Box::new(RemoveRandom::new())),
                "interswap" => Arg::Mut(Box::new(InterSwap)),
                "intraswap" => Arg::Mut(Box::new(IntraSwap)),
                "redcost" => Arg::Mut(Box::new(RedCost::red_cost_mutation(int(&mut stack)?))),
                "twerk" => Arg::Mut(Box::new(Twerk::everybody())),
                "2opt" => Arg::Mut(Box::new(TwoOpt::new(TwoOptMode::IntraRandom))),
                "time-setter" => Arg::Mut(Box::new(TimeSetter::new(float(&mut stack)?).unwrap())),
                "replace-node" => Arg::Mut(Box::new(ReplaceNode::new(float(&mut stack)?))),
                "swap*" => Arg::Mut(Box::new(SwapStar)),
                "dedup" => Arg::Mut(Box::new(Dedup(DedupPolicy::KeepFirst))),
                "bounce" => Arg::Mut(Box::new(Bounce::new(int(&mut stack)?, BounceMode::All))),
                "add-smart" => Arg::Mut(Box::new(AddSmart)),
                "rr-period" => Arg::Mut(Box::new(Period::new(
                    float(&mut stack)?,
                    float(&mut stack)?,
                    int(&mut stack)?,
                    int(&mut stack)?,
                ))),
                "rr-vessel" => Arg::Mut(Box::new(Vessel::new(
                    float(&mut stack)?,
                    float(&mut stack)?,
                    int(&mut stack)?,
                ))),
                "sisr" => Arg::Mut(Box::new(SlackInductionByStringRemoval::new(sisr::Config {
                    average_removal: int(&mut stack)?,
                    max_cardinality: int(&mut stack)?,
                    alpha: float(&mut stack)?,
                    blink_rate: float(&mut stack)?,
                    first_n: int(&mut stack)?,
                    epsilon: (float(&mut stack)?, float(&mut stack)?),
                }))),
                "?" => Arg::Mut(Box::new(Stochastic::new(
                    float(&mut stack)?,
                    mutation(&mut stack)?,
                ))),
                x => match x.contains('.') {
                    true => Arg::Float(x.parse().map_err(|e| ParseError::ParseFloatError(e))?),
                    false => Arg::Int(x.parse().map_err(|e| ParseError::ParseIntError(e))?),
                },
            };

            // At this point, we will chain them all together
            stack.push(new);
        }

        // Can't wait for try_reduce to hit stable...
        let mut it = stack.into_iter();
        let mut mutation = match it.next() {
            Some(Arg::Mut(m)) => m,
            _ => return Err(ParseError::ExpectedMutation),
        };

        for arg in it {
            match arg {
                Arg::Mut(m) => mutation = Box::new(Chain(mutation, m)),
                _ => return Err(ParseError::ExpectedMutation),
            }
        }

        Ok(mutation)
    }
}
