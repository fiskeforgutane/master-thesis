use rand::{prelude::StdRng, Rng, SeedableRng};

use crate::{problem::Problem, solution::routing::RoutingSolution};

pub trait ParentSelection {
    fn init(&mut self, fitness_values: Vec<f64>);

    fn sample(&mut self) -> usize;
}

pub trait Recombination {
    fn apply(&mut self, problem: &Problem, left: &mut RoutingSolution, right: &mut RoutingSolution);

    fn with_probability(self, p: f64) -> Stochastic<Self>
    where
        Self: Sized,
    {
        Stochastic::new(p, self)
    }
}

pub trait Mutation {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution);

    fn with_probability(self, p: f64) -> Stochastic<Self>
    where
        Self: Sized,
    {
        Stochastic::new(p, self)
    }
}

pub trait SurvivalSelection {
    fn select_survivors<F>(
        &mut self,
        count: usize,
        objective_fn: F,
        population: &Vec<RoutingSolution>,
        parents: &Vec<&RoutingSolution>,
        children: &Vec<RoutingSolution>,
        out: &mut Vec<RoutingSolution>,
    ) where
        F: Fn(&RoutingSolution) -> (f64, bool);
}

pub trait Penalty {
    fn penalty(&self, problem: &Problem, solution: &RoutingSolution) -> f64;
}

// Some actual implemtations

/// Applies both mutations
pub struct Chain<A, B>(pub A, pub B);

#[macro_export]
macro_rules! chain {
    // By using both a zero and one element as base cases, make trailing commas optional and allowed
    () => {crate::ga::traits::Nop};
    // If >= 2: chain head and chain!(tail)
    ($a:expr $(, $b:expr)*)=>{
        {
            ga::Chain($a, chain!($($b),*))
        }
    };
}

impl<A, B> Mutation for Chain<A, B>
where
    A: Mutation,
    B: Mutation,
{
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        self.0.apply(problem, solution);
        self.1.apply(problem, solution);
    }
}

impl<A, B> Recombination for Chain<A, B>
where
    A: Recombination,
    B: Recombination,
{
    fn apply(
        &mut self,
        problem: &Problem,
        left: &mut RoutingSolution,
        right: &mut RoutingSolution,
    ) {
        self.0.apply(problem, left, right);
        self.1.apply(problem, left, right);
    }
}

/// Applies a mutation stochastically with probability `p`
pub struct Stochastic<M>(pub f64, pub M, StdRng);

impl<M> Stochastic<M> {
    pub fn new(p: f64, inner: M) -> Self {
        Stochastic(p, inner, StdRng::from_entropy())
    }
}

impl<M> Mutation for Stochastic<M>
where
    M: Mutation,
{
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        if self.2.gen_range(0.0..1.0) < self.0 {
            self.1.apply(problem, solution);
        }
    }
}

impl<R> Recombination for Stochastic<R>
where
    R: Recombination,
{
    fn apply(
        &mut self,
        problem: &Problem,
        left: &mut RoutingSolution,
        right: &mut RoutingSolution,
    ) {
        if self.2.gen_range(0.0..1.0) < self.0 {
            self.1.apply(problem, left, right);
        }
    }
}

pub struct Nop;

impl Recombination for Nop {
    fn apply(&mut self, _: &Problem, _: &mut RoutingSolution, _: &mut RoutingSolution) {}
}

impl Mutation for Nop {
    fn apply(&mut self, _: &Problem, _: &mut RoutingSolution) {}
}

impl Penalty for Nop {
    fn penalty(&self, _: &Problem, _: &RoutingSolution) -> f64 {
        0.0
    }
}
