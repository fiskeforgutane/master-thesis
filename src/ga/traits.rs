use pyo3::pyclass;
use rand::{
    prelude::{SliceRandom, StdRng},
    Rng, SeedableRng,
};

use crate::{problem::Problem, solution::routing::RoutingSolution};

pub trait ParentSelection {
    fn init(&mut self, fitness_values: Vec<f64>);

    fn sample(&mut self) -> usize;
}

pub trait Recombination {
    fn apply(&mut self, problem: &Problem, left: &mut RoutingSolution, right: &mut RoutingSolution);
}

pub trait Mutation {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution);
}

impl<M> Mutation for Box<M>
where
    M: Mutation + ?Sized,
{
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        M::apply(self, problem, solution)
    }
}

pub trait SurvivalSelection {
    fn select_survivors<F>(
        &mut self,
        objective_fn: F,
        population: &[RoutingSolution],
        parents: &[&RoutingSolution],
        children: &[RoutingSolution],
        out: &mut [RoutingSolution],
    ) where
        F: Fn(&RoutingSolution) -> f64;
}

pub trait Penalty {
    fn penalty(&self, problem: &Problem, solution: &RoutingSolution) -> f64;
}

pub trait Fitness {
    fn of(&self, problem: &Problem, solution: &RoutingSolution) -> f64;
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
#[derive(Debug, Clone)]
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

#[pyclass]
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

/// Vectors are used for "choice", i.e. choose one of (unweighted)
impl<M> Mutation for Vec<M>
where
    M: Mutation,
{
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        let mutation = self.choose_mut(&mut rand::thread_rng()).unwrap();
        mutation.apply(problem, solution)
    }
}

/// Vector of tuples are used for "weighted choice".
impl<M> Mutation for Vec<(M, f64)>
where
    M: Mutation,
{
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        let (mutation, _) = self
            .choose_weighted_mut(&mut rand::thread_rng(), |(_, w)| *w)
            .unwrap();
        mutation.apply(problem, solution)
    }
}

/// Vectors are used for "choice", i.e. choose inweighted
impl<R> Recombination for Vec<R>
where
    R: Recombination,
{
    fn apply(
        &mut self,
        problem: &Problem,
        left: &mut RoutingSolution,
        right: &mut RoutingSolution,
    ) {
        let recombination = self.choose_mut(&mut rand::thread_rng()).unwrap();
        recombination.apply(problem, left, right)
    }
}

/// Vectors of tuples are used for weighted choice
impl<R> Recombination for Vec<(R, f64)>
where
    R: Recombination,
{
    fn apply(
        &mut self,
        problem: &Problem,
        left: &mut RoutingSolution,
        right: &mut RoutingSolution,
    ) {
        let (recombination, _) = self
            .choose_weighted_mut(&mut rand::thread_rng(), |(_, w)| *w)
            .unwrap();
        recombination.apply(problem, left, right)
    }
}

// Array implementations for Mutation and Recombination
macro_rules! impl_array_choice {
    ($n:expr) => {
        impl<M> Mutation for [M; $n]
        where
            M: Mutation,
        {
            fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
                let mutation = self.choose_mut(&mut rand::thread_rng()).unwrap();
                mutation.apply(problem, solution)
            }
        }

        impl<R> Recombination for [R; $n]
        where
            R: Recombination,
        {
            fn apply(
                &mut self,
                problem: &Problem,
                left: &mut RoutingSolution,
                right: &mut RoutingSolution,
            ) {
                let recombination = self.choose_mut(&mut rand::thread_rng()).unwrap();
                recombination.apply(problem, left, right)
            }
        }
    };
}

impl_array_choice!(1);
impl_array_choice!(2);
impl_array_choice!(3);
impl_array_choice!(4);
impl_array_choice!(5);
impl_array_choice!(6);
impl_array_choice!(7);
impl_array_choice!(8);
impl_array_choice!(9);
impl_array_choice!(10);
impl_array_choice!(11);
impl_array_choice!(12);
impl_array_choice!(13);
impl_array_choice!(14);
impl_array_choice!(15);
impl_array_choice!(16);
