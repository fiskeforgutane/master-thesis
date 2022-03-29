#[macro_use]
pub mod mutations;
pub mod chromosome;
pub mod initialization;
pub mod parent_selection;
pub mod penalizers;
pub mod recombinations;
pub mod survival_selection;
pub mod traits;

use std::sync::Arc;

pub use traits::*;

use crate::{problem::Problem, solution::routing::RoutingSolution};

/// A general implementation of a genetic algorithm.
pub struct GeneticAlgorithm<PS, R, M, S, P> {
    /// The Multi-Depot Vehicle Routing Problem specification
    pub problem: Arc<Problem>,
    /// The current population of solution candidates
    pub population: Vec<RoutingSolution>,
    /// Will contain the generated population of children
    child_population: Vec<RoutingSolution>,
    /// Will house the next generation of solution candidates, selected from (population, parent_population, child_population)
    next_population: Vec<RoutingSolution>,

    /// Size of population (equal to population.len())
    pub population_size: usize,
    /// Number of children to generate before selecting
    pub child_count: usize,

    /// Select a parent for reproduction from the current population
    pub parent_selection: PS,
    /// Recombine two individuals into one or several offsprings
    pub recombination: R,
    /// Mutate an offspring
    pub mutation: M,
    /// Select the next population based on the parents and children
    pub selection: S,
    /// Penalty for violations of the objective function
    pub penalizer: P,
}

impl<PS, R, M, S, P> GeneticAlgorithm<PS, R, M, S, P>
where
    PS: ParentSelection,
    R: Recombination,
    M: Mutation,
    S: SurvivalSelection,
    P: Penalty,
{
    /// Constructs a new GeneticAlgorithm with the given configuration.
    pub fn new<I>(
        problem: Arc<Problem>,
        population_size: usize,
        child_count: usize,
        initialization: I,
        parent_selection: PS,
        recombination: R,
        mutation: M,
        selection: S,
        penalizer: P,
    ) -> Self
    where
        I: initialization::Initialization<Out = RoutingSolution>,
    {
        assert!(population_size > 0);
        assert!(child_count >= population_size);

        let population = (0..population_size)
            .map(|_| initialization.new(&problem))
            .collect();

        GeneticAlgorithm {
            problem,
            population,
            child_population: Vec::new(),
            next_population: Vec::new(),
            population_size,
            child_count,
            parent_selection,
            recombination,
            mutation,
            selection,
            penalizer,
        }
    }

    /// The objective function that the GA attempts to minimize. It consists of a solutions `cost` plus a penalty term given by the instance's `penalizer`,
    /// typically by putting some cost on any violations of number of vehicles, duration or load.
    pub fn objective_fn(&self, _solution: &RoutingSolution) -> f64 {
        todo!()
    }

    /// Returns the individual in the population with the minimal objective function.
    pub fn best_individual<'a>(&'a self) -> &'a RoutingSolution {
        let mut best = &self.population[0];
        let mut best_z = std::f64::INFINITY;

        for solution in &self.population {
            let z = self.objective_fn(solution);
            if z < best_z {
                best = solution;
                best_z = z;
            }
        }

        best
    }

    pub fn epoch(&mut self) {
        let problem = &self.problem;
        let population = &mut self.population;
        let penalizer = &self.penalizer;
        let children = &mut self.child_population;
        let next = &mut self.next_population;
        let mut parents = Vec::with_capacity(self.child_count);
        let z = |x: &RoutingSolution| {
            let _penalty = penalizer.penalty(problem, x);
            todo!()
        };

        children.clear();
        next.clear();

        // Initialize the parent selection with the current population
        self.parent_selection
            .init(population.iter().map(z).collect());

        // Sample `child_count` parents from the parent selection strategy, which will be the base for offsprings
        while children.len() < self.child_count {
            let p = self.parent_selection.sample();
            let p = &population[p];
            parents.push(p);
            children.push(p.clone());
        }

        // Recombine the children, which are currently direct copies of the parents.
        // We then apply a mutation to each of them.
        for w in children.chunks_exact_mut(2) {
            if let [left, right] = w {
                self.recombination.apply(problem, left, right);
                self.mutation.apply(problem, left);
                self.mutation.apply(problem, right);
            }
        }

        // After having generated the parents and children, we will select the new population based on it
        self.selection.select_survivors(
            self.population_size,
            |_: &RoutingSolution| todo!(),
            population,
            &parents,
            children,
            next,
        );

        // And then we'll switch to the new generation
        std::mem::swap(population, next);
    }
}
