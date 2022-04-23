#[macro_use]
pub mod mutations;
pub mod chromosome;
pub mod fitness;
pub mod initialization;
pub mod parent_selection;
pub mod penalizers;
pub mod recombinations;
pub mod survival_selection;
pub mod traits;

use std::sync::Arc;

use log::{info, trace};
pub use traits::*;

use crate::{problem::Problem, solution::routing::RoutingSolution};

#[derive(Debug, Clone)]
pub struct Config<PS, R, M, S, F> {
    /// The problem specification
    pub problem: Arc<Problem>,
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
    /// The fitness function of the genetic algorithm
    pub fitness: F,
}

/// A general implementation of a genetic algorithm.
pub struct GeneticAlgorithm<PS, R, M, S, F> {
    /// The current population of solution candidates
    pub population: Vec<RoutingSolution>,
    /// Will contain the generated population of children
    child_population: Vec<RoutingSolution>,
    /// Will house the next generation of solution candidates, selected from (population, parent_population, child_population)
    next_population: Vec<RoutingSolution>,

    /// The configuration of the GA
    pub config: Config<PS, R, M, S, F>,
}

impl<PS, R, M, S, F> GeneticAlgorithm<PS, R, M, S, F>
where
    PS: ParentSelection,
    R: Recombination,
    M: Mutation,
    S: SurvivalSelection,
    F: Fitness,
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
        fitness: F,
    ) -> Self
    where
        I: initialization::Initialization<Out = RoutingSolution>,
    {
        trace!("Initializing population");
        let population = (0..population_size)
            .map(|_| initialization.new(problem.clone()))
            .collect::<Vec<_>>();

        // We need a strictly positive population size for this to make sense
        assert!(population_size > 0);
        // We also need to generate `at least` as many children as there are parents, since we'll swapping the populations
        assert!(child_count >= population_size);
        // Check validity that each initial individual has the correct number of vehicles
        assert!(population
            .iter()
            .all(|x| x.len() == problem.vessels().len()));
        // Check that all individuals point to the exact same `problem`
        assert!(population
            .iter()
            .all(|x| std::ptr::eq(x.problem(), &*problem)));
        // It doesn't matter what solution we use for `child_population` and `next_population`
        let dummy = population.first().unwrap().clone();

        GeneticAlgorithm {
            population,
            child_population: vec![dummy.clone(); child_count],
            next_population: vec![dummy; population_size],
            config: Config {
                problem,
                population_size,
                child_count,
                parent_selection,
                recombination,
                mutation,
                selection,
                fitness,
            },
        }
    }

    /// Returns the individual in the population with the minimal objective function.
    pub fn best_individual<'a>(&'a self) -> &'a RoutingSolution {
        let mut best = &self.population[0];
        let mut best_z = std::f64::INFINITY;

        for solution in &self.population {
            let z = self.config.fitness.of(&self.config.problem, solution);
            if z < best_z {
                best = solution;
                best_z = z;
            }
        }

        best
    }

    pub fn epoch(&mut self) {
        trace!("Start of epoch");
        let problem = &self.config.problem;
        let fitness = &self.config.fitness;
        let population = &mut self.population;
        let children = &mut self.child_population;
        let next = &mut self.next_population;
        // This could potentially be reused, but I don't think it's worth it.
        let mut parents = Vec::with_capacity(self.config.child_count);

        // Initialize the parent selection with the current population
        trace!("Initializing parent selection");
        self.config.parent_selection.init(
            population
                .iter()
                .map(|x| self.config.fitness.of(problem, x))
                .collect(),
        );

        assert!(self.config.child_count == children.len());

        trace!("Selecting parents");
        // Sample `child_count` parents from the parent selection strategy, which will be the base for offsprings
        for i in 0..self.config.child_count {
            let p = &population[self.config.parent_selection.sample()];
            parents.push(p);
            children[i].clone_from(p);
        }

        // Recombine the children, which are currently direct copies of the parents.
        // We then apply a mutation to each of them.
        trace!("Applying recombination and mutation");
        for w in children.chunks_exact_mut(2) {
            if let [left, right] = w {
                trace!("Applying recombination");
                self.config.recombination.apply(problem, left, right);
                trace!("Applying mutation to left");
                self.config.mutation.apply(problem, left);
                trace!("Applying mutation to right");
                self.config.mutation.apply(problem, right);
                trace!("finished with recomb and mutations")
            }
        }

        // After having generated the parents and children, we will select the new population based on it
        assert!(self.config.population_size == next.len());
        // TODO: actually use feasibility of problem (currently just set to `true`).
        trace!("Selecting survivors");
        self.config.selection.select_survivors(
            |x: &RoutingSolution| fitness.of(problem, x),
            population,
            &parents,
            children,
            next,
        );

        // And then we'll switch to the new generation
        std::mem::swap(population, next);
        let best = population
            .iter()
            .min_by(|a, b| {
                fitness
                    .of(problem, a)
                    .partial_cmp(&fitness.of(problem, b))
                    .unwrap()
            })
            .unwrap();

        info!("Lowest fitness: {:?}", fitness.of(problem, best));
        info!("Time warp: {:?}", best.warp());
        info!("Shortage: {:?}", best.violation());
        info!("Revenue: {:?}", best.revenue());
        info!("Cost: {:?}", best.cost());

        trace!("End of epoch");
    }
}
