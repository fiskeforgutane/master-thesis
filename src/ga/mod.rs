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

use std::{sync::Arc, thread::JoinHandle};

use log::{info, trace};
pub use traits::*;

use crate::{problem::Problem, solution::routing::RoutingSolution};

use self::initialization::Initialization;

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

pub enum Strategy {
    Islands,
    Shared,
}

/// Master-to-slave message
pub enum MTS {
    Terminate,
    Echo,
}
/// Slave-to-master message
pub enum STM {
    Ack,
}

pub struct ConcurrentGA<PS, R, M, S, F> {
    /// The config used for each GA instance
    pub config: Config<PS, R, M, S, F>,
    /// The handle of each spawned thread
    handles: Vec<JoinHandle<()>>,
    /// Transmitter-end of channels to each spawned thread
    txs: Vec<std::sync::mpsc::Sender<MTS>>,
    /// Receiver-end of the transmitter channel which each spawned thread has available.
    rx: std::sync::mpsc::Receiver<STM>,
}

impl<PS, R, M, S, F> ConcurrentGA<PS, R, M, S, F>
where
    PS: ParentSelection + Send + 'static,
    R: Recombination + Send + 'static,
    M: Mutation + Send + 'static,
    S: SurvivalSelection + Send + 'static,
    F: Fitness + Send + 'static,
    Config<PS, R, M, S, F>: Clone + Send + 'static,
{
    pub fn new<I: Initialization<Out = RoutingSolution> + Clone + Send + 'static>(
        init: I,
        config: Config<PS, R, M, S, F>,
        count: usize,
    ) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<STM>();
        let mut handles = Vec::with_capacity(count);
        let mut txs = Vec::with_capacity(count);

        for _ in 0..count {
            let (init, config, stm) = (init.clone(), config.clone(), tx.clone());
            let (tx, rx) = std::sync::mpsc::channel::<MTS>();

            handles.push(std::thread::spawn(move || {
                let mut ga = GeneticAlgorithm::new(init, config);

                loop {
                    match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                        // TODO: do something
                        Ok(MTS::Terminate) => return (),
                        Ok(MTS::Echo) => stm.send(STM::Ack).unwrap(),
                        // If we timeout, there is simply no message available for us
                        Err(_) => (),
                    }
                    ga.epoch();
                }
            }));

            txs.push(tx);
        }

        Self {
            config,
            handles,
            txs,
            rx,
        }
    }
}

impl<PS, R, M, S, F> Drop for ConcurrentGA<PS, R, M, S, F> {
    fn drop(&mut self) {
        for tx in self.txs {
            tx.send(MTS::Terminate);
        }

        for handle in self.handles {
            handle.join().expect("worker thread panic");
        }
    }
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
    pub fn new<I>(initialization: I, config: Config<PS, R, M, S, F>) -> Self
    where
        I: initialization::Initialization<Out = RoutingSolution>,
    {
        trace!("Initializing population");
        let population = (0..config.population_size)
            .map(|_| initialization.new(config.problem.clone()))
            .collect::<Vec<_>>();

        // We need a strictly positive population size for this to make sense
        assert!(config.population_size > 0);
        // We also need to generate `at least` as many children as there are parents, since we'll swapping the populations
        assert!(config.child_count >= config.population_size);
        // Check validity that each initial individual has the correct number of vehicles
        assert!(population
            .iter()
            .all(|x| x.len() == config.problem.vessels().len()));
        // Check that all individuals point to the exact same `problem`
        assert!(population
            .iter()
            .all(|x| std::ptr::eq(x.problem(), &*config.problem)));
        // It doesn't matter what solution we use for `child_population` and `next_population`
        let dummy = population.first().unwrap().clone();

        GeneticAlgorithm {
            population,
            child_population: vec![dummy.clone(); config.child_count],
            next_population: vec![dummy; config.population_size],
            config,
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
