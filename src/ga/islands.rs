use std::{
    sync::{atomic::AtomicU64, mpsc::TryRecvError, Arc},
    thread::JoinHandle,
};

use itertools::Itertools;
use rand::prelude::SliceRandom;

use crate::solution::{routing::RoutingSolution, Visit};

use super::{
    initialization::Initialization, Config, Fitness, GeneticAlgorithm, Mutation, ParentSelection,
    Recombination, SurvivalSelection,
};

/// Master-to-slave message
pub enum Mts<F> {
    /// Tell the slave to terminate
    Terminate,
    /// Ask for the current population from the slave
    GetPopulation,
    /// A migration of individuals that should be included in the slave's population
    Migration(Vec<ThinSolution>),
    /// Tell the slave to update its fitness function
    SetFitness(F),
    /// Ask the slave for its best individual
    GetBest,
}
/// Slave-to-master message
pub struct Stm {
    slave: usize,
    message: StmMessage,
}

pub enum StmMessage {
    /// The current population of the GA
    Population(Vec<ThinSolution>),
    /// The best individual in the population
    Best(ThinSolution),
}

/// A set of plans without the reference to GRBEnv. GRBEnv makes it impossible to send RoutingSolutions between threads
type ThinSolution = Vec<Vec<Visit>>;

pub struct IslandGA<PS, R, M, S, F> {
    /// The config used for each GA instance
    pub config: Config<PS, R, M, S, F>,
    /// The handle of each spawned thread
    handles: Vec<JoinHandle<()>>,
    /// Transmitter-end of channels to each spawned thread
    txs: Vec<std::sync::mpsc::Sender<Mts<F>>>,
    /// Receiver-end of the transmitter channel which each spawned thread has available.
    rx: std::sync::mpsc::Receiver<Stm>,
    /// The total number of epochs done across all islands
    total_epochs: Arc<AtomicU64>,
    // TODO:
    // The latest population for each island
    // populations: Vec<Arc<Mutex<Vec<ThinSolution>>>>,
}

impl<PS, R, M, S, F> IslandGA<PS, R, M, S, F>
where
    PS: ParentSelection,
    R: Recombination,
    M: Mutation,
    S: SurvivalSelection,
    F: Fitness + Clone + Send + 'static,
{
    /// The number of islands
    pub fn island_count(&self) -> usize {
        return self.handles.len();
    }

    /// Construct and start a new island GA.
    pub fn new<I: Initialization + Clone + Send + 'static, Func>(
        init: I,
        config: Func,
        count: usize,
    ) -> Self
    where
        Func: Send + Fn() -> Config<PS, R, M, S, F> + 'static + Clone,
    {
        let (tx, rx) = std::sync::mpsc::channel::<Stm>();
        let mut handles = Vec::with_capacity(count);
        let mut txs = Vec::with_capacity(count);
        let total_epochs = Arc::new(AtomicU64::new(0));

        for i in 0..count {
            let (init, stm) = (init.clone(), tx.clone());
            let (tx, rx) = std::sync::mpsc::channel::<Mts<F>>();
            let total_epochs = total_epochs.clone();
            let config = config.clone();

            handles.push(std::thread::spawn(move || {
                let config = config();
                let problem = config.problem.clone();
                let mut ga = GeneticAlgorithm::new(init, config);

                loop {
                    match rx.try_recv() {
                        // TODO: do something
                        Ok(Mts::Terminate) => return (),
                        Ok(Mts::GetBest) => stm
                            .send(Stm {
                                slave: i,
                                message: StmMessage::Best(ga.best_individual().to_vec()),
                            })
                            .unwrap(),
                        Ok(Mts::SetFitness(f)) => {
                            ga.config.fitness = f;
                        }
                        Ok(Mts::GetPopulation) => stm
                            .send(Stm {
                                slave: i,
                                message: StmMessage::Population(
                                    ga.population
                                        .iter()
                                        .map(|solution| solution.to_vec())
                                        .collect(),
                                ),
                            })
                            .unwrap(),
                        Ok(Mts::Migration(migration)) => {
                            ga.population.extend(migration.into_iter().map(|plans| {
                                RoutingSolution::new_with_model(
                                    problem.clone(),
                                    plans,
                                    ga.quantities.clone(),
                                )
                            }))
                        }
                        // If there is nothing awaiting processing, we will just continue to the GA epoch
                        Err(TryRecvError::Empty) => (),
                        // This should not happen
                        Err(TryRecvError::Disconnected) => panic!("master tx disconnected"),
                    }

                    ga.epoch();
                    // Increment the number of epochs done
                    total_epochs.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }));

            txs.push(tx);
        }

        Self {
            config: config(),
            handles,
            txs,
            rx,
            total_epochs,
        }
    }

    /// The number of epochs done across all islands
    pub fn epochs(&self) -> u64 {
        self.total_epochs.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns the current best solution
    pub fn best(&self) -> RoutingSolution {
        for tx in &self.txs {
            tx.send(Mts::GetBest).unwrap();
        }

        let mut best = RoutingSolution::empty(self.config.problem.clone());
        let mut best_fitness = f64::INFINITY;

        let fitness = &self.config.fitness;
        let problem = self.config.problem.as_ref();

        for _ in 0..self.island_count() {
            let solution = match self.rx.recv().unwrap() {
                Stm {
                    message: StmMessage::Best(solution),
                    ..
                } => RoutingSolution::new(self.config.problem.clone(), solution),
                _ => panic!("unexpected response"),
            };

            let solution_fitness = fitness.of(problem, &solution);
            if solution_fitness < best_fitness {
                best = solution;
                best_fitness = solution_fitness;
            }
        }

        best
    }

    /// Tell each island to replace their fitness function
    pub fn set_fitness(&mut self, fitness: F) {
        for tx in &self.txs {
            tx.send(Mts::SetFitness(fitness.clone())).unwrap();
        }

        self.config.fitness = fitness;
    }

    /// Get the population of each island
    pub fn populations(&mut self) -> Vec<Vec<ThinSolution>> {
        let mut populations = vec![Vec::new(); self.island_count()];
        // Ask for the population of each island
        for tx in &self.txs {
            tx.send(Mts::GetPopulation).unwrap();
        }

        // Wait for a response from each island.
        for _ in 0..self.island_count() {
            let Stm { slave, message } = self.rx.recv().unwrap();

            match message {
                StmMessage::Population(population) => {
                    populations[slave] = population
                        .into_iter()
                        .map(|x| x.into_iter().map(|x| x[..].to_vec()).collect())
                        .collect()
                }
                StmMessage::Best(_) => panic!("expected population message"),
            }
        }

        populations
    }

    /// Initiate migrations between islands
    pub fn migrate(&mut self, count: usize) {
        let mut populations = vec![Vec::new(); self.island_count()];
        // Ask for the population of each island
        for tx in &self.txs {
            tx.send(Mts::GetPopulation).unwrap();
        }

        // Wait for a response from each island.
        for _ in 0..self.island_count() {
            let Stm { slave, message } = self.rx.recv().unwrap();

            match message {
                StmMessage::Population(population) => populations[slave] = population,
                StmMessage::Best(_) => panic!("expected population message"),
            }
        }

        // We will now transfer some individuals between random pairs of islands
        let mut order = (0..self.island_count()).collect::<Vec<_>>();
        order.shuffle(&mut rand::thread_rng());
        let first = order.first();

        // We will migrate some individuals from `i` to `j`s population.
        for (&i, &j) in order.iter().chain(first).tuple_windows() {
            let population = &populations[i];
            let tx = &self.txs[j];

            tx.send(Mts::Migration(
                population
                    .choose_multiple(&mut rand::thread_rng(), count)
                    .cloned()
                    .collect(),
            ))
            .unwrap();
        }
    }

    /// Migrate a provided set of solutions into the islands
    pub fn insert(&mut self, mut individuals: Vec<Vec<Vec<Visit>>>) {
        individuals.shuffle(&mut rand::thread_rng());

        let size = individuals.len() / (self.island_count() + 1);
        for (chunk, tx) in individuals.chunks(size).zip(&self.txs) {
            tx.send(Mts::Migration(chunk.to_vec())).unwrap();
        }
    }
}

impl<PS, R, M, S, F> Drop for IslandGA<PS, R, M, S, F> {
    fn drop(&mut self) {
        for tx in &self.txs {
            tx.send(Mts::Terminate).expect("sending failed");
        }

        // We need the join handles by value
        for handle in std::mem::replace(&mut self.handles, Vec::new()) {
            handle.join().expect("worker thread panic");
        }
    }
}
