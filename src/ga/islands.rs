use std::{
    sync::{atomic::AtomicU64, mpsc::TryRecvError, Arc, Mutex, MutexGuard},
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
pub enum Mts {
    /// Tell the slave to terminate
    Terminate,
    /// Ask for the current population from the slave
    GetPopulation,
    /// A migration of individuals that should be included in the slave's population
    Migration(Vec<ThinSolution>),
}
/// Slave-to-master message
pub struct Stm {
    slave: usize,
    message: StmMessage,
}

pub enum StmMessage {
    /// The current population of the GA
    Population(Vec<ThinSolution>),
}

/// A set of plans without the reference to GRBEnv. GRBEnv makes it impossible to send RoutingSolutions between threads
type ThinSolution = Vec<Vec<Visit>>;

pub struct IslandGA<PS, R, M, S, F> {
    /// The config used for each GA instance
    pub config: Config<PS, R, M, S, F>,
    /// The handle of each spawned thread
    handles: Vec<JoinHandle<()>>,
    /// Transmitter-end of channels to each spawned thread
    txs: Vec<std::sync::mpsc::Sender<Mts>>,
    /// Receiver-end of the transmitter channel which each spawned thread has available.
    rx: std::sync::mpsc::Receiver<Stm>,
    /// The best individual ever recorded across all islands.
    best: Arc<Mutex<ThinSolution>>,
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
    F: Fitness + Clone,
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
        let problem = config.clone()().problem;

        let best = Arc::new(Mutex::new(RoutingSolution::empty(problem.clone()).to_vec()));

        for i in 0..count {
            let (init, stm) = (init.clone(), tx.clone());
            let (tx, rx) = std::sync::mpsc::channel::<Mts>();
            let total_epochs = total_epochs.clone();
            let mutex = best.clone();
            let config = config.clone();

            handles.push(std::thread::spawn(move || {
                let config = config();
                let problem = config.problem.clone();
                let fitness = config.fitness.clone();
                let mut ga = GeneticAlgorithm::new(init, config);

                loop {
                    match rx.try_recv() {
                        // TODO: do something
                        Ok(Mts::Terminate) => return (),
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

                    // Update the best individual, if a new best is found.
                    let island_best = ga.best_individual();
                    let island_fitness = fitness.of(&problem, island_best);
                    let mut best = mutex.lock().unwrap();
                    // Not ideal.
                    let best_fitness = fitness.of(
                        &problem,
                        &RoutingSolution::new(problem.clone(), best.clone()),
                    );

                    if island_fitness <= best_fitness {
                        best.clone_from_slice(&island_best.to_vec());
                    }
                }
            }));

            txs.push(tx);
        }

        Self {
            config: config(),
            handles,
            txs,
            rx,
            best,
            total_epochs,
        }
    }

    /// The number of epochs done across all islands
    pub fn epochs(&self) -> u64 {
        self.total_epochs.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns the current best solution
    pub fn best(&self) -> MutexGuard<ThinSolution> {
        self.best.lock().unwrap()
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
