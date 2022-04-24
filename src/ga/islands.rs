use std::{
    sync::{mpsc::TryRecvError, Arc, Mutex},
    thread::JoinHandle,
};

use crate::solution::routing::{Plan, RoutingSolution};

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
    Migration(Vec<()>),
}
/// Slave-to-master message
pub struct Stm {
    slave: usize,
    message: StmMessage,
}

pub enum StmMessage {
    /// The current population of the GA
    Population(Vec<()>),
}

pub struct IslandGA<PS, R, M, S, F> {
    /// The config used for each GA instance
    pub config: Config<PS, R, M, S, F>,
    /// The handle of each spawned thread
    handles: Vec<JoinHandle<()>>,
    /// Transmitter-end of channels to each spawned thread
    txs: Vec<std::sync::mpsc::Sender<Mts>>,
    /// Receiver-end of the transmitter channel which each spawned thread has available.
    rx: std::sync::mpsc::Receiver<Stm>,
    /// The best individual ever recorded accross all islands.
    best: Arc<Mutex<(Vec<Plan>, f64)>>,
}

impl<PS, R, M, S, F> IslandGA<PS, R, M, S, F>
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
        let (tx, rx) = std::sync::mpsc::channel::<Stm>();
        let mut handles = Vec::with_capacity(count);
        let mut txs = Vec::with_capacity(count);
        let best = Arc::new(Mutex::new((
            RoutingSolution::empty(config.problem.clone()).to_vec(),
            std::f64::INFINITY,
        )));

        for i in 0..count {
            let (init, config, stm) = (init.clone(), config.clone(), tx.clone());
            let (tx, rx) = std::sync::mpsc::channel::<Mts>();
            let mutex = best.clone();

            handles.push(std::thread::spawn(move || {
                let mut ga = GeneticAlgorithm::new(init, config.clone());

                loop {
                    match rx.try_recv() {
                        // TODO: do something
                        Ok(Mts::Terminate) => return (),
                        Ok(Mts::GetPopulation) => stm
                            .send(Stm {
                                slave: i,
                                message: StmMessage::Population(Vec::new()),
                            })
                            .unwrap(),
                        Ok(Mts::Migration(migration)) => {}
                        // If there is nothing awaiting processing, we will just continue to the GA epoch
                        Err(TryRecvError::Empty) => (),
                        // This should not happen
                        Err(TryRecvError::Disconnected) => panic!("master tx disconnected"),
                    }
                    ga.epoch();
                    let island_best = ga.best_individual();
                    let island_fitness = config.fitness.of(&config.problem, island_best);
                    let mut best = mutex.lock().unwrap();

                    if island_fitness <= best.1 {
                        best.0.clone_from_slice(&island_best[..]);
                    }
                }
            }));

            txs.push(tx);
        }

        Self {
            config,
            handles,
            txs,
            rx,
            best,
        }
    }
}

impl<PS, R, M, S, F> Drop for IslandGA<PS, R, M, S, F> {
    fn drop(&mut self) {
        for tx in &self.txs {
            tx.send(Mts::Terminate).expect("sending failed");
        }

        // We need the join handles by value
        let handles = std::mem::replace(&mut self.handles, Vec::new());

        for handle in handles {
            handle.join().expect("worker thread panic");
        }
    }
}
