use std::sync::{Arc, Mutex};

use pyo3::{pyclass, pymethods};

use crate::{
    chain,
    ga::{
        self,
        fitness::Weighted,
        initialization::InitRoutingSolution,
        islands::IslandGA,
        mutations::{
            AddRandom, AddSmart, Bounce, BounceMode, Dedup, DedupPolicy, InterSwap, IntraSwap,
            RedCost, RemoveRandom, ReplaceNode, TimeSetter, Twerk, TwoOpt, TwoOptMode,
        },
        parent_selection::Tournament,
        recombinations::PIX,
        survival_selection::{self, Elite},
        Fitness, Mutation, ParentSelection, Recombination, Stochastic, SurvivalSelection,
    },
    problem::Problem,
    solution::Visit,
};

/// To avoid having to type out the humongous GA type.
trait AnyIslandGa {
    /// Perform a migration between the islands
    fn intra_migration(&mut self, count: usize);
    /// Migrate a set of external individuals
    fn migrate(&mut self, individuals: Vec<Vec<Vec<Visit>>>);
    /// Return the populations in each island.
    fn populations(&mut self) -> Vec<Vec<Vec<Vec<Visit>>>>;
}

impl<PS, R, M, S, F> AnyIslandGa for IslandGA<PS, R, M, S, F>
where
    PS: ParentSelection,
    R: Recombination,
    M: Mutation,
    S: SurvivalSelection,
    F: Fitness + Clone + Send + 'static,
{
    fn intra_migration(&mut self, count: usize) {
        self.migrate(count)
    }

    fn populations(&mut self) -> Vec<Vec<Vec<Vec<Visit>>>> {
        self.populations()
    }

    fn migrate(&mut self, individuals: Vec<Vec<Vec<Visit>>>) {
        self.insert(individuals);
    }
}

// Oh well
unsafe impl Send for ComputeNode {}

/// A node running an Islanding GA.
#[pyclass]
pub struct ComputeNode {
    ga: Arc<Mutex<dyn AnyIslandGa>>,
}

#[pymethods]
impl ComputeNode {
    pub fn intra_migration(&self, count: usize) {
        let mut ga = self.ga.lock().unwrap();
        ga.intra_migration(count);
    }

    pub fn migrate(&self, individuals: Vec<Vec<Vec<Visit>>>) {
        let mut ga = self.ga.lock().unwrap();
        ga.migrate(individuals)
    }

    #[new]
    pub fn new(problem: Problem, islands: usize) -> Self {
        let problem = Arc::new(problem);
        let max_revenue = problem.max_revenue();
        let config = move || ga::Config {
            problem: problem.clone(),
            population_size: 100,
            child_count: 100,
            parent_selection: Tournament::new(3).unwrap(),
            recombination: Stochastic::new(0.10, PIX),
            mutation: chain!(
                Stochastic::new(0.03, AddRandom::new()),
                Stochastic::new(0.03, RemoveRandom::new()),
                Stochastic::new(0.03, InterSwap),
                Stochastic::new(0.03, IntraSwap),
                Stochastic::new(0.03, RedCost::red_cost_mutation(10)),
                Stochastic::new(0.03, Twerk::everybody()),
                Stochastic::new(0.03, Twerk::some_random_person()),
                Stochastic::new(0.03, TwoOpt::new(TwoOptMode::IntraRandom)),
                Stochastic::new(0.03, TimeSetter::new(0.4).unwrap()),
                Stochastic::new(0.03, TimeSetter::new(0.0).unwrap()),
                Stochastic::new(0.03, Bounce::new(3, BounceMode::All)),
                Stochastic::new(0.03, Bounce::new(3, BounceMode::Random)),
                Stochastic::new(0.03, AddSmart),
                Stochastic::new(0.05, ReplaceNode::new(0.10)),
                Stochastic::new(0.10, Dedup(DedupPolicy::KeepFirst)),
                Stochastic::new(0.10, Dedup(DedupPolicy::KeepLast))
            ),
            selection: Elite(1, survival_selection::Proportionate(|x| 1.0 / (1.0 + x))),
            fitness: Weighted {
                warp: 1e8,
                violation: 1e4,
                revenue: -1.0,
                cost: 1.0,
                approx_berth_violation: 1e8,
                spot: 1.0,
                travel_empty: 1e5,
                travel_at_cap: 1e5,
                offset: max_revenue + 1.0,
            },
        };

        Self {
            ga: Arc::new(Mutex::new(IslandGA::new(
                InitRoutingSolution,
                config,
                islands,
            ))),
        }
    }
}
