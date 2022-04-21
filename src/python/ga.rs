use crate::ga;

use crate::ga::chromosome::InitRoutingSolution;
use crate::ga::fitness::Weighted;
use crate::ga::mutations::AddRandom;
use crate::ga::mutations::AddSmart;
use crate::ga::mutations::Bounce;
use crate::ga::mutations::BounceMode;
use crate::ga::mutations::DistanceReduction;
use crate::ga::mutations::DistanceReductionMode;
use crate::ga::mutations::InterSwap;
use crate::ga::mutations::IntraSwap;
use crate::ga::mutations::RedCost;
use crate::ga::mutations::RemoveRandom;
use crate::ga::mutations::TimeSetter;
use crate::ga::mutations::Twerk;
use crate::ga::mutations::TwoOpt;
use crate::ga::mutations::TwoOptMode;
use crate::ga::parent_selection;
use crate::ga::Chain;
use crate::ga::Fitness;
use crate::ga::GeneticAlgorithm;
use crate::ga::ParentSelection;
use crate::ga::Recombination;
use crate::ga::SurvivalSelection;

use crate::ga::Mutation;
use crate::ga::Nop;
use crate::ga::Stochastic;

use crate::ga::recombinations::PIX;
use crate::ga::survival_selection;
use crate::ga::survival_selection::Elite;
use crate::models::quantity::F64Variables;
use crate::models::utils::ConvertVars;
use crate::problem::Problem;
use crate::python::Solution;
use crate::solution::Delivery;
use crate::solution::Visit;
use log::trace;
use pyo3::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;

use crate::solution::routing::RoutingSolution;

#[pyclass]
#[derive(Clone)]
pub struct PyMut {
    inner: Arc<Mutex<dyn ga::Mutation + Send>>,
}

impl Mutation for PyMut {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        self.inner.lock().unwrap().apply(problem, solution)
    }
}

#[pymethods]
impl PyMut {
    pub fn py_apply(&self, problem: Problem, routes: Vec<Vec<Visit>>) -> PyResult<Solution> {
        let arc = Arc::new(problem);
        let mut solution = RoutingSolution::new(arc.clone(), routes);
        self.inner
            .lock()
            .unwrap()
            .apply(&arc.clone(), &mut solution);
        let deliveries = solution
            .iter()
            .map(|r| {
                r.iter()
                    .map(|visit| Delivery::new(visit.node, 0, visit.time, 0.0))
                    .collect()
            })
            .collect();
        Ok(Solution::new(deliveries))
    }
}

#[pyfunction]
pub fn twerk() -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(Twerk::everybody())),
    }
}

#[pyfunction]
pub fn red_cost_mutation(max_visits: usize) -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(RedCost::red_cost_mutation(max_visits))),
    }
}

#[pyfunction]
pub fn local_search_red_cost(max_visits: usize, iterations: usize) -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(RedCost::red_cost_local_search(
            max_visits, iterations,
        ))),
    }
}

#[pyfunction]
pub fn bounce(passes: usize, mode: BounceMode) -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(Bounce::new(passes, mode))),
    }
}

#[pyfunction]
pub fn distance_reduction(mode: DistanceReductionMode) -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(DistanceReduction::new(mode))),
    }
}

#[pyfunction]
pub fn intra_swap() -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(IntraSwap {})),
    }
}

#[pyfunction]
/// 2-opt local search mutation
///
/// ## Arguments
///
/// * `time_limit` - The time limit in seconds for every local search in a **voyage**
pub fn two_opt_local(time_limit: u64, epsilon: f64) -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(TwoOpt::new(TwoOptMode::LocalSerach(
            time_limit, epsilon,
        )))),
    }
}

#[pyfunction]
pub fn add_random() -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(AddRandom::new())),
    }
}

#[pyfunction]
pub fn remove_random() -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(RemoveRandom::new())),
    }
}

#[pyfunction]
pub fn two_opt_intra() -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(TwoOpt::new(TwoOptMode::IntraRandom))),
    }
}

#[pyfunction]
pub fn inter_swap() -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(InterSwap {})),
    }
}

pub struct TimeSetterWrapper {
    inner: Arc<Mutex<TimeSetter>>,
}

impl TimeSetterWrapper {
    pub fn new(delay: f64) -> TimeSetterWrapper {
        let res = TimeSetterWrapper {
            inner: Arc::new(Mutex::new(TimeSetter::new(delay).unwrap())),
        };
        trace!("wrapper succesfully built");
        res
    }
}

// careful
unsafe impl Send for TimeSetterWrapper {}

impl Mutation for TimeSetterWrapper {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        self.inner.lock().unwrap().apply(problem, solution)
    }
}

#[pyfunction]
pub fn time_setter(delay: f64) -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(TimeSetterWrapper::new(delay))),
    }
}

#[pyfunction]
pub fn add_smart() -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(AddSmart {})),
    }
}

#[pyfunction]
pub fn stochastic(probability: f64, mutation: PyMut) -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(Stochastic::new(probability, mutation))),
    }
}

#[pyfunction]
pub fn chain(mutations: Vec<PyMut>) -> PyMut {
    let nop = || PyMut {
        inner: Arc::new(Mutex::new(Nop)),
    };

    mutations.into_iter().fold(nop(), |acc, x| PyMut {
        inner: Arc::new(Mutex::new(Chain(acc, x))),
    })
}

#[pyclass]
#[derive(Clone)]
pub struct PyRecombination {
    inner: Arc<Mutex<dyn ga::Recombination + Send>>,
}

#[pyfunction]
pub fn pix() -> PyRecombination {
    PyRecombination {
        inner: Arc::new(Mutex::new(PIX)),
    }
}

#[pyfunction]
pub fn recomb_stochastic(probability: f64, recombination: PyRecombination) -> PyRecombination {
    PyRecombination {
        inner: Arc::new(Mutex::new(Stochastic::new(probability, recombination))),
    }
}

#[pyfunction]
pub fn recomb_chain(mutations: Vec<PyRecombination>) -> PyRecombination {
    let nop = || PyRecombination {
        inner: Arc::new(Mutex::new(Nop)),
    };

    mutations.into_iter().fold(nop(), |acc, x| PyRecombination {
        inner: Arc::new(Mutex::new(Chain(acc, x))),
    })
}

impl Recombination for PyRecombination {
    fn apply(
        &mut self,
        problem: &Problem,
        left: &mut RoutingSolution,
        right: &mut RoutingSolution,
    ) {
        self.inner.lock().unwrap().apply(problem, left, right)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyParentSelection {
    inner: Arc<Mutex<dyn ga::ParentSelection + Send>>,
}

impl ParentSelection for PyParentSelection {
    fn init(&mut self, fitness_values: Vec<f64>) {
        self.inner.lock().unwrap().init(fitness_values)
    }

    fn sample(&mut self) -> usize {
        self.inner.lock().unwrap().sample()
    }
}

/// Select proportionate to 1 / (1 + fitness)
#[pyfunction]
pub fn proportionate() -> PyParentSelection {
    PyParentSelection {
        inner: Arc::new(Mutex::new(parent_selection::Proportionate::with_fn(|x| {
            1.0 / (1.0 + x)
        }))),
    }
}

/// Tournament selection
#[pyfunction]
pub fn tournament(k: usize) -> Option<PyParentSelection> {
    Some(PyParentSelection {
        inner: Arc::new(Mutex::new(parent_selection::Tournament::new(k)?)),
    })
}

#[pyfunction]
pub fn greedy() -> survival_selection::Greedy {
    survival_selection::Greedy
}

#[pyclass]
#[derive(Clone)]
pub struct PyElite {
    inner: Elite<survival_selection::Proportionate<fn(f64) -> f64>>,
}

fn key(x: f64) -> f64 {
    1.0 / (1.0 + x)
}

#[pyfunction]
pub fn elite(k: usize) -> PyElite {
    PyElite {
        inner: Elite(k, survival_selection::Proportionate(key)),
    }
}

impl SurvivalSelection for PyElite {
    fn select_survivors<F>(
        &mut self,
        objective_fn: F,
        population: &[RoutingSolution],
        parents: &[&RoutingSolution],
        children: &[RoutingSolution],
        out: &mut [RoutingSolution],
    ) where
        F: Fn(&RoutingSolution) -> f64,
    {
        self.inner
            .select_survivors(objective_fn, population, parents, children, out)
    }
}

#[pyfunction]
pub fn weighted(warp: f64, violation: f64, revenue: f64, cost: f64) -> Weighted {
    Weighted {
        warp,
        violation,
        revenue,
        cost,
    }
}

#[pyclass(name = "GeneticAlgorithm")]
pub struct PyGA {
    inner:
        Arc<Mutex<GeneticAlgorithm<PyParentSelection, PyRecombination, PyMut, PyElite, Weighted>>>,
}

// Don't do this.
unsafe impl Send for PyGA {}

#[pymethods]
impl PyGA {
    #[new]
    pub fn new(
        problem: Problem,
        population_size: usize,
        child_count: usize,
        parent_selection: PyParentSelection,
        recombination: PyRecombination,
        mutation: PyMut,
        selection: PyElite,
        fitness: Weighted,
    ) -> Self {
        PyGA {
            inner: Arc::new(Mutex::new(GeneticAlgorithm::new(
                Arc::new(problem),
                population_size,
                child_count,
                InitRoutingSolution,
                parent_selection,
                recombination,
                mutation,
                selection,
                fitness,
            ))),
        }
    }

    pub fn epoch(&mut self) {
        self.inner.lock().unwrap().epoch()
    }

    pub fn population(&self) -> Vec<(Vec<Vec<Visit>>, F64Variables, f64, (f64, f64, f64, f64))> {
        let ga = self.inner.lock().unwrap();
        let problem = &ga.problem;
        ga.population
            .iter()
            .map(|solution| {
                let routing = solution
                    .iter_with_origin()
                    .map(|it| it.collect::<Vec<_>>())
                    .collect::<Vec<_>>();

                let lp = solution.quantities();
                let x = lp.vars.x.convert(&lp.model).unwrap();
                let s = lp.vars.s.convert(&lp.model).unwrap();
                let l = lp.vars.l.convert(&lp.model).unwrap();
                let w = lp.vars.w.convert(&lp.model).unwrap();

                let v = F64Variables { w, x, s, l };

                let obj = (
                    solution.warp() as f64,
                    solution.violation(),
                    solution.revenue(),
                    solution.cost(),
                );

                (routing, v, ga.fitness.of(problem, solution), obj)
            })
            .collect::<Vec<_>>()
    }
}
