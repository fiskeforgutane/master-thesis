use crate::ga;

use crate::ga::mutations::Bounce;
use crate::ga::mutations::InterSwap;
use crate::ga::mutations::IntraSwap;
use crate::ga::mutations::RedCost;
use crate::ga::mutations::Twerk;
use crate::ga::mutations::TwoOpt;
use crate::ga::mutations::{BounceMode, RedCostMode};
use crate::ga::parent_selection;
use crate::ga::Chain;
use crate::ga::ParentSelection;
use crate::ga::Recombination;
use crate::ga::SurvivalSelection;

use crate::ga::Mutation;
use crate::ga::Nop;
use crate::ga::Stochastic;

use crate::ga::recombinations::PIX;
use crate::ga::survival_selection;
use crate::ga::survival_selection::Elite;
use crate::problem::Problem;
use pyo3::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;

use crate::solution::routing::RoutingSolution;

use super::pyerr;

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

#[pyfunction]
pub fn twerk() -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(Twerk::everybody())),
    }
}

#[pyfunction]
pub fn red_cost(mode: RedCostMode, max_visits: usize) -> PyMut {
    match mode {
        RedCostMode::Mutate => PyMut {
            inner: Arc::new(Mutex::new(RedCost::red_cost_mutation(max_visits))),
        },
        RedCostMode::LocalSerach => PyMut {
            inner: Arc::new(Mutex::new(RedCost::red_cost_local_search(max_visits))),
        },
    }
}

#[pyfunction]
pub fn bounce(passes: usize, mode: BounceMode) -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(Bounce::new(passes, mode))),
    }
}

#[pyfunction]
pub fn intra_swap() -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(IntraSwap {})),
    }
}

#[pyfunction]
pub fn two_opt() -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(TwoOpt {})),
    }
}

#[pyfunction]
pub fn inter_swap() -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(InterSwap {})),
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
