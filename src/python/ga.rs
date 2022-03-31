use crate::ga;
use crate::ga::chromosome::Chromosome;
use crate::ga::mutations::Bounce;
use crate::ga::mutations::InterSwap;
use crate::ga::mutations::IntraSwap;
use crate::ga::mutations::RedCost;
use crate::ga::mutations::Twerk;
use crate::ga::mutations::TwoOpt;
use crate::ga::mutations::{BounceMode, RedCostMode};
use crate::ga::Chain;
use crate::ga::GeneticAlgorithm;
use crate::ga::Mutation;
use crate::ga::Nop;
use crate::ga::Stochastic;
use crate::models::quantity::F64Variables;
use crate::models::quantity::QuantityLp;
use crate::problem::Compartment;
use crate::problem::Cost;
use crate::problem::Distance;
use crate::problem::Inventory;
use crate::problem::Node;
use crate::problem::NodeType;
use crate::problem::Problem;
use crate::problem::Quantity;
use crate::problem::TimeIndex;
use crate::problem::Vessel;
use crate::problem::VesselIndex;
use crate::quants::Order;
use crate::quants::Quantities;
use crate::solution::Visit;
use crate::solution::{Delivery, Evaluation};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_log;
use pyo3_log::Logger;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::Mutex;

use crate::solution::routing::RoutingSolution;

#[pyclass]
#[derive(Clone)]
pub struct PyMut {
    inner: Arc<Mutex<dyn ga::Mutation + Send>>,
}

#[pyclass]
#[derive(Clone)]
pub struct PyRecombination {
    inner: Arc<Mutex<dyn ga::Recombination + Send>>,
}

#[pyclass]
#[derive(Clone)]
pub struct PyParentSelection {
    inner: Arc<Mutex<dyn ga::ParentSelection + Send>>,
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
