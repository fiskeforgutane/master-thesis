pub mod ga;
pub mod models;
pub mod parse;
pub mod problem;
pub mod python;
pub mod quants;
pub mod rolling_horizon;
pub mod solution;
pub mod termination;
pub mod utils;

use crate::python::ga::*;
use crate::python::*;
use ga::{
    fitness::Weighted,
    mutations::{BounceMode, DistanceReductionMode},
};
use models::{exact_model::model::ExactModelSolver, quantity::F64Variables};

use problem::Compartment;

use problem::Node;
use problem::NodeType;
use problem::Problem;
use problem::Vessel;

use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use pyo3_log;
use pyo3_log::Logger;
use python::distributed::ComputeNode;
use quants::Order;
use solution::Visit;

#[pyfunction]
pub fn test_logging() {
    log::error!("This is an error");
    log::warn!("This is a warning");
    log::info!("This is some info");
    log::debug!("This is a debug message");
    log::trace!("This is a trace message");
    log::info!("Is trace enabled: {}", log::log_enabled!(log::Level::Trace));
}

#[pyfunction]
pub fn write_exact(problem: Problem, path: &str) -> PyResult<()> {
    ExactModelSolver::build_and_write(&problem, path).map_err(pyerr)
}

/// A submodule for the GA
#[pymodule]
fn ga(_py: Python, m: &PyModule) -> PyResult<()> {
    // Mutations
    m.add_function(wrap_pyfunction!(twerk, m)?)?;
    m.add_function(wrap_pyfunction!(local_search_red_cost, m)?)?;
    m.add_function(wrap_pyfunction!(red_cost_mutation, m)?)?;
    m.add_function(wrap_pyfunction!(bounce, m)?)?;
    m.add_function(wrap_pyfunction!(intra_swap, m)?)?;
    m.add_function(wrap_pyfunction!(two_opt_local, m)?)?;
    m.add_function(wrap_pyfunction!(two_opt_intra, m)?)?;
    m.add_function(wrap_pyfunction!(inter_swap, m)?)?;
    m.add_function(wrap_pyfunction!(distance_reduction, m)?)?;
    m.add_function(wrap_pyfunction!(add_random, m)?)?;
    m.add_function(wrap_pyfunction!(remove_random, m)?)?;
    m.add_function(wrap_pyfunction!(add_smart, m)?)?;
    m.add_function(wrap_pyfunction!(time_setter, m)?)?;
    m.add_function(wrap_pyfunction!(python::ga::swap_star, m)?)?;
    m.add_function(wrap_pyfunction!(write_model, m)?)?;

    // Mutation combinators
    m.add_function(wrap_pyfunction!(chain, m)?)?;
    m.add_function(wrap_pyfunction!(stochastic, m)?)?;

    // Recombinations
    m.add_function(wrap_pyfunction!(pix, m)?)?;

    // Recombination combinators
    m.add_function(wrap_pyfunction!(recomb_chain, m)?)?;

    // Survival and parent selection
    m.add_function(wrap_pyfunction!(proportionate, m)?)?;
    m.add_function(wrap_pyfunction!(tournament, m)?)?;
    m.add_function(wrap_pyfunction!(greedy, m)?)?;
    m.add_function(wrap_pyfunction!(elite, m)?)?;

    // Fitness
    m.add_function(wrap_pyfunction!(weighted, m)?)?;

    m.add_class::<BounceMode>()?;
    m.add_class::<DistanceReductionMode>()?;
    m.add_class::<PyMut>()?;
    m.add_class::<PyRecombination>()?;
    m.add_class::<PyParentSelection>()?;
    m.add_class::<PyElite>()?;
    m.add_class::<Weighted>()?;
    m.add_class::<PyGA>()?;
    m.add_class::<ComputeNode>()?;

    Ok(())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn master(_py: Python, m: &PyModule) -> PyResult<()> {
    let _handle = Logger::new(_py, pyo3_log::Caching::LoggersAndLevels)?
        .filter(log::LevelFilter::Trace)
        .install()
        .expect("A logger has already been installed:(");

    m.add_wrapped(wrap_pymodule!(ga))?;

    m.add_function(wrap_pyfunction!(test_logging, m)?)?;
    m.add_function(wrap_pyfunction!(initial_orders, m)?)?;
    m.add_function(wrap_pyfunction!(initial_quantities, m)?)?;
    m.add_function(wrap_pyfunction!(solve_quantities, m)?)?;
    m.add_function(wrap_pyfunction!(objective_terms, m)?)?;
    m.add_function(wrap_pyfunction!(solve_multiple_quantities, m)?)?;
    m.add_function(wrap_pyfunction!(python::swap_star_test, m)?)?;
    m.add_function(wrap_pyfunction!(write_exact, m)?)?;
    m.add_class::<Problem>()?;
    m.add_class::<Vessel>()?;
    m.add_class::<Node>()?;
    m.add_class::<NodeType>()?;
    m.add_class::<Compartment>()?;
    m.add_class::<Order>()?;
    m.add_class::<Visit>()?;
    m.add_class::<F64Variables>()?;
    m.add_class::<ObjectiveTerms>()?;

    Ok(())
}

#[allow(unused_macros)]
macro_rules! impl_repr {
    ($type:ident) => {
        #[pymethods]
        impl $type {
            pub fn __str__(&self) -> String {
                format!("{:#?}", self)
            }

            pub fn __repr__(&self) -> String {
                self.__str__()
            }
        }
    };
}
