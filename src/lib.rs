pub mod destroy_and_repair;
pub mod ga;
pub mod models;
pub mod problem;
pub mod quants;
pub mod route_pool;
pub mod solution;
pub mod utils;

use ga::chromosome::Chromosome;
use ga::mutations::Twerk;
use ga::mutations::TwoOpt;
use ga::mutations::TwoOptMode;
use ga::Mutation;
use ga::Nop;
use ga::Stochastic;
use models::quantity::F64Variables;
use models::quantity::QuantityLp;
use problem::Compartment;
use problem::Cost;
use problem::Distance;
use problem::Inventory;
use problem::Node;
use problem::NodeType;
use problem::Problem;
use problem::Quantity;
use problem::TimeIndex;
use problem::Vessel;
use problem::VesselIndex;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_log;
use pyo3_log::Logger;
use quants::Order;
use quants::Quantities;
use solution::Visit;
use solution::{Delivery, Evaluation};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::Mutex;

use crate::solution::routing::RoutingSolution;

#[pyfunction]
pub fn test_logging() {
    log::error!("This is an error");
    log::warn!("This is a warning");
    log::info!("This is some info");
    log::debug!("This is a debug message");
    log::trace!("This is a trace message");
    log::info!("Is trace enabled: {}", log::log_enabled!(log::Level::Trace));
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Solution {
    #[pyo3(get, set)]
    pub routes: Vec<Vec<Delivery>>,
}

#[pymethods]
impl Solution {
    #[new]
    pub fn new(routes: Vec<Vec<Delivery>>) -> Self {
        Self { routes }
    }

    pub fn __len__(&self) -> usize {
        self.routes.len()
    }

    pub fn __getitem__(&self, idx: usize) -> Vec<Delivery> {
        self.routes[idx].clone()
    }

    pub fn evaluate(&self, problem: &Problem) -> PyResult<solution::Evaluation> {
        let solution = solution::FullSolution::new(problem, self.routes.clone())
            .map_err(|err| PyErr::new::<PyValueError, _>(format!("{:?}", err)))?;

        Ok(solution.evaluation())
    }

    pub fn vessel_inventory_at(
        &self,
        problem: &Problem,
        vessel: VesselIndex,
        time: TimeIndex,
    ) -> PyResult<Inventory> {
        let solution = solution::FullSolution::new(problem, self.routes.clone())
            .map_err(|err| PyErr::new::<PyValueError, _>(format!("{:?}", err)))?;

        Ok(solution.vessel_inventory_at(vessel, time))
    }

    pub fn node_product_inventory_at(
        &self,
        problem: &Problem,
        node: usize,
        product: usize,
        time: usize,
    ) -> PyResult<f64> {
        let solution = solution::FullSolution::new(problem, self.routes.clone())
            .map_err(|err| PyErr::new::<PyValueError, _>(format!("{:?}", err)))?;

        Ok(solution.node_product_inventory_at(node, product, time))
    }

    pub fn __str__(&self) -> String {
        format!("{:#?}", self.routes)
    }

    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

#[pymethods]
impl Problem {
    #[new]
    pub fn new_py(
        vessels: Vec<Vessel>,
        nodes: Vec<Node>,
        timesteps: usize,
        products: usize,
        distances: Vec<Vec<Distance>>,
    ) -> PyResult<Problem> {
        Problem::new(vessels, nodes, timesteps, products, distances)
            .map_err(|err| PyErr::new::<PyValueError, _>(format!("{:?}", err)))
    }

    pub fn __str__(&self) -> String {
        format!("{:#?}", self)
    }

    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

#[pymethods]
impl Vessel {
    #[new]
    pub fn new_py(
        compartments: Vec<Quantity>,
        speed: f64,
        travel_unit_cost: Cost,
        empty_travel_unit_cost: Cost,
        time_unit_cost: Cost,
        available_from: usize,
        initial_inventory: Vec<f64>,
        origin: usize,
        class: String,
        index: usize,
    ) -> PyResult<Vessel> {
        let inventory = Inventory::new(&initial_inventory)
            .ok_or(PyErr::new::<PyValueError, _>("invalid inventory"))?;

        Ok(Vessel::new(
            compartments.iter().map(|&x| Compartment(x)).collect(),
            speed,
            travel_unit_cost,
            empty_travel_unit_cost,
            time_unit_cost,
            available_from,
            inventory.fixed(),
            origin,
            class,
            index,
        ))
    }

    pub fn __str__(&self) -> String {
        format!("{:#?}", self)
    }

    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

#[pymethods]
impl Order {
    pub fn __str__(&self) -> String {
        format!("{:#?}", self)
    }

    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

#[pymethods]
impl Node {
    #[new]
    pub fn new_py(
        name: String,
        kind: NodeType,
        index: usize,
        port_capacity: Vec<usize>,
        min_unloading_amount: Quantity,
        max_loading_amount: Quantity,
        port_fee: Cost,
        capacity: Vec<f64>,
        inventory_changes: Vec<Vec<f64>>,
        revenue: Cost,
        initial_inventory: Vec<f64>,
    ) -> PyResult<Node> {
        let err = || PyErr::new::<PyValueError, _>("invalid inventory");

        let capacity = Inventory::new(&capacity).ok_or(err())?;
        let initial_inventory = Inventory::new(&initial_inventory).ok_or(err())?;

        let mut changes = Vec::new();

        for x in &inventory_changes {
            let inventory = Inventory::new(x).ok_or(err())?;
            changes.push(inventory.fixed());
        }

        Ok(Node::new(
            name,
            kind,
            index,
            port_capacity,
            min_unloading_amount,
            max_loading_amount,
            port_fee,
            capacity.fixed(),
            changes,
            revenue,
            initial_inventory.fixed(),
        ))
    }

    pub fn __str__(&self) -> String {
        format!("{:#?}", self)
    }

    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

#[pymethods]
impl Compartment {
    pub fn __str__(&self) -> String {
        format!("{:#?}", self)
    }

    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

#[pymethods]
impl Chromosome {
    pub fn __str__(&self) -> String {
        format!("{:#?}", self)
    }

    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

fn pyerr<D: Debug>(err: D) -> PyErr {
    PyErr::new::<PyValueError, _>(format!("{:?}", err))
}

#[pymethods]
impl Evaluation {
    pub fn __str__(&self) -> String {
        format!("{:#?}", self)
    }

    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

#[pymethods]
impl Inventory {
    pub fn __str__(&self) -> String {
        format!(
            "Inventory({:?})",
            (0..self.num_products())
                .map(|i| self[i])
                .collect::<Vec<_>>()
        )
    }

    pub fn __repr__(&self) -> String {
        self.__str__()
    }

    pub fn __len__(&self) -> usize {
        self.num_products()
    }

    pub fn __getitem__(&self, idx: usize) -> f64 {
        self[idx]
    }
}

#[pymethods]
impl Visit {
    #[new]
    /// Construct a new visit without checkout its validity.
    /// Prefer to use `Visit::new` unless you have ensured that `node` and `time` are within bounds.
    pub fn new_py(node: usize, time: usize) -> Visit {
        Visit::new_unchecked(node, time)
    }

    pub fn __str__(&self) -> String {
        format!("Visit(n = {}, t = {})", self.node, self.time)
    }

    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

#[pyfunction]
fn chromosome(problem: &Problem) -> PyResult<Chromosome> {
    Chromosome::new(problem).map_err(pyerr)
}

#[pyfunction]
fn initial_quantities(problem: &Problem, product: usize) -> HashMap<usize, f64> {
    Quantities::quantities(problem, product)
}

#[pyfunction]
fn initial_orders(problem: &Problem) -> PyResult<Vec<Order>> {
    quants::initial_orders(&problem).map_err(pyerr)
}

#[pyfunction]
fn solve_quantities(problem: Problem, routes: Vec<Vec<Visit>>) -> PyResult<F64Variables> {
    let mut lp = QuantityLp::new(&problem).map_err(pyerr)?;
    let solution = RoutingSolution::new(Arc::new(problem), routes);
    lp.configure(&solution).map_err(pyerr)?;
    let res = lp.solve_python().map_err(pyerr)?;
    Ok(res)
}

#[pyfunction]
fn solve_multiple_quantities(
    problem: Problem,
    solutions: Vec<Vec<Vec<Visit>>>,
) -> PyResult<Vec<F64Variables>> {
    let mut lp = QuantityLp::new(&problem).map_err(pyerr)?;

    let mut results = Vec::new();
    let arc = Arc::new(problem);
    for routes in solutions {
        let solution = RoutingSolution::new(arc.clone(), routes);
        lp.configure(&solution).map_err(pyerr)?;
        results.push(lp.solve_python().map_err(pyerr)?);
    }

    Ok(results)
}

#[pyclass]
#[derive(Clone)]
pub struct PyMut {
    inner: Arc<Mutex<dyn Mutation + Send>>,
}

impl Mutation for PyMut {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        self.inner.lock().unwrap().apply(problem, solution)
    }
}

#[pymethods]
impl PyMut {
    fn py_apply(&mut self, problem: Problem, routes: Vec<Vec<Visit>>) -> Vec<Vec<usize>> {
        let arc = Arc::new(problem);
        let mut solution = RoutingSolution::new(arc.clone(), routes);
        self.inner
            .lock()
            .unwrap()
            .apply(&arc.clone(), &mut solution);
        solution
            .iter()
            .map(|r| r.iter().map(|v| v.node).collect())
            .collect()
    }
}

#[pyfunction]
fn twerk() -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(Twerk::everybody())),
    }
}

#[pyfunction]
fn two_opt_local(iters: usize, improvement: f64) -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(TwoOpt::new(TwoOptMode::LocalSerach(
            improvement,
            iters,
        )))),
    }
}

#[pyfunction]
fn two_opt_intra() -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(TwoOpt::new(TwoOptMode::IntraRandom))),
    }
}

#[pyfunction]
fn stochastic(probability: f64, mutation: PyMut) -> PyMut {
    PyMut {
        inner: Arc::new(Mutex::new(Stochastic::new(probability, mutation))),
    }
}

#[pyfunction]
fn chain(mutations: Vec<PyMut>) -> PyMut {
    let nop = || PyMut {
        inner: Arc::new(Mutex::new(Nop)),
    };

    mutations.into_iter().fold(nop(), |acc, x| PyMut {
        inner: Arc::new(Mutex::new(chain!(acc, x))),
    })
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

    m.add_function(wrap_pyfunction!(test_logging, m)?)?;
    m.add_function(wrap_pyfunction!(initial_orders, m)?)?;
    m.add_function(wrap_pyfunction!(initial_quantities, m)?)?;
    m.add_function(wrap_pyfunction!(solve_quantities, m)?)?;
    m.add_function(wrap_pyfunction!(solve_multiple_quantities, m)?)?;
    m.add_function(wrap_pyfunction!(twerk, m)?)?;
    m.add_function(wrap_pyfunction!(chain, m)?)?;
    m.add_function(wrap_pyfunction!(stochastic, m)?)?;
    m.add_class::<Problem>()?;
    m.add_class::<Solution>()?;
    m.add_class::<Vessel>()?;
    m.add_class::<Node>()?;
    m.add_class::<NodeType>()?;
    m.add_class::<Compartment>()?;
    m.add_class::<Delivery>()?;
    m.add_class::<Evaluation>()?;
    m.add_class::<Order>()?;
    m.add_class::<Chromosome>()?;
    m.add_class::<Visit>()?;
    m.add_class::<F64Variables>()?;
    m.add_class::<PyMut>()?;

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
