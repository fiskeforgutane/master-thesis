pub mod distributed;
pub mod ga;

use crate::ga::chromosome::Chromosome;
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
use crate::quants;
use crate::quants::Order;
use crate::quants::Quantities;
use crate::solution;
use crate::solution::Visit;
use crate::solution::{Delivery, Evaluation};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use crate::solution::routing::RoutingSolution;

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

    pub fn solution_from_json(&self, string: &str) -> PyResult<Vec<Vec<Visit>>> {
        serde_json::from_str(string).map_err(pyerr)
    }

    pub fn json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(self).map_err(pyerr)
    }

    #[staticmethod]
    pub fn from_json(string: &str) -> PyResult<Problem> {
        serde_json::from_str(string).map_err(pyerr)
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
        spot_market_limit_per_time: f64,
        spot_market_limit: f64,
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
            spot_market_limit_per_time,
            spot_market_limit,
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
    #[new]
    pub fn new_py(problem: &Problem) -> PyResult<Chromosome> {
        Chromosome::new(problem).map_err(pyerr)
    }

    pub fn __str__(&self) -> String {
        format!("{:#?}", self)
    }

    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

pub fn pyerr<D: Debug>(err: D) -> PyErr {
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
pub fn chromosome(problem: &Problem) -> PyResult<Chromosome> {
    Chromosome::new(problem).map_err(pyerr)
}

#[pyfunction]
pub fn initial_quantities(problem: &Problem, product: usize) -> HashMap<usize, f64> {
    Quantities::quantities(problem, product)
}

#[pyfunction]
pub fn initial_orders(problem: &Problem) -> PyResult<Vec<Order>> {
    quants::initial_orders(&problem).map_err(pyerr)
}

#[pyfunction]
pub fn solve_quantities(
    problem: Problem,
    routes: Vec<Vec<Visit>>,
    semicont: bool,
    berth: bool,
) -> PyResult<F64Variables> {
    let mut lp = QuantityLp::new(&problem).map_err(pyerr)?;
    let solution = RoutingSolution::new(Arc::new(problem), routes);
    lp.configure(&solution, semicont, berth).map_err(pyerr)?;
    let res = lp.solve_python().map_err(pyerr)?;
    Ok(res)
}

#[pyfunction]
pub fn solve_multiple_quantities(
    problem: Problem,
    solutions: Vec<Vec<Vec<Visit>>>,
    semicont: bool,
    berth: bool,
) -> PyResult<Vec<F64Variables>> {
    let mut lp = QuantityLp::new(&problem).map_err(pyerr)?;

    let mut results = Vec::new();
    let arc = Arc::new(problem);
    for routes in solutions {
        let solution = RoutingSolution::new(arc.clone(), routes);
        lp.configure(&solution, semicont, berth).map_err(pyerr)?;
        results.push(lp.solve_python().map_err(pyerr)?);
    }

    Ok(results)
}
