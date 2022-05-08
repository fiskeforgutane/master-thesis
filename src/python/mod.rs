pub mod distributed;
pub mod ga;

use crate::ga::mutations::SwapStar;
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
use crate::problem::Vessel;
use crate::quants;
use crate::quants::Order;
use crate::quants::Quantities;
use crate::solution::Visit;
use log::trace;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use crate::solution::routing::RoutingSolution;

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
        spot_market_unit_price: f64,
        spot_market_discount_factor: f64,
        coordinates: (f64, f64),
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
            spot_market_unit_price,
            spot_market_discount_factor,
            coordinates,
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

pub fn pyerr<D: Debug>(err: D) -> PyErr {
    PyErr::new::<PyValueError, _>(format!("{:?}", err))
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
    load_restrictions: bool,
) -> PyResult<F64Variables> {
    let mut lp = QuantityLp::new(&problem).map_err(pyerr)?;
    let solution = RoutingSolution::new(Arc::new(problem), routes);
    lp.configure(&solution, semicont, berth, load_restrictions)
        .map_err(pyerr)?;
    let res = lp.solve_python().map_err(pyerr)?;
    Ok(res)
}

#[pyfunction]
pub fn solve_multiple_quantities(
    problem: Problem,
    solutions: Vec<Vec<Vec<Visit>>>,
    semicont: bool,
    berth: bool,
    load_restrictions: bool,
) -> PyResult<Vec<F64Variables>> {
    let mut lp = QuantityLp::new(&problem).map_err(pyerr)?;

    let mut results = Vec::new();
    let arc = Arc::new(problem);
    for routes in solutions {
        let solution = RoutingSolution::new(arc.clone(), routes);
        lp.configure(&solution, semicont, berth, load_restrictions)
            .map_err(pyerr)?;
        results.push(lp.solve_python().map_err(pyerr)?);
    }

    Ok(results)
}

#[pyfunction]
pub fn swap_star_test(
    r1: usize,
    r2: usize,
    routes: Vec<Vec<Visit>>,
    problem: Problem,
) -> Vec<Vec<Visit>> {
    trace!("here");
    let arc = Arc::new(problem);
    trace!("here");
    let mut solution = RoutingSolution::new(arc.clone(), routes);
    trace!("testing_overlap");
    let overlapping = SwapStar::overlapping(r1, r2, &solution, &arc);
    trace!("overlapping: {:?}", overlapping);
    let best_swap = SwapStar::best_swap(&solution[r1], &solution[r2], &arc);
    trace!("Best identified swap: {:?}", best_swap);
    if let Some(((v1, p1), (v2, p2))) = best_swap {
        let into_plan2 = SwapStar::new_visit(r2, p1, solution[r1][v1], &solution);
        let into_plan1 = SwapStar::new_visit(r1, p2, solution[r2][v2], &solution);
        SwapStar::apply_swap(&mut solution, r1, r2, into_plan1, into_plan2, v1, v2);
    }
    solution
        .iter()
        .map(|plan| plan.iter().cloned().collect())
        .collect()
}

#[pyfunction]
pub fn objective_terms(
    problem: Problem,
    routes: Vec<Vec<Visit>>,
    semicont: bool,
    berth: bool,
) -> PyResult<ObjectiveTerms> {
    let mut lp = QuantityLp::new(&problem).map_err(pyerr)?;
    let solution = RoutingSolution::new(Arc::new(problem), routes);
    lp.configure(&solution, semicont, berth).map_err(pyerr)?;
    let res = lp.solve_python().map_err(pyerr)?;

    Ok(ObjectiveTerms {
        revenue: res.revenue,
        violation: res.violation,
        spot: res.spot,
        cost: solution.cost(),
        timing: res.timing,
        warp: solution.warp() as f64,
    })
}

#[pyfunction]
pub fn write_model(
    filename: &str,
    problem: Problem,
    routes: Vec<Vec<Visit>>,
    semicont: bool,
    berth: bool,
) -> PyResult<()> {
    let mut lp = QuantityLp::new(&problem).map_err(pyerr)?;
    let solution = RoutingSolution::new(Arc::new(problem), routes);
    lp.configure(&solution, semicont, berth).map_err(pyerr)?;
    lp.model.write(filename).map_err(pyerr)?;

    Ok(())
}

#[pyclass]
pub struct ObjectiveTerms {
    #[pyo3(get)]
    pub revenue: f64,
    #[pyo3(get)]
    pub violation: f64,
    #[pyo3(get)]
    pub spot: f64,
    #[pyo3(get)]
    pub cost: f64,
    #[pyo3(get)]
    pub timing: f64,
    #[pyo3(get)]
    pub warp: f64,
}
