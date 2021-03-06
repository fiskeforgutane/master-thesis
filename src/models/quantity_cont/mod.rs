use std::collections::HashMap;

use grb::{attr, c, expr::GurobiSum, Constr, Model, Var};
use itertools::{iproduct, Itertools};
use log::trace;
use pyo3::pyclass;
use rand::{prelude::SliceRandom, thread_rng};

use crate::{
    models::utils::AddVars,
    problem::{NodeIndex, NodeType, Problem},
    solution::routing::RoutingSolution,
};

use super::utils::ConvertVars;

type VisitIndex = usize;

/// Variables for the continuous time quantity deciding LP
pub struct Variables {
    /// The shortage or overflow at the **beginning** of visit (node: i,visitnumber: m) of product p.
    /// Indexed: (i,m,p)
    pub w: Vec<Vec<Vec<Var>>>,

    /// The quantity delivered or picked up at visit (node: i,visitnumber: m) of product p.
    /// Indexed: (i,m,p)
    pub x: Vec<Vec<Vec<Var>>>,

    /// The inventory level at the **beginning** of visit (node: i,visitnumber: m) of product p.
    /// /// Indexed: (i,m,p)
    pub s: Vec<Vec<Vec<Var>>>,

    /// The load of the vessel performing visit (node: i,visitnumber: m) of product p **after** the visit is finished.
    /// /// Indexed: (i,m,p)
    pub l: Vec<Vec<Vec<Var>>>,

    /// The time at which visit (node: i,visitnumber: m) begins.
    /// Indexed (i,m)
    pub t: Vec<Vec<Var>>,
}

#[pyclass]
/// A Pyclass enabling exposing of `quantity_cont::Variables` to Python through pyo3.
pub struct F64VariablesCont {
    #[pyo3(get)]
    pub w: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub x: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub s: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub l: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub t: Vec<Vec<f64>>,
}

/// The `QuantityLpCont` is continuous time linear program that is capable of deciding the arrival times and quantites picked up or deliverd
/// given a sequence of nodes to visit for every vessel. The objective is to minimize shortage and overflow.
pub struct QuantityLpCont {
    pub model: Model,
    pub vars: Variables,
    pub delay: f64,
}

#[allow(non_snake_case)]
impl QuantityLpCont {
    /// Create a QuantityLPCont
    ///
    /// ## Arguments
    ///
    /// * `delay` - The mandatory delay that is added between visits for a vessel. A nonzero value will hopefully make the output from the continuous model fit a discrete time representation better.
    pub fn new(delay: f64) -> grb::Result<QuantityLpCont> {
        let mut model = Model::new(&format!("cont quant model"))?;
        model.set_param(grb::param::OutputFlag, 0)?;

        let vars = Variables {
            w: Vec::new(),
            x: Vec::new(),
            s: Vec::new(),
            l: Vec::new(),
            t: Vec::new(),
        };

        Ok(QuantityLpCont { model, vars, delay })
    }

    /// Calculates the path through a graph of visits for every vessel.
    /// A visit is represented as (i,m), where i is the node index and m indicates that it is the m'th visit at node i
    ///
    /// The paths are constructed such that a vessel visiting node (i,m) is visit number m to port i if all vessels were to sail
    /// their plans as quickly as possible. Meaning that no loading or unloading time is taken into account.
    /// Ties are handled by shuffling before sorting
    /// ## Arguments
    ///
    /// * `solution` - A sequence of visits for every vessel, given as a RoutingSolution
    pub fn paths(solution: &RoutingSolution) -> Vec<Vec<(NodeIndex, VisitIndex)>> {
        let problem = solution.problem();
        // hash map vesselindex, visit -> rank
        let mut ranks = HashMap::new();
        for n in 0..problem.nodes().len() {
            let mut earliest_visit_times = Vec::new();
            for (vessel_idx, plan) in solution.iter().enumerate() {
                let vessel = &problem.vessels()[vessel_idx];
                let mut t = vessel.available_from();
                for vis in 0..plan.len() {
                    let visit = plan[vis];
                    if visit.node == n {
                        earliest_visit_times.push((vessel_idx, visit, t));
                    }
                    if vis < plan.len() - 1 {
                        t += problem.travel_time(visit.node, plan[vis + 1].node, vessel);
                    }
                }
            }
            // shuffle to handle ties stochastically
            let mut rng = thread_rng();
            earliest_visit_times.shuffle(&mut rng);

            // sort by visit time and add rank
            earliest_visit_times.sort_by_key(|x| x.2);
            earliest_visit_times.into_iter().enumerate().for_each(
                |(rank, (vessel_idx, visit, _))| {
                    ranks.insert((vessel_idx, visit), rank);
                },
            );
        }

        solution
            .iter()
            .enumerate()
            .map(|(vessel_idx, plan)| {
                plan.iter()
                    .map(|visit| {
                        let n = visit.node;
                        let m = ranks[&(vessel_idx, *visit)];
                        (n, m)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Configure the model such that it is ready to solve for the given solution
    /// This can perhaps be done quicker if we update only relevant variables and constraints when changes are made to the solution
    pub fn configure(
        &mut self,
        solution: &RoutingSolution,
        paths: &Vec<Vec<(usize, usize)>>,
    ) -> grb::Result<()> {
        let problem = solution.problem();
        // clear model of current variables and constraints
        Self::clear_model(&mut self.model)?;

        // The count of visits to every node
        let mut M = solution.iter().flatten().map(|visit| visit.node).counts();

        // go through all nodes and add an entry with value 0 if the node isnt visited
        (0..problem.nodes().len()).for_each(|i| {
            if !M.contains_key(&i) {
                M.insert(i, 0);
            }
        });

        // update the paths according to the given solution
        //let paths = Self::paths(solution, problem, &M);

        trace!("paths: {:?}", paths);

        // set new variables
        self.vars = Self::create_vars(&mut self.model, problem, &M)?;

        let s = &self.vars.s;
        let x = &self.vars.x;
        let t = &self.vars.t;
        let l = &self.vars.l;
        let w = &self.vars.w;

        // add constraints
        Self::inventory_constraints(&mut self.model, problem, &s, &x, &t, &M)?;
        Self::load_constraints(&mut self.model, problem, &paths, &l, &x)?;
        Self::time_constraints(&mut self.model, problem, &paths, self.delay, &x, &t)?;
        Self::shortage_constraints(&mut self.model, problem, &s, &w, &x, &t, &M)?;

        // set objective
        let obj = w.iter().flatten().flatten().grb_sum();
        self.model.set_objective(obj, grb::ModelSense::Minimize)?;

        Ok(())
    }

    /// Solves the model for the given `solution` and returns the optimized variables in the Python exposed Pyclass `F64VariablesCont`
    pub fn py_solve(&mut self, solution: &RoutingSolution) -> grb::Result<F64VariablesCont> {
        let paths = Self::paths(solution);
        self.configure(solution, &paths)?;

        self.model.optimize()?;

        let x = self.vars.x.convert(&self.model)?;
        let l = self.vars.l.convert(&self.model)?;
        let w = self.vars.w.convert(&self.model)?;
        let t = self.vars.t.convert(&self.model)?;
        let s = self.vars.s.convert(&self.model)?;

        Ok(F64VariablesCont { w, x, s, l, t })
    }

    /// Solves the model for the given solution and returns the optimized variables
    pub fn solve(
        &mut self,
        solution: &RoutingSolution,
        paths: &Vec<Vec<(usize, usize)>>,
    ) -> grb::Result<()> {
        //let paths = Self::paths(solution);
        self.configure(solution, &paths)?;
        self.model.optimize()?;
        Ok(())
    }

    /// Returns the optimized arrival times of the given `solution`.
    /// First, the linear program is solved using continuous time, and then the final time variables are rounded up to the closest integer.
    pub fn get_visit_times(&mut self, solution: &RoutingSolution) -> grb::Result<Vec<Vec<usize>>> {
        let paths = Self::paths(solution);
        self.solve(solution, &paths)?;

        let res = paths
            .iter()
            .map(|path| {
                path.iter()
                    .map(|(i, m)| {
                        let value = self.model.get_obj_attr(attr::X, &self.vars.t[*i][*m])?;
                        Ok(f64::ceil(value) as usize)
                    })
                    .collect()
            })
            .collect::<grb::Result<Vec<Vec<usize>>>>()?;

        Ok(res)
    }

    /// Removes all variables and constraints from the model
    fn clear_model(model: &mut Model) -> grb::Result<()> {
        let constrs = model
            .get_constrs()?
            .iter()
            .cloned()
            .collect::<Vec<Constr>>();

        let vars = model.get_vars()?.iter().cloned().collect::<Vec<Var>>();

        for i in 0..constrs.len() {
            model.remove(constrs[i])?;
        }
        for i in 0..vars.len() {
            model.remove(vars[i])?;
        }

        Ok(())
    }

    /// Creates and returns the variables for the model
    ///
    /// ## Arguments
    ///
    /// * `model` - The underlying gurobi model to use.
    /// * `problem` - The underlying problem.
    /// * `M` - The number of visits at every node (key: nodeindex, value: number ov vists to key)
    fn create_vars(
        model: &mut Model,
        problem: &Problem,
        M: &HashMap<usize, usize>,
    ) -> grb::Result<Variables> {
        let P = problem.products();

        let x = (0..problem.nodes().len())
            .map(|i| {
                let m = *M.get(&i).unwrap();
                (m, P).cont(model, &format!("x_{i}"))
            })
            .collect::<grb::Result<Vec<Vec<Vec<Var>>>>>()?;

        let s = (0..problem.nodes().len())
            .map(|i| {
                let m = *M.get(&i).unwrap();
                (m, P).free(model, &format!("s_{i}"))
            })
            .collect::<grb::Result<Vec<Vec<Vec<Var>>>>>()?;

        let w = (0..problem.nodes().len())
            .map(|i| {
                let m = *M.get(&i).unwrap() + 1;
                (m, P).cont(model, &format!("w_{i}"))
            })
            .collect::<grb::Result<Vec<Vec<Vec<Var>>>>>()?;

        let t = (0..problem.nodes().len())
            .map(|i| {
                let m = *M.get(&i).unwrap();
                (m).cont(model, &format!("t_{i}"))
            })
            .collect::<grb::Result<Vec<Vec<Var>>>>()?;

        let l = (0..problem.nodes().len())
            .map(|i| {
                let m = *M.get(&i).unwrap();
                (m, P).cont(model, &format!("l_{i}"))
            })
            .collect::<grb::Result<Vec<Vec<Vec<Var>>>>>()?;

        Ok(Variables { w, x, s, l, t })
    }

    /// Adds the inventory constrains.
    ///
    /// ## Constraints
    /// * Set the inventory at the beginning of the first visit according to the initial inventory and time of first visit
    /// * Balancing equations for visits at the same node
    ///
    /// ## Arguments
    ///
    /// * `model` - Gurobi model
    /// * `problem` - Underlying problem
    /// * `s` - The node inventory variables, s
    /// * `x` - The quantity variables, x
    /// * `t` - The arrival time variables, t
    /// * `M` - The number of visits at every node (key: nodeindex, value: number ov vists to key)
    fn inventory_constraints(
        model: &mut Model,
        problem: &Problem,
        s: &[Vec<Vec<Var>>],
        x: &[Vec<Vec<Var>>],
        t: &[Vec<Var>],
        M: &HashMap<usize, usize>,
    ) -> grb::Result<()> {
        let N = problem.nodes().len();
        let P = problem.products();

        // initial inventory
        for (i, p) in iproduct!(0..N, 0..P) {
            // if the node do not have any visits, move on
            if M.get(&i).unwrap() == &0 {
                continue;
            }
            let rate = f64::abs(problem.nodes()[i].inventory_changes()[0][p]);
            let kind = problem.nodes()[i].r#type();
            let initial = problem.nodes()[i].initial_inventory()[p];
            // amount produced or consumed before the first visit
            let change = t[i][0] * rate;

            model.add_constr(
                &format!("init_inv_{}_{}_{}", i, 0, p),
                c!(s[i][0][p] == initial + Self::multiplier(kind) * change),
            )?;

            // the remaining visits

            // the the number of visits, guaranteed to be at least one
            let M_i = *M.get(&i).unwrap();
            for m in 1..M_i {
                // consumed or produced between the two visits
                let change = (t[i][m] - t[i][m - 1]) * rate;

                // delivered/picked up in last visit
                let external_change = x[i][m - 1][p];

                model.add_constr(
                    &format!("inv_{}_{}_{}", i, m, p),
                    c!(s[i][m][p]
                        == s[i][m - 1][p] + Self::multiplier(kind) * change
                            - Self::multiplier(kind) * external_change),
                )?;
            }
        }

        Ok(())
    }

    /// Adds the load constraints.
    ///
    /// ## Constraints
    /// * Set the initial load
    /// * Balancing equations for load
    /// * Bound the load to not exceed the capacity of the vessel
    ///
    /// ## Arguments
    ///
    /// * `model` - Gurobi model
    /// * `problem` - Underlying problem
    /// * `paths` - The paths for every vessel. A path consists of a path through a graph consisting of nodes *(i,m)*, where i is the node index in the given `problem` and *m* is the number of visits to *i*
    /// * `l` - The vessel load variables, l
    /// * `x` - The quantity variables, x
    fn load_constraints(
        model: &mut Model,
        problem: &Problem,
        paths: &Vec<Vec<(NodeIndex, VisitIndex)>>,
        l: &[Vec<Vec<Var>>],
        x: &[Vec<Vec<Var>>],
    ) -> grb::Result<()> {
        let V = problem.vessels().len();
        let P = problem.products();

        // balancing
        for (v, p) in iproduct!(0..V, 0..P) {
            // v doesn't perform any visits, continue
            if paths[v].is_empty() {
                continue;
            }

            let initial_load = problem.vessels()[v].initial_inventory()[p];

            // node and call number of first visit
            let (i, m) = paths[v][0];

            let kind = problem.nodes()[i].r#type();

            // get the first visit to set the initial load
            let lhs = l[i][m][p];
            let rhs = initial_load + Self::multiplier(kind) * x[i][m][p];
            model.add_constr(&format!("init_load_{}_{}", v, p), c!(lhs == rhs))?;

            // set the load for the remaining visits
            for win in paths[v].windows(2) {
                let (i, m) = win[0];
                let (j, n) = win[1];
                let next_kind = problem.nodes()[j].r#type();
                model.add_constr(
                    &format!("load_{}_{}_i:{}_m:{}_j:{}_n_{}", n, p, i, m, j, n),
                    c!(l[i][m][p] == l[j][n][p] - Self::multiplier(next_kind) * x[j][n][p]),
                )?;
            }
        }

        // bound the load to never exceed the capacity of the vessel
        for v in 0..V {
            for (i, m) in &paths[v] {
                let lhs = (0..P).map(|p| l[*i][*m][p]).grb_sum();
                let rhs = problem.vessels()[v].capacity();
                model.add_constr(&format!("bound_load_{}_{}_{}", i, m, v), c!(lhs <= rhs))?;
            }
        }

        Ok(())
    }

    /// Adds the time constraints to the model.
    ///
    /// ## Constraints
    ///
    /// * Set up the relation between the time of the scheduled visits for each vessel
    /// * Bound the time of the last visit to not exceed the duration of the planning period
    /// * Set the time of the origin visits to be the time the vessels become available
    ///
    /// ## Arguments
    ///
    /// * `model` - Gurobi model
    /// * `problem` - Underlying problem
    /// * `paths` - The paths for every vessel. A path consists of a path through a graph consisting of nodes *(i,m)*, where i is the node index in the given `problem` and *m* is the number of visits to *i*
    /// * `delay` - The mandatory delay between visits, intended to make conversion to discrete time smoother.
    /// * `x` - The quantity variables, x
    /// * `t` - The arrival time variables, t
    fn time_constraints(
        model: &mut Model,
        problem: &Problem,
        paths: &Vec<Vec<(NodeIndex, VisitIndex)>>,
        delay: f64,
        x: &[Vec<Vec<Var>>],
        t: &[Vec<Var>],
    ) -> grb::Result<()> {
        let V = problem.vessels().len();
        let P = problem.products();
        let T = problem.timesteps();
        for (v, p) in iproduct!(0..V, 0..P) {
            let vessel = &problem.vessels()[v];
            // if the vessel does not have a path, continue
            let path = &paths[v];
            for win in path.windows(2) {
                let (i, m) = win[0];
                let (j, n) = win[1];

                // force the vessel to use at most one time period when two consecutive visits are to the same node
                let epsilon = if i == j { 1.0 } else { 0.0 };

                // unloading rate as time per quantity
                let time_per_quant = 1.0 / (problem.nodes()[i].max_loading_amount() as f64);

                // time taken to load/unload at the previous visit
                let visit_time = time_per_quant * x[i][m][p];

                // sailing time from i to j
                let sail_time = problem.travel_time(i, j, vessel);

                // allow loading/unloading in the same period as leaving for the next port
                let adjuster = if sail_time > 0 { 1.0 } else { 0.0 };

                model.add_constr(
                    &format!("time_{v}_{i}_{m}_{j}_{n}"),
                    c!(t[j][n] == t[i][m] + visit_time + sail_time + delay + epsilon - adjuster),
                )?;
            }

            // bound the last visit to be before the end of the planning period, time steps are counted from 0
            let last = paths[v].iter().last();
            if let Some((i, m)) = last {
                model.add_constr(
                    &format!("upper_bound_time_{}_{}_{}", v, i, m),
                    c!(t[*i][*m] <= T - 1),
                )?;
            }

            // set the initial visit times to the time period in which the vessels become available
            let (node, visit) = path[0];
            model.add_constr(
                &format!("set time of initial visit"),
                c!(t[node][visit] == vessel.available_from()),
            )?;
        }

        Ok(())
    }

    /// Add the shortage and overflow constraints
    ///
    /// ## Constraints
    ///
    /// * Set up the last shortage variable for each node to reflect the shortage/overflow at the end of the plannign period
    /// * Hard upper bound on the inventory of consumption nodes
    /// * Soft lower bound for consumption nodes, but registering shortage
    /// * Hard lower bound on the inventory of production nodes
    /// * Soft upper bound for production nodes, but registering overflow
    ///
    /// ## Arguments
    /// * `model` - Gurobi model
    /// * `problem` - Underlying problem
    /// * `paths` - The paths for every vessel. A path consists of a path through a graph consisting of nodes *(i,m)*, where i is the node index in the given `problem` and *m* is the number of visits to *i*
    /// * `delay` - The mandatory delay between visits, intended to make conversion to discrete time smoother.
    /// * `s` - The node inventory variables, s
    /// * `w` - The shortage and overflow variables, w
    /// * `x` - The quantity variables, x
    /// * `t` - The arrival time variables, t
    /// * `M` - The number of visits at every node (key: nodeindex, value: number ov vists to key)
    fn shortage_constraints(
        model: &mut Model,
        problem: &Problem,
        s: &[Vec<Vec<Var>>],
        w: &[Vec<Vec<Var>>],
        x: &[Vec<Vec<Var>>],
        t: &[Vec<Var>],
        M: &HashMap<usize, usize>,
    ) -> grb::Result<()> {
        let N = problem.nodes().len();
        let P = problem.products();
        let T = problem.timesteps() as f64;

        for (i, p) in iproduct!(0..N, 0..P) {
            let kind = problem.nodes()[i].r#type();

            // end shortage

            // last vist + 1 to indicate the artificial visit at the end
            let m = *M.get(&i).unwrap();

            // change rate
            let change_rate = f64::abs(problem.nodes()[i].inventory_changes()[0][p]);

            let lhs = if m == 0 {
                let change = change_rate * T;
                let initial = problem.nodes()[i].initial_inventory()[p];
                initial + Self::multiplier(kind) * change - Self::multiplier(kind) * w[i][m][p]
            } else {
                s[i][m - 1][p] - Self::multiplier(kind) * x[i][m - 1][p]
                    + Self::multiplier(kind) * change_rate * (T - t[i][m - 1])
                    - Self::multiplier(kind) * w[i][m][p]
            };

            match kind {
                NodeType::Consumption => {
                    model.add_constr(&format!("end_shortage_{i}_{m}_{p}"), c!(lhs >= 0.0))?;
                }
                NodeType::Production => {
                    model.add_constr(
                        &format!("end_overflow_{i}_{m}_{p}"),
                        c!(lhs <= problem.nodes()[i].capacity()[p]),
                    )?;
                }
            }

            // check that the node is actually visited
            if m == 0 {
                continue;
            }

            // set shortage for every visit
            for m in 0..*M.get(&i).unwrap() {
                match kind {
                    // set hard limit on upper bound and allow shortage
                    NodeType::Consumption => {
                        let lhs = s[i][m][p];
                        // hard upper
                        model.add_constr(
                            &format!("cons_upper_{i}_{m}_{p}"),
                            c!(lhs <= problem.nodes()[i].capacity()[p]),
                        )?;

                        // allow shortage
                        let lhs = s[i][m][p] + w[i][m][p];
                        let rhs = 0.0;
                        model.add_constr(&format!("cons_lower_{i}_{m}_{p}"), c!(lhs >= rhs))?;
                    }
                    // set hard limit on lower bound and allow overflow
                    NodeType::Production => {
                        let lhs = s[i][m][p];
                        // hard lower limit
                        model
                            .add_constr(&format!("prod_lower_{}_{}_{}", i, m, p), c!(lhs >= 0.0))?;

                        // allow overflow
                        let lhs = s[i][m][p] - w[i][m][p];
                        model.add_constr(
                            &format!("prod_upper_{}_{}_{}", i, m, p),
                            c!(lhs <= problem.nodes()[i].capacity()[p]),
                        )?;
                    }
                }
            }
        }

        Ok(())
    }

    /// A positive or negative multiplier depending on the node kind
    fn multiplier(kind: NodeType) -> f64 {
        match kind {
            NodeType::Consumption => -1.0,
            NodeType::Production => 1.0,
        }
    }
}
