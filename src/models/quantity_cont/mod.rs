use std::collections::HashMap;

use grb::{attr, c, expr::GurobiSum, Constr, Model, Var};
use itertools::{iproduct, Itertools};
use pyo3::pyclass;

use crate::{
    models::utils::AddVars,
    problem::{NodeIndex, NodeType, Problem, VesselIndex},
    solution::routing::RoutingSolution,
};

use super::utils::ConvertVars;

type VisitIndex = usize;

pub struct Variables {
    pub w: Vec<Vec<Vec<Var>>>, // (i,m,p)
    pub x: Vec<Vec<Vec<Var>>>, // (i,m,p)
    pub s: Vec<Vec<Vec<Var>>>, // (i,m,p)
    pub l: Vec<Vec<Vec<Var>>>, // (i,m,p), load at the associated vessel performing (i,m) of product p after the visit is finished
    pub t: Vec<Vec<Var>>,      // (i,m)
}

#[pyclass]
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

pub struct QuantityLpCont {
    pub model: Model,
    pub vars: Variables,
    /// map from a vessel index and a visit index for that vessel to the visits associated node index
    //pub paths: HashMap<VesselIndex, Vec<(NodeIndex, VisitIndex)>>,
    pub delay: f64,
    //pub problem: Arc<Problem>,
}

#[allow(non_snake_case)]
impl QuantityLpCont {
    pub fn new(delay: f64) -> grb::Result<QuantityLpCont> {
        let model = Model::new(&format!("cont quant model"))?;
        let vars = Variables {
            w: Vec::new(),
            x: Vec::new(),
            s: Vec::new(),
            l: Vec::new(),
            t: Vec::new(),
        };
        //let paths = HashMap::new();

        Ok(QuantityLpCont {
            model,
            vars,
            //paths,
            delay,
            //problem,
        })
    }

    pub fn paths(
        solution: &RoutingSolution,
        problem: &Problem,
        M: &HashMap<usize, usize>,
    ) -> HashMap<VesselIndex, Vec<(NodeIndex, VisitIndex)>> {
        // helper to hold the number of visits assigned a vessel
        let mut _b = (0..problem.nodes().len())
            .map(|n| *M.get(&n).unwrap())
            .collect::<Vec<_>>();

        solution
            .iter()
            .enumerate()
            .map(|(vessel, plan)| {
                (
                    vessel,
                    plan.iter()
                        .map(|visit| {
                            let n = visit.node;
                            let res = (n, M.get(&n).unwrap() - _b[n]);
                            _b[n] -= 1;
                            res
                        })
                        .collect(),
                )
            })
            .collect()
    }

    /// Configure the model such that it is ready to solve for the given solution
    /// This can perhaps be done quicker if we update only relevant variables and constraints when changes are made to the solution
    pub fn configure(&mut self, solution: &RoutingSolution) -> grb::Result<()> {
        let problem = solution.problem();
        // clear model of current variables and constraints
        Self::clear_model(&mut self.model)?;

        let mut M = solution.iter().flatten().map(|visit| visit.node).counts();
        // go through all nodes and add a entry with value 0 if the node isnt visited
        (0..problem.nodes().len()).for_each(|i| {
            if !M.contains_key(&i) {
                M.insert(i, 0);
            }
        });

        // update the paths according to the given solution
        let paths = Self::paths(solution, problem, &M);
        // set new variables
        self.vars = Self::create_vars(&mut self.model, problem, &M)?;

        let s = &self.vars.s;
        let x = &self.vars.x;
        let t = &self.vars.t;
        let l = &self.vars.l;
        let w = &self.vars.w;

        let P = problem.products();
        let N = problem.nodes().len();
        let V = problem.vessels().len();
        let T = problem.timesteps();

        // add constraints
        Self::inventory_constraints(&mut self.model, problem, &s, &x, &t, N, P, &M)?;
        Self::load_constraints(&mut self.model, problem, &paths, &l, &x, V, P)?;
        Self::time_constraints(
            &mut self.model,
            problem,
            &paths,
            self.delay,
            &x,
            &t,
            V,
            P,
            T as f64,
        )?;
        Self::shortage_constraints(&mut self.model, problem, &s, &w, &x, &t, N, P, &M, T as f64)?;

        // set objective
        let obj = w.iter().flatten().flatten().grb_sum();
        self.model.set_objective(obj, grb::ModelSense::Minimize)?;

        Ok(())
    }

    pub fn py_solve(&mut self, solution: &RoutingSolution) -> grb::Result<F64VariablesCont> {
        self.configure(solution)?;
        self.model.optimize()?;

        let x = self.vars.x.convert(&self.model)?;
        let l = self.vars.l.convert(&self.model)?;
        let w = self.vars.w.convert(&self.model)?;
        let t = self.vars.t.convert(&self.model)?;
        let s = self.vars.s.convert(&self.model)?;

        Ok(F64VariablesCont { w, x, s, l, t })
    }

    /// Solves the model for the given solution
    pub fn solve(&mut self, solution: &RoutingSolution) -> grb::Result<&Variables> {
        self.configure(solution)?;

        self.model.optimize()?;

        Ok(&self.vars)
    }

    pub fn get_visit_times(
        &mut self,
        solution: &RoutingSolution,
        problem: &Problem,
    ) -> grb::Result<Vec<Vec<usize>>> {
        let variables = self.solve(solution)?;
        let t: Vec<Vec<Var>> = variables.t.iter().cloned().collect();

        // counter for every node
        let mut counter = vec![0; problem.nodes().len()];
        let mut res: Vec<Vec<usize>> = Vec::new();

        for (v, plan) in solution.iter().enumerate() {
            for visit in plan {
                // the nodeindex
                let i = visit.node;
                let count = counter[i];
                counter[i] += 1;
                let calculated_visit_time =
                    f64::ceil(self.model.get_obj_attr(attr::X, &t[i][count])?) as usize;

                if let Some(x) = res.get_mut(v) {
                    x.push(calculated_visit_time);
                } else {
                    res.push(Vec::new());
                    res[v].push(calculated_visit_time);
                }
            }
        }

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
                (m, P).cont(model, &format!("s_{i}"))
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
                let m = *M.get(&i).unwrap() + 1;
                (m, P).cont(model, &format!("l_{i}"))
            })
            .collect::<grb::Result<Vec<Vec<Vec<Var>>>>>()?;

        Ok(Variables { w, x, s, l, t })
    }

    fn inventory_constraints(
        model: &mut Model,
        problem: &Problem,
        s: &[Vec<Vec<Var>>],
        x: &[Vec<Vec<Var>>],
        t: &[Vec<Var>],
        N: usize,
        P: usize,
        M: &HashMap<usize, usize>,
    ) -> grb::Result<()> {
        // initial inventory
        for (i, p) in iproduct!(0..N, 0..P) {
            // if the node do not have any visits, move on
            if M.get(&i).unwrap() == &0 {
                continue;
            }
            let rate = problem.nodes()[i].inventory_changes()[0][p];
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

    fn load_constraints(
        model: &mut Model,
        problem: &Problem,
        paths: &HashMap<VesselIndex, Vec<(NodeIndex, VisitIndex)>>,
        l: &[Vec<Vec<Var>>],
        x: &[Vec<Vec<Var>>],
        V: usize,
        P: usize,
    ) -> grb::Result<()> {
        // balancing
        for (v, p) in iproduct!(0..V, 0..P) {
            // v doesn't perform any visits, continue

            if paths.get(&v).unwrap().is_empty() {
                continue;
            }
            let initial_load = problem.vessels()[v].initial_inventory()[p];

            // node and call number of first visit
            let (i, m) = (paths.get(&v).unwrap()[0].0, paths.get(&v).unwrap()[0].1);
            let kind = problem.nodes()[i].r#type();

            // get the first visit
            let lhs = l[i][m][p];
            let rhs = initial_load + Self::multiplier(kind) * x[i][m][p];
            model.add_constr(&format!("init_load_{}_{}", v, p), c!(lhs == rhs))?;

            // the remaining visits
            for win in paths.get(&v).unwrap().windows(2) {
                let (i, m) = win[0];
                let (j, n) = win[1];
                let next_kind = problem.nodes()[j].r#type();
                model.add_constr(
                    &format!("load_{}_{}", n, p),
                    c!(l[i][m][p] == l[j][n][p] - Self::multiplier(next_kind) * x[j][n][p]),
                )?;
            }
        }

        for v in 0..V {
            for (i, m) in paths.get(&v).unwrap() {
                let lhs = (0..P).map(|p| l[*i][*m][p]).grb_sum();
                let rhs = problem.vessels()[v].capacity();
                model.add_constr(&format!("bound_load_{}_{}_{}", i, m, v), c!(lhs <= rhs))?;
            }
        }

        Ok(())
    }

    fn time_constraints(
        model: &mut Model,
        problem: &Problem,
        paths: &HashMap<VesselIndex, Vec<(NodeIndex, VisitIndex)>>,
        delay: f64,
        x: &[Vec<Vec<Var>>],
        t: &[Vec<Var>],
        V: usize,
        P: usize,
        T: f64,
    ) -> grb::Result<()> {
        for (v, p) in iproduct!(0..V, 0..P) {
            let vessel = &problem.vessels()[v];
            // if the vessel does not have a path, continue
            let path = paths.get(&v).unwrap();
            for win in path.windows(2) {
                let (i, m) = win[0];
                let (j, n) = win[1];

                // unloading rate as time per quantity
                let time_per_quant = problem.nodes()[i].time_per_quantity();

                // time taken to load/unload at the previous visit
                let visit_time = time_per_quant * x[i][m][p];

                // sailing time from i to j
                let sail_time = problem.travel_time(i, j, vessel);

                model.add_constr(
                    &format!("time_{v}_{i}_{m}_{j}_{n}"),
                    c!(t[j][m] == t[i][m] + visit_time + sail_time + delay),
                )?;
            }

            // bound the last visit to be before the end of the planning period
            let last = paths.get(&v).unwrap().iter().last();
            if let Some((i, m)) = last {
                model.add_constr(
                    &format!("upper_bound_time_{}_{}_{}", v, i, m),
                    c!(t[*i][*m] <= T),
                )?;
            }
        }

        Ok(())
    }

    fn shortage_constraints(
        model: &mut Model,
        problem: &Problem,
        s: &[Vec<Vec<Var>>],
        w: &[Vec<Vec<Var>>],
        x: &[Vec<Vec<Var>>],
        t: &[Vec<Var>],
        N: usize,
        P: usize,
        M: &HashMap<usize, usize>,
        T: f64,
    ) -> grb::Result<()> {
        for (i, p) in iproduct!(0..N, 0..P) {
            let kind = problem.nodes()[i].r#type();

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
                        model.add_constr(&format!("cons_upper_{i}_{m}_{p}"), c!(lhs >= rhs))?;
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

            // end shortage
            // last vist + 1 to indicate the artificial visit at the end
            let m = *M.get(&i).unwrap();

            // change rate
            let change_rate = problem.nodes()[i].inventory_changes()[0][p];

            let lhs = s[i][m - 1][p] - Self::multiplier(kind) * x[i][m - 1][p]
                + change_rate * (T - t[i][m - 1])
                + w[i][m][p];

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
        }

        Ok(())
    }

    fn multiplier(kind: NodeType) -> f64 {
        match kind {
            NodeType::Consumption => -1.0,
            NodeType::Production => 1.0,
        }
    }
}
