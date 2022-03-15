use crate::models::utils::{AddVars, ConvertVars, NObjectives};
use grb::{prelude::*, Result};
use itertools::iproduct;
use log::{debug, info, warn};

use super::sets_and_parameters::{Parameters, Sets};

pub struct PathFlowSolver {}

impl PathFlowSolver {
    /// adds a route to the model
    pub fn add_routes(
        from_route: usize,
        variables: &mut Variables,
        sets: &Sets,
        parameters: &Parameters,
        model: &mut Model,
    ) -> Result<()> {
        debug!("adding new routes from route: {:?}", from_route);
        // add variables associated with the new routes
        let new_vars = Self::add_route_vars(from_route, sets, parameters, model)?;
        // merge in the new variables into the new variables
        variables.merge_in(new_vars);
        let v = &variables;

        debug!(
            "new x_vars shape: {:?}",
            (v.x.len(), v.x[0].len(), v.x[0][0].len(), v.x[0][0][0].len())
        );

        // add constraints associated with the new routes
        Self::add_route_constrs(from_route, variables, sets, parameters, model)?;

        // update objectives
        Self::add_objectives(variables, sets, parameters, model)?;
        model.update()?;

        debug!(
            "\nsuccessfully added new routes from route: {:?}",
            from_route
        );

        Ok(())
    }

    /// Adds all variables to the given model according to the given sets and parameters
    fn add_variables(sets: &Sets, parameters: &Parameters, model: &mut Model) -> Result<Variables> {
        let route_vars = Self::add_route_vars(0, sets, parameters, model)?;
        let ind_vars = Self::add_route_ind_vars(sets, model)?;
        Ok(Variables {
            x: route_vars.x,
            s: ind_vars.s,
            v_plus: ind_vars.v_plus,
            v_minus: ind_vars.v_minus,
            l: ind_vars.l,
            q: route_vars.q,
        })
    }

    /// adds route independent variables to the given model according to the given sets and parameters
    /// ## Note:
    /// Since these variables are route independent it is only necessary to call once
    fn add_route_ind_vars(sets: &Sets, model: &mut Model) -> Result<IndVars> {
        // inventory at node n at *the end* time step t of product p
        let s: Vec<Vec<Vec<Var>>> = (sets.N, sets.T, sets.P).free(model, &"s")?;

        // overflow at node n at time step t of product p
        let v_plus: Vec<Vec<Vec<Var>>> = (sets.N, sets.T, sets.P).cont(model, &"v_plus")?;

        // shortage at node n at time step t of product p
        let v_minus: Vec<Vec<Vec<Var>>> = (sets.N, sets.T, sets.P).cont(model, &"v_minus")?;

        // load of vessel v in time period t of product p
        let l: Vec<Vec<Vec<Var>>> = (sets.V, sets.T, sets.P).cont(model, &"l")?;

        Ok(IndVars {
            s,
            v_plus,
            v_minus,
            l,
        })
    }

    /// adds route depending variables to the given model according to the given sets and parameters
    fn add_route_vars(
        from_route: usize,
        sets: &Sets,
        parameters: &Parameters,
        model: &mut Model,
    ) -> Result<RouteVars> {
        // 1 if vessel v follows route r and is at the route's i'th stop at the beginning of time step t, indexed (r,i,v,t)
        let x: Vec<Vec<Vec<Vec<Var>>>> = (from_route..sets.R)
            .map(|r| (sets.I_r[r], sets.V, sets.T).binary(model, &format!("x_{}", r)))
            .collect::<Result<Vec<Vec<Vec<Vec<Var>>>>>>()?;

        // loaded or unloaded at the i'th visit of route r by vessel v at time step t, indexed (r,i,v,t,p)
        let q: Vec<Vec<Vec<Vec<Vec<Var>>>>> = (from_route..sets.R)
            .map(|r| {
                (sets.I_r[r], sets.V, sets.T, sets.P).vars_with(|(i, v, t, p)| {
                    model.add_var(
                        &format!("q_{}_{}_{}_{}_{}", r, i, v, t, p),
                        SemiCont,
                        0.0,
                        parameters.F_min[parameters.node(r, i).unwrap()][p],
                        parameters.F_max[parameters.node(r, i).unwrap()][p],
                        std::iter::empty(),
                    )
                })
            })
            .collect::<Result<Vec<Vec<Vec<Vec<Vec<Var>>>>>>>()?;
        Ok(RouteVars { x, q })
    }

    /// adds all constraints to the model according to the given sets and parameters and variables
    fn add_constrs(
        variables: &Variables,
        sets: &Sets,
        parameters: &Parameters,
        model: &mut Model,
    ) -> Result<()> {
        Self::add_route_constrs(0, variables, sets, parameters, model)?;
        Self::add_route_ind_constrs(variables, sets, parameters, model)?;
        Ok(())
    }

    /// adds the route dependent constraints to the given model according to the given `from_route`, sets and parameters.
    fn add_route_constrs(
        from_route: usize,
        variables: &Variables,
        sets: &Sets,
        parameters: &Parameters,
        model: &mut Model,
    ) -> Result<()> {
        let x = variables.x();
        let s = variables.s();
        let q = variables.q();
        let l = variables.l();

        // ************* ROUTING ************

        // A vessel at the i'th stop of a route has two chioces
        // 1. Travel to the next stop
        // 2. Stay at the current node until the next time period
        // If the vessel is at a visit in a route in the last possible time step allowing the vessel
        // to finish within the planning period, the only option is to go to the next visit/stop
        for r in from_route..sets.R {
            for (i, v) in iproduct!(0..(sets.I_r[r] - 1), 0..sets.V) {
                for t in 0..(sets.T_r[r][v][i] + 1) {
                    // travel time between the i'th and (i+1)'th visit of route r for vessel v
                    let t_time = parameters.travel[r][i][v];
                    let lhs = x[r][i][v][t];
                    let rhs = if t < sets.T_r[r][v][i] {
                        x[r][i + 1][v][t + t_time] + x[r][i][v][t + 1] // if it is possible to stay
                    } else {
                        x[r][i + 1][v][t + t_time] + 0 // must move on to finish in time
                    };

                    model.add_constr(&format!("1_{}_{}_{}_{}", r, i, v, t), c!(lhs <= rhs))?;
                }
            }
        }

        //model.add_constr(&format!("test"), c!(x[0][0][0][0] == 1))?;
        //model.add_constr(&format!("test"), c!(x[0][0][0][5] == 1))?;
        // Ensure that the lhs of constraint 1 cannot exceed one.
        for r in from_route..sets.R {
            for (i, v) in iproduct!(0..(sets.I_r[r] - 1), 0..sets.V) {
                for t in 0..(sets.T_r[r][v][i] + 1) {
                    // travel time between the i'th and (i+1)'th visit of route r for vessel v
                    let t_time = parameters.travel[r][i][v];
                    let lhs = x[r][i + 1][v][t + t_time] + x[r][i][v][t + 1];

                    model.add_constr(
                        &format!("bound_1_lhs_{}_{}_{}_{}", r, i, v, t),
                        c!(lhs <= 1),
                    )?;
                }
            }
        }

        // if a route is finished, the vessel can start a new route
        for r in from_route..sets.R {
            // routes with the same start as the end of route r
            let possible_routes: Vec<usize> = (0..sets.R)
                .filter(|n_r| parameters.N_r[r][sets.I_r[r] - 1] == parameters.N_r[*n_r][0])
                .collect();
            for (v, t) in iproduct!(0..sets.V, 0..sets.T - 1) {
                let lhs = x[r][sets.I_r[r] - 1][v][t];
                let rhs = possible_routes.iter().map(|n_r| x[*n_r][0][v][t]).grb_sum()
                    + x[r][sets.I_r[r] - 1][v][t + 1];
                model.add_constr(&format!("new_route_{}_{}_{}", r, v, t), c!(lhs <= rhs))?;
            }
        }

        // bound the lhs of the new route constraint
        for r in from_route..sets.R {
            // routes with the same start as the end of route r
            let possible_routes: Vec<usize> = (0..sets.R)
                .filter(|n_r| parameters.N_r[r][sets.I_r[r] - 1] == parameters.N_r[*n_r][0])
                .collect();
            for (v, t) in iproduct!(0..sets.V, 0..sets.T - 1) {
                let lhs = possible_routes.iter().map(|n_r| x[*n_r][0][v][t]).grb_sum()
                    + x[r][sets.I_r[r] - 1][v][t + 1];
                model.add_constr(
                    &format!("bound_new_route_lhs_{}_{}_{}", r, v, t),
                    c!(lhs <= 1),
                )?;
            }
        }

        // remove the following constraints if present in the model
        Self::remove_constrs((0..sets.V).map(|v| format!("start_{}", v)).collect(), model)?;
        // a vessel can choose to start a route after it has become available
        for v in 0..sets.V {
            // routes with the same start as the origin of the vessel v
            let possible_routes =
                (from_route..sets.R).filter(|r| parameters.N_r[*r][0] == parameters.O[v]);
            let lhs = possible_routes
                .map(|r| x[r][0][v][parameters.T_0[v]])
                .grb_sum();
            model.add_constr(&format!("start_{}", v), c!(lhs <= 1))?;
        }

        // remove the following constraints if present
        Self::remove_constrs(
            (0..sets.V)
                .map(|v| format!("end_criterion_{}", v))
                .collect(),
            model,
        )?;
        // a vessel must reside in the end of a route at the end of the planning period if a route is initiated
        for v in 0..sets.V {
            let possible_routes =
                (from_route..sets.R).filter(|r| parameters.N_r[*r][0] == parameters.O[v]);
            let lhs = possible_routes
                .map(|r| x[r][0][v][parameters.T_0[v]])
                .grb_sum();
            let rhs = (0..sets.R)
                .map(|r| x[r][sets.I_r[r] - 1][v][sets.T - 1])
                .grb_sum();
            model.add_constr(&format!("end_criterion_{}", v), c!(lhs == rhs))?;
        }

        // a vessel can not be at a visit in a route at a time where it cannot reach the end of the route
        for r in from_route..sets.R {
            for i in 0..sets.I_r[r] {
                for v in 0..sets.V {
                    for t in (sets.T_r[r][v][i] + 1)..sets.T {
                        model.add_constr(
                            &format!("force_zero_{}_{}_{}_{}", r, i, v, t),
                            c!(x[r][i][v][t] == 0),
                        )?;
                    }
                }
            }
        }

        // ensure that x-variables for last visits in routes are zero before it is possible to arrive there
        // (this is not taken care of in constraint 1, as it is not defined for certain time stesp)
        for (r, v, t) in iproduct!(from_route..sets.R, 0..sets.V, 0..sets.T) {
            // time from second last to last visit
            let t_time = parameters.travel[r][sets.I_r[r] - 2][v];
            if t < t_time {
                let i = sets.I_r[r] - 1;
                model.add_constr(
                    &format!("force_zero_{}_{}_{}_{}", r, i, v, t),
                    c!(x[r][i][v][t] == 0),
                )?;
            }
        }

        // remove constraints if already present
        Self::remove_constrs(
            iproduct!((0..sets.N), (0..sets.P), (1..sets.T))
                .map(|(n, p, t)| format!("node_inv_{}_{}_{}", n, p, t))
                .collect(),
            model,
        )?;

        // logging of inventory at node n
        for (n, p, t) in iproduct!(0..sets.N, 0..sets.P, 1..sets.T) {
            // delivered in time period t
            let delivered = (0..sets.R)
                .map(|r| {
                    let visits = (0..sets.I_r[r]).filter(|i| parameters.node(r, *i).unwrap() == n);
                    let subsum = visits.map(|i| (0..sets.V).map(|v| q[r][i][v][t][p]).grb_sum());
                    subsum.grb_sum()
                })
                .grb_sum();
            let lhs = s[n][t][p];
            let rhs = s[n][t - 1][p] + parameters.kind(n) * (parameters.D[n][t][p] as isize)
                - parameters.kind(n) * delivered;
            model.add_constr(&format!("node_inv_{}_{}_{}", n, p, t), c!(lhs == rhs))?;
        }

        // remove constraints if already present
        Self::remove_constrs(
            iproduct!(0..sets.V, 1..sets.T, 0..sets.P)
                .map(|(v, t, p)| format!("load_{}_{}_{}", v, t, p))
                .collect(),
            model,
        )?;

        // load logging
        for (v, t, p) in iproduct!(0..sets.V, 1..sets.T, 0..sets.P) {
            // loaded or unloaded in the time period
            let change = (from_route..sets.R)
                .map(|r| {
                    (0..sets.I_r[r])
                        .map(|i| parameters.visit_kind(r, i).unwrap() * q[r][i][v][t][p])
                        .grb_sum()
                })
                .grb_sum();
            let lhs = l[v][t][p];
            let rhs = l[v][t - 1][p] + change;
            model.add_constr(&format!("load_{}_{}_{}", v, t, p), c!(lhs == rhs))?;
        }

        // ************* LOADING/UNLOADING ************

        // do not load or unload unless actually there
        for r in from_route..sets.R {
            for (i, v, t) in iproduct!((0..sets.I_r[r]), (0..sets.V), (0..sets.T - 1)) {
                let lhs = (0..sets.P).map(|p| q[r][i][v][t][p]).grb_sum();
                warn!("tighten this bound");
                let rhs = 10000 * x[r][i][v][t];
                model.add_constr(&format!("loading_{r}_{i}_{v}_{t}"), c!(lhs <= rhs))?;
                let lhs = (0..sets.P).map(|p| q[r][i][v][t][p]).grb_sum();
                let rhs = 10000 * x[r][i][v][t + 1];
                model.add_constr(&format!("loading2_{r}_{i}_{v}_{t}"), c!(lhs <= rhs))?;
            }
        }

        Ok(())
    }

    /// Adds the route independent constraints to the given model according to the given sets and parameters
    /// ## Note:
    /// Since these constraints are route independent, it is only necessary to call this once.
    fn add_route_ind_constrs(
        variables: &Variables,
        sets: &Sets,
        parameters: &Parameters,
        model: &mut Model,
    ) -> Result<()> {
        let s = variables.s();
        let v_minus = variables.v_minus();
        let v_plus = variables.v_plus();
        let l = variables.l();

        // ************* NODE INVENTORY ************

        // inventory in first time step at node n
        for (n, p) in iproduct!(0..sets.N, 0..sets.P) {
            model.add_constr(
                &format!("initial_inv_{}_{}", n, p),
                c!(s[n][0][p] == parameters.S_0[n][p]),
            )?;
        }

        // shortage logging
        for (n, t, p) in iproduct!(&sets.N_C, 0..sets.T, 0..sets.P) {
            let lhs = parameters.S_min[*n][p][t];
            let rhs = s[*n][t][p] + v_minus[*n][t][p];
            model.add_constr(&format!("overflow_{}_{}_{}", n, t, p), c!(lhs <= rhs))?;
        }

        // shortage logging
        for (n, t, p) in iproduct!(&sets.N_P, 0..sets.T, 0..sets.P) {
            let lhs = parameters.S_max[*n][p][t];
            let rhs = s[*n][t][p] - v_plus[*n][t][p];
            model.add_constr(&format!("shortage_{}_{}_{}", n, t, p), c!(lhs >= rhs))?;
        }

        // ************* VESSEL LOAD ************

        // initial load
        for (v, p) in iproduct!(0..sets.V, 0..sets.P) {
            let lhs = l[v][parameters.T_0[v]][p];
            let rhs = parameters.L_0[v][p];
            model.add_constr(&format!("initial_load_{}_{}", v, p), c!(lhs == rhs))?;
        }

        Ok(())
    }

    fn add_objectives(
        variables: &Variables,
        sets: &Sets,
        parameters: &Parameters,
        model: &mut Model,
    ) -> Result<()> {
        let x = variables.x();
        let v_minus = variables.v_minus();
        let v_plus = variables.v_plus();

        // reset current objectives
        model.set_param(param::ObjNumber, 0)?;
        model.set_attr(attr::NumObj, 0)?;
        let shortage = iproduct!(0..sets.N, 0..sets.T, 0..sets.P)
            .map(|(n, t, p)| v_plus[n][t][p] + v_minus[n][t][p])
            .grb_sum();

        let cost = iproduct!(0..sets.R, 0..sets.V, 0..sets.T)
            .map(|(r, v, t)| parameters.C_r[r][v] * x[r][0][v][t])
            .grb_sum();

        model.set_objective_N(cost, 0, 0, &"cost")?;
        model.set_objective_N(shortage, 1, 1, &"shortage")?;
        Ok(())
    }

    /// builds the path flow model
    pub fn build(sets: &Sets, parameters: &Parameters) -> grb::Result<(Model, Variables)> {
        info!("Building path flow model");

        let mut model = Model::new(&format!("Path_flow_model"))?;

        let variables = Self::add_variables(&sets, &parameters, &mut model)?;
        Self::add_objectives(&variables, &sets, &parameters, &mut model)?;
        Self::add_constrs(&variables, &sets, &parameters, &mut model)?;

        info!("Successfully built path-flow model");

        Ok((model, variables))
    }

    pub fn solve(variables: &Variables, model: &mut Model) -> Result<PathFlowResult> {
        info!("Solving path flow model");
        model.optimize()?;
        info!("Finished optimizing path flow model");
        PathFlowResult::new(variables, model)
    }

    /// Removes the constraints given as input if present in the model
    fn remove_constrs(names: Vec<String>, model: &mut Model) -> Result<()> {
        model.update()?;
        for name in names {
            let constr = model.get_constr_by_name(&name)?;
            if let Some(x) = constr {
                model.remove(x)?;
                model.update()?;
            }
        }
        Ok(())
    }
}

pub struct Variables {
    /// 1 if vessel v follows route r and is at the route's i'th stop at the beginning of time step t, indexed (r,i,v,t)
    pub x: Vec<Vec<Vec<Vec<Var>>>>,
    /// inventory at node n at *the end* time step t of product p
    pub s: Vec<Vec<Vec<Var>>>,
    /// overflow at node n at time step t of product p
    pub v_plus: Vec<Vec<Vec<Var>>>,
    /// shortage at node n at time step t of product p
    pub v_minus: Vec<Vec<Vec<Var>>>,
    /// load of vessel v in time period t of product p
    pub l: Vec<Vec<Vec<Var>>>,
    /// semicontinuous variable indicated quantity loaded or unloaded at the i'th visit of route r by vessel v at time step t, indexed (r,i,v,t)
    pub q: Vec<Vec<Vec<Vec<Vec<Var>>>>>,
}

impl Variables {
    pub fn x(&self) -> &Vec<Vec<Vec<Vec<Var>>>> {
        &self.x
    }
    pub fn s(&self) -> &Vec<Vec<Vec<Var>>> {
        &self.s
    }
    pub fn v_plus(&self) -> &Vec<Vec<Vec<Var>>> {
        &self.v_plus
    }
    pub fn v_minus(&self) -> &Vec<Vec<Vec<Var>>> {
        &self.v_minus
    }
    pub fn l(&self) -> &Vec<Vec<Vec<Var>>> {
        &self.l
    }
    pub fn q(&self) -> &Vec<Vec<Vec<Vec<Vec<Var>>>>> {
        &self.q
    }
    /// Merges in the x and q variables from the variables given as input into this variable object
    fn merge_in(&mut self, vars: RouteVars) {
        let x = vars.x;
        let q = vars.q;
        x.into_iter().for_each(|v| self.x.push(v));
        q.into_iter().for_each(|v| self.q.push(v));
    }
}

struct IndVars {
    /// inventory at node n at *the end* time step t of product p
    pub s: Vec<Vec<Vec<Var>>>,
    /// overflow at node n at time step t of product p
    pub v_plus: Vec<Vec<Vec<Var>>>,
    /// shortage at node n at time step t of product p
    pub v_minus: Vec<Vec<Vec<Var>>>,
    /// load of vessel v in time period t of product p
    pub l: Vec<Vec<Vec<Var>>>,
}

struct RouteVars {
    /// 1 if vessel v follows route r and is at the route's i'th stop at the beginning of time step t, indexed (r,i,v,t)
    pub x: Vec<Vec<Vec<Vec<Var>>>>,
    /// semicontinuous variable indicated quantity loaded or unloaded at the i'th visit of route r by vessel v at time step t, indexed (r,i,v,t)
    pub q: Vec<Vec<Vec<Vec<Vec<Var>>>>>,
}

pub struct PathFlowResult {
    /// 1 if vessel v follows route r and is at the route's i'th stop at the beginning of time step t, indexed (r,i,v,t)
    pub x: Vec<Vec<Vec<Vec<f64>>>>,
    /// inventory at node n at *the end* time step t of product p
    pub s: Vec<Vec<Vec<f64>>>,
    /// overflow at node n at time step t of product p
    pub v_plus: Vec<Vec<Vec<f64>>>,
    /// shortage at node n at time step t of product p
    pub v_minus: Vec<Vec<Vec<f64>>>,
    /// load of vessel v in time period t of product p
    pub l: Vec<Vec<Vec<f64>>>,
    /// semicontinuous variable indicated quantity loaded or unloaded at the i'th visit of route r by vessel v at time step t, indexed (r,i,v,t)
    pub q: Vec<Vec<Vec<Vec<Vec<f64>>>>>,
}

impl PathFlowResult {
    pub fn new(variables: &Variables, model: &Model) -> Result<PathFlowResult> {
        Ok(PathFlowResult {
            x: variables.x.convert(model)?,
            s: variables.s.convert(model)?,
            v_plus: variables.v_plus.convert(model)?,
            v_minus: variables.v_minus.convert(model)?,
            l: variables.l.convert(model)?,
            q: variables.q.convert(model)?,
        })
    }

    pub fn non_zero_4_d(vec: Vec<Vec<Vec<Vec<f64>>>>) -> Vec<((usize, usize, usize, usize), f64)> {
        let mut res = Vec::new();
        for (r, r_) in vec.iter().enumerate() {
            for (i, i_) in r_.iter().enumerate() {
                for (v, v_) in i_.iter().enumerate() {
                    for (t, t_) in v_.iter().enumerate() {
                        if f64::round(*t_) > 0.0 {
                            res.push(((r, i, v, t), f64::round(*t_)));
                        }
                    }
                }
            }
        }
        res.sort_by(|a, b| a.0 .3.cmp(&b.0 .3));
        //res.sort_by(|a, b| a.0 .2.cmp(&b.0 .2));
        //res.sort_by(|a, b| a.0 .1.cmp(&b.0 .1));
        //res.sort_by(|a, b| a.0 .0.cmp(&b.0 .0));

        res
    }
}
