use crate::models::utils::{AddVars, ConvertVars, NObjectives};
use grb::{prelude::*, Result};
use itertools::iproduct;
use log::info;

use super::sets_and_parameters::{Parameters, Sets};

pub struct PathFlowSolver {}

impl PathFlowSolver {
    /// builds the path flow model
    pub fn build(sets: Sets, parameters: Parameters) -> grb::Result<(Model, Variables)> {
        info!("Building path flow model");

        let mut model = Model::new(&format!("Path_flow_model"))?;

        //*************CREATE VARIABLES*************//

        // 1 if vessel v follows route r and is at the route's i'th stop at the beginning of time step t, indexed (r,i,v,t)
        let x: Vec<Vec<Vec<Vec<Var>>>> = (0..sets.R)
            .map(|r| (sets.I_R[r], sets.V, sets.T).binary(&mut model, &format!("x_{}", r)))
            .collect::<Result<Vec<Vec<Vec<Vec<Var>>>>>>()?;

        // inventory at node n at *the end* time step t of product p
        let s: Vec<Vec<Vec<Var>>> = (sets.N, sets.T, sets.P).free(&mut model, &"s")?;

        // overflow at node n at time step t of product p
        let v_plus: Vec<Vec<Vec<Var>>> = (sets.N, sets.T, sets.P).cont(&mut model, &"v_plus")?;

        // shortage at node n at time step t of product p
        let v_minus: Vec<Vec<Vec<Var>>> = (sets.N, sets.T, sets.P).cont(&mut model, &"v_minus")?;

        // load of vessel v in time period t of product p
        let l: Vec<Vec<Vec<Var>>> = (sets.V, sets.T, sets.P).cont(&mut model, &"l")?;

        // loaded or unloaded at the i'th visit of route r by vessel v at time step t, indexed (r,i,v,t)
        let q: Vec<Vec<Vec<Vec<Vec<Var>>>>> = (0..sets.R)
            .map(|r| {
                (sets.I_R[r], sets.V, sets.T, sets.P).vars_with(|(i, v, t, p)| {
                    model.add_var(
                        &format!("q_{}_{}_{}_{}_{}", r, i, v, t, p),
                        SemiCont,
                        0.0,
                        parameters.F_min[parameters.node(r, i)][p],
                        parameters.F_max[parameters.node(r, i)][p],
                        std::iter::empty(),
                    )
                })
            })
            .collect::<Result<Vec<Vec<Vec<Vec<Vec<Var>>>>>>>()?;

        // ******************** ADD OBJECTIVES ********************
        let shortage = iproduct!(0..sets.N, 0..sets.T, 0..sets.P)
            .map(|(n, t, p)| v_plus[n][t][p] + v_minus[n][t][p])
            .grb_sum();

        let cost = iproduct!(0..sets.R, 0..sets.V, 0..sets.T)
            .map(|(r, v, t)| parameters.C_r[r][v] * x[r][0][v][t])
            .grb_sum();

        model.set_objective_N(cost, 0, 0, &"cost")?;
        model.set_objective_N(shortage, 1, 1, &"shortage")?;

        // ******************** ADD CONSTRAINTS ********************

        // ************* ROUTING ************

        // A vessel at the i'th stop of a route has two chioces
        // 1. Travel to the next stop
        // 2. Stay at the current node until the next time period
        for r in 0..sets.R {
            for (i, v) in iproduct!(0..(sets.I_R[r] - 1), 0..sets.V) {
                for t in 0..(sets.T_r[r][v][i] - 1) {
                    // travel time between the i'th and (i+1)'th visit of route r for vessel v
                    let t_time = parameters.travel[r][i][v];
                    let lhs = x[r][i][v][t];
                    let rhs = x[r][i + 1][v][t + t_time] + x[r][i][v][t + 1];
                    model.add_constr(&format!("1_{}_{}_{}_{}", r, i, v, t), c!(lhs == rhs))?;
                }
            }
        }

        // if a route is finished, the vessel can start a new route
        for r in 0..sets.R {
            // routes with the same start as the end of route r
            let possible_routes: Vec<usize> = (0..sets.R)
                .filter(|n_r| parameters.N_R[r][sets.I_R[r] - 1] == parameters.N_R[*n_r][0])
                .collect();

            for (v, t) in iproduct!(0..sets.V, 0..sets.T) {
                let lhs = x[r][sets.I_R[r] - 1][v][t];
                let rhs = possible_routes.iter().map(|n_r| x[*n_r][0][v][t]).grb_sum()
                    + x[r][sets.I_R[r] - 1][v][t + 1];
                model.add_constr(&format!("2_{}_{}_{}", r, v, t), c!(lhs == rhs))?;
            }
        }

        // a vessel can choose to start a route after it has become available
        for v in 0..sets.V {
            // routes with the same start as the origin of the vessel v
            let possible_routes = (0..sets.R).filter(|r| parameters.N_R[*r][0] == parameters.O[v]);
            let lhs = possible_routes
                .map(|r| x[r][0][v][parameters.T_0[v]])
                .grb_sum();
            model.add_constr(&format!("start_{}", v), c!(lhs <= 1))?;
        }

        // a vessel must reside in the end of a route at the end of the planning period if a route is initiated
        for v in 0..sets.V {
            let possible_routes = (0..sets.R).filter(|r| parameters.N_R[*r][0] == parameters.O[v]);
            let lhs = possible_routes
                .map(|r| x[r][0][v][parameters.T_0[v]])
                .grb_sum();
            let rhs = (0..sets.R)
                .map(|r| x[r][sets.I_R[r] - 1][v][sets.T - 1])
                .grb_sum();
            model.add_constr(&format!("end_criterion_{}", v), c!(lhs == rhs))?;
        }

        // ************* NODE INVENTORY ************

        // inventory in first time step at node n
        for (n, p) in iproduct!(0..sets.N, 0..sets.P) {
            model.add_constr(
                &format!("initial_inv_{}_{}", n, p),
                c!(s[n][0][p] == parameters.S_0[n][p]),
            )?;
        }

        // logging of inventory at node n
        for (n, p, t) in iproduct!(0..sets.N, 0..sets.P, 1..sets.T) {
            // delivered in time period t
            let delivered = (0..sets.R)
                .map(|r| {
                    let visits = (0..sets.I_R[r]).filter(|i| parameters.node(r, *i) == n);
                    let subsum = visits.map(|i| (0..sets.V).map(|v| q[r][i][v][t][p]).grb_sum());
                    subsum.grb_sum()
                })
                .grb_sum();
            let lhs = s[n][t][p];
            let rhs =
                s[n][t - 1][p] + parameters.kind(n) * (parameters.D[n][t][p] as isize) - delivered;
            model.add_constr(&format!("node_inv_{}_{}_{}", n, p, t), c!(lhs == rhs))?;
        }

        // shortage logging
        for (n, t, p) in iproduct!(sets.N_C, 0..sets.T, 0..sets.P) {
            let lhs = parameters.S_min[n][p][t];
            let rhs = s[n][p][t] + v_minus[n][t][p];
            model.add_constr(&format!("overflow_{}_{}_{}", n, t, p), c!(lhs <= rhs))?;
        }

        // shortage logging
        for (n, t, p) in iproduct!(sets.N_P, 0..sets.T, 0..sets.P) {
            let lhs = parameters.S_max[n][p][t];
            let rhs = s[n][p][t] - v_plus[n][t][p];
            model.add_constr(&format!("shortage_{}_{}_{}", n, t, p), c!(lhs <= rhs))?;
        }

        // ************* VESSEL LOAD ************

        // initial load
        for (v, p) in iproduct!(0..sets.V, 0..sets.P) {
            let lhs = l[v][parameters.T_0[v]][p];
            let rhs = parameters.L_0[v][p];
            model.add_constr(&format!("initial_load_{}_{}", v, p), c!(lhs == rhs))?;
        }

        // load logging
        for (v, t, p) in iproduct!(0..sets.V, 1..sets.T, 0..sets.P) {
            // loaded or unloaded in the time period
            let change = (0..sets.R)
                .map(|r| {
                    (0..sets.I_R[r])
                        .map(|i| parameters.visit_kind(r, i) * q[r][i][v][t][p])
                        .grb_sum()
                })
                .grb_sum();
            let lhs = l[v][t][p];
            let rhs = l[v][t - 1][p] + change;
            model.add_constr(&format!("load_{}_{}_{}", v, t, p), c!(lhs == rhs))?;
        }

        info!("Successfully built path-flow model");

        Ok((
            model,
            Variables {
                x,
                s,
                v_plus,
                v_minus,
                l,
                q,
            },
        ))
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
    pub fn new(variables: Variables, model: &Model) -> Result<PathFlowResult> {
        Ok(PathFlowResult {
            x: variables.x.convert(model)?,
            s: variables.s.convert(model)?,
            v_plus: variables.v_plus.convert(model)?,
            v_minus: variables.v_minus.convert(model)?,
            l: variables.l.convert(model)?,
            q: variables.q.convert(model)?,
        })
    }
}
