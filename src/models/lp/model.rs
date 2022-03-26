use std::collections::HashMap;

use crate::models::utils::{vars, vars2, AddVars, ConvertVars};

use super::sets_and_parameters::{NodeIndex, Parameters, ProductIndex, Sets, VisitIndex};
use derive_more::Constructor;
use grb::prelude::*;
use itertools::iproduct;
use log::trace;

#[derive(Constructor)]
pub struct Variables {
    pub x: Vec<Vec<Var>>,
    pub s: Vec<Vec<Var>>,
    pub l: Vec<Vec<Var>>,
    pub w_minus: HashMap<(VisitIndex, ProductIndex), Var>,
    pub w_plus: HashMap<(VisitIndex, ProductIndex), Var>,
    pub w_minus_end: HashMap<(NodeIndex, ProductIndex), Var>,
    pub w_plus_end: HashMap<(NodeIndex, ProductIndex), Var>,
}

pub struct LpSolver {}

#[allow(non_snake_case)]
impl LpSolver {
    /// Builds the lp determining the quantities
    pub fn build(sets: &Sets, parameters: &Parameters) -> grb::Result<(Model, Variables)> {
        trace!("Building lp for determining quantities");

        let mut model = Model::new(&format!("quant_lp"))?;

        // assign som variables to save some space later
        let P = &sets.P;
        let V = &sets.V;
        let N = &sets.N;
        let J = &sets.J;
        let J_n = &sets.J_n;
        let J_v = &sets.J_v;

        //*************CREATE VARIABLES*************//

        // loaded/unloaded at visit j of product p
        let x = (J.len(), P.len()).cont(&mut model, &"x")?;

        // inventory at the node assiciated with the visit at the beginning of the visit
        let s = (J.len(), P.len()).free(&mut model, &"s")?;

        // load of the vessel at the beginning of a vessel of every product type
        let l = (J.len(), P.len()).cont(&mut model, &"l")?;

        // shortage at the node associated with the visit at the beginning of the visit
        let indices = iproduct!(sets.consumption_visits(), P.into_iter().map(|p| *p)).collect();
        let w_minus = vars(
            indices,
            &mut model,
            VarType::Continuous,
            &(0.0..f64::INFINITY),
            "w_minus",
        )?;

        // shortage at the node associated with the visit at the beginning of the visit
        let indices = iproduct!(sets.production_visits(), P.into_iter().map(|p| *p)).collect();
        let w_plus = vars(
            indices,
            &mut model,
            VarType::Continuous,
            &(0.0..f64::INFINITY),
            "w_plus",
        )?;

        // shortage at the node at ending of the planning period
        let indices = iproduct!(sets.consumption_nodes(), P.into_iter().map(|p| *p)).collect();
        let w_minus_end = vars2(
            indices,
            &mut model,
            VarType::Continuous,
            &(0.0..f64::INFINITY),
            "w_end_minus",
        )?;

        // overflow at the node at ending of the planning period
        let indices = iproduct!(sets.production_nodes(), P.into_iter().map(|p| *p)).collect();
        let w_plus_end = vars2(
            indices,
            &mut model,
            VarType::Continuous,
            &(0.0..f64::INFINITY),
            "w_end_plus",
        )?;

        // ******************** ADD CONSTRAINTS ********************

        // INVENTORY CONSTRAINTS

        // set initial inventory at all nodes
        for n in N {
            let j = J_n[*n].first();
            if let Some(j) = j {
                for p in P {
                    let lhs = s[**j][**p];
                    let rhs = parameters.S_0[*n][*p];
                    model.add_constr(&format!("init_inv_{:?}_{:?}", j, p), c!(lhs == rhs))?;
                }
            }
        }

        // update inventories at the remaining visits
        for n in &N[1..] {
            for win in J_n[*n].windows(2) {
                let (i, j) = (win[0], win[1]);
                for p in P {
                    let lhs = s[*j][**p];
                    let D = parameters.D(i, j, *p).unwrap();
                    let rhs =
                        s[*i][**p] - parameters.kind(*n) * x[*i][**p] + parameters.kind(*n) * D;
                    model.add_constr(&format!("inv_{:?}_{:?}", j, p), c!(lhs == rhs))?;
                }
            }
        }

        // LOAG SHORTAGE AND OVERFLOW

        // log overflow
        for (j, p) in iproduct!(sets.production_visits(), P.into_iter().map(|p| *p)) {
            let lhs = s[*j][*p] - *w_plus.get(&(j, p)).unwrap();
            let rhs = parameters.S_max[j][p];
            model.add_constr(&format!("overflow_{:?}_{:?}", j, p), c!(lhs <= rhs))?;
        }

        // log shortage
        for (j, p) in iproduct!(sets.consumption_visits(), P.into_iter().map(|p| *p)) {
            let lhs = s[*j][*p] + *w_minus.get(&(j, p)).unwrap();
            let rhs = parameters.S_min[j][p];
            model.add_constr(&format!("overflow_{:?}_{:?}", j, p), c!(lhs >= rhs))?;
        }

        // BOUND DELIVERY / PICKED UP

        // upper bound given by (un)loading rate
        for (j, p) in iproduct!(J, P) {
            let lhs = x[**j][**p];
            let rhs = parameters.A[*j] * parameters.R[*j];
            model.add_constr(&format!("l_rate_{:?}_{:?}", j, p), c!(lhs <= rhs))?;
        }

        // upper bound given from node and vessel inventory restrictions
        for (j, p) in iproduct!(J, P) {
            // production
            if parameters.v_kind(*j) > 0.0 {
                // dont pick up more than available
                let lhs = x[**j][**p];
                let rhs = s[**j][**p] - parameters.S_min[*j][*p];
                model.add_constr(&format!("prod_available_{:?}_{:?}", j, p), c!(lhs <= rhs))?;

                // do not load more onto a ship than there is room for
                let lhs = x[**j][**p];
                let rhs = parameters.Q[parameters.V_j[*j]] - l[**j].iter().grb_sum();
                model.add_constr(&format!("vessel_space_{:?}_{:?}", j, p), c!(lhs <= rhs))?;
            }
            // consumption node
            else {
                // do not deliver more than there is room for at the node
                let lhs = x[**j][**p];
                let rhs = parameters.S_max[*j][*p] - s[**j][**p];
                model.add_constr(&format!("consump_space_{:?}_{:?}", j, p), c!(lhs <= rhs))?;

                // do not deliver more than available in the vessel
                let lhs = x[**j][**p];
                let rhs = l[**j][**p];
                model.add_constr(&format!("vessel_available_{:?}_{:?}", j, p), c!(lhs <= rhs))?;
            }
        }

        // log the vessel load
        for (v, p) in iproduct!(V, P) {
            // load of the first visit
            let j = J_v[*v].first();
            if let Some(j) = j {
                let lhs = l[**j][**p];
                let rhs = parameters.L_0[*v][*p];
                model.add_constr(&format!("initial_load_{:?}_{:?}", v, p), c!(lhs == rhs))?;
            }

            // remaining visitis
            for win in J_v[*v].windows(2) {
                let (i, j) = (win[0], win[1]);
                let lhs = l[*j][**p];
                let rhs = l[*i][**p] + parameters.v_kind(i) * x[*i][**p];
                model.add_constr(&format!("log_load_{:?}_{:?}", v, p), c!(lhs == rhs))?;
            }
        }

        // log end shortage
        for (n, p) in iproduct!(sets.consumption_nodes(), P) {
            let lhs = *w_minus_end.get(&(n, *p)).unwrap();

            // has no visits
            if J_n[n].is_empty() {
                let rhs =
                    parameters.D_tot(n, *p) - (parameters.S_0[n][*p] - parameters.S_min_n[n][*p]);
                model.add_constr(&format!("end_minus_{:?}_{:?}", n, p), c!(lhs >= rhs))?;
            } else {
                let j = J_n[n].first().unwrap();
                let rhs = parameters.remaining(*j, *p)
                    - (s[**j][**p] + x[**j][**p] - parameters.S_min_n[n][*p]);
                model.add_constr(&format!("end_minus_{:?}_{:?}", n, p), c!(lhs >= rhs))?;
            }
        }

        // log end overflow
        for (n, p) in iproduct!(sets.production_nodes(), P) {
            let lhs = *w_plus_end.get(&(n, *p)).unwrap();

            // has no visits
            if J_n[n].is_empty() {
                let rhs =
                    parameters.D_tot(n, *p) - (parameters.S_max_n[n][*p] - parameters.S_0[n][*p]);
                model.add_constr(&format!("end_plus_{:?}_{:?}", n, p), c!(lhs >= rhs))?;
            } else {
                let j = J_n[n].first().unwrap();
                let rhs = parameters.remaining(*j, *p)
                    - (parameters.S_max_n[n][*p] - s[**j][**p] - x[**j][**p]);
                model.add_constr(&format!("end_plus_{:?}_{:?}", n, p), c!(lhs >= rhs))?;
            }
        }

        // SET OBJECTIVE

        // shortage occuring during or before visits begin
        let shortage = iproduct!(J, P)
            .map(|(j, p)| match parameters.v_kind(*j) as isize {
                1 => w_plus.get(&(*j, *p)).unwrap(),
                -1 => w_minus.get(&(*j, *p)).unwrap(),
                _ => unreachable!("fuck off - this is impossible"),
            })
            .grb_sum();

        // end shortage
        let end_shortage = iproduct!(N, P)
            .map(|(n, p)| match parameters.kind(*n) as isize {
                1 => w_plus_end.get(&(*n, *p)).unwrap(),
                -1 => w_minus_end.get(&(*n, *p)).unwrap(),
                _ => unreachable!("fuck off - this is impossible"),
            })
            .grb_sum();

        model.set_objective(shortage + end_shortage, Minimize)?;

        model.update()?;

        trace!("Successfully built lp");

        Ok((
            model,
            Variables::new(x, s, l, w_minus, w_plus, w_minus_end, w_plus_end),
        ))
    }

    pub fn solve(sets: &Sets, parameters: &Parameters) -> Result<LpResult, grb::Error> {
        let (m, vars) = LpSolver::build(sets, parameters)?;
        let mut model = m;

        model.optimize()?;

        LpResult::new(&vars, &model)
    }
}

#[derive(Debug, Clone)]
pub struct LpResult {
    pub x: Vec<Vec<f64>>,
    pub s: Vec<Vec<f64>>,
    pub l: Vec<Vec<f64>>,
    pub w_minus: HashMap<(usize, usize), f64>,
    pub w_plus: HashMap<(usize, usize), f64>,
    pub w_minus_end: HashMap<(usize, usize), f64>,
    pub w_plus_end: HashMap<(usize, usize), f64>,
}

impl LpResult {
    pub fn new(variables: &Variables, model: &Model) -> Result<LpResult, grb::Error> {
        let x = variables.x.convert(model)?;
        let s = variables.s.convert(model)?;
        let l = variables.l.convert(model)?;
        let w_minus = variables
            .w_minus
            .iter()
            .map(|((v, p), var)| Ok(((**v, **p), model.get_obj_attr(attr::X, var)?)))
            .collect::<Result<HashMap<(usize, usize), f64>, grb::Error>>()?;
        let w_plus = variables
            .w_plus
            .iter()
            .map(|((v, p), var)| Ok(((**v, **p), model.get_obj_attr(attr::X, var)?)))
            .collect::<Result<HashMap<(usize, usize), f64>, grb::Error>>()?;
        let w_minus_end = variables
            .w_minus_end
            .iter()
            .map(|((v, p), var)| Ok(((**v, **p), model.get_obj_attr(attr::X, var)?)))
            .collect::<Result<HashMap<(usize, usize), f64>, grb::Error>>()?;
        let w_plus_end = variables
            .w_plus_end
            .iter()
            .map(|((v, p), var)| Ok(((**v, **p), model.get_obj_attr(attr::X, var)?)))
            .collect::<Result<HashMap<(usize, usize), f64>, grb::Error>>()?;

        Ok(LpResult {
            x,
            s,
            l,
            w_minus,
            w_plus,
            w_minus_end,
            w_plus_end,
        })
    }
}
