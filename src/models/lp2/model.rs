use std::collections::HashMap;

use crate::models::{
    lp::sets_and_parameters::{NodeIndex, ProductIndex, TimeIndex, VesselIndex},
    utils::{better_vars, AddVars, ConvertVars},
};

use derive_more::Constructor;
use grb::prelude::*;
use itertools::iproduct;
use log::trace;
use pyo3::pyclass;

use super::sets_and_parameters::{Parameters, Sets};

#[derive(Constructor)]
pub struct Variables {
    pub x: HashMap<(TimeIndex, NodeIndex, VesselIndex, ProductIndex), Var>,
    pub s: HashMap<(TimeIndex, NodeIndex, ProductIndex), Var>,
    pub l: HashMap<(TimeIndex, VesselIndex, ProductIndex), Var>,
    pub w_minus: HashMap<(TimeIndex, NodeIndex, ProductIndex), Var>,
    pub w_plus: HashMap<(TimeIndex, NodeIndex, ProductIndex), Var>,
    pub w_end_minus: HashMap<(NodeIndex, ProductIndex), Var>,
    pub w_end_plus: HashMap<(NodeIndex, ProductIndex), Var>,
}

pub struct LpSolver2 {}

#[allow(non_snake_case)]
impl LpSolver2 {
    /// Builds the lp determining the quantities
    pub fn build(sets: &Sets, parameters: &Parameters) -> grb::Result<(Model, Variables)> {
        trace!("Building lp for determining quantities");

        let mut model = Model::new(&format!("quant_lp"))?;

        // assign som variables to save some space later
        let P = &sets.P;
        let V = &sets.V;
        let N = &sets.N;
        let T = &sets.T;
        let N_t = &sets.N_t;
        let N_tP = &sets.N_tP;
        let N_tC = &sets.N_tC;
        let N_P = &sets.N_P;
        let N_C = &sets.N_C;
        let T_n = &sets.T_n;
        let V_nt = &sets.V_nt;

        //*************CREATE VARIABLES*************//

        let indices = iproduct!(
            T.iter().cloned(),
            N.iter().cloned(),
            V.iter().cloned(),
            P.iter().cloned()
        )
        .collect();
        let x = better_vars(
            indices,
            &mut model,
            VarType::Continuous,
            &(0.0..f64::INFINITY),
            "x",
        )?;

        let indices = iproduct!(T.iter().cloned(), N.iter().cloned(), P.iter().cloned()).collect();
        let s = better_vars(
            indices,
            &mut model,
            VarType::Continuous,
            &(f64::NEG_INFINITY..f64::INFINITY),
            "s",
        )?;

        let indices = iproduct!(T.iter().cloned(), V.iter().cloned(), P.iter().cloned()).collect();
        let l = better_vars(
            indices,
            &mut model,
            VarType::Continuous,
            &(0.0..f64::INFINITY),
            "l",
        )?;

        let indices = T
            .iter()
            .cloned()
            .flat_map(|t| {
                iproduct!(N_tC.get(&t).unwrap().iter().cloned(), P.iter().cloned())
                    .map(move |(n, p)| (t, n, p))
            })
            .collect();
        let w_minus = better_vars(
            indices,
            &mut model,
            VarType::Continuous,
            &(0.0..f64::INFINITY),
            "w_minus",
        )?;

        let indices = T
            .iter()
            .cloned()
            .flat_map(|t| {
                iproduct!(N_tP.get(&t).unwrap().iter().cloned(), P.iter().cloned())
                    .map(move |(n, p)| (t, n, p))
            })
            .collect();
        let w_plus = better_vars(
            indices,
            &mut model,
            VarType::Continuous,
            &(0.0..f64::INFINITY),
            "w_plus",
        )?;

        let indices = iproduct!(N_P.iter().cloned(), P.iter().cloned()).collect();
        let w_end_plus = better_vars(
            indices,
            &mut model,
            VarType::Continuous,
            &(0.0..f64::INFINITY),
            "w_end_plus",
        )?;

        let indices = iproduct!(N_C.iter().cloned(), P.iter().cloned()).collect();
        let w_end_minus = better_vars(
            indices,
            &mut model,
            VarType::Continuous,
            &(0.0..f64::INFINITY),
            "w_end_minus",
        )?;

        // ******************** ADD CONSTRAINTS ********************

        // INVENTORY CONSTRAINTS

        // Set initial inventory at all nodes
        for (n, p) in iproduct!(N, P) {
            // check that the node actually is visite
            if !T_n[*n].is_empty() {
                // time period of first vist
                let t_0 = T_n[*n].first().unwrap();
                let lhs = *s.get(&(*t_0, *n, *p)).unwrap();
                let rhs = parameters.S_0[*n][*p];
                model.add_constr(&format!("init_inv_{:?}_{:?}", n, p), c!(lhs == rhs))?;
            }
        }

        // inv for remaining time peirods
        for (n, p) in iproduct!(N, P) {
            for t in T_n[*n].windows(2) {
                let i = t[0];
                let j = t[1];
                let lhs = *s.get(&(j, *n, *p)).unwrap();
                let a = V_nt
                    .get(&(*n, i))
                    .iter()
                    .flat_map(|vessels| vessels.iter().map(|v| *x.get(&(i, *n, *v, *p)).unwrap()))
                    .grb_sum();
                let rhs = *s.get(&(i, *n, *p)).unwrap()
                    - parameters.I[*n]
                        * V_nt
                            .get(&(*n, i))
                            .iter()
                            .flat_map(|vessels| {
                                vessels.iter().map(|v| *x.get(&(i, *n, *v, *p)).unwrap())
                            })
                            .grb_sum()
                    + parameters.I[*n] * parameters.D(*n, i, j, *p).unwrap();

                model.add_constr(&format!("inv_{:?}_{:?}", n, p), c!(lhs == rhs))?;
            }
        }

        // LOG SHORTAGE AND OVERFLOW

        // overflow
        for t in T.iter().cloned() {
            for (n, p) in iproduct!(N_tP.get(&t).unwrap().iter().cloned(), P.iter().cloned()) {
                let lhs = *s.get(&(t, n, p)).unwrap() - *w_plus.get(&(t, n, p)).unwrap();
                let rhs = parameters.S_max.get(&(n, p, t)).unwrap();
                model.add_constr(&format!("overflow_{:?}_{:?}_{:?}", t, n, p), c!(lhs <= rhs))?;
            }
        }

        // shortage
        for t in T.iter().cloned() {
            for (n, p) in iproduct!(N_tC.get(&t).unwrap().iter().cloned(), P.iter().cloned()) {
                let lhs = *s.get(&(t, n, p)).unwrap() + *w_minus.get(&(t, n, p)).unwrap();
                let rhs = parameters.S_min.get(&(n, p, t)).unwrap();
                model.add_constr(&format!("shortage_{:?}_{:?}_{:?}", t, n, p), c!(lhs >= rhs))?;
            }
        }

        // bound inventory
        // overflow
        for t in T.iter().cloned() {
            for (n, p) in iproduct!(N_tP.get(&t).unwrap().iter().cloned(), P.iter().cloned()) {
                let lhs = *s.get(&(t, n, p)).unwrap();
                let rhs = parameters.S_min.get(&(n, p, t)).unwrap();
                model.add_constr(
                    &format!("lower_inv_bound_prod_{:?}_{:?}_{:?}", t, n, p),
                    c!(lhs >= rhs),
                )?;
            }
        }

        // upper inventory bound consumpiton nodes
        for t in T.iter().cloned() {
            for (n, p) in iproduct!(N_tC.get(&t).unwrap().iter().cloned(), P.iter().cloned()) {
                let lhs = *s.get(&(t, n, p)).unwrap();
                let rhs = parameters.S_max.get(&(n, p, t)).unwrap();
                model.add_constr(
                    &format!("upper_inv_bound_cons_{:?}_{:?}_{:?}", t, n, p),
                    c!(lhs <= rhs),
                )?;
            }
        }

        // BOUND DELIVERD/PICKED UP

        for t in T.iter().cloned() {
            for n in N_t.get(&t).unwrap().iter().cloned() {
                for v in V_nt.get(&(n, t)).unwrap().iter().cloned() {
                    for p in P.iter().cloned() {
                        // bound from (un)loading rate
                        let lhs = *x.get(&(t, n, v, p)).unwrap();
                        let rhs = parameters.R.get(&(n, v, t)).unwrap();
                        model.add_constr(
                            &format!("load_rate_bound_{:?}_{:?}_{:?}_{:?}", t, n, v, p),
                            c!(lhs <= rhs),
                        )?;

                        // bound from node inventory
                        let rhs = match parameters.I[n] as isize {
                            //prodution
                            1 => {
                                *s.get(&(t, n, p)).unwrap()
                                    - *parameters.S_min.get(&(n, p, t)).unwrap()
                            }
                            // consumption
                            -1 => {
                                *parameters.S_max.get(&(n, p, t)).unwrap()
                                    - *s.get(&(t, n, p)).unwrap()
                            }
                            _ => unreachable!(),
                        };
                        model.add_constr(
                            &format!("load_bound_inv_{:?}_{:?}_{:?}_{:?}", t, n, v, p),
                            c!(lhs <= rhs),
                        )?;

                        // bound from vessel load and capacity
                        let rhs = match parameters.I[n] as isize {
                            //prodution
                            1 => {
                                parameters.Q[v]
                                    - P.iter()
                                        .cloned()
                                        .map(|p| *l.get(&(t, v, p)).unwrap())
                                        .grb_sum()
                            }
                            // consumption
                            -1 => *l.get(&(t, v, p)).unwrap() + 0.0,
                            _ => unreachable!(),
                        };

                        model.add_constr(
                            &format!("load_bound_load_{:?}_{:?}_{:?}_{:?}", t, n, v, p),
                            c!(lhs <= rhs),
                        )?;
                    }
                }
            }
        }

        // END SHORTAGE AND OVERFLOW

        // end overflow
        for (n, p) in iproduct!(N_P.iter().cloned(), P.iter().cloned()) {
            let lhs = *w_end_plus.get(&(n, p)).unwrap();
            let rhs = match T_n[n].is_empty() {
                // no visits
                true => Expr::Constant(
                    parameters.D_tot(n, p).unwrap()
                        - (parameters.S_max(n, p) - parameters.S_0[n][p]),
                ),
                // visits
                false => {
                    let last_t = *T_n[n].last().unwrap();
                    let remaining = parameters.D_rem(n, p, last_t).unwrap();
                    let max = parameters.S_max(n, p);
                    let current = *s.get(&(last_t, n, p)).unwrap();
                    let delivered = V_nt
                        .get(&(n, last_t))
                        .iter()
                        .flat_map(|vessels| {
                            vessels.iter().map(|v| *x.get(&(last_t, n, *v, p)).unwrap())
                        })
                        .grb_sum();
                    remaining - (max - current - delivered)
                }
            };
            model.add_constr(&format!("end_overflow_{:?}_{:?}", n, p), c!(lhs >= rhs))?;
        }

        // end shortage
        for (n, p) in iproduct!(N_C.iter().cloned(), P.iter().cloned()) {
            let lhs = *w_end_minus.get(&(n, p)).unwrap();
            let rhs = match T_n[n].is_empty() {
                // no visits
                true => Expr::Constant(
                    parameters.D_tot(n, p).unwrap()
                        - (parameters.S_0[n][p] - parameters.S_min(n, p)),
                ),
                // visits
                false => {
                    let last_t = *T_n[n].last().unwrap();
                    let remaining = parameters.D_rem(n, p, last_t).unwrap();
                    let current = *s.get(&(last_t, n, p)).unwrap();
                    let delivered = V_nt
                        .get(&(n, last_t))
                        .iter()
                        .flat_map(|vessels| {
                            vessels.iter().map(|v| *x.get(&(last_t, n, *v, p)).unwrap())
                        })
                        .grb_sum();
                    let min = parameters.S_min(n, p);

                    remaining - (current + delivered - min)
                }
            };
            model.add_constr(&format!("end_shortage_{:?}_{:?}", n, p), c!(lhs >= rhs))?;
        }

        // SET OBJECTIVE

        // shortage during visits
        let shortage = T
            .iter()
            .cloned()
            .map(|t| {
                iproduct!(N_t.get(&t).unwrap().iter().cloned(), P.iter().cloned())
                    .map(|(n, p)| match parameters.I[n] as isize {
                        1 => *w_plus.get(&(t, n, p)).unwrap(),
                        -1 => *w_minus.get(&(t, n, p)).unwrap(),
                        _ => unreachable!(),
                    })
                    .grb_sum()
            })
            .grb_sum();

        let end_shortage = iproduct!(N.iter().cloned(), P.iter().cloned())
            .map(|(n, p)| match parameters.I[n] as isize {
                1 => *w_end_plus.get(&(n, p)).unwrap(),
                -1 => *w_end_minus.get(&(n, p)).unwrap(),
                _ => unreachable!(),
            })
            .grb_sum();

        model.set_objective(shortage + end_shortage, Minimize)?;

        model.update()?;

        trace!("Successfully built lp");

        Ok((
            model,
            Variables::new(x, s, l, w_minus, w_plus, w_end_minus, w_end_plus),
        ))
    }

    pub fn solve(sets: &Sets, parameters: &Parameters) -> Result<LpResult2, grb::Error> {
        let (m, vars) = LpSolver2::build(sets, parameters)?;
        let mut model = m;

        model.optimize()?;

        LpResult2::new(&vars, &model)
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct LpResult2 {
    #[pyo3(get)]
    pub x: HashMap<(usize, usize, usize, usize), f64>,
    #[pyo3(get)]
    pub s: HashMap<(usize, usize, usize), f64>,
    #[pyo3(get)]
    pub l: HashMap<(usize, usize, usize), f64>,
    #[pyo3(get)]
    pub w_minus: HashMap<(usize, usize, usize), f64>,
    #[pyo3(get)]
    pub w_plus: HashMap<(usize, usize, usize), f64>,
    #[pyo3(get)]
    pub w_end_minus: HashMap<(usize, usize), f64>,
    #[pyo3(get)]
    pub w_end_plus: HashMap<(usize, usize), f64>,
}

impl LpResult2 {
    pub fn new(variables: &Variables, model: &Model) -> Result<LpResult2, grb::Error> {
        let x = variables
            .x
            .iter()
            .map(|((t, n, v, p), var)| {
                Ok(((**t, **n, **v, **p), model.get_obj_attr(attr::X, var)?))
            })
            .collect::<Result<HashMap<(usize, usize, usize, usize), f64>, grb::Error>>()?;
        let s = variables
            .s
            .iter()
            .map(|((t, n, p), var)| Ok(((**t, **n, **p), model.get_obj_attr(attr::X, var)?)))
            .collect::<Result<HashMap<(usize, usize, usize), f64>, grb::Error>>()?;
        let l = variables
            .l
            .iter()
            .map(|((t, v, p), var)| Ok(((**t, **v, **p), model.get_obj_attr(attr::X, var)?)))
            .collect::<Result<HashMap<(usize, usize, usize), f64>, grb::Error>>()?;
        let w_minus = variables
            .w_minus
            .iter()
            .map(|((t, n, p), var)| Ok(((**t, **n, **p), model.get_obj_attr(attr::X, var)?)))
            .collect::<Result<HashMap<(usize, usize, usize), f64>, grb::Error>>()?;
        let w_plus = variables
            .w_plus
            .iter()
            .map(|((t, n, p), var)| Ok(((**t, **n, **p), model.get_obj_attr(attr::X, var)?)))
            .collect::<Result<HashMap<(usize, usize, usize), f64>, grb::Error>>()?;
        let w_end_minus = variables
            .w_end_minus
            .iter()
            .map(|((t, n), var)| Ok(((**t, **n), model.get_obj_attr(attr::X, var)?)))
            .collect::<Result<HashMap<(usize, usize), f64>, grb::Error>>()?;
        let w_end_plus = variables
            .w_end_plus
            .iter()
            .map(|((t, n), var)| Ok(((**t, **n), model.get_obj_attr(attr::X, var)?)))
            .collect::<Result<HashMap<(usize, usize), f64>, grb::Error>>()?;

        Ok(LpResult2 {
            x,
            s,
            l,
            w_minus,
            w_plus,
            w_end_minus,
            w_end_plus,
        })
    }
}
