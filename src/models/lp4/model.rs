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
    pub x: Vec<Vec<Vec<Vec<Var>>>>,
    pub s: Vec<Vec<Vec<Var>>>,
    pub l: Vec<Vec<Vec<Var>>>,
    pub w_minus: HashMap<(TimeIndex, NodeIndex, ProductIndex), Var>,
    pub w_plus: HashMap<(TimeIndex, NodeIndex, ProductIndex), Var>,
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

        let x = (T.len(), N.len(), V.len(), P.len()).cont(&mut model, "x")?;
        let s = (T.len(), N.len(), P.len()).free(&mut model, &"s")?;
        let l = (T.len(), V.len(), P.len()).cont(&mut model, &"l")?;

        let indices = T
            .iter()
            .cloned()
            .flat_map(|t| {
                iproduct!(N_C.iter().cloned(), P.iter().cloned()).map(move |(n, p)| (t, n, p))
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
                iproduct!(N_P.iter().cloned(), P.iter().cloned()).map(move |(n, p)| (t, n, p))
            })
            .collect();
        let w_plus = better_vars(
            indices,
            &mut model,
            VarType::Continuous,
            &(0.0..f64::INFINITY),
            "w_plus",
        )?;

        // ******************** ADD CONSTRAINTS ********************

        // INVENTORY CONSTRAINTS

        // Set initial inventory at all nodes
        for (n, p) in iproduct!(N.iter().cloned(), P.iter().cloned()) {
            let lhs = s[0][*n][*p];
            let rhs = parameters.S_0[n][p];
            model.add_constr(&format!("init_inv_{:?}_{:?}", n, p), c!(lhs == rhs))?;
        }

        // inv for remaining time peirods
        for (n, p) in iproduct!(N.iter().cloned(), P.iter().cloned()) {
            for t in T_n[n].windows(2) {
                let i = t[0];
                let j = t[1];
                let lhs = s[*j][*n][*p];
                let rhs = s[*i][*n][*p]
                    - parameters.I[n] * V.iter().map(|v| x[*i][*n][**v][*p]).grb_sum()
                    + parameters.I[n] * parameters.D(n, i, j, p).unwrap();
                model.add_constr(&format!("inv_{:?}_{:?}", n, p), c!(lhs == rhs))?;
            }
        }

        // LOG SHORTAGE AND OVERFLOW

        // overflow
        for t in T.iter().cloned() {
            for (n, p) in iproduct!(N_tP.get(&t).unwrap().iter().cloned(), P.iter().cloned()) {
                let lhs = s[*t][*n][*p] - *w_plus.get(&(t, n, p)).unwrap();
                let rhs = parameters.S_max.get(&(n, p, t)).unwrap();
                model.add_constr(&format!("overflow_{:?}_{:?}_{:?}", t, n, p), c!(lhs <= rhs))?;
            }
        }

        // shortage
        for t in T.iter().cloned() {
            for (n, p) in iproduct!(N_tC.get(&t).unwrap().iter().cloned(), P.iter().cloned()) {
                let lhs = s[*t][*n][*p] + *w_minus.get(&(t, n, p)).unwrap();
                let rhs = parameters.S_min.get(&(n, p, t)).unwrap();
                model.add_constr(&format!("shortage_{:?}_{:?}_{:?}", t, n, p), c!(lhs >= rhs))?;
            }
        }

        // bound inventory
        // overflow
        for t in T.iter().cloned() {
            for (n, p) in iproduct!(N_P.iter().cloned(), P.iter().cloned()) {
                let lhs = s[*t][*n][*p];
                let rhs = parameters.S_min.get(&(n, p, t)).unwrap();
                model.add_constr(
                    &format!("lower_inv_bound_prod_{:?}_{:?}_{:?}", t, n, p),
                    c!(lhs >= rhs),
                )?;
            }
        }

        // upper inventory bound consumpiton nodes
        for t in T.iter().cloned() {
            for (n, p) in iproduct!(N_C.iter().cloned(), P.iter().cloned()) {
                let lhs = s[*t][*n][*p];
                let rhs = parameters.S_max.get(&(n, p, t)).unwrap();
                model.add_constr(
                    &format!("upper_inv_bound_cons_{:?}_{:?}_{:?}", t, n, p),
                    c!(lhs <= rhs),
                )?;
            }
        }

        // BOUND DELIVERD/PICKED UP
        for (t, n, v, p) in iproduct!(
            T.iter().cloned(),
            N.iter().cloned(),
            V.iter().cloned(),
            P.iter().cloned()
        ) {
            // bound from (un)loading rate
            let lhs = x[*t][*n][*v][*p];
            let rhs = parameters.R.get(&(n, v, t)).unwrap();
            model.add_constr(
                &format!("load_rate_bound_{:?}_{:?}_{:?}_{:?}", t, n, v, p),
                c!(lhs <= rhs),
            )?;
        }

        // VESSE LOAD CONSTRAINTS

        // initial load
        for (v, p) in iproduct!(V.iter().cloned(), P.iter().cloned()) {
            let lhs = l[0][*v][*p];
            let rhs = parameters.L_0[v][p];
            model.add_constr(&format!("vessel_init_load_{:?}_{:?}", v, p), c!(lhs == rhs))?;
        }

        // set load for remaining time periods
        for (v, p) in iproduct!(V.iter().cloned(), P.iter().cloned()) {
            for t in T.windows(2) {
                let t1 = t[0];
                let t2 = t[1];

                let lhs = l[*t2][*v][*p];
                let rhs = l[*t1][*v][*p] + N.iter().map(|n| x[*t1][**n][*v][*p]).grb_sum();
                model.add_constr(&format!("vessel_load_{:?}_{:?}", v, p), c!(lhs == rhs))?;
            }
        }

        // load bounds
        for (v, t) in iproduct!(V.iter().cloned(), T.iter().cloned()) {
            let lhs = P.iter().map(|p| l[*t][*v][**p]).grb_sum();
            let rhs = parameters.Q[v];
            model.add_constr(
                &format!("vessel_load_bound_{:?}_{:?}", v, t),
                c!(lhs <= rhs),
            )?;
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

        /* let end_shortage = iproduct!(N.iter().cloned(), P.iter().cloned())
        .map(|(n, p)| match parameters.I[n] as isize {
            1 => *w_end_plus.get(&(n, p)).unwrap(),
            -1 => *w_end_minus.get(&(n, p)).unwrap(),
            _ => unreachable!(),
        })
        .grb_sum(); */

        model.set_objective(shortage, Minimize)?;

        model.update()?;

        trace!("Successfully built lp");

        Ok((model, Variables::new(x, s, l, w_minus, w_plus)))
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
    pub x: Vec<Vec<Vec<Vec<f64>>>>,
    #[pyo3(get)]
    pub s: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub l: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub w_minus: HashMap<(usize, usize, usize), f64>,
    #[pyo3(get)]
    pub w_plus: HashMap<(usize, usize, usize), f64>,
}

impl LpResult2 {
    pub fn new(variables: &Variables, model: &Model) -> Result<LpResult2, grb::Error> {
        let x = variables.x.convert(model)?;
        let s = variables.s.convert(model)?;
        let l = variables.l.convert(model)?;

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

        Ok(LpResult2 {
            x,
            s,
            l,
            w_minus,
            w_plus,
        })
    }
}
