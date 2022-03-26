use crate::models::utils::{vars, vars2, AddVars};

use super::sets_and_parameters::{Parameters, Sets};
use grb::prelude::*;
use itertools::iproduct;
use log::trace;

pub struct Variables {}

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
        let w_plus_end = vars2(
            indices,
            &mut model,
            VarType::Continuous,
            &(0.0..f64::INFINITY),
            "w_end_minus",
        )?;

        // shortage at the node at ending of the planning period
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

        todo!()
    }
}
