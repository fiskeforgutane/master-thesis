use crate::problem::ProductIndex;

use super::sets_and_params::{Parameters, Sets};
use grb::prelude::*;
use itertools::iproduct;

pub struct TransportationSolver {}

impl TransportationSolver {
    /// builds the transportation model
    fn build(
        sets: &Sets,
        parameters: &Parameters,
        product: usize,
    ) -> grb::Result<(Model, Variables)> {
        let mut model = Model::new(&format!("transport_model_{}", product))?;

        //*************CREATE VARIABLES*************//

        // quantity transported from node i to node j with cargo h
        let x: Vec<Vec<Vec<Var>>> = sets
            .N
            .iter()
            .map(|i| {
                sets.N
                    .iter()
                    .map(|j| {
                        sets.H
                            .iter()
                            .map(|h| {
                                add_ctsvar!(model, name: &format!("x_{}_{}_{}",i,j,h), bounds:0.0..)
                                    .unwrap()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        // 1 if the h'th cargo from node i to node j exists, 0 otherwise
        let y: Vec<Vec<Vec<Var>>> = sets
            .N
            .iter()
            .map(|i| {
                sets.N
                    .iter()
                    .map(|j| {
                        sets.H
                            .iter()
                            .map(|h| {
                                add_binvar!(model, name: &format!("y_{}_{}_{}", i, j, h)).unwrap()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        // itegrate all the variables into the model
        model.update().unwrap();

        // ******************** ADD CONSTRAINTS ********************

        // ensure that enough is picked up at the production nodes
        for i in &sets.N {
            if parameters.J[*i] < 0 {
                continue;
            }
            let lhs = iproduct!(&sets.N, &sets.H)
                .map(|(j, h)| &x[*i][*j][*h])
                .grb_sum();
            model.add_constr(&format!("1_{i}"), c!(lhs >= parameters.Q[*i][product]))?;
        }

        // ensure that enough is delivered at the consumption nodes
        for j in &sets.N {
            if parameters.J[*j] > 0 {
                continue;
            }
            let lhs = iproduct!(&sets.N, &sets.H)
                .map(|(i, h)| &x[*i][*j][*h])
                .grb_sum();
            model.add_constr(&format!("2_{j}"), c!(lhs >= parameters.Q[*j][product]))?;
        }

        // lower limit on transportation amount
        for (i, j, h) in iproduct!(&sets.N, &sets.N, &sets.H) {
            // lower bound on how much can be transported from node i to node j
            let lhs = parameters.lower_Q[*i][*j][product] * y[*i][*j][*h];
            model.add_constr(
                &format!("lower_transportation_limit_{i}_{j}_{h}"),
                c!(lhs <= x[*i][*j][*h]),
            )?;
        }

        // lower limit on transportation amount
        for (i, j, h) in iproduct!(&sets.N, &sets.N, &sets.H) {
            // lower bound on how much can be transported from node i to node j
            let rhs = parameters.upper_Q[*i][*j][product] * y[*i][*j][*h];
            model.add_constr(
                &format!("upper_transportation_limit_{i}_{j}_{h}"),
                c!(x[*i][*j][*h] <= rhs),
            )?;
        }

        // symmetry braking constraint stating that if cargo h from node i to node j with h > 1, then cargo h-1 must also exist
        for (i, j, h) in iproduct!(&sets.N, &sets.N, &sets.H[1..]) {
            model.add_constr(
                &format!("symmetry_{}_{}_{}", i, j, h),
                c!(y[*i][*j][h - 1] >= y[*i][*j][*h]),
            )?;
        }

        model.update()?;

        Ok((model, Variables::new(x, y)))
        //Ok(model)
    }

    pub fn solve(
        sets: &Sets,
        parameters: &Parameters,
        product: ProductIndex,
    ) -> Result<TransportationResult, grb::Error> {
        // build model
        let (m, vars) = TransportationSolver::build(sets, parameters, product)?;
        let mut model = m;

        // optimize model
        model.optimize()?;

        TransportationResult::new(&vars, &model)
    }
}

pub struct TransportationResult {
    /// nonzero quantities transported from node i to j with cargo h
    pub x: Vec<Vec<Vec<f64>>>,
    /// nonzero y variables, not necessary fo results but good for testing
    pub y: Vec<Vec<Vec<f64>>>,
}
impl TransportationResult {
    pub fn new(variables: &Variables, model: &Model) -> Result<TransportationResult, grb::Error> {
        let x = TransportationResult::convert(&variables.x, model)?;
        let y = TransportationResult::convert(&variables.y, model)?;

        Ok(TransportationResult { x, y })
    }

    fn convert(vars: &Vec<Vec<Vec<Var>>>, model: &Model) -> Result<Vec<Vec<Vec<f64>>>, grb::Error> {
        let get_value = |var| model.get_obj_attr(attr::X, var);
        vars.iter()
            .map(|i| {
                i.iter()
                    .map(|e| e.iter().map(|v| get_value(v)).collect())
                    .collect()
            })
            .collect()
    }
}

pub struct Variables {
    x: Vec<Vec<Vec<Var>>>,
    y: Vec<Vec<Vec<Var>>>,
}
impl Variables {
    pub fn new(x: Vec<Vec<Vec<Var>>>, y: Vec<Vec<Vec<Var>>>) -> Variables {
        Variables { x, y }
    }
}
