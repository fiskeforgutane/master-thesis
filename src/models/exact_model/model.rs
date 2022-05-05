use crate::models::exact_model::sets_and_parameters::NetworkNodeType;
use crate::models::utils::{AddVars, ConvertVars};
use grb::prelude::*;
use itertools::iproduct;
use log::info;

use super::sets_and_parameters::{Parameters, Sets};

pub struct ExactModelSolver {}

#[allow(non_snake_case)]
impl ExactModelSolver {
    pub fn build(sets: &Sets, parameters: &Parameters) -> grb::Result<(Model, Variables)> {
        info!("Building exact model.");

        let mut model = Model::new("exact_model")?;
        model.set_param(grb::param::OutputFlag, 0)?;

        // Disable console output
        model.set_param(param::OutputFlag, 0)?;

        //*****************CREATE VARIABLES*****************//
        let vessels = sets.V.len();
        let arcs = sets.A.len();
        let products = sets.P.len();
        let ports = sets.I.len();
        let nodes = sets.N.len();
        let timesteps = sets.T.len();

        // 1 if the vessel traverses the arc, 0 otherwise
        let x: Vec<Vec<Var>> = (vessels, arcs).binary(&mut model, &"x")?;
        // 1 if the vessel is able to unload at the node, 0 otherwise
        let z: Vec<Vec<Var>> = (nodes, vessels).binary(&mut model, &"z")?;
        // Quantity unloaded of product at node by the vessel
        let q: Vec<Vec<Vec<Vec<Var>>>> = (ports, vessels, timesteps, products).cont(&mut model, &"q")?;
        // The quantity sold of product by port in timestep
        let a: Vec<Vec<Vec<Var>>> = (ports, timesteps, products).cont(&mut model, &"q")?;
        // The current inventory of product in port in timestep
        let s_port: Vec<Vec<Vec<Var>>> =
            (ports, timesteps, products).cont(&mut model, &"s_port")?;
        // The current inventory of product in vessel in timestep
        let s_vessel: Vec<Vec<Vec<Var>>> =
            (vessels, timesteps, products).cont(&mut model, &"s_vessels")?;

        model.update()?;

        //*****************ADD CONSTRAINTS*****************//

        // ensure that the all "normal" nodes have as many arcs entering as those leaving
        // and that the source just has one leaving, and that the sink has one entering
        for (n, v) in iproduct!(&sets.N, &sets.V) {
            let lhs = sets.Fs[n.get_index()]
                .iter()
                .map(|a| &x[*a][*v])
                .grb_sum()
                - sets.Rs[n.get_index()]
                    .iter()
                    .map(|a| &x[*a][*v])
                    .grb_sum();
            let rhs: isize;
            match n.get_kind() {
                NetworkNodeType::Source => rhs = 1,
                NetworkNodeType::Sink => rhs = -1,
                NetworkNodeType::Normal => rhs = 0,
            }

            let node_index = n.get_index();
            model.add_constr(&format!("travel_{v}_{node_index}"), c!(lhs == rhs))?;
        }

        // port storage balance
        for n in &sets.N {
            if n.get_time() == 0 {
                continue;
            }
            for p in &sets.P {
                let i = n.get_port();
                let t = n.get_time();
                let lhs = s_port[i][t][*p]
                    - s_port[i][t - 1][*p]
                    - parameters.port_type[i]
                        * (parameters.consumption[i][t][*p]
                            - sets.V.iter().map(|v| &q[i][*v][t][*p]).grb_sum()
                            - a[i][t][*p]);

                model.add_constr(&format!("port_inventory_{i}_{t}_{p}"), c!(lhs == 0.0_f64))?;
            }
        }

        // initial port storage
        for (i,p) in iproduct!(&sets.I, &sets.P) {
            let lhs = s_port[*i][0][*p]
                - parameters.initial_port_inventory[*i][*p]
                - parameters.port_type[*i]
                    * (parameters.consumption[*i][0][*p]
                        - sets.V.iter().map(|v| &q[*i][*v][0][*p]).grb_sum()
                        - a[*i][0][*p]);

            model.add_constr(&format!("initial_port_inventory_{i}_0_{p}"), c!(lhs == 0.0_f64))?;
        }

        // ensure that the inventory in all nodes remain within the lower and upper boundaries
        for (i, t, p) in iproduct!(&sets.I, &sets.T, &sets.P) {
            model.add_constr(&format!("lower_limit_port_{i}_{t}_{p}"), c!(s_port[*i][*t][*p] >= parameters.min_inventory_port[*i][*p]))?;
            model.add_constr(&format!("upper_limit_port_{i}_{t}_{p}"), c!(s_port[*i][*t][*p] <= parameters.port_capacity[*i][*p]))?;
        }

        // ensure balance of vessel storage
        for (v, t, p) in iproduct!(&sets.V, &sets.T, &sets.P) {
            if *t > 0 {
                let lhs = s_vessel[*v][*t][*p] - s_vessel[*v][*t-1][*p] - sets.I.iter().map(
                    |i| parameters.port_type[*i] * q[*i][*v][*t][*p]
                ).grb_sum();

                model.add_constr(&format!("storage_balance_vessel_{v}_{t}_{p}"), c!(lhs == 0.0_f64))?;
            }
        }

        // ensure initial storage in all vessels
        for (v, p) in iproduct!(&sets.V, &sets.P) {
            let lhs = s_vessel[*v][0][*p] - parameters.initial_inventory[*v][*p] - sets.I.iter().map(
                |i| parameters.port_type[*i] * q[*i][*v][0][*p]
            ).grb_sum();

            model.add_constr(&format!("initial_storage_balance_vessel_{v}_0_{p}"), c!(lhs == 0.0_f64))?;
        }

        // ensure that the load at the vessels is within the capacity limits
        for (v, t) in iproduct!(&sets.V, &sets.T) {
            let lhs = sets.P.iter().map(
                |p| s_vessel[*v][*t][*p]
            ).grb_sum();

            let rhs = 0.9 * parameters.vessel_capacity[*v];

            model.add_constr(&format!("vessel_storage_capacity_{v}_{t}"), c!(lhs <= rhs))?;
        }

        // berth capacities must be respected
        for n in &sets.N {
            let i = n.get_port();
            let t = n.get_time();

            let lhs = sets.V.iter().map(
                |v| z[n.get_index()][*v]
            ).grb_sum();

            let rhs = parameters.berth_capacity[i][t];

            model.add_constr(&format!("berth_capacity_{i}_{t}"), c!(lhs <= rhs))?;
        }

        // ensure that all vessels that deliver something to a node are located at that node
        for (n,v) in iproduct!(&sets.N, &sets.V) {
            let lhs = z[n.get_index()][*v];

            let rhs = sets.Rs[*v].iter().map(
                |a| x[*v][*a]
            ).grb_sum();

            let time = n.get_time();
            let port = n.get_port();

            model.add_constr(&format!("present_at_node_{port}_{time}"), c!(lhs <= rhs))?;
        }

        // the delivered quantity must lie within the boundaries for the vessel
        for (n, v) in iproduct!(&sets.N, &sets.V) {
            let i = n.get_port();
            let t = n.get_time();

            let lower = parameters.min_loading_rate[i] * z[n.get_index()][*v];
            let upper = parameters.max_loading_rate[i] * z[n.get_index()][*v];
            let loading_rate = sets.P.iter().map(
                |p| q[i][*v][t][*p]
            ).grb_sum();

            model.add_constr(&format!("lower_loading_rate_{i}_{t}"), c!(loading_rate.clone() >= lower))?;
            model.add_constr(&format!("upper_loading_rate_{i}_{t}"), c!(loading_rate <= upper))?;
        }

        // the total amount of products sold/bought from the spot market must within 
        // the limits
        for i in &sets.I {
            let lhs = iproduct!(&sets.T, &sets.P).map(
                |(t, p)| a[*i][*t][*p]
            ).grb_sum();

            model.add_constr(&format!("spot_market_upper_{i}"), c!(lhs <= parameters.max_spot_horizon[*i]))?;
        }

        // we must ensure that the amount bought from the spot market in each time period
        // is within the limits
        for (i,t) in iproduct!(&sets.I, &sets.T) {
            let lhs = sets.P.iter().map(
                |p| a[*i][*t][*p]
            ).grb_sum();

            model.add_constr(&format!("spot_market_period_{i}_{t}"), c!(lhs <= parameters.max_spot_period[*i][*t]))?;
        }

        let revenue = iproduct!(&sets.N, &sets.V, &sets.P).map(
            |(n, v, p)| parameters.revenue[n.get_port()] * q[n.get_port()][*v][n.get_time()][*p]
        ).grb_sum();

        let transportation_cost = iproduct!(&sets.V, &sets.A).map(
            |(v, a)| parameters.travel_cost[a.get_from().get_port()][a.get_to().get_port()][*v] * x[*v][a.get_index()]
        ).grb_sum();

        let spot_market_cost = iproduct!(&sets.I, &sets.T, &sets.P).map(
            |(i, t, p)| parameters.spot_market_cost[*t] * a[*i][*t][*p]
        ).grb_sum();

        let delay_cost = iproduct!(&sets.N, &sets.V).map(
            |(i, v)| (i.get_time() as f64) * parameters.epsilon * z[i.get_index()][*v]
        ).grb_sum();
        
        model.set_objective(revenue - transportation_cost - spot_market_cost - delay_cost, Maximize)?;

        model.update()?;

        info!(
            "Successfully built exact model",
        );

        Ok((model, Variables::new(x, z, q, a, s_port, s_vessel)))
    }
    
    pub fn solve(
        sets: &Sets,
        parameters: &Parameters,
    ) -> Result<ExactModelResults, grb::Error>{
        let (m, variables) = ExactModelSolver::build(sets, parameters)?;
        let mut model = m;

        model.optimize()?;

        ExactModelResults::new(&variables, &model)
    }
}

pub struct ExactModelResults {
    x: Vec<Vec<f64>>,
    z: Vec<Vec<f64>>,
    q: Vec<Vec<Vec<Vec<f64>>>>,
    a: Vec<Vec<Vec<f64>>>,
    s_port: Vec<Vec<Vec<f64>>>,
    s_vessel: Vec<Vec<Vec<f64>>>,
}

impl ExactModelResults {
    pub fn new(variables: &Variables, model: &Model) -> Result<ExactModelResults, grb::Error> {
        let x = variables.x.convert(model)?;
        let z = variables.z.convert(model)?;
        let q = variables.q.convert(model)?;
        let a = variables.a.convert(model)?;
        let s_port = variables.s_port.convert(model)?;
        let s_vessel = variables.s_vessel.convert(model)?;

        Ok(ExactModelResults {x, z, q, a, s_port, s_vessel} )
    }
}



pub struct Variables {
    x: Vec<Vec<Var>>,
    z: Vec<Vec<Var>>,
    q: Vec<Vec<Vec<Vec<Var>>>>,
    a: Vec<Vec<Vec<Var>>>,
    s_port: Vec<Vec<Vec<Var>>>,
    s_vessel: Vec<Vec<Vec<Var>>>,
}

impl Variables {
    pub fn new(
        x: Vec<Vec<Var>>,
        z: Vec<Vec<Var>>,
        q: Vec<Vec<Vec<Vec<Var>>>>,
        a: Vec<Vec<Vec<Var>>>,
        s_port: Vec<Vec<Vec<Var>>>,
        s_vessel: Vec<Vec<Vec<Var>>>,
    ) -> Variables {
        Variables {
            x,
            z,
            q,
            a,
            s_port,
            s_vessel,
        }
    }
}
