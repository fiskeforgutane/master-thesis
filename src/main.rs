pub mod models;
pub mod problem;
pub mod quants;
pub mod sisrs;
pub mod solution;
use grb::prelude::*;

use itertools::iproduct;
use models::path_flow::model::{PathFlowResult, PathFlowSolver};
use models::path_flow::sets_and_parameters::Route;
use models::transportation_model::model::TransportationSolver;
use problem::*;
use quants::*;

//use models::transportation_model::sets_and_params::{Parameters, Sets};
use models::path_flow::sets_and_parameters::{Parameters, Sets};

fn vessels() -> Vec<Vessel> {
    let vessel = Vessel::new(
        vec![Compartment(10.0), Compartment(15.0)],
        100.0,
        1.0,
        1.0,
        1.0,
        vec![1.0; 4],
        0,
        FixedInventory::from(Inventory::new(&[0.0]).unwrap()),
        0,
        "class 1".to_string(),
        0,
    );
    return vec![vessel];
}

fn nodes() -> Vec<Node> {
    let node_0 = Node::new(
        "node 0".to_string(),
        NodeType::Production,
        0,
        vec![1],
        5.0,
        50.0,
        10.0,
        FixedInventory::from(Inventory::new(&[50.0]).unwrap()),
        vec![FixedInventory::from(Inventory::new(&[3.0]).unwrap()); 50],
        5.0,
        vec![cumulative(vec![3.0; 50], 10.0)],
        FixedInventory::from(Inventory::new(&[10.0]).unwrap()),
    );

    let node_1 = Node::new(
        "node 1".to_string(),
        NodeType::Consumption,
        1,
        vec![1],
        5.0,
        50.0,
        10.0,
        FixedInventory::from(Inventory::new(&[50.0]).unwrap()),
        vec![FixedInventory::from(Inventory::new(&[3.0]).unwrap()); 50],
        5.0,
        vec![cumulative(vec![3.0; 50], 10.0)],
        FixedInventory::from(Inventory::new(&[10.0]).unwrap()),
    );

    let node_2 = Node::new(
        "node 2".to_string(),
        NodeType::Consumption,
        2,
        vec![1],
        5.0,
        50.0,
        10.0,
        FixedInventory::from(Inventory::new(&[50.0]).unwrap()),
        vec![FixedInventory::from(Inventory::new(&[3.0]).unwrap()); 50],
        5.0,
        vec![cumulative(vec![3.0; 50], 10.0)],
        FixedInventory::from(Inventory::new(&[10.0]).unwrap()),
    );

    let node_3 = Node::new(
        "node 3".to_string(),
        NodeType::Production,
        3,
        vec![1],
        5.0,
        50.0,
        10.0,
        FixedInventory::from(Inventory::new(&[50.0]).unwrap()),
        vec![FixedInventory::from(Inventory::new(&[3.0]).unwrap()); 50],
        5.0,
        vec![cumulative(vec![3.0; 50], 10.0)],
        FixedInventory::from(Inventory::new(&[10.0]).unwrap()),
    );
    return vec![node_0, node_1, node_2, node_3];
}

fn distances() -> Vec<Vec<Distance>> {
    vec![
        vec![0.0, 5.0, 6.0, 10.0],
        vec![5.0, 0.0, 4.0, 6.0],
        vec![6.0, 4.0, 0.0, 4.0],
        vec![10.0, 6.0, 4.0, 0.0],
    ]
}

fn cumulative(x: Vec<f64>, initial: Quantity) -> Vec<f64> {
    let mut y = Vec::new();
    y.push(initial + x[0]);
    for i in 1..x.len() {
        y.push(y[i - 1] + x[i]);
    }
    y
}

/* fn main() {
    let problem = Problem::new(vessels(), nodes(), 50, 1, distances());
    let params = Parameters::new(&prob, &sets);
    let mut res = TransportationSolver::build(&sets, &params, 0)?;
    let mut model = res.0;
    let vars = res.1;
    let res = TransportationSolver::solve(&sets, &params, 0)?;
    println!("x: {:?}", res.x);
    println!("y: {:?}", res.y);
    //let test = (Inventory::new(&[1.0]).unwrap());
    //println!("{:?}", prob.nodes()[0].inventory_without_deliveries(0));
    //let quants = Quantities::new(prob);
    //let orders = quants.initial_orders();
    println!("{:?}", sets);
    println!("Hello, world!");
    Ok(())
}  */

/* use crate::models::utils::NObjectives;

fn main() -> grb::Result<()> {
    let mut model = Model::new("model1")?;

    // add decision variables with no bounds
    let x1 = add_ctsvar!(model, name: "x1", bounds: 0..)?;
    let x2 = add_intvar!(model, name: "x2", bounds: 0..)?;

    // add linear constraints
    let c0 = model.add_constr("c0", c!(x1 <= 10))?;
    let c1 = model.add_constr("c1", c!(x2 <= 10))?;
    let c2 = model.add_constr("c2", c!(x1 + x2 <= 10))?;

    // model is lazily updated by default
    assert_eq!(
        model.get_obj_attr(attr::VarName, &x1).unwrap_err(),
        grb::Error::ModelObjectPending
    );
    assert_eq!(model.get_attr(attr::IsMIP)?, 0);

    // set the objective function, which updates the model objects (variables and constraints).
    // One could also call `model.update()`
    //model.set_objective(8 * x1 + x2, Maximize)?;

    //model.set_objective_N(8 * x1 + x2, 0, 1, &"test")?;
    //model.set_param(param::ObjNumber, 1)?;
    model.set_objective_N(x1, 0, 0, &"min x1")?;
    //model.set_objective_N(x2, 1, 1, &"min x2")?;
    model.set_attr(attr::ModelSense, Maximize)?;

    model.update()?;
    println!("{:?}", model.get_attr(attr::ObjNName)?);
    assert_eq!(model.get_obj_attr(attr::VarName, &x1)?, "x1");
    assert_eq!(model.get_attr(attr::IsMIP)?, 1);

    // write model to the file.
    model.write("model.lp")?;

    // optimize the model
    model.optimize()?;
    assert_eq!(model.status()?, Status::Optimal);

    model.set_param(param::ObjNumber, 0)?;
    model.set_attr(attr::NumObj, 0)?;
    model.set_objective_N(x2, 1, 1, &"min x2")?;
    model.optimize()?;
    // Querying a model attribute
    //assert_eq!(model.get_attr(attr::ObjVal)?, 59.0);

    // Querying a model object attributes
    //assert_eq!(model.get_obj_attr(attr::Slack, &c0)?, -34.5);
    let x1_name = model.get_obj_attr(attr::VarName, &x1)?;

    // Querying an attribute for multiple model objects
    let val = model.get_obj_attr_batch(attr::X, vec![x1, x2])?;
    //assert_eq!(val, [6.5, 7.0]);
    println!("val: {:?}", val);
    // Querying variables by name
    //assert_eq!(model.get_var_by_name(&x1_name)?, Some(x1));

    Ok(())
}
 */

fn main() {
    let prob = Problem::new(vessels(), nodes(), 10, 1, distances());

    let r = Route::new(vec![0, 1, 3], &prob);
    let r2 = Route::new(vec![3, 2, 3], &prob);
    let routes = vec![r, r2];
    let mut sets = Sets::new(&prob, &routes);
    println!("{:?}", sets.T_r[0][0]);
    let mut params = Parameters::new(&prob, &sets, &routes);

    let res = PathFlowSolver::build(&sets, &params).unwrap();
    let mut m = res.0;
    let mut variables = res.1;
    m.write("model.lp");
    let result = PathFlowSolver::solve(&variables, &mut m).unwrap();
    println!(
        "s: {:?},{:?},{:?}",
        result.s[0][0][0], result.s[1][0][0], result.s[2][0][0]
    );
    let mut a: Vec<(usize, usize)> = vec![(1, 2), (1, 1), (2, 1)];
    a.sort_by(|a, b| a.1.cmp(&b.1));
    a.sort_by(|a, b| a.0.cmp(&b.0));

    for r in 0..sets.R {
        for (i, v) in iproduct!(0..(sets.I_r[r] - 1), 0..sets.V) {
            for t in 0..(sets.T_r[r][v][i]) {
                // travel time between the i'th and (i+1)'th visit of route r for vessel v
                let t_time = params.travel[r][i][v];
                let lhs = &result.x[r][i][v][t];
                let rhs = &result.x[r][i + 1][v][t + t_time] + &result.x[r][i][v][t + 1];
                println!("1_{}_{}_{}_{}: lhs: {:?}, rhs: {:?}", r, i, v, t, lhs, rhs);
            }
        }
    }

    println!("q: {:?}", result.q);
    println!("x {:?}", result.x);
    println!("x_non_zero: {:?}", PathFlowResult::non_zero_4_d(result.x));
    println!("q 50 {:?}", result.q[0][0][0][1][0]);

    /* let r1 = Route::new(vec![0, 2, 0], &prob);
    sets.add_route(&prob, &r1);
    println!("r: {:?}\n T_r:{:?}\nI_r: {:?}", sets.R, sets.T_r, sets.I_r);

    params.add_route(&r1, &prob);
    println!(
        "C_r: {:?}\n travel: {:?}\n N_r: {:?}",
        params.C_r, params.travel, params.N_r
    );
    let a = m.get_constr_by_name("start_0");
    println!("{:?}", a);
    PathFlowSolver::add_routes(1, &mut variables, &sets, &params, &mut m).unwrap();

    //println!("hei");
    m.write("model2.lp");
    let res2 = PathFlowSolver::solve(&variables, &mut m).unwrap(); */
}
