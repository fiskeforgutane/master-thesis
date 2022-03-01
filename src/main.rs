pub mod problem;
pub mod quants;
pub mod sisrs;
use problem::*;
use quants::*;

fn vessels() -> Vec<Vessel> {
    let vessel = Vessel::new(
        vec![Compartment(10.0), Compartment(15.0)],
        10.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0,
        FixedInventory::from(Inventory::new(&[0.0]).unwrap()),
        0,
        "class 1".to_string(),
    );
    return vec![vessel];
}

fn nodes() -> Vec<Node> {
    /* let node_0 = Node::new(
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
    ); */

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
    return vec![node_1, node_2];
}

fn distances() -> Vec<Vec<Distance>> {
    vec![
        vec![1.0, 1.0, 1.0],
        vec![1.0, 1.0, 1.0],
        vec![1.0, 1.0, 1.0],
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

fn main() {
    let prob = Problem::new(vessels(), nodes(), 50, 1, distances());
    //let test = (Inventory::new(&[1.0]).unwrap());
    println!("{:?}", prob.nodes()[0].inventory_without_deliveries(0));
    let quants = Quantities::new(prob);
    let orders = quants.initial_orders();
    println!("{:?}", orders);
    println!("Hello, world!");
}
