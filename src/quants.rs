use std::collections::HashMap;

use grb::Result;
use pyo3::pyclass;

use crate::models::transportation_model::model::TransportationSolver;
use crate::models::transportation_model::sets_and_parameters::{Parameters, Sets};
use crate::problem::{Node, NodeIndex, NodeType, Problem, ProductIndex, Quantity, TimeIndex};

/// Generates the initial orders for the given problem
pub fn initial_orders(problem: &Problem) -> Result<Vec<Order>> {
    let mut out = Vec::new();
    for p in 0..problem.products() {
        let sets = Sets::new(problem);
        let parameters = Parameters::new(problem, &sets);
        let res = TransportationSolver::solve(&sets, &parameters, p)?;
        let mut quantities: HashMap<NodeIndex, Vec<f64>> = problem
            .production_nodes()
            .iter()
            .map(|n| (n.index(), res.picked_up(n.index())))
            .collect();
        problem.consumption_nodes().iter().for_each(|n| {
            let k = n.index();
            let v = res.delivered(k);
            quantities.insert(k, v);
        });
        for node in problem.nodes() {
            let quants = quantities.get(&node.index()).unwrap();
            let windows = Quantities::time_windows(node, quants, p);
            for i in 0..quants.len() {
                let order = Order::new(node.index(), windows[i].0, windows[i].1, p, quants[i]);
                out.push(order);
            }
        }
    }

    Ok(out)
}

pub struct Quantities {}

impl Quantities {
    /// Calculates the quantities that are to be either delivered or picked up at the nodes
    pub fn quantities(problem: &Problem, product: ProductIndex) -> HashMap<NodeIndex, Quantity> {
        /*
        1. Assign quantites to production nodes such that all owerflow is handled and to consumption nodes such that all shortage is covered
        2. If the total demanded quanitity is larger than the quanitity pickup up at production nodes, increase the amount picked up at the production nodes
            by the quantity they have available but that is not already picked up.
        */

        // calculate how much of the given product type that must be (un)loaded at each node
        let consumption = |node: &Node| {
            node.inventory_change(0, problem.timesteps() - 1, product)
                .abs()
        };

        let total_consumption_demand: Quantity = problem
            .nodes()
            .iter()
            .map(|n| match n.r#type() {
                NodeType::Consumption => consumption(n),
                NodeType::Production => 0.0,
            })
            .sum();

        let mut quantities: HashMap<NodeIndex, Quantity> = problem
            .nodes()
            .iter()
            .map(|node| {
                (
                    node.index(),
                    match node.r#type() {
                        // consumption not covered by initial inventory
                        NodeType::Consumption => {
                            f64::max(consumption(node) - node.initial_inventory()[product], 0.0)
                        }
                        // production that there isn't room for
                        NodeType::Production => {
                            let production = consumption(node);
                            f64::max(
                                production
                                    - (node.capacity()[product]
                                        - node.initial_inventory()[product]),
                                0.0,
                            )
                        }
                    },
                )
            })
            .collect();

        let picked_up: Quantity = problem
            .nodes()
            .iter()
            .map(|n| match n.r#type() {
                NodeType::Consumption => 0.0,
                NodeType::Production => quantities[&n.index()],
            })
            .sum();

        if picked_up < total_consumption_demand {
            let mut not_covered = total_consumption_demand - picked_up;
            let num_prod_nodes = problem.production_nodes().len();

            // production nodes sorted on the capacity for the given product
            let mut prod_nodes = problem.production_nodes().clone();

            prod_nodes.sort_by(|a, b| {
                a.capacity()[product]
                    .partial_cmp(&b.capacity()[product])
                    .unwrap()
            });
            let mut count = 0;
            for prod_node in prod_nodes {
                // the remaining demand not covered equally distributed over the remaining production nodes
                let eq_share = not_covered / ((num_prod_nodes - count) as f64);
                count += 1;

                // current quantity being picked up at the production node
                let curr = quantities[&prod_node.index()];

                // excess product that hasn't been picked up
                let excess = if curr > 0.0 {
                    prod_node.capacity()[product]
                } else {
                    prod_node.initial_inventory()[product] + consumption(prod_node)
                };

                // extra product to be picked up at the production node to help satisfy the total consumption demand
                let added_pickup = f64::min(excess, eq_share);

                // update the demand not covered
                not_covered -= added_pickup;

                // update the demand picked up at the production node
                *quantities.get_mut(&prod_node.index()).unwrap() += added_pickup;
            }
        }
        quantities
    }

    /// Find appropriate time windoes for each delivery
    pub fn time_windows(
        node: &Node,
        deliveries: &Vec<Quantity>,
        product: ProductIndex,
    ) -> Vec<(TimeIndex, TimeIndex)> {
        match node.r#type() {
            NodeType::Consumption => Self::consumption_windows(node, deliveries, product),
            NodeType::Production => Self::production_windows(node, deliveries, product),
        }
    }

    /// Returns the time windows for the deliveries of the given product at the given consumption node.
    ///
    /// ## Note:
    /// Node should be a consumption node.
    fn consumption_windows(
        node: &Node,
        deliveries: &Vec<Quantity>,
        product: ProductIndex,
    ) -> Vec<(TimeIndex, TimeIndex)> {
        let mut windows = Vec::new();
        // accumulated deliveries
        let mut accumulated_deliveries = 0.0;
        for delivery in deliveries {
            accumulated_deliveries += delivery;

            // the window opens in time period 0 if there is capacity to receive all previous orders plus the current one right away
            // otherwise, it is set to open when the node has consumed enough to have capacity for the current delivery as well
            let open = if node.capacity()[product] - node.initial_inventory()[product]
                >= accumulated_deliveries
            {
                0
            } else {
                // the quantity that must have been consumed before there is room for the new delivery
                let excess = accumulated_deliveries + node.initial_inventory()[product]
                    - node.capacity()[product];
                node.inventory_change_at_least(product, excess)
            };

            // the window must close before the node breaches the lower limit (0)
            let close = {
                // The initial inventory plus the deliveries made so far
                let excess = node.initial_inventory()[product] + accumulated_deliveries - delivery;

                node.inventory_change_at_least(product, excess)
            };
            windows.push((open, close))
        }
        windows
    }

    /// Returns the time windows for the pickups of the given product at the given production production node.
    ///
    /// ### Note:
    /// Node should be a production node.
    fn production_windows(
        node: &Node,
        pickups: &Vec<Quantity>,
        product: ProductIndex,
    ) -> Vec<(TimeIndex, TimeIndex)> {
        let mut windows = Vec::new();
        // pickups so far
        let mut accumulated_pickups = 0.0;
        for pickup in pickups {
            accumulated_pickups += pickup;

            // if the pickups so far can be supported by the initial inventory, the window can be open from the first time period
            // otherwise, the windows opens when the node has produced enough to support all previous pickups and the current pickup
            let open = if accumulated_pickups <= node.initial_inventory()[product] {
                0
            } else {
                node.inventory_change_at_least(
                    product,
                    accumulated_pickups - node.initial_inventory()[product],
                )
            };

            // the window must close when production cause a capacity breach.
            //This is calculated based the initial inventory, the capacity and the pickups already performed
            let close = {
                let excess_capacity = node.capacity()[product] - node.initial_inventory()[product]
                    + accumulated_pickups
                    - pickup;
                node.inventory_change_at_least(product, excess_capacity)
            };
            windows.push((open, close));
        }
        windows
    }
}

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub struct Order {
    #[pyo3(get, set)]
    node: NodeIndex,
    #[pyo3(get, set)]
    open: TimeIndex,
    #[pyo3(get, set)]
    close: TimeIndex,
    #[pyo3(get, set)]
    product: ProductIndex,
    #[pyo3(get, set)]
    quantity: Quantity,
}

impl Order {
    pub fn new(
        node: NodeIndex,
        open: TimeIndex,
        close: TimeIndex,
        product: ProductIndex,
        quantity: Quantity,
    ) -> Order {
        Order {
            node,
            open,
            close,
            product,
            quantity,
        }
    }

    pub fn node(&self) -> NodeIndex {
        self.node
    }
    pub fn open(&self) -> TimeIndex {
        self.open
    }
    pub fn close(&self) -> TimeIndex {
        self.close
    }
    pub fn product(&self) -> ProductIndex {
        self.product
    }
    pub fn quantity(&self) -> Quantity {
        self.quantity
    }
}

pub struct SisrsOutput {
    routes: Vec<Vec<(Node, TimeIndex)>>,
    loads: Vec<Vec<Vec<Quantity>>>,
}

impl SisrsOutput {
    pub fn routes(&self) -> &Vec<Vec<(Node, TimeIndex)>> {
        &self.routes
    }

    pub fn loads(&self) -> &Vec<Vec<Vec<Quantity>>> {
        &self.loads
    }
}
