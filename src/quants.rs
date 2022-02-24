use std::collections::HashMap;

use crate::problem::{InventoryType, Node, NodeIndex, NodeType, Problem, ProductIndex, TimeIndex};

pub struct Quantities {
    pub problem: Problem,
}

impl Quantities {
    /// Returns the initial orders only considering the problem
    pub fn initial_orders(&self) -> Vec<Order> {
        return (0..self.problem.products())
            .flat_map(|p| self.initial_orders_per(p))
            .collect();
    }

    /// Returns the initial orders for the given product and only considering the problem
    fn initial_orders_per(&self, product: ProductIndex) -> Vec<Order> {
        /*
         - Calculate necessary deliveries to each consumption node for the entire planning period
         - Split the total delivery into deliveries with an origin and a destination
         - Assign a time window for the delivery, ensuring that no capacity breach occurs
         - return orders
        */
        // {k.key(): sum(k.consumption) for k in problem.nodes()}

        // calculate how much of the given product type that must be (un)loaded at each node
        let t = self.problem.timesteps();
        let consumption = |node: &Node| {
            (0..t)
                .map(|time| node.inventory_changes()[time][product])
                .sum::<f64>()
        };

        let quantities: HashMap<NodeIndex, InventoryType> = self
            .problem
            .nodes()
            .iter()
            .map(|node| {
                (
                    node.index(),
                    consumption(node) - node.initial_inventory()[product],
                )
            })
            .collect();

        let deliveries = self.find_deliveries(quantities);

        let mut orders = Vec::new();
        for node in self.problem.nodes() {
            let windows = self.time_windows(node, &deliveries[&node.index()], product);

            for ((open, close), quantity) in windows.iter().zip(deliveries[&node.index()]) {
                let order = Order::new(node.index(), *open, *close, product, quantity);

                orders.push(Order::new(node.index(), *open, *close, product, quantity));
            }
        }
        orders
    }

    /// Calculate the number of deliveries per node and the quantities
    fn find_deliveries(
        &self,
        quantities: HashMap<NodeIndex, InventoryType>,
    ) -> HashMap<NodeIndex, Vec<InventoryType>> {
        /*
        Find the vessel with the lowest capacity -> given that this vessel were to deliver full loads to the nodes,
        how many visits are needed for the nodes not to breach?
        */
        let min_capacity = self
            .problem
            .vessels()
            .iter()
            .map(|v| v.compartments().iter().map(|c| c.0).sum())
            .reduce(f64::min);

        self.problem
            .nodes()
            .iter()
            .map(|node| {
                (
                    node.index(),
                    vec![min_capacity; 1 + quantities[node.index()] / min_capacity as usize],
                )
            })
            .collect();
    }

    /// Find appropriate time windoes for each delivery
    fn time_windows(
        &self,
        node: &Node,
        deliveries: &Vec<InventoryType>,
    ) -> Vec<(TimeIndex, TimeIndex)> {
        match node.r#type {
            NodeType::Consumption => self.consumption_windows(node, deliveries),
            NodeType::Production => self.production_windows(node, deliveries),
        }
    }

    fn consumption_windows(
        &self,
        node: &Node,
        deliveries: &Vec<InventoryType>,
    ) -> Vec<TimeIndex, TimeIndex> {
    }

    fn production_windows(
        &self,
        node: &Node,
        pickups: &Vec<InventoryType>,
        product: ProductIndex,
    ) -> Vec<TimeIndex, TimeIndex> {
        let mut accumulated_pickups = 0;
        for pickup in pickups {
            let open;
            let close;
            let open = if accumulated_pickups == 0 {
                0
            } else {
                node.inventory_change_at_least(accumulated_pickups)
            };
        }

        todo!()
    }

    pub fn orders(&self, sisrs_out: SisrsOutput) {
        todo!()
    }
}

pub struct Order {
    node: NodeIndex,
    open: TimeIndex,
    close: TimeIndex,
    product: ProductIndex,
    quantity: InventoryType,
}

impl Order {
    pub fn new(
        node: NodeIndex,
        open: TimeIndex,
        close: TimeIndex,
        product: ProductIndex,
        quantity: InventoryType,
    ) -> Order {
        Order {
            node,
            open,
            close,
            product,
            quantity,
        }
    }
}

pub struct SisrsOutput {
    pub routes: Vec<Vec<(Node, TimeIndex)>>,
}
