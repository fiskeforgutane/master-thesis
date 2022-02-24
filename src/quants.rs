use std::collections::HashMap;

use crate::problem::{Node, NodeIndex, NodeType, Problem, ProductIndex, Quantity, TimeIndex};

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

        let quantities: HashMap<NodeIndex, Quantity> = self
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

            for ((open, close), quantity) in windows.iter().zip(&deliveries[&node.index()]) {
                orders.push(Order::new(node.index(), *open, *close, product, *quantity));
            }
        }
        orders
    }

    /// Calculate the number of deliveries per node and the quantities
    fn find_deliveries(
        &self,
        quantities: HashMap<NodeIndex, Quantity>,
    ) -> HashMap<NodeIndex, Vec<Quantity>> {
        /*
        Find the vessel with the lowest capacity -> given that this vessel were to deliver full loads to the nodes,
        how many visits are needed for the nodes not to breach?
        */
        let min_capacity = self
            .problem
            .vessels()
            .iter()
            .map(|v| v.compartments().iter().map(|c| c.0).sum())
            .reduce(f64::min)
            .unwrap();

        self.problem
            .nodes()
            .iter()
            .map(|node| {
                (
                    node.index(),
                    vec![min_capacity; (quantities[&node.index()] / min_capacity).ceil() as usize],
                )
            })
            .collect()
    }

    /// Find appropriate time windoes for each delivery
    fn time_windows(
        &self,
        node: &Node,
        deliveries: &Vec<Quantity>,
        product: ProductIndex,
    ) -> Vec<(TimeIndex, TimeIndex)> {
        match node.r#type() {
            NodeType::Consumption => self.consumption_windows(node, deliveries, product),
            NodeType::Production => self.production_windows(node, deliveries, product),
        }
    }

    /// Returns the time windows for the deliveries of the given product at the given consumption node.
    ///
    /// ## Note:
    /// Node should be a consumption node.
    fn consumption_windows(
        &self,
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
        &self,
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
                node.inventory_change_at_least(product, accumulated_pickups)
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

    pub fn orders(&self, sisrs_out: SisrsOutput) {
        todo!()
    }
}

pub struct Order {
    node: NodeIndex,
    open: TimeIndex,
    close: TimeIndex,
    product: ProductIndex,
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
    pub routes: Vec<Vec<(Node, TimeIndex)>>,
}
