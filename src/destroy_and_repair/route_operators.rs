use crate::problem::{NodeIndex, Problem, Quantity};
use rand::prelude::*;

#[derive(Clone, Debug)]
/// A Route is defined as a path for a vehicle starting and ending in a production nodex
pub struct Route {
    nodes: Vec<NodeIndex>,
    quantities: Vec<Vec<Quantity>>,
}

impl Route {
    pub fn get_nodes(&self) -> &Vec<NodeIndex> {
        &self.nodes
    }

    pub fn get_nodes_mut(&mut self) -> &mut Vec<NodeIndex> {
        &mut self.nodes
    }

    pub fn add_node(&mut self, node: NodeIndex) {
        self.nodes.push(node);
    }

    pub fn get_quantities(&self) -> &Vec<Vec<Quantity>> {
        &self.quantities
    }
}

pub trait RemoveNode {
    /// Performs some operation on a route in order to create a new route
    /// without changing the route given as input
    fn apply(&self, route: &Route, problem: &Problem) -> (Route, NodeIndex);

    /// Calculates the decreased distance by removing a node at a particular position
    fn decreased_distance(&self, position: usize, route: &Route, problem: &Problem) -> f64 {
        let r = route.get_nodes();

        problem.distance(r[position - 1], r[position])
            + problem.distance(r[position], r[position + 1])
            - problem.distance(r[position - 1], r[position + 1])
    }
}

pub struct RemoveHighestCost {}

impl RemoveNode for RemoveHighestCost {
    /// Removes the node associated with the highest extra distance and return it
    fn apply(&self, route: &Route, problem: &Problem) -> (Route, NodeIndex) {
        let mut route_copy = route.clone();

        let index = route_copy
            .get_nodes()
            .iter()
            .enumerate()
            .max_by(|a, b| {
                self.decreased_distance(a.0, &route_copy, problem)
                    .partial_cmp(&self.decreased_distance(b.0, &route_copy, problem))
                    .unwrap()
            })
            .unwrap()
            .0;

        let removed_node = route_copy.get_nodes_mut().remove(index);

        (route_copy, removed_node)
    }
}
pub struct RemoveLargestDelivery {}

impl RemoveNode for RemoveLargestDelivery {
    /// Removes the node associated with the highest delivery quanitity in the route
    fn apply(&self, route: &Route, _: &Problem) -> (Route, NodeIndex) {
        let mut route_copy = route.clone();

        let node_index = route_copy
            .get_quantities()
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.iter()
                    .sum::<f64>()
                    .partial_cmp(&b.1.iter().sum())
                    .unwrap()
            })
            .unwrap()
            .0;

        let removed_node = route_copy.get_nodes_mut().remove(node_index);

        (route_copy, removed_node)
    }
}
pub struct RemoveSmallestDelivery {}

impl RemoveNode for RemoveSmallestDelivery {
    /// Removes the node associated with the lowest delivery quantity in the route
    fn apply(&self, route: &Route, _: &Problem) -> (Route, NodeIndex) {
        let mut route_copy = route.clone();

        let node_index = route_copy
            .get_quantities()
            .iter()
            .enumerate()
            .min_by(|a, b| {
                a.1.iter()
                    .sum::<f64>()
                    .partial_cmp(&b.1.iter().sum())
                    .unwrap()
            })
            .unwrap()
            .0;

        let removed_node = route_copy.get_nodes_mut().remove(node_index);

        (route_copy, removed_node)
    }
}
pub struct RemoveRandom {}

impl RemoveNode for RemoveRandom {
    fn apply(&self, route: &Route, _: &Problem) -> (Route, NodeIndex) {
        let mut route_copy = route.clone();

        let node_index = (0..route_copy.get_nodes().len())
            .choose(&mut thread_rng())
            .unwrap();

        let removed_node = route_copy.get_nodes_mut().remove(node_index);

        (route_copy, removed_node)
    }
}

pub trait InsertNode {
    fn apply(&self, node: NodeIndex, routes: Vec<&mut Route>, problem: &Problem);

    /// Calculates the increased distance by inserting a node at a particular position
    fn increased_distance(
        &self,
        node_index: NodeIndex,
        position: usize,
        problem: &Problem,
        route: &Route,
    ) -> f64 {
        let r: &Vec<NodeIndex> = route.get_nodes();

        problem.distance(r[position - 1], node_index) + problem.distance(node_index, r[position])
            - problem.distance(r[position - 1], r[position])
    }
}

pub struct InsertAtLowestCost {}

impl InsertAtLowestCost {
    fn find_insertion_point(
        &self,
        node: NodeIndex,
        route: &Route,
        problem: &Problem,
    ) -> (usize, f64) {
        let position = route.get_nodes()[1..route.get_nodes().len()]
            .iter()
            .enumerate()
            .min_by(|a, b| {
                self.increased_distance(node, a.0, problem, route)
                    .partial_cmp(&self.increased_distance(node, b.0, problem, route))
                    .unwrap()
            })
            .unwrap()
            .0;

        let cost = self.increased_distance(node, position, problem, route);

        (position, cost)
    }
}

impl InsertNode for InsertAtLowestCost {
    fn apply(&self, node: NodeIndex, routes: Vec<&mut Route>, problem: &Problem) {
        let best_route = routes
            .into_iter()
            .min_by(|a, b| {
                self.find_insertion_point(node, a, problem)
                    .1
                    .partial_cmp(&self.find_insertion_point(node, b, problem).1)
                    .unwrap()
            })
            .unwrap();

        let insertion_point = self.find_insertion_point(node, best_route, problem).0;

        best_route.get_nodes_mut().insert(insertion_point, node);
    }
}
