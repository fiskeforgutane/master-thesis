use std::collections::HashMap;

use crate::problem::{NodeIndex, Quantity, Problem, Node};
use crate::destroy_and_repair::route_operators::{RemoveNode, InsertNode};

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

/// A Pool is the set of routes currently being optimized by the path flow MIP
struct RoutePool<'a>{
    routes: Vec<&'a mut Route>,
}

impl <'a> RoutePool<'a> {
    fn new(problem: &Problem) -> RoutePool{
        RoutePool {
            routes: Vec::new(),
        }
    }

    fn add_route(&mut self, route: &'a mut Route) {
        self.routes.push(route);
    }

    fn add_routes(&mut self, routes: Vec<&'a mut Route>) {
        for route in routes {
            todo!("Filter out routes already in pool");
            self.add_route(route);
        }
    }

    pub fn get_routes_mut(&mut self) -> &mut [&'a mut Route] {
        &mut self.routes
    }
}

pub struct UpdateRoutePool {
    neighbor_map: HashMap<NodeIndex, NodeIndex>,
    destroy_operators: Vec<Box<dyn RemoveNode>>,
    repair_operator: Box<dyn InsertNode>,
}

impl UpdateRoutePool {
    fn new(problem: &Problem, destroy_operators: Vec<Box<dyn RemoveNode>>, repair_operator: Box<dyn InsertNode>) -> Self{
        UpdateRoutePool {
            neighbor_map: UpdateRoutePool::generate_neighbor_map(problem),
            destroy_operators,
            repair_operator
        }
    }

    fn generate_neighbor_map(problem: &Problem) -> HashMap<NodeIndex, NodeIndex> {
        let mut neighbor_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();

        for node in problem.nodes() {
            let index = node.index();

            let nearest_neighbor: &Node = problem
                .nodes()
                .iter()
                .filter(|p| p.index() != index)
                .min_by(|a, b| {
                    problem
                        .distance(index, a.index())
                        .partial_cmp(&problem.distance(index, b.index()))
                        .unwrap()
                })
                .unwrap();
            
            neighbor_map.insert(index, nearest_neighbor.index());
        }

        neighbor_map
    }

    /// Updates the pool of routes used by the MIP by appending more routes
    pub fn update_pool(&mut self) {
        todo!()
    }
}