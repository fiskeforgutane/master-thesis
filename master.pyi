from typing import List

Quantity = float
Cost = float
Distance = float


class NodeType:
    pass


class Node:
    def __init__(self, name: str, kind: NodeType, index: int, port_capacity: List[int], min_unloading_amount: Quantity, max_loading_amount: Quantity, port_fee: Cost, capacity: List[float], inventory_changes: List[List[float]], revenue: Cost, initial_inventory: List[float]):
        pass

class Vessel:
    def __init__(
        compartments: List[Quantity],
        speed: float,
        travel_unit_cost: Cost,
        empty_travel_unit_cost: Cost,
        time_unit_cost: Cost,
        available_from: int,
        initial_inventory: List[float],
        origin: int,
        vclass: str,
        index: int,
    ):
        pass

class Problem:
    def __init__(self,        
        vessels: List[Vessel],
        nodes: List[Node],
        timesteps: int,
        products: int,
        distances: List[List[Distance]],
    ):
        pass

class Visit:
    node: int
    product: int
    time: int
    quantity: Quantity


class Solution:
    def __init__(self,
        routes: List[List[Visit]],
    ):
        pass