from itertools import combinations
from z3 import Bool, BoolVal, Xor, Or, And, Not
import math
import re
from collections import defaultdict
from typing import List, Tuple
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

### READ DAT INSTANCE and COMPUTE HEURISTIC BOUNDS
def read_dat_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Read m and n
    m = int(lines[0].strip())
    n = int(lines[1].strip())

    # Read l array
    l = list(map(int, lines[2].strip().split()))

    # Read s array
    s = list(map(int, lines[3].strip().split()))

    # Read D matrix
    D = []
    for i in range(4, len(lines)):
        D.append(list(map(int, lines[i].strip().split())))
    
    maxD = float('-inf')
    minD = float('inf')

    for i in range(len(D)):
        for j in range(len(D[i])):
            if i != j:  # Exclude diagonal elements
                if D[i][j] > maxD:
                    maxD = D[i][j]
                if D[i][j] < minD:
                    minD = D[i][j]
    
    # Heuristic number of nodes per courier
    h_num_nodes_per_courier = n//m +3

    # Compute naive bounds
    lb = h_num_nodes_per_courier * minD
    ub = h_num_nodes_per_courier * maxD

    # Safety assignment
    lb = 1 if lb == 0 else lb

    return {
        'm': m,
        'n': n,
        'l': l,
        's': s,
        'D': D,
        'lb': lb,
        'ub': ub
    }

### FIND MORE ENGINEERED LOWER AND UPPER BOUNDS ###
def compute_bounds(m, n, l, s, D, lb_old, ub_old):
    # Define new lower-bound (LB)
    lb_radial = max(D[n][i] + D[i][n] for i in range(n))

    # Define new upper-bound (UB)
    try:
        # Clarke-Wright UB
        cw_routes = clarke_wright_seed(m, n, s, l, D)
        cw_dists = compute_path_dist(cw_routes, n, D)
        ub_cw = sum(cw_dists)

        # Ortools UB
        ot_routes = ortools_seed(n, m, s, l, D)
        ot_dists = compute_path_dist(ot_routes, n, D)
        ub_ot = sum(ot_dists)

    # Safety assignment: if Clarke-Wright doesn't find any route
    except ValueError:
        ub_cw = ub_old
        ub_ot = ub_old
    
    # Pick the highest lb and the lowest ub
    lb = max(lb_old, lb_radial)
    ub = min(ub_old, ub_cw, ub_ot)
    return lb, ub

def clarke_wright_seed(m: int, n: int, s: List[int], l: List[int], D: List[List[int]]) -> Tuple[List[List[int]], int]:
    """
    Finds an heuristic starting route configuration. It always find a solution.

    Args:
        m (int): Number of couriers.
        n (int): Number of nodes.
        s (list[int]): Item weights in correspondence of nodes.
        l (list[int]): Maximum load a courier is able to carry.
        D (list[list[int]]): Distance matrix.

    Return:
        list[list[int]]: A list containing one route per courier, where the route is represented by a list of node indices.
    """

    depot = n  # D is (n+1)x(n+1), depot is at index n
    max_cap = max(l)

    # Start with one route per customer
    routes = [[i] for i in range(n)]
    route_load = [s[i] for i in range(n)]
    route_of = list(range(n))

    # Compute Clarke-Wright savings
    savings = [(D[i][depot] + D[depot][j] - D[i][j], i, j)
               for i in range(n) for j in range(i+1, n)]
    savings.sort(reverse=True)

    for _, i, j in savings:
        ri, rj = route_of[i], route_of[j]
        if ri == rj:
            continue
        if route_load[ri] + route_load[rj] > max_cap:
            continue
        if (routes[ri][0] in (i, j) or routes[ri][-1] in (i, j)) and \
           (routes[rj][0] in (i, j) or routes[rj][-1] in (i, j)):
            if routes[ri][-1] != i:
                routes[ri].reverse()
            if routes[rj][0] != j:
                routes[rj].reverse()
            routes[ri].extend(routes[rj])
            route_load[ri] += route_load[rj]
            for cust in routes[rj]:
                route_of[cust] = ri
            routes[rj] = []

    routes = [r for r in routes if r]

    # Add routes to reach m (splitting longest if needed)
    while len(routes) < m:
        idx = max(range(len(routes)), key=lambda r: sum(s[c] for c in routes[r]))
        cust = routes[idx].pop()
        routes.append([cust])

    # Merge routes if too many
    while len(routes) > m:
        routes.sort(key=lambda r: sum(s[c] for c in r))
        a = routes.pop(0)
        b = routes.pop(0)
        if sum(s[c] for c in a + b) <= max_cap:
            routes.append(a + b)
        else:
            routes.extend([a, b])
            break

    # Balance capacity
    changed = True
    while changed:
        changed = False
        for idx, r in enumerate(routes[:m]):
            over = sum(s[i] for i in r) - l[idx]
            if over <= 0:
                continue
            spare = [l[j] - sum(s[i] for i in routes[j]) for j in range(m)]
            best_tgt = max((j for j in range(m) if spare[j] >= 1),
                           key=lambda j: spare[j], default=None)
            if best_tgt is None:
                raise ValueError("total capacity < total demand ‒ infeasible instance")
            cand = max((c for c in r if s[c] <= spare[best_tgt]),
                       key=lambda c: s[c], default=None)
            if cand is None:
                raise ValueError("cannot rebalance routes within capacities")
            r.remove(cand)
            routes[best_tgt].append(cand)
            changed = True

    # Sort routes
    def route_key(route):
        load = sum(s[i] for i in route)
        dist = sum(D[a][b] for a, b in zip([depot] + route, route + [depot]))
        first = route[0]
        return (-load, -dist, first)

    routes.sort(key=route_key)

    # Final seed (1-based customer IDs)
    seed = [[cust + 1 for cust in r] for r in routes[:m]]

    # Sanity checks
    assert len(seed) == m
    assert sorted(c for tour in seed for c in tour) == list(range(1, n + 1))

    return seed

def ortools_seed(n:int, m:int, demand:list[int], capacity:list[int], dist:list[list[int]], depot:int=None, seconds:int=8) -> list[list[int]]:
    """
    Finds an heuristic starting route configuration.

    Args:
        n (int): Number of nodes.
        m (int): Number of couriers.
        demand (list[int]): Item weights in correspondence of nodes.
        capacity (list[int]): Maximum load a courier is able to carry.
        dist (list[list[int]]): Distance matrix.
        depot (int): Depot index.
        seconds (int): Time allocated to seed searching.

    Return:
        list[list[int]]: A list containing one route per courier, where the route is represented by a list of node indices.
    """

    depot   = depot if depot is not None else n
    manager = pywrapcp.RoutingIndexManager(n + 1, m, depot)
    routing = pywrapcp.RoutingModel(manager)

    # Distance matrix
    def dcb(from_idx, to_idx):
        i, j = manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)
        return int(dist[i][j])
    transit = routing.RegisterTransitCallback(dcb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    # Capacity
    def qcb(idx):
        node = manager.IndexToNode(idx)
        return 0 if node == depot else demand[node]
    qidx = routing.RegisterUnaryTransitCallback(qcb)
    routing.AddDimensionWithVehicleCapacity(qidx, 0, capacity, True, "Load")

    # Distance dimension, so we can penalise the longest route
    routing.AddDimension(transit, 0, sum(map(max, dist)), True, "RouteLen")
    route_len = routing.GetDimensionOrDie("RouteLen")
    
    # Minimise the maximum “RouteLen” over all vehicles
    route_len.SetGlobalSpanCostCoefficient(1000)

    # Allow vehicles to stay idle 
    # (drop Visited >= 1 and fixed costs, helping min-max objectives)
    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search.time_limit.FromSeconds(seconds)

    sol = routing.SolveWithParameters(search)
    if sol is None:
        raise RuntimeError("OR-Tools could not find a seed.")

    # Extract routes
    tours = []
    for v in range(m):
        idx, tour = routing.Start(v), []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != depot:
                tour.append(node + 1)
            idx = sol.Value(routing.NextVar(idx))
        if tour:                               
            tours.append(tour)

    # Respect the symmetry-breaking order
    tours.sort(key=lambda r: -sum(demand[i-1] for i in r))
    return tours

def compute_path_dist(routes, n, D):
    distances = []
    depot = n
    for route in routes:
        dist = 0
        if route:
            dist += D[depot][route[0]-1]  # depot to first customer
            for i in range(len(route) - 1):
                dist += D[route[i]-1][route[i+1]-1]
            dist += D[route[-1]-1][n]  # last customer back to depot
        distances.append(dist)
    return distances

####### SAT UTILIITES #######

def number_of_bits(x):
    return math.floor(math.log2(x)) + 1

def int_to_bool_list(x, digits):
    bits = [(x % (2 ** (i + 1)) // (2 ** i)) == 1 for i in range(digits - 1, -1, -1)]
    return [BoolVal(bit) for bit in bits]

def bool_list_to_int(bool_list):
    value = 0
    for i, b in enumerate(bool_list):
        if b:
            value += 2 ** (len(bool_list) - 1 - i)
    return value

def pad_bool_list(bool_list, target_length):
    return [BoolVal(False)] * (target_length - len(bool_list)) + bool_list

def at_least_one(vars_list):
    return Or(vars_list)

def at_most_one(vars_list, name=""):
    return And([Not(And(pair[0], pair[1])) for pair in combinations(vars_list, 2)])

def exactly_one(vars_list, name=""):
    return And(at_least_one(vars_list), at_most_one(vars_list, name))

def full_adder_bit(x, y, carry_in, result_bit, carry_out):
    eq_res = result_bit == Xor(Xor(x, y), carry_in)
    eq_carry = carry_out == Or(And(Xor(x, y), carry_in), And(x, y))
    return And(eq_res, eq_carry)

def binary_adder(x_bits, y_bits, result_bits, name=""):
    max_len = max(len(x_bits), len(y_bits))
    x_bits = pad_bool_list(x_bits, max_len)
    y_bits = pad_bool_list(y_bits, max_len)
    carries = [Bool(f"carry_{name}_{i}") for i in range(max_len)] + [BoolVal(False)]
    constraints = []
    for i in range(max_len):
        constraints.append(
            full_adder_bit(
                x=x_bits[max_len - i - 1],
                y=y_bits[max_len - i - 1],
                carry_in=carries[max_len - i],
                result_bit=result_bits[max_len - i - 1],
                carry_out=carries[max_len - i - 1]
            )
        )
    constraints.append(Not(carries[0]))
    return And(constraints)

def mask_bool_list(bool_list, mask_value):
    return [And(bit, mask_value) for bit in bool_list]

def conditional_binary_sum(num_list, mask, result_bits, name=""):
    constraints = []
    temp_results = [[BoolVal(False) for _ in range(len(result_bits))]] + [
        [Bool(f"{name}_temp_{i}_{j}") for j in range(len(result_bits))]
        for i in range(len(num_list))
    ]
    for i in range(len(num_list)):
        constraints.append(
            binary_adder(
                x_bits=temp_results[i],
                y_bits=mask_bool_list(num_list[i], mask[i]),
                result_bits=temp_results[i + 1],
                name=f"{name}_{i}"
            )
        )
    constraints.append(equal_bool_lists(temp_results[-1], result_bits))
    return And(constraints)

def geq_bool_lists(x_bits, y_bits):
    if len(x_bits) != len(y_bits):
        max_len = max(len(x_bits), len(y_bits))
        x_bits = pad_bool_list(x_bits, max_len)
        y_bits = pad_bool_list(y_bits, max_len)
    if len(x_bits) == 1:
        return Or(x_bits[0] == y_bits[0], And(Not(y_bits[0]), x_bits[0]))
    else:
        return Or(And(Not(y_bits[0]), x_bits[0]), And(x_bits[0] == y_bits[0], geq_bool_lists(x_bits[1:], y_bits[1:])))

def equal_bool_lists(x_bits, y_bits):
    max_len = max(len(x_bits), len(y_bits))
    x_bits = pad_bool_list(x_bits, max_len)
    y_bits = pad_bool_list(y_bits, max_len)
    return And([x_bits[i] == y_bits[i] for i in range(max_len)])

def max_bool_variable(bool_vars_list, max_var_bits):
    equality_constraints = Or([equal_bool_lists(var_bits, max_var_bits) for var_bits in bool_vars_list])
    geq_constraints = And([geq_bool_lists(max_var_bits, var_bits) for var_bits in bool_vars_list])
    return And(equality_constraints, geq_constraints)

def lexicographical_less(v1, v2):
    if not v1:
        return BoolVal(True)
    if not v2:
        return BoolVal(False)
    return Or(And(Not(v1[0]), v2[0]), And(v1[0] == v2[0], lexicographical_less(v1[1:], v2[1:])))

def refine_solution(time_taken, obj_value, solution, search_method, depot, time_limit):
    time_taken = int(time_taken)
    time_limit_sec = time_limit / 1000  
    is_optimal = False
    courier_routes = defaultdict(list)
    package_pattern = re.compile(r'p_(\d+)_(\d+)_(\d+)')
    solution_dict = {str(var): solution[var] for var in solution}
    
    if time_taken != time_limit_sec:
        is_optimal = True

    for entry, value in solution_dict.items():
        match = package_pattern.match(entry)
        if match and value:
            courier, package, step = map(int, match.groups())
            if package != depot:
                package = package + 1  
                courier_routes[courier].append((step, package))
    
    refined_routes = [
        [pkg for _, pkg in sorted(routes, key=lambda x: x[0])]
        for _, routes in sorted(courier_routes.items())
    ]

    return {search_method: {
        "time": time_taken,
        "optimal": is_optimal,
        "obj": obj_value,
        "sol": refined_routes
    }}
