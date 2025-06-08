import numpy as np
from datetime import datetime
#from utils.utils import read_dat_file, clarke_wright_seed, ortools_seed
import utils.utils as utils
import json
import os
import copy
from pyscipopt import Model, quicksum
from pyscipopt import scip
import itertools
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def extract_path(adj_matrix: list[list[int]], depot_idx:int) -> list[int]:
    """
    Extracts the path followed by the courier from an adjacency matrix. 
    The depot n is not included.

    Args:
        adj_matrix (list[list[int]]): The adjacency matrix encoding the courier path.
        depot_idx (int): The depot index.
    
    Returns:
        list[int]: A list containing the indices of the nodes visited by the courier.
    """
    res = []
    source = depot_idx
    
    # Empty path
    if sum(adj_matrix[source]) == 0:
        return res
    
    while True:
        for dest in range(depot_idx+1):
            if adj_matrix[source][dest] == 1:
                if dest == depot_idx:
                    return res
                res.append(dest+1)
                source = dest
                break

# Solution checker purposes only
def order_solution(sol:list[list[int]], m:int, permuted_loads:list[int], original_loads:list[int]):
    """
    Orders the solution in-place according to the initial loads order of the couriers provided in the instance file.
    Since the courier id doesn't matter from a mathematical perspective, this method doesn't care about exact ids.

    Args:
        sol (list[list[int]]): A list containing the courier's paths expressed as a list of node indices.
        m (int): The number of couriers in the instance.
        permuted_loads (list[int]): The loads order of the couriers after some permutation.
        original_loads (list[int]): The original loads order of the couriers.
    """
    c = 0
    unordered_idxs = []
    # The courier identity doesn't matter as long as they have the same load 
    for c in range(m):
        if permuted_loads[c] != original_loads[c]:
            unordered_idxs.append(c)
    
    while len(unordered_idxs) > 1:
        idx1 = unordered_idxs[-1]
        
        for idx in unordered_idxs:
            if permuted_loads[idx1] == original_loads[idx]:
                park = copy.deepcopy(sol[idx1])
                sol[idx1] = copy.deepcopy(sol[idx])
                sol[idx] = park

                park1 = permuted_loads[idx1]
                permuted_loads[idx1] = permuted_loads[idx]
                permuted_loads[idx] = park1
                
                unordered_idxs = []
                for c in range(m):
                    if permuted_loads[c] != original_loads[c]:
                        unordered_idxs.append(c)
                break

def solve_MIP(parsed_data:tuple[int, int, list[int], list[int], list[list[int]]], time_limit:int, seed:list[list[int]]) -> tuple[int, str, list[list[list[int]]]]:
    """
    Builds and solve the MIP problem.

    Args:
        parsed_data (tuple[int, int, list[int], list[int], list[list[int]]]): The data contained in the instance following that order.
        time_limit (int): The total time allocated to the model to find a solution.
        seed (list[list[int]]): The initial solution found by the presolving algorithm (if any), containing list of node indices.

    Returns:
        tuple[int, str, list[list[list[int]]]]: The tuple containing the objective value, the status, and the adjacency matrices found by the model.
    """

    m, n, s, l, D = parsed_data
    D = np.array(D)
    depot_idx = n

    model = Model("MCP")
    
    # Finds the maximum distance between any arc in the seed solution
    ARCLIM = np.inf
    if seed:
        UB_seed = max(
            sum(D[i_prev][i_curr] for i_prev, i_curr in zip([n] + tour, tour + [n]))
            for tour in seed
        )
        ARCLIM = UB_seed

    # Removes the self-loops and the arcs exceeding the upper bound (if any)
    edges = [
            (i, j) for i in range(n + 1) for j in range(n + 1)
            if i != j and (i == n or j == n or D[i][j] < ARCLIM)
        ]
    
    # Build the edge lists ONCE
    IN  = {j: [] for j in range(n+1)}
    OUT = {j: [] for j in range(n+1)}
    for (i, j) in edges:
        OUT[i].append(j)
        IN[j].append(i)

    ############################################################################################
    ######################################## Variables #########################################
    ############################################################################################
    
    # Decision Variables
    x = {} 
    for c in range(m):
        for (i,j) in edges:
            x[c, i, j] = model.addVar(vtype='B', name=f"x_{c}_{i}_{j}")

    # Flow Variables
    f = {}                                                            
    for c in range(m):
        for i, j in edges:
            f[c, i, j] = model.addVar(lb=0, ub=l[c], name=f"f_{c}_{i}_{j}")

    # Courier Activity Variables
    y = [model.addVar(vtype='B', name=f"use_{c}") for c in range(m)]

    # Intermediate Variables
    load = [model.addVar(lb=0, name=f"load_{c}") for c in range(m)]
    dist = [model.addVar(lb=0, name=f"dist_{c}") for c in range(m)]

    # Objective Variable
    max_dist = model.addVar(lb=0, name="max_dist")

    # Objective: minimize maximum distance
    model.setObjective(max_dist, sense="minimize")

    ############################################################################################
    ############################### Static Capacity Subset Cuts ################################
    ############################################################################################
    max_cap = max(l)
    V = list(range(n))   # just the customer nodes (0..n-1)
    for r in (2,3):
        for S in itertools.combinations(V, r):
            if sum(s[i] for i in S) <= max_cap:
                continue

            cross_arcs = [
                (c, i, j)
                for c in range(m)
                for i in S
                for j in range(n + 1)
                if j not in S and (i, j) in x          # (i,j) survived filtering
            ]

            if not cross_arcs:
                continue

            lhs = quicksum(x[idx] for idx in cross_arcs)
            model.addCons(lhs >= 1)
        
    ############################################################################################
    ####################################### Constraints ########################################
    ############################################################################################

    # Each customer is visited exactly once
    for j in range(n):
        model.addCons(quicksum(x[c, i, j] for c in range(m) for i in IN[j]) == 1)

    # The flow passing thoughout a node cannot exceed the courier load capacity
    for c in range(m):
        for (i,j) in edges:
            model.addCons(f[c,i,j] <= l[c]*x[c,i,j])
        
        # Flow balance
        for j in range(n):
            model.addCons(
                quicksum(f[c,i,j] for i in IN[j]) -
                quicksum(f[c,j,k] for k in OUT[j])
                == s[j] * quicksum(x[c,i,j] for i in IN[j])
            )

            # Flow conservation
            model.addCons(quicksum(x[c, h, j] for h in IN[j]) ==
                          quicksum(x[c, j, h] for h in OUT[j]))

        # Outgoing flow from depot equals courier load
        model.addCons(quicksum(f[c, n, j] for j in OUT[n]) == load[c])

        # No load on the return arcs to the depot
        for i in IN[n]:
            model.addCons(f[c, i, n] == 0)

        # Leave depot at most once, but only if y_c == 1
        model.addCons(
            quicksum(x[c, n, j] for j in range(n)) == y[c]
        )
        # Come back at most once, but only if y_c == 1
        model.addCons(
            quicksum(x[c, i, n] for i in range(n)) == y[c]
        )

        # Load capacity constraint enabled only if the courier is used
        model.addCons(load[c] <= l[c] * y[c])
        
        # Load constraints
        model.addCons(load[c] == quicksum(s[j] * quicksum(x[c, i, j] for i in IN[j]) for j in range(n)))
        model.addCons(load[c] <= l[c])

        # Distance constraints
        model.addCons(dist[c] == quicksum(D[i][j] * x[c, i, j] for i in range(n+1) for j in OUT[i]))
        model.addCons(dist[c] <= max_dist)

    # Symmetry breaking: Load ordering
    for c in range(m - 1):
        model.addCons(load[c] >= load[c + 1])

    # Trivial lower bound
    lb_radial = max(D[depot_idx][j] + D[j][depot_idx] for j in range(n))                   
    model.addCons(max_dist >= lb_radial)

    ############################################################################################
    ######################################## Warm Start ########################################
    ############################################################################################
    if seed:
        init_sol = model.createSol()
        dist_c_values = []

        for c, tour_nodes in enumerate(seed):
            # Unused courier
            if not tour_nodes:          
                model.setSolVal(init_sol, y[c], 0)
                continue
            
            model.setSolVal(init_sol, y[c], 1)
            
            # Indexing purposes
            tour_nodes = [t - 1 for t in seed[c]]
            prev = n
            dist_c = 0

            remain = load_c = sum(s[node] for node in tour_nodes)

            # Flow on arc depot -> first node
            first = tour_nodes[0]
            model.setSolVal(init_sol, f[c, n, first], remain)

            for node in tour_nodes:
                # Activating the node in the solution
                model.setSolVal(init_sol, x[c, prev, node], 1)

                # We just set the depot -> first node flow
                if prev != n:
                    model.setSolVal(init_sol, f[c, prev, node], remain)
                remain -= s[node]

                dist_c += D[prev][node]
                prev = node

            dist_c += D[prev][n]
            dist_c_values.append(dist_c)

            model.setSolVal(init_sol, x[c, prev, n], 1)
            model.setSolVal(init_sol, load[c], load_c)  
            model.setSolVal(init_sol, dist[c], dist_c)

            # Explicitly set unused arcs to zero
            for (i,j) in edges:
                if (i,j) not in zip([n] + tour_nodes, tour_nodes + [n]):
                    model.setSolVal(init_sol, x[c, i, j], 0)

        max_d = max(dist_c_values)

        # Check whether the heuristic UB is valid, assuming that the LB is always valid
        if max_d >= lb_radial:
            model.setObjlimit(max_d)
            for c in range(m):
                model.addCons(dist[c] <= max_d)
                model.addCons(dist[c] <= max_d * y[c])    

        model.setSolVal(init_sol, max_dist, float(max_d))
        # Keep the warm start solution as incumbent
        model.addSol(init_sol, free=False) 

    ############################################################################################
    ##################################### Solver Settings ######################################
    ############################################################################################
    model.setParam("limits/time", time_limit)
    model.setParam('presolving/maxrounds', 20)
    model.setParam('heuristics/undercover/freq', 10)
    model.setParam("parallel/maxnthreads", 8)

    # Getting a tigher dual bound
    model.setParam("separating/maxcutsroot", 1000)
    model.setParam("separating/maxcuts", 200)	

    # Getting a better primal bound
    model.setParam("heuristics/localbranching/freq", 1)	
    model.setParam("heuristics/rins/freq", 5)

    model.setParam("display/verblevel", 0)

    model.optimize()

    paths = {(c, i, j): int(model.getVal(x[c, i, j])) for (c, i, j) in x}
    return (model.getObjVal(), model.getStatus(), paths)

def MIP(instance_number:str):
    """
    Solves the MIP instance and saves the result in a json file named {instance_number}.json in the folder res/MIP.

    Args:
        instance_number (int): The index of the instance to solve, named inst{instance_number}.dat
    """
    file_path = os.path.join('instances', "dat_instances", str(f'inst{instance_number}.dat'))
    parsed_data = utils.read_dat_file(file_path)

    m = parsed_data['m']
    n = parsed_data['n']
    s = parsed_data['s']
    l = sorted(parsed_data['l'], reverse=True)
    D = parsed_data['D']

    starting_time = datetime.now()
    time_limit = 5*60
    safe_bound = 10
    
    try:
        seed = utils.ortools_seed(n, m, s, l, D)
    except RuntimeError:
        # Always guarantees to find a seed
        seed = utils.clarke_wright_seed(m, n, s, l, D)
    
    presolving_time = datetime.now() - starting_time

    solution = solve_MIP(parsed_data=(m, n, s, l, D), 
                         time_limit=time_limit - presolving_time.seconds - safe_bound, 
                         seed=seed)
    
    end_time = datetime.now() - starting_time
    status = "undefined"
    sol = []

    if solution:
        objective, status, paths = solution

        # Robustness towards numerical errors
        if str(objective).split('.')[-1] != '0':
            status = "undefined"
        
        if status == "optimal" or status == "timelimit":
            for c in range(m):
                matrix_values = [[0 for _ in range(n+1)] for _ in range(n+1)]
                for i in range(n+1):
                    for j in range(n+1):
                        if i != j:
                            matrix_values[i][j] = int(paths[c,i,j])
                sol.append(extract_path(matrix_values,n))
            order_solution(sol,m,l,parsed_data['l'])

    json_dict = {}
    json_dict['MIP'] = {}
    json_dict['MIP']['time'] = end_time.seconds if (end_time.seconds+safe_bound<time_limit) else time_limit
    json_dict['MIP']['optimal'] = True if status == 'optimal' else False
    json_dict['MIP']['obj'] = int(objective) if (status == "optimal" or status == "timelimit") else None
    json_dict['MIP']['sol'] = sol

    with open(f'res/MIP/{str(int(instance_number))}.json', 'w') as outfile:
        json.dump(json_dict, outfile)

if __name__ == "__main__":
    MIP('09')