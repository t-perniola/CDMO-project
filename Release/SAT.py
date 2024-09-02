import time
from z3 import *
import numpy as np
import json
import os
#from itertools import combinations
from encodings_utils import *

# Modelling MCP problem
def mcp(instance, timeout=3000000):
    m = instance["m"]  # Number of couriers
    n = instance["n"]  # Number of items
    l = instance["l"]  # Weights each courier can carry
    s = instance["s"]  # Items' sizes
    max_times = instance["max_times"]  # Time horizon: a courier can carry at most max_times items..
    # .. and since at each timestep a courier can pick exactly one item, max_times refers to the max time steps possible

    solver = Solver()
    solver.set("timeout", timeout)
    solver.set("random_seed", 42)

    # VARIABLES
    # - Represent that each courier c picks the item i at the time step t (add 1 for depot)
    pickup_schedule = [[[Bool(f"x_{c}_{i}_{t}") for t in range(max_times + 1)] for i in range(n + 1)] for c in range(m)]
    
    # CONSTRAINTS
    # 1) Each courier c can carry at most l[c] kg
    for c in range(m):
        load_list = []
        for i in range(n):
            for t in range(1, max_times):
                for _ in range(s[i]):
                    load_list.append(pickup_schedule[c][i][t])
        solver.add(at_most_k_seq(load_list, l[c], f"courier_{c}_load"))
        
    # 2) Each courier c starts and ends at position i = n
    for c in range(m):
        solver.add(pickup_schedule[c][n][0] == True)  # Start at position n at time 0
        solver.add(pickup_schedule[c][n][max_times] == True)  # End at position n at max_times
    
    # 3) Each courier must deliver at least one item
    for c in range(m):
        solver.add(at_least_one_bw([pickup_schedule[c][i][t] for t in range(1, max_times) for i in range(n)]))
    
    # 4) Each courier cannot pick the same item more than once
    for c in range(m):
        for i in range(n):
            solver.add(at_most_one_bw([pickup_schedule[c][i][t] for t in range(1, max_times)], f"exactly_once_courier{c}"))

    # 5) Each item is taken exactly once among all couriers and across all timesteps
    for i in range(n):
        solver.add(exactly_one_bw([pickup_schedule[c][i][t] for t in range(1, max_times) for c in range(m)], f"exactly_once_{i}"))
    
    # 6) All items should be picked up
    for i in range(n):
        solver.add(at_least_one_bw([pickup_schedule[c][i][t] for t in range(1, max_times) for c in range(m)]))

    '''# 7) Symmetry breaking: two couriers have the same load size
    for c1, c2 in combinations(range(m), 2):
        if l[c1] == l[c2]:
            for t in range(1, max_times):
                for i in range(n):
                    solver.add(Or(Not(v[c1][i][t]), v[c2][i][t]))'''
    
    # return model and the updated 3D variable
    return solver, pickup_schedule

# Calculate distances relatively to the paths retrieved as solution
def compute_distances(solution, instance):
    distances = instance["D"]
    n = instance["n"]
    distances_dict = {}

    for i, courier_solution in enumerate(solution):
        if not courier_solution:
            continue
        total_distance = 0
        depot = n  # Starting node
        for node in courier_solution:
            if node != depot:
                total_distance += distances[depot][node-1]
                depot = node-1
        total_distance += distances[depot][n]  # Return to starting node
        distances_dict[i] = total_distance

    return distances_dict

# Objective function
def compute_max_dist(distances_dict):
    return max(distances_dict.values())

# Retrieve the solution
def extract_solution(model, instance, v):
    m = instance["m"]  # Number of couriers
    n = instance["n"]  # Number of packages
    max_times = instance["max_times"]
    solution = []
    for c in range(m):
        courier_path = []
        for t in range(max_times):
            for i in range(n):
                if model[v[c][i][t]]:
                    courier_path.append(i+1)
        solution.append(courier_path)
    return solution

# Additional info for the instance
def compute_additional_info(instance):
    n = instance["n"]
    m = instance["m"]

    # Number of max_times steps
    max_times = (n // m) + 3
    additional_info = {"max_times": max_times}
    return additional_info

def write_results_to_json(instance_number, solution, max_dist, time_taken):
    TIME_LIMIT = 300
    opt_values = {"01": 14, "02": 226, "03": 12, "04": 220, "05": 206, "06": 322, "07": 167,
                  "08": 186, "09": 436, "10": 244}

    # Prepare the paths list based on the solution
    paths = []
    for courier_path in solution:
        path = [item for item in courier_path]
        paths.append(path)

    # Prepare JSON dictionary
    json_dict = {}
    json_dict['SAT'] = {}
    json_dict['SAT']['time'] = int(time_taken)
    json_dict['SAT']['optimal'] = True if (time_taken < TIME_LIMIT and opt_values[instance_number] == max_dist) else False
    json_dict['SAT']['obj'] = max_dist if max_dist is not None else None
    json_dict['SAT']['sol'] = paths

    # Write JSON to file
    with open(f'res/SAT/{str(int(instance_number))}.json', 'w') as outfile:
        json.dump(json_dict, outfile)

def SAT(instance_number):
    start_time = time.time()
    
    # Importing specified instance
    try:
        file_path = os.path.join('Instances', f'inst{instance_number}.dat')
        instance = read_dat_file(file_path)
    except Exception as e:
        print(f"Error reading the instance file: {e}")
        return None
    
    # Add useful info to instance
    instance.update(compute_additional_info(instance))

    # Running the model
    solver, v = mcp(instance=instance)
    
    if solver.check() == sat:
        model = solver.model()
        solution = extract_solution(model, instance, v)
        distances_dict = compute_distances(solution, instance)
        max_dist = compute_max_dist(distances_dict)
        time_taken = time.time() - start_time
        
        print(f"\nObj. function value, max dist: {max_dist}")

        write_results_to_json(instance_number, solution, max_dist, time_taken)
    else:
        print("unsat")

