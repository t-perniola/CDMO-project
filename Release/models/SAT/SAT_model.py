import time
import os
from math import floor
import json
import multiprocessing as mp
from z3 import BoolVal, Bool, Implies, And, Solver, sat, is_true
from utils.utils import *

TIME_LIMIT = 300000  # 5 minutes

def json_fun(instance_number, obj, paths, time_taken, TIME_LIMIT, symm_break, search_strategy):
    file_path = f'res/SAT/{str(int(instance_number))}.json'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    json_dict = {
        "time": int(floor(time_taken)),
        "optimal": True if time_taken < TIME_LIMIT else False,
        "obj": obj,
        "sol": paths
    }

    if os.path.exists(file_path):
        with open(file_path, 'r') as infile:
            try:
                existing_data = json.load(infile)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    # Determine model type
    if symm_break:
        model_type = "SAT_SB"
    else:
        model_type = "SAT_noSB"

    if search_strategy == "binary":
        model_type += "_BS"
    else:
        model_type += "_BB"
    
    # Update the existing data
    existing_data[model_type] = json_dict

    # Write updated data to file
    with open(file_path, 'w') as outfile:
        json.dump(existing_data, outfile, indent=4)

# Serialize the model
def serialize_model(model, vars_list):
    serialized = {}
    for var in vars_list:
        eval_var = model.eval(var, model_completion=True)
        if str(eval_var.sort()) == "Bool":
            serialized[str(var)] = is_true(eval_var)
        else:
            serialized[str(var)] = eval_var.as_long()
    return serialized

# Model function
def model(m, n, l, s, D, lb, ub, time_limit, init_flag, result_queue, search_method='binary', symm_break=False):

    # Initialize the solver 
    start_time = time.time()
    solver = Solver()
    solver.set("random_seed", 42)

    # Define constants
    depot = n  
    max_load = sum(s)
    s_max = max(s)
    l_max = max(l)

    # Define variables
    path = [[[Bool(f'p_{courier}_{package}_{step}') for step in range(n + 2)]
             for package in range(n + 1)] for courier in range(m)]
    cw = [[Bool(f'w_{courier}_{package}') for package in range(n)]
                       for courier in range(m)]
    cl = [[Bool(f'cl_{courier}_{bit}') for bit in range(number_of_bits(max_load))]
                     for courier in range(m)]
    cd = [[Bool(f'cdt_{courier}_{bit}') for bit in range(number_of_bits(ub))]
                  for courier in range(m)]
    max_d = [Bool(f'max_d_{bit}') for bit in range(number_of_bits(ub))]
    c_step_d = [[[Bool(f"cpt_{courier}_{step}_{bit}") for bit in range(number_of_bits(ub))]
                   for step in range(n+1)] for courier in range(m)]

    # Convert integer values to binary lists
    s_b = [int_to_bool_list(s_value, number_of_bits(s_max)) for s_value in s]
    l_b = [int_to_bool_list(l_value, number_of_bits(l_max)) for l_value in l]
    D_b = [[int_to_bool_list(D[i][j], number_of_bits(ub)) for j in range(n+1)]
           for i in range(n+1)]
    
    # Exactly one package is delivered at each step
    for courier in range(m):
        for step in range(n + 2):
            solver.add(exactly_one([path[courier][package][step] for package in range(n + 1)]))
    
    # Each package is delivered exactly once
    for package in range(n):
        solver.add(exactly_one([path[courier][package][step] for courier in range(m)
                                   for step in range(n + 2)]))

    # Each courier starts and ends at the depot
    for courier in range(m):
        solver.add(path[courier][depot][0] == True)
        solver.add(path[courier][depot][n + 1] == True)
        solver.add(conditional_binary_sum(s_b, cw[courier], cl[courier], f"courier_load_{courier}"))
        solver.add(geq_bool_lists(l_b[courier], cl[courier]))
        
        for step in range(n + 2):
            for package in range(n):
                solver.add(Implies(path[courier][package][step], cw[courier][package]))
    
    # Symmetry breaking constraints
    if symm_break:
        for c1 in range(m):
            for c2 in range(c1+1,m):
                if l[c1] == l[c2]:
                    solver.add(lexicographical_less(
                        [cw[c1][p] for p in range(n)],
                        [cw[c2][p] for p in range(n)]
                    ))
                elif l[c1] > l[c2]:
                    solver.add(lexicographical_less(cl[c2], cl[c1]))   

    # Distance computation
    for courier in range(m):
        for step in range(n + 1):
            for package_start in range(n + 1):
                for package_end in range(n + 1):
                    solver.add(Implies(
                        And(path[courier][package_start][step],
                            path[courier][package_end][step+1]),
                        equal_bool_lists(D_b[package_start][package_end],
                                         c_step_d[courier][step])
                    ))
        solver.add(conditional_binary_sum(c_step_d[courier],
                                [BoolVal(True) for _ in range(n+1)],
                                cd[courier],
                                f"def_courier_dist_{courier}"))
    
    # Objective function
    solver.add(max_bool_variable(cd, max_d))
    solver.add(geq_bool_lists(max_d, int_to_bool_list(lb, number_of_bits(lb))))
    
    init_flag.value = True
    best_model = None
    
    # Search methods
    if search_method == 'branch_and_bound':
        solver.push()
        while True:
            remaining_time = max(1, (time_limit - int((time.time() - start_time) * 1000)))
            solver.set("timeout", remaining_time)
            solver.push()
            solver.add(geq_bool_lists(int_to_bool_list(ub, number_of_bits(ub)), max_d))
            if solver.check() == sat:
                best_model = solver.model()
                ub = bool_list_to_int([best_model[b] for b in max_d]) - 1  
            else:
                solver.pop()
                break  
            solver.pop()
        solver.pop()

    elif search_method == 'binary':
        while ub - lb > 1:
            remaining_time = max(1, (time_limit - int((time.time() - start_time) * 1000)))
            solver.set("timeout", remaining_time)
            mid = (ub + lb) // 2
            solver.push()
            solver.add(geq_bool_lists(int_to_bool_list(mid, number_of_bits(ub)), max_d))
            if solver.check() == sat:
                best_model = solver.model()
                ub = bool_list_to_int([best_model[b] for b in max_d])
            else:
                lb = mid  
            solver.pop()
    
    # Output the results
    if best_model is None:
        result_queue.put((time.time() - start_time, None, None, search_method))
        return
    
    # Serialize the model
    obj_value = bool_list_to_int([best_model[b] for b in max_d])
    flat_vars = [var for courier in path for package in courier for var in package]
    serialized_solution = serialize_model(best_model, flat_vars)
    result_queue.put((time.time() - start_time, obj_value, serialized_solution, search_method))

# Refine the solution
def solve_problem(m, n, l, s, D, lb, ub, time_limit, search_method="binary", symm_break=False):
    result_queue = mp.Queue()
    init_flag = mp.Value('b', False)  
    process = mp.Process(target=model, args=(m, n, l, s, D, lb, ub, time_limit, init_flag, result_queue, search_method, symm_break))
    process.start()
    
    start_time = time.time()
    while time.time() - start_time < time_limit / 1000:
        time.sleep(1)
        if init_flag.value:
            break
    else:
        process.terminate()
        process.join()
        return time.time() - start_time, None, None, search_method
    
    process.join() 
    
    try:
        result = result_queue.get_nowait()
    except Exception as e:
        print(f"Error: {e}")
        return time.time() - start_time, None, None, search_method
    
    return result

# Main function
def SAT(instance_num, sb_bool, search_method="branch_and_bound"):
    mp.set_start_method("spawn")

    # IMPORTING INSTANCE
    try:
        file_path = os.path.join('instances','dat_instances', f'inst{instance_num}.dat')
        instance = read_dat_file(file_path)
    except Exception as e:
        print(f"Error reading the instance file: {e}")
        return None
    
    m = instance["m"]
    n = instance["n"]
    l = instance["l"]
    s = instance["s"]
    D = instance["D"]
    lb = instance["lb"]
    ub = instance["ub"]

    time_taken, obj_value, solution, search_method = solve_problem(
        m, n, l, s, D, lb, ub, TIME_LIMIT,
        search_method = search_method,
        symm_break = sb_bool
    )

    print("\nRun summary:")
    print(f"- Approach: SAT")
    print(f"- Instance: {instance_num}")
    print(f"- Solver: {'Binary Search' if search_method == 'binary' else 'Branch and Bound'}")
    print(f"- Symmetry breaking: {'Yes' if sb_bool else 'No'}")

    if solution:
        depot = n
        solution = refine_solution(time_taken, obj_value, solution, search_method, depot, TIME_LIMIT)
        print(f"- Objective value (max dist): {solution[search_method]['obj']}\n")

        # Output the results to a JSON file
        json_fun(instance_num, solution[search_method]['obj'], solution[search_method]['sol'], time_taken, TIME_LIMIT, sb_bool, search_method)
    else:
        print("- Objective value (max dist): No feasible solution found (UNSAT).\n")
        json_fun(instance_num, None, None, time_taken, TIME_LIMIT, sb_bool, search_method)
