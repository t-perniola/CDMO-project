from z3 import BoolVal, Bool, Implies, And, Solver, sat, is_true
import time
import os
import multiprocessing as mp
from utils import *

def serialize_model(model, vars_list):
    serialized = {}
    for var in vars_list:
        eval_var = model.eval(var, model_completion=True)
        if str(eval_var.sort()) == "Bool":
            serialized[str(var)] = is_true(eval_var)
        else:
            serialized[str(var)] = eval_var.as_long()
    return serialized

def model(m, n, l, s, D, lb, ub, time_limit, init_flag, result_queue, search_method='binary', symm=False):
    start_time = time.time()
    solver = Solver()
    solver.set("random_seed", 42)

    depot = n  
    max_load = sum(s)
    s_max = max(s)
    l_max = max(l)
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

    s_b = [int_to_bool_list(s_value, number_of_bits(s_max)) for s_value in s]
    l_b = [int_to_bool_list(l_value, number_of_bits(l_max)) for l_value in l]
    D_b = [[int_to_bool_list(D[i][j], number_of_bits(ub)) for j in range(n+1)]
           for i in range(n+1)]
    
    for courier in range(m):
        for step in range(n + 2):
            solver.add(exactly_one([path[courier][package][step] for package in range(n + 1)]))
    
    for package in range(n):
        solver.add(exactly_one([path[courier][package][step] for courier in range(m)
                                   for step in range(n + 2)]))

    for courier in range(m):
        solver.add(path[courier][depot][0] == True)
        solver.add(path[courier][depot][n + 1] == True)
        solver.add(conditional_binary_sum(s_b, cw[courier], cl[courier], f"courier_load_{courier}"))
        solver.add(geq_bool_lists(l_b[courier], cl[courier]))
        
        for step in range(n + 2):
            for package in range(n):
                solver.add(Implies(path[courier][package][step], cw[courier][package]))
    
    if symm:
        for c1 in range(m):
            for c2 in range(m):
                if c1 < c2 and l[c1] == l[c2]:
                    solver.add(lexicographical_less(
                        [cw[c1][p] for p in range(n)],
                        [cw[c2][p] for p in range(n)]
                    ))

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
    
    solver.add(max_bool_variable(cd, max_d))
    solver.add(geq_bool_lists(max_d, int_to_bool_list(lb, number_of_bits(lb))))
    
    init_flag.value = True
    best_model = None
    
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
    
    if best_model is None:
        result_queue.put((time.time() - start_time, None, None, search_method))
        return
    
    obj_value = bool_list_to_int([best_model[b] for b in max_d])
    flat_vars = [var for courier in path for package in courier for var in package]
    serialized_solution = serialize_model(best_model, flat_vars)
    result_queue.put((time.time() - start_time, obj_value, serialized_solution, search_method))

def solve_problem(m, n, l, s, D, lb, ub, time_limit, search_method="binary", symm=False):
    result_queue = mp.Queue()
    init_flag = mp.Value('b', False)  
    process = mp.Process(target=model, args=(m, n, l, s, D, lb, ub, time_limit, init_flag, result_queue, search_method, symm))
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

if __name__ == "__main__":
    mp.set_start_method("spawn")
    instance_num = "01"
    time_limit = 300000  
    file_path = os.path.join('Instances', f'inst{instance_num}.dat')
    instance = read_data_file(file_path)
    m = instance["m"]
    n = instance["n"]
    l = instance["l"]
    s = instance["s"]
    D = instance["D"]
    lb = instance["lb"]
    ub = instance["ub"]

    time_taken, obj_value, solution, search_method = solve_problem(
        m, n, l, s, D, lb, ub, time_limit,
        search_method='branch_and_bound',
        symm=True
    )
    if solution:
        depot = n
        solution = refine_solution(time_taken, obj_value, solution, search_method, depot, time_limit)
        print('Solution:', solution)
    else:
        print(f'"{search_method}": {{\n"obj": null,\n"sol": null,\n"optimal": false,\n"time": {int(time_taken)}\n}}')
