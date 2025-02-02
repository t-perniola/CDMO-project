from z3 import BoolVal, Bool, Implies, And, Solver, sat, is_true
import time
import os
import multiprocessing as mp
from utils2 import *  


def serialize_model(model, vars_list):
    serialized = {}
    for var in vars_list:
        eval_var = model.eval(var, model_completion=True)
        if str(eval_var.sort()) == "Bool":
            serialized[str(var)] = is_true(eval_var)
        else:
            serialized[str(var)] = eval_var.as_long()
    return serialized

def mcp_sat(m, n, l, s, D, lb, ub, time_limit, init, result_queue, search='binary', symm_constr=False):
    start_time = time.time()
    solver = Solver()
    solver.set("random_seed", 42)

    deposit = n  
    max_load = sum(s)
    s_max = max(s)
    l_max = max(l)

    # Variabili
    path = [[[Bool(f'p_{courier}_{package}_{step}') for step in range(n + 2)]
             for package in range(n + 1)] for courier in range(m)]
    courier_weights = [[Bool(f'w_{courier}_{package}') for package in range(n)]
                       for courier in range(m)]
    courier_loads = [[Bool(f'cl_{courier}_{bit}') for bit in range(num_bits(max_load))]
                     for courier in range(m)]
    c_dist_tot = [[Bool(f'cdt_{courier}_{bit}') for bit in range(num_bits(ub))]
                  for courier in range(m)]
    max_dist_b = [Bool(f'max_d_{bit}') for bit in range(num_bits(ub))]
    c_dist_par = [[[Bool(f"cpt_{courier}_{step}_{bit}") for bit in range(num_bits(ub))]
                   for step in range(n+1)] for courier in range(m)]

    s_b = [int_to_bin(s_value, num_bits(s_max)) for s_value in s]
    l_b = [int_to_bin(l_value, num_bits(l_max)) for l_value in l]
    D_b = [[int_to_bin(D[i][j], num_bits(ub)) for j in range(n+1)]
           for i in range(n+1)]
    
    # Vincoli
    for courier in range(m):
        for step in range(n + 2):
            solver.add(exactly_one_np([path[courier][package][step] for package in range(n + 1)]))
    
    for package in range(n):
        solver.add(exactly_one_np([path[courier][package][step] for courier in range(m)
                                   for step in range(n + 2)]))
    
    for courier in range(m):
        solver.add(path[courier][deposit][0] == True)
        solver.add(path[courier][deposit][n + 1] == True)
        solver.add(cond_sum_bin(s_b, courier_weights[courier], courier_loads[courier], f"courier_load_{courier}"))
        solver.add(geq(l_b[courier], courier_loads[courier]))
        
        for step in range(n + 2):
            for package in range(n):
                solver.add(Implies(path[courier][package][step], courier_weights[courier][package]))
    
    if symm_constr:
        for c1 in range(m):
            for c2 in range(m):
                if c1 < c2 and l[c1] == l[c2]:
                    solver.add(lex_less([courier_weights[c1][p1] for p1 in range(n)],
                                         [courier_weights[c2][p2] for p2 in range(n)]))
    
    for courier in range(m):
        for step in range(n + 1):
            for package_start in range(n + 1):
                for package_end in range(n + 1):
                    solver.add(Implies(And(path[courier][package_start][step],
                                           path[courier][package_end][step+1]),
                                       eq_bin(D_b[package_start][package_end],
                                              c_dist_par[courier][step])))
        solver.add(cond_sum_bin(c_dist_par[courier],
                                [BoolVal(True) for _ in range(n+1)],
                                c_dist_tot[courier],
                                f"def_courier_dist_{courier}"))
    solver.add(max_var(c_dist_tot, max_dist_b))
    solver.add(geq(max_dist_b, int_to_bin(lb, num_bits(lb))))
    
    init.value = True
    best_model = None
    
    if search == 'branch_and_bound':
        solver.push()
        while True:
            remaining_time = max(1, (time_limit - int((time.time() - start_time) * 1000)))
            solver.set("timeout", remaining_time)
            solver.push()
            solver.add(geq(int_to_bin(ub, num_bits(ub)), max_dist_b))
            if solver.check() == sat:
                best_model = solver.model()
                ub = bin_to_int([best_model[b] for b in max_dist_b]) - 1  
            else:
                solver.pop()
                break  
            solver.pop()
        solver.pop()
    elif search == 'binary':
        while ub - lb > 1:
            remaining_time = max(1, (time_limit - int((time.time() - start_time) * 1000)))
            solver.set("timeout", remaining_time)
            mid = (ub + lb) // 2
            solver.push()
            solver.add(geq(int_to_bin(mid, num_bits(ub)), max_dist_b))
            if solver.check() == sat:
                best_model = solver.model()
                ub = bin_to_int([best_model[b] for b in max_dist_b])
            else:
                lb = mid  
            solver.pop()
    
    if best_model is None:
        result_queue.put((time.time() - start_time, None, None, search))
        return
    
    obj_value = bin_to_int([best_model[b] for b in max_dist_b])
    flat_vars = [var for courier in path for package in courier for var in package]
    serialized_solution = serialize_model(best_model, flat_vars)
    result_queue.put((time.time() - start_time, obj_value, serialized_solution, search))


def solve_problem(m, n, l, s, D, lb, ub, time_limit, search="binary", symm_constr=False):
    result_queue = mp.Queue()
    init_flag = mp.Value('b', False)  
    process = mp.Process(target=mcp_sat, args=(m, n, l, s, D, lb, ub, time_limit, init_flag, result_queue, search, symm_constr))
    process.start()
    
    start_time = time.time()
    while time.time() - start_time < time_limit / 1000:
        time.sleep(1)
        if init_flag.value:
            break
    else:
        print("Timeout reached without initialization. Terminating process.")
        process.terminate()
        process.join()
        return None, None, None, search
    
    process.join() 
    
    try:
        result = result_queue.get_nowait()
    except Exception as e:
        print(f"Errore nel recupero del risultato dalla coda: {e}")
        return None, None, None, search
    
    return result


if __name__ == "__main__":
    mp.set_start_method("spawn")
    instance_num = "09"
    time_limit = 21000  
    file_path = os.path.join('Instances', f'inst{instance_num}.dat')
    instance = read_dat_file(file_path)
    m = instance["m"]
    n = instance["n"]
    l = instance["l"]
    s = instance["s"]
    D = instance["D"]
    lb = instance["lb"]
    ub = instance["ub"]

    time_taken, obj_value, solution, search = solve_problem(m, n, l, s, D, lb, ub, time_limit,
                                                             search='branch_and_bound',
                                                             symm_constr=True)
    if solution:
        depot = n
        solution = refineSol(time_taken, obj_value, solution, search, depot, time_limit)
        print('Solution:', solution)
    else:
        print(f'"{search}": {{\n"obj": null,\n"sol": null,\n"optimal": false,\n"time": 300\n}}')