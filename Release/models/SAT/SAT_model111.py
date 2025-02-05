import time
import os
from math import floor
import json
import multiprocessing as mp
from z3 import BoolVal, Bool, Implies, And, Solver, sat, is_true
from utils.utils import *

TIME_LIMIT = 21000  # 21 secondi in millisecondi

def json_fun(instance_number, obj, paths, time_taken, TIME_LIMIT, symm_break, search_strategy):
    """
    Salva i risultati in un file JSON. Se il file esiste già, aggiorna il modello relativo.
    """
    file_path = f'res/SAT/{int(instance_number)}.json'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    json_dict = {
        "time": int(floor(time_taken)),
        "optimal": time_taken < TIME_LIMIT,
        "obj": obj,
        "sol": paths
    }
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as infile:
                existing_data = json.load(infile)
        except json.JSONDecodeError:
            existing_data = {}
    else:
        existing_data = {}

    model_type = f"SAT_{'SB' if symm_break else 'noSB'}_{search_strategy.upper()}"
    existing_data[model_type] = json_dict

    with open(file_path, 'w') as outfile:
        json.dump(existing_data, outfile, indent=4)


def serialize_model(model, vars_list):
    """
    Serializza il modello Z3 in un dizionario leggibile.
    """
    return {
        str(var): is_true(model.eval(var, model_completion=True)) if str(var.sort()) == "Bool" 
        else model.eval(var).as_long()
        for var in vars_list
    }


def search_branch_and_bound(solver, start_time, time_limit, max_d, ub):
    """
    Esegue la ricerca branch-and-bound aggiornando l'upper bound.
    Restituisce il miglior modello trovato (se presente) e l'upper bound aggiornato.
    """
    best_model = None
    while True:
        remaining_time = max(1, time_limit - int((time.time() - start_time) * 1000))
        solver.set("timeout", remaining_time)
        solver.push()
        # Aggiunge la condizione che il valore corrente di max_d sia almeno uguale all'upper bound corrente.
        solver.add(geq_bool_lists(int_to_bool_list(ub, number_of_bits(ub)), max_d))
        if solver.check() == sat:
            best_model = solver.model()
            ub = bool_list_to_int([best_model[b] for b in max_d]) - 1
        else:
            solver.pop()
            break
        solver.pop()
    return best_model, ub


def search_binary(solver, start_time, time_limit, max_d, lb, ub):
    """
    Esegue la ricerca binaria per ottimizzare l'obiettivo.
    Restituisce il miglior modello trovato (se presente) e gli estremi aggiornati.
    """
    best_model = None
    while ub - lb > 1:
        remaining_time = max(1, time_limit - int((time.time() - start_time) * 1000))
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
    return best_model, lb, ub


def model(m, n, l, s, D, lb, ub, time_limit, init_flag, result_queue, search_method='binary', symm_break=False):
    start_time = time.time()
    solver = Solver()
    solver.set("random_seed", 42)

    # Costanti e pre-calcoli
    depot = n
    max_load = sum(s)
    s_max = max(s)
    l_max = max(l)
    nbits_ub = number_of_bits(ub)

    # Definizione delle variabili
    path = [[[Bool(f'p_{c}_{p}_{step}') for step in range(n + 2)]
             for p in range(n + 1)]
            for c in range(m)]
    cw = [[Bool(f'w_{c}_{p}') for p in range(n)]
          for c in range(m)]
    cl = [[Bool(f'cl_{c}_{bit}') for bit in range(number_of_bits(max_load))]
          for c in range(m)]
    cd = [[Bool(f'cdt_{c}_{bit}') for bit in range(nbits_ub)]
          for c in range(m)]
    max_d = [Bool(f'max_d_{bit}') for bit in range(nbits_ub)]
    c_step_d = [[[Bool(f"cpt_{c}_{step}_{bit}") for bit in range(nbits_ub)]
                  for step in range(n+1)]
                 for c in range(m)]
    
    # Conversione dei parametri in rappresentazioni booleane
    s_b = [int_to_bool_list(s_val, number_of_bits(s_max)) for s_val in s]
    l_b = [int_to_bool_list(l_val, number_of_bits(l_max)) for l_val in l]
    D_b = [[int_to_bool_list(D[i][j], nbits_ub) for j in range(n+1)]
           for i in range(n+1)]
    
    # Vincolo: esattamente un pacco per ogni step per ogni courier
    for c in range(m):
        for step in range(n + 2):
            solver.add(exactly_one([path[c][p][step] for p in range(n + 1)]))
    
    # Vincolo: ogni pacco viene consegnato esattamente una volta
    for p in range(n):
        solver.add(exactly_one([path[c][p][step] for c in range(m) for step in range(n + 2)]))
    
    # Vincoli relativi ai courier
    for c in range(m):
        solver.add(path[c][depot][0])
        solver.add(path[c][depot][n + 1])
        solver.add(conditional_binary_sum(s_b, cw[c], cl[c], f"courier_load_{c}"))
        solver.add(geq_bool_lists(l_b[c], cl[c]))
        for step in range(n + 2):
            for p in range(n):
                solver.add(Implies(path[c][p][step], cw[c][p]))
    
    # Vincoli di symmetry breaking
    if symm_break:
        for c1 in range(m):
            for c2 in range(c1 + 1, m):
                if l[c1] == l[c2]:
                    solver.add(lexicographical_less([cw[c1][p] for p in range(n)],
                                                    [cw[c2][p] for p in range(n)]))
                elif l[c1] > l[c2]:
                    solver.add(lexicographical_less(cl[c2], cl[c1]))
    
    # Calcolo delle distanze per ogni step e courier
    for c in range(m):
        for step in range(n + 1):
            for p_start in range(n + 1):
                for p_end in range(n + 1):
                    solver.add(Implies(
                        And(path[c][p_start][step], path[c][p_end][step + 1]),
                        equal_bool_lists(D_b[p_start][p_end], c_step_d[c][step])
                    ))
        # Somma condizionale delle distanze per ogni courier
        solver.add(conditional_binary_sum(c_step_d[c],
                                          [BoolVal(True)] * (n + 1),
                                          cd[c],
                                          f"def_courier_dist_{c}"))
    
    # Definizione dell'obiettivo: massimizzare la distanza
    solver.add(max_bool_variable(cd, max_d))
    solver.add(geq_bool_lists(max_d, int_to_bool_list(lb, nbits_ub)))

    # Comunica che il solver è stato inizializzato
    init_flag.value = True
    best_model = None

    # Strategia di ricerca
    if search_method == 'branch_and_bound':
        best_model, ub = search_branch_and_bound(solver, start_time, time_limit, max_d, ub)
    elif search_method == 'binary':
        best_model, lb, ub = search_binary(solver, start_time, time_limit, max_d, lb, ub)

    if best_model is None:
        result_queue.put((time.time() - start_time, None, None, search_method))
        return

    obj_value = bool_list_to_int([best_model[b] for b in max_d])
    # Estrazione e serializzazione delle variabili 'path'
    flat_vars = [var for courier in path for package in courier for var in package]
    serialized_solution = serialize_model(best_model, flat_vars)
    result_queue.put((time.time() - start_time, obj_value, serialized_solution, search_method))


def solve_problem(m, n, l, s, D, lb, ub, time_limit, search_method="binary", symm_break=False):
    """
    Avvia il processo che esegue il modello Z3 e gestisce il timeout.
    """
    result_queue = mp.Queue()
    init_flag = mp.Value('b', False)
    
    start_time = time.time()  # Memorizzo il tempo di inizio
    process = mp.Process(
        target=model, 
        args=(m, n, l, s, D, lb, ub, time_limit, init_flag, result_queue, search_method, symm_break)
    )
    process.start()

    # Attende l'inizializzazione del solver (con timeout)
    process.join(timeout=time_limit / 1000)
    if not init_flag.value:
        process.terminate()
        process.join()
        return time.time() - start_time, None, None, search_method

    process.join()
    try:
        return result_queue.get_nowait()
    except Exception as e:
        print(f"Error retrieving result: {e}")
        return time.time() - start_time, None, None, search_method


def SAT(instance_num, sb_bool, search_method="branch_and_bound"):
    """
    Funzione principale per risolvere l'istanza con il modello SAT.
    """
    mp.set_start_method("spawn", force=True)
    try:
        file_path = os.path.join('instances', 'dat_instances', f'inst{instance_num}.dat')
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

    time_taken, obj_value, solution, used_search_method = solve_problem(
        m, n, l, s, D, lb, ub, TIME_LIMIT,
        search_method=search_method,
        symm_break=sb_bool
    )

    print("\nRun summary:")
    print(f"- Approach: SAT")
    print(f"- Instance: {instance_num}")
    print(f"- Solver: {'Binary Search' if used_search_method == 'binary' else 'Branch and Bound'}")
    print(f"- Symmetry breaking: {'Yes' if sb_bool else 'No'}")

    if solution:
        depot = n
        solution = refine_solution(time_taken, obj_value, solution, used_search_method, depot, TIME_LIMIT)
        print(f"- Objective value (max dist): {solution[used_search_method]['obj']}\n")
        json_fun(instance_num, solution[used_search_method]['obj'], solution[used_search_method]['sol'], time_taken, TIME_LIMIT, sb_bool, used_search_method)
    else:
        print("- Objective value (max dist): No feasible solution found (UNSAT).\n")
        json_fun(instance_num, None, None, time_taken, TIME_LIMIT, sb_bool, used_search_method)
