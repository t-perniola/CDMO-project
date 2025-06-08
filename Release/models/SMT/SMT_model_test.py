from z3 import *
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.utils import read_dat_file, compute_bounds
import time

# Time limit CONSTANT
TIME_LIMIT = 300

# Initialize the solver
def init_solver(instance_number, sb_bool):
    # Start the count
    start_time = time.time()

    # IMPORTING INSTANCE
    try:
        os.chdir(os.path.join(os.getcwd(), "Release"))
        print(os.getcwd())
        file_path = os.path.join('instances', 'dat_instances', f'inst{instance_number}.dat')
        instance = read_dat_file(file_path)
    except Exception as e:
        print(f"Error reading the instance file: {e}")
        return None

    # DECLARING CONSTANTS
    m = instance['m']
    n = instance['n']
    l = instance['l']
    s = instance['s']
    D = instance['D']
    lb = instance['lb']
    ub = instance['ub']

    MAX_ITEMS = (n // m) + 3
    Couriers = range(1, m+1)
    Items = range(1, n+1)

    # DECLARING VARIABLES USING SMT SORTS
    Z = IntSort()
    B = BoolSort()
    b_path = Array("b_path", Z, ArraySort(Z, B))
    load = Array('load', Z, Z)
    size = Array('size', Z, Z)
    path = Array('path', Z, ArraySort(Z, Z))
    path_length = Array('path', Z, Z)
    total_distance = Array('total_distance', Z, Z)
    D_func = Function('D_func', Z, Z, Z)

    # Compute bounds
    lb, ub = compute_bounds(m, n, l, s, D, lb, ub)

    # INITIALIZE the OPTIMIZER
    optimizer = Solver()
    optimizer.set("random_seed", 42)

    # HELPER FUNCTIONS
    def distinct_except(values, forbidden_values):
        non_forbidden_values = [v for v in values if v not in forbidden_values]
        return Distinct(non_forbidden_values)

    def lexleq(a1, a2):
        n = len(a1)
        assert n == len(a2)
        if n == 0:
            return BoolVal(True)
        clauses = []
        for i in range(n):
            equal_prev = And([a1[j] == a2[j] for j in range(i)])
            less = And(equal_prev, Or(Not(a1[i]), a2[i]))
            clauses.append(less)
        return Or(clauses)

    # CONSTRAINTS
    for i in range(n+1):
        for j in range(n+1):
            optimizer.add(D_func(i, j) == D[i][j])
    for i in range(n):
        optimizer.add(size[i+1] == s[i])
    for c in range(m):
        optimizer.add(load[c+1] == l[c])
    for c in Couriers:
        for j in range(1, MAX_ITEMS+1):
            optimizer.add(And(path[c][j] >= IntVal(0), path[c][j] <= IntVal(n+1)))
    for c in Couriers:
        optimizer.add(And(path_length[c] >= IntVal(3), path_length[c] <= IntVal(MAX_ITEMS)))
    for c in Couriers:
        optimizer.add(path[c][1] == IntVal(n + 1))
        optimizer.add(path[c][path_length[c]] == IntVal(n + 1))
    for c in Couriers:
        for i in range(1, MAX_ITEMS + 1):
            optimizer.add(Implies(i > path_length[c], path[c][i] == IntVal(0)))
    for j in Items:
        optimizer.add(Sum([If(b_path[c][j], 1, 0) for c in Couriers]) == 1)
    for c in Couriers:
        optimizer.add(Sum([If(b_path[c][j], 1, 0) for j in Items]) <= MAX_ITEMS)
    for c in Couriers:
        optimizer.add(distinct_except([path[c][j] for j in range(1, MAX_ITEMS)], [0]))
    for c in Couriers:    
        load_expr = Sum([If(b_path[c][j], size[j], 0) for j in Items])
        optimizer.add(load_expr <= load[c])
    for c in Couriers:
        for i in Items:
            optimizer.add(Implies(b_path[c][i], Or([path[c][j] == i for j in range(1, MAX_ITEMS+1)])))
            optimizer.add(Implies(Not(b_path[c][i]), And([path[c][j] != i for j in range(1, MAX_ITEMS+1)])))
    for c in Couriers:
        dist_expr = Sum([If(And(path[c][j] != 0, path[c][j+1] != 0), D_func(path[c][j]-IntVal(1), path[c][j+1]-IntVal(1)), 0)
                        for j in range(1, MAX_ITEMS)])
        dist_expr += Sum([If(And(path[c][j] == 0, path[c][j-1] != 0, path[c][j+1] != 0), D_func(path[c][j-1]-IntVal(1), path[c][j+1]-IntVal(1)), 0)
                        for j in range(1, MAX_ITEMS)])
        optimizer.add(total_distance[c] == dist_expr)

    if sb_bool:
        for c1 in Couriers:
            for c2 in Couriers:
                if c1 < c2:
                    sym_break_constraint = If(load[c1] == load[c2], lexleq([b_path[c1][j] for j in Items], [b_path[c2][j] for j in Items]), True)
                    optimizer.add(sym_break_constraint)

    max_dist = Int('max_dist')
    optimizer.add([max_dist >= total_distance[c] for c in Couriers])

    return optimizer, {
        "m": m, "n": n, "MAX_ITEMS": MAX_ITEMS, "Couriers": Couriers, "Items": Items, "path": path, 
        "path_length": path_length, "total_distance": total_distance, "b_path": b_path, "size": size, "load": load,
        "D_func": D_func, "lb": lb, "ub": ub, "start_time": start_time, "max_dist": max_dist
    }

def branch_and_bound(optimizer, params):
    current_best_max_dist = params['ub']
    max_dist = params['max_dist']
    paths = []
    found_solution = False

    optimizer.add(max_dist >= params['lb'])

    while True:
        elapsed_time = time.time() - params['start_time']
        if elapsed_time > TIME_LIMIT:
            print("\nTime limit reached.")
            break

        optimizer.set(timeout=int(max(TIME_LIMIT - elapsed_time, 1) * 1000))

        if optimizer.check() == sat:
            found_solution = True
            model = optimizer.model()
            current_max_dist = max(model.eval(params['total_distance'][c]).as_long() for c in params['Couriers'])

            if current_best_max_dist is None or current_max_dist < current_best_max_dist:
                current_best_max_dist = current_max_dist
                paths = []

                for c in params['Couriers']:
                    path_length_c = model.eval(params['path_length'][c]).as_long()
                    path_values = [
                        model.eval(params['path'][c][j]).as_long()
                        for j in range(2, path_length_c) if model.eval(params['path'][c][j]).as_long() != 0
                    ]
                    paths.append(path_values)

            if current_best_max_dist > params['lb']:
                optimizer.add(Int('max_dist') < current_best_max_dist)
        else:
            break

    if not found_solution:
        return None, []
    
    return current_best_max_dist, paths

def binary_search(optimizer, params):
    lb, ub = params['lb'], params['ub']
    max_dist = params['max_dist']
    best_solution = None
    best_paths = []
    found_solution = False

    while lb < ub:
        elapsed_time = time.time() - params['start_time']
        if elapsed_time > TIME_LIMIT:
            print("\nTime limit reached.")
            break

        mid = (lb + ub) // 2
        optimizer.push()
        optimizer.add(max_dist <= mid)

        optimizer.set(timeout=int(max(TIME_LIMIT - elapsed_time, 1) * 1000))

        if optimizer.check() == sat:
            found_solution = True
            model = optimizer.model()
            current_max_dist = max(model.eval(params['total_distance'][c]).as_long() for c in params['Couriers'])
            best_solution = current_max_dist

            best_paths = []
            for c in params['Couriers']:
                path_length_c = model.eval(params['path_length'][c]).as_long()
                path_values = [
                    model.eval(params['path'][c][j]).as_long()
                    for j in range(2, path_length_c) if model.eval(params['path'][c][j]).as_long() != 0
                ]
                best_paths.append(path_values)

            ub = mid
        else:
            lb = mid + 1
        optimizer.pop()

    if not found_solution:
        return None, []
    
    return best_solution, best_paths

def SMT(instance_number, bin_search_bool=False, sb_bool=False):
    optimizer, params = init_solver(instance_number, sb_bool)
    if optimizer is None:
        return

    if bin_search_bool:
        best_solution, paths = binary_search(optimizer, params)
    else:
        best_solution, paths = branch_and_bound(optimizer, params)

    print("\nRun summary:")
    print(f"- Approach: SMT (test version, no file output)")
    print(f"- Symmetry breaking: {'Yes' if sb_bool else 'No'}")
    print(f"- Search type: {'Binary' if bin_search_bool else 'Branch and Bound'}")
    print(f"- Instance: {instance_number}")
    print(f"- Execution time: {time.time() - params['start_time']:.2f} seconds")

    if best_solution is None:
        print("- Objective value (max dist): No feasible solution found (UNSAT).")
    else:
        print(f"- Objective value (max dist): {best_solution}")
        print(f"- Paths: {paths}\n")

# Example usage:
SMT("05", bin_search_bool=False, sb_bool=True)