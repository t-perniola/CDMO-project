from numpy import floor
from z3 import *
from utils.utils import read_dat_file, compute_bounds
import json
import time

# Time limit CONSTANT
TIME_LIMIT = 300

# Initialize the solver
def init_solver(instance_number, sb_bool):

    # Start the count
    start_time = time.time()
    
    # IMPORTING INSTANCE
    try:
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
    lb = instance['lb'] # naive bound
    ub = instance['ub'] # naive bound

    MAX_ITEMS = (n // m) + 3
    Couriers = range(1, m+1)
    Items = range(1, n+1)

    # DECLARING VARIABLES USING SMT SORTS
    Z = IntSort()
    B = BoolSort()
    b_path = Array("b_path", Z, ArraySort(Z, B)) # create a boolean matrix
    load = Array('load', Z, Z)
    size = Array('size', Z, Z)
    path = Array('path', Z, ArraySort(Z, Z)) # create an integer matrix
    path_length = Array('path', Z, Z) # how many steps (including depots) are in courier's path
    total_distance = Array('total_distance', Z, Z)
    D_func = Function('D_func', Z, Z, Z) # takes two integer arguments (indices) and returns an integer (distance)

    # Compute lower and upper bounds
    lb, ub = compute_bounds(m, n, l, s, D, lb, ub)

    # INITIALIZE the OPTIMIZER
    optimizer = Solver()
    optimizer.set("random_seed", 42)

    # HELPER FUNCTIONS
    # - all_different
    def distinct_except(values, forbidden_values):
        non_forbidden_values = [v for v in values if v not in forbidden_values]
        return Distinct(non_forbidden_values) # enforcing uniqueness

    # - lexicographic order (iterative)
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
    # Mappings (due to 0-indexing): Z3 Arrays <-> Python arrays
    # - distance matrix
    for i in range(n+1):
        for j in range(n+1):
            optimizer.add(D_func(i, j) == D[i][j])
    # items' sizes
    for i in range(n):
        optimizer.add(size[i+1] == s[i])
    # Couriers' load capacities
    for c in range(m):
        optimizer.add(load[c+1] == l[c])

    # Path should range between 0 and n+1        
    for c in Couriers:
        for j in range(1, MAX_ITEMS+1):
            optimizer.add(And(path[c][j] >= IntVal(0), path[c][j] <= IntVal(n+1)))

    # Boundaries of path_length's values
    for c in Couriers:
        optimizer.add(And(path_length[c] >= IntVal(3), path_length[c] <= IntVal(MAX_ITEMS)))

    # Define initial node and final node
    for c in Couriers:
        optimizer.add(path[c][1] == IntVal(n + 1)) # Initial node
        optimizer.add(path[c][path_length[c]] == IntVal(n + 1))  # Ending node

    # Set unvisited items to zero
    for c in Couriers:
        for i in range(1, MAX_ITEMS + 1):
            optimizer.add(Implies(i > path_length[c], path[c][i] == IntVal(0)))

    # Exactly one item assignment to a courier
    for j in Items:
        optimizer.add(Sum([If(b_path[c][j], 1, 0) for c in Couriers]) == 1)

    # A courier cannot take more than MAX_ITEMS items
    for c in Couriers:
        optimizer.add(Sum([If(b_path[c][j], 1, 0) for j in Items]) <= MAX_ITEMS)

    # Couriers cannot visit same node twice (nor stay in the same node)
    for c in Couriers:
        optimizer.add(distinct_except([path[c][j] for j in range(1, MAX_ITEMS)], [0]))
    
    # The items weight cannot exceed the load capacity
    for c in Couriers:
        optimizer.add(Sum([If(b_path[c][j], size[j], 0) for j in Items]) <= load[c])

    # Channeling: 
    # - b_path -> path
    for c in Couriers:
        for i in Items: # if b_bath[c][i] is true, then there must be true also at least one path[c][j] == i
            optimizer.add(Implies(b_path[c][i], Or([path[c][j] == i for j in range(1, MAX_ITEMS+1)])))
            # if b_bath[c][i] is false, then there won't be any path[c][j] == i
            optimizer.add(Implies(Not(b_path[c][i]), And([path[c][j] != i for j in range(1, MAX_ITEMS+1)])))

    # Distance computation
    for c in Couriers:
        # Sum for distances between two non-zero Items
        dist_expr = Sum([If(And(path[c][j] != 0, path[c][j+1] != 0), D_func(path[c][j]-IntVal(1), path[c][j+1]-IntVal(1)), 0)
                        for j in range(1, MAX_ITEMS)])
        
        # Sum for distances when there's a zero node between two non-zero items
        dist_expr += Sum([If(And(path[c][j] == 0, path[c][j-1] != 0, path[c][j+1] != 0), D_func(path[c][j-1]-IntVal(1), path[c][j+1]-IntVal(1)), 0)
                        for j in range(1, MAX_ITEMS)])
        
        optimizer.add(total_distance[c] == dist_expr)
    
    # SB constraint: two couriers with the same load size
    if sb_bool:
        for c1 in Couriers:
            for c2 in Couriers:
                if c1 < c2:  # Ensure c1 < c2 to avoid redundant comparisons
                    # Add symmetry-breaking constraint if load sizes are equal
                    sym_break_constraint = If(load[c1] == load[c2], lexleq([b_path[c1][j] for j in Items], [b_path[c2][j] for j in Items]), True)
                    optimizer.add(sym_break_constraint)

    # OPTIMIZATION OBJECTIVE - Minimize the maximum distance traveled by any courier
    max_dist = Int('max_dist')
    optimizer.add([max_dist >= total_distance[c] for c in Couriers])

    return optimizer, {
        
        "m": m, "n": n, "MAX_ITEMS": MAX_ITEMS, "Couriers": Couriers, "Items": Items, "path": path, 
        "path_length": path_length, "total_distance": total_distance, "b_path": b_path, "size": size, "load": load,
        "D_func": D_func, "lb": lb, "ub": ub, "start_time": start_time, "max_dist": max_dist
    }

# IMPLEMENT BRANCH-AND-BOUND SEARCH
def branch_and_bound(optimizer, params):
    current_best_max_dist = params['ub']  # Use upper bound as initial best solution
    max_dist = params['max_dist']
    paths = []
    found_solution = False  # Track if we ever find a valid solution

    # Add initial lower bound constraint
    optimizer.add(max_dist >= params['lb'])

    while True:
        elapsed_time = time.time() - params['start_time']
        if elapsed_time > TIME_LIMIT:
            print("\nTime limit reached.")
            break

        optimizer.set(timeout=int(max(TIME_LIMIT - elapsed_time, 1) * 1000))

        if optimizer.check() == sat:
            found_solution = True  # Mark that we found at least one solution
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

            if current_best_max_dist > params['lb']:  # Ensure we don't go below lb
                optimizer.add(Int('max_dist') < current_best_max_dist)
        else:
            break  # No better solution found, terminate search

    if not found_solution:
        return None, []

    return current_best_max_dist, paths

# IMPLEMENT BINARY SEARCH
def binary_search(optimizer, params):
    lb, ub = params['lb'], params['ub']
    max_dist = params['max_dist']
    best_solution = None
    best_paths = []
    found_solution = False  # Track if we ever find a valid solution

    while lb < ub:
        elapsed_time = time.time() - params['start_time']
        if elapsed_time > TIME_LIMIT:
            print("\nTime limit reached.")
            break

        mid = (lb + ub) // 2
        optimizer.push()  # Save current solver state
        optimizer.add(max_dist <= mid)

        optimizer.set(timeout=int(max(TIME_LIMIT - elapsed_time, 1) * 1000))

        if optimizer.check() == sat:
            found_solution = True  # Mark that we found at least one solution
            model = optimizer.model()
            # Here, we compute the actual maximum distance from the model
            current_max_dist = max(model.eval(params['total_distance'][c]).as_long() for c in params['Couriers'])
            best_solution = current_max_dist  # Use the computed max distance as the objective value

            best_paths = []
            for c in params['Couriers']:
                path_length_c = model.eval(params['path_length'][c]).as_long()
                path_values = [
                    model.eval(params['path'][c][j]).as_long()
                    for j in range(2, path_length_c) if model.eval(params['path'][c][j]).as_long() != 0
                ]
                best_paths.append(path_values)

            # Now, update ub to mid since we found a solution with this bound.
            ub = mid
        else:
            lb = mid + 1  # Increase the lower bound since no solution exists with mid as the bound.
        optimizer.pop()  # Restore solver state

    if not found_solution:
        return None, []

    return best_solution, best_paths


# Saves results into a JSON file
def save_results(instance_number, model_type, best_solution, paths, start_time):
    file_path = f'res/SMT/{str(int(instance_number))}.json'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    end_time = int(floor(time.time() - start_time))

    json_dict = {
        'time': TIME_LIMIT if end_time > TIME_LIMIT else end_time,
        'optimal': True if end_time < TIME_LIMIT else False,
        'obj': int(best_solution) if best_solution is not None else None,
        'sol': paths
    }

    if os.path.exists(file_path):
        with open(file_path, 'r') as infile:
            existing_data = json.load(infile)
    else:
        existing_data = {}

    existing_data[model_type] = json_dict

    with open(file_path, 'w') as outfile:
        json.dump(existing_data, outfile, indent=4, separators=(",", ": "))


# Main function
def SMT(instance_number, bin_search_bool=False, sb_bool=True):

    # Initialize the solver
    optimizer, params = init_solver(instance_number, sb_bool)
    if optimizer is None:
        return

    model_type = f"SMT_{'SB' if sb_bool else 'noSB'}_{'bin' if bin_search_bool else 'BB'}"

    if bin_search_bool:
        best_solution, paths = binary_search(optimizer, params)
    else:
        best_solution, paths = branch_and_bound(optimizer, params)

    print("\nRun summary:")
    print(f"- Approach: SMT")
    print(f"- Symmetry breaking: {'Yes' if sb_bool else 'No'}")
    print(f"- Search type: {'Binary' if bin_search_bool else 'Branch and Bound'}")
    print(f"- Instance: {instance_number}")

    if best_solution is None:
        print("- Objective value (max dist): No feasible solution found (UNSAT).")
    else:
        print(f"- Objective value (max dist): {best_solution}\n")

    # Save results to JSON file
    save_results(instance_number, model_type, best_solution, paths, params['start_time'])
    