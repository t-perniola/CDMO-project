from math import floor
import z3
import json
import utils
import os
import time

# NOTE: THEORIES USED
# - Boolean Logic
# - Arithmetic
# - EUF (uninterpreted functions)
# - Arrays

def SMT(instance_number):

    start_time = time.time()

    # IMPORTING INSTANCE
    file_path = os.path.join('Instances', f'inst{instance_number}.dat')
    instance = utils.read_dat_file(file_path)

    m = instance['m']
    n = instance['n']
    l = instance['l']
    s = instance['s']
    D = instance['D']

    # DECLARING CONSTANTS
    MAX_ITEMS = (n // m) + 3
    Couriers = range(1, m+1)
    Items = range(1, n+1)
    print(f"Instance {instance_number} chosen \nnum Couriers: {m}, num items: {n}")

    # DECLARING VARIABLES USING SMT SORTS
    Z = z3.IntSort()
    B = z3.BoolSort()
    b_path = z3.Array("b_path", Z, z3.ArraySort(Z, B)) # create a boolean matrix
    load = z3.Array('load', Z, Z)
    size = z3.Array('size', Z, Z)
    path = z3.Array('path', Z, z3.ArraySort(Z, Z)) # create an integer matrix
    path_length = z3.Array('path', Z, Z)
    total_distance = z3.Array('total_distance', Z, Z)
    D_func = z3.Function('D_func', Z, Z, Z) # takes two integer arguments (indices) and returns an integer (distance)

    # INITIALIZE the OPTIMIZER
    optimizer = z3.Solver()

    # HELPER FUNCTIONS
    # - all_different
    def distinct_except(values, forbidden_values):
        non_forbidden_values = [v for v in values if v not in forbidden_values]
        return z3.Distinct(non_forbidden_values) # enforcing uniqueness

    # - lexicographic order
    def lexleq(a1, a2):
        if not a1:  # If a1 is an empty list, it precedes a2
            return True
        if not a2:  # If a2 is an empty list, a1 does not precede a2
            return False
        # Compare the first elements and recursively check the rest
        return z3.Or(z3.And(z3.Not(a1[0]), a2[0]), z3.And(a1[0] == a2[0], lexleq(a1[1:], a2[1:])))
        #return Or(And(a1[0], Not(a2[0])), And(a1[0] == a2[0], lexleq(a1[1:], a2[1:])))

    # CONSTRAINTS
    # Mappings (due to 0-indexing): Z3 Arrays <-> Python arrays
    # - distance matrix
    for i in range(n+1):
        for j in range(n+1):
            optimizer.add(D_func(i, j) == D[i][j])
    # items' sizes
    for i in range(n):
        optimizer.add(size[i+1] == s[i])
    # Couriers' laod capacities
    for c in range(m):
        optimizer.add(load[c+1] == l[c])

    # Path should range between 0 and n+1        
    for c in Couriers:
        for j in range(1, MAX_ITEMS+1):
            optimizer.add(z3.And(path[c][j] >= 0, path[c][j] <= n+1))

    # Boundaries of path_length's values
    for c in Couriers:
        optimizer.add(z3.And(path_length[c] >= 3, path_length[c] <= MAX_ITEMS))

    # Define initial node and final node
    for c in Couriers:
        optimizer.add(path[c][1] == n + 1) # Initial node
        optimizer.add(path[c][path_length[c]] == n + 1)  # Ending node

    # Set unvisited Items to zero
    for c in Couriers:
        for i in range(1, MAX_ITEMS + 1):
            optimizer.add(z3.Implies(i > path_length[c], path[c][i] == 0))

    # No courier exceeds its load capacity
    for c in Couriers:    
        load_expr = z3.Sum([z3.If(b_path[c][j], size[j], 0) for j in Items])
        optimizer.add(load_expr <= load[c])

    # Exactly one item assignment to a courier
    for j in Items:
        optimizer.add(z3.Sum([z3.If(b_path[c][j], 1, 0) for c in Couriers]) == 1)

    # A courier cannot take more than MAX_ITEMS items
    for c in Couriers:
        optimizer.add(z3.Sum([z3.If(b_path[c][j], 1, 0) for j in Items]) <= MAX_ITEMS)

    # Couriers cannot visit same node twice (nor stay in the same node)
    for c in Couriers:
        optimizer.add(distinct_except([path[c][j] for j in range(1, MAX_ITEMS)], [0]))

    # Channeling: 
    # - b_path -> path
    for c in Couriers:
        for i in Items: # if b_bath[c][i] is true, then there must be true also at least one path[c][j] == i
            optimizer.add(z3.Implies(b_path[c][i], z3.Or([path[c][j] == i for j in range(1, MAX_ITEMS+1)])))
            # if b_bath[c][i] is false, then there won't be any path[c][j] == i
            optimizer.add(z3.Implies(z3.Not(b_path[c][i]), z3.And([path[c][j] != i for j in range(1, MAX_ITEMS+1)])))

    # The items weight cannot exceed the load size
    for c in Couriers:
        optimizer.add(z3.Sum([z3.If(b_path[c][j], size[j], 0) for j in Items]) <= load[c])
        optimizer.add(z3.Sum([z3.If(j < path_length[c]-1, size[path[c][j]], 0) for j in range(2, MAX_ITEMS)]) <= load[c])
                
    # Distance computation
    for c in Couriers:
        # Sum for distances between two non-zero Items
        dist_expr = z3.Sum([z3.If(z3.And(path[c][j] != 0, path[c][j+1] != 0), D_func(path[c][j]-1, path[c][j+1]-1), 0)
                        for j in range(1, MAX_ITEMS)])
        
        # Sum for distances when there's a zero node between two non-zero items
        # FIXME: allowing zeros in the paths, we have to do this additional computation
        dist_expr += z3.Sum([z3.If(z3.And(path[c][j] == 0, path[c][j-1] != 0, path[c][j+1] != 0), D_func(path[c][j-1]-1, path[c][j+1]-1), 0)
                        for j in range(1, MAX_ITEMS)])
        
        optimizer.add(total_distance[c] == dist_expr)

    # Symmetry breaking: two couriers with the same load size
    for c1 in Couriers:
        for c2 in Couriers:
            if c1 < c2:  # Ensure c1 < c2 to avoid redundant comparisons
                # Add symmetry-breaking constraint if load sizes are equal
                sym_break_constraint = z3.If(load[c1] == load[c2], lexleq([b_path[c1][j] for j in Items], [b_path[c2][j] for j in Items]), True)
                optimizer.add(sym_break_constraint)

    # OPTIMIZATION OBJECTIVE - Minimize the maximum distance traveled by any courier
    max_dist = z3.Int('max_dist')
    optimizer.add([max_dist >= total_distance[c] for c in Couriers])

    TIME_LIMIT = 300
    current_best_max_dist = None

    while True:
        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > TIME_LIMIT:
            print("\nTime limit reached.")
            break

        # Set timeout for each solver call to the remaining time
        remaining_time = int(max(TIME_LIMIT - elapsed_time, 1) * 1000)  # in milliseconds
        # we enforce a strict time limit on each individual call to optimizer.check(),
        # s.t. the timout is adjusted dynamically based on the remaining time.
        optimizer.set(timeout=remaining_time)

        # CHECK SATISFIABILITY
        if optimizer.check() == z3.sat:
            model = optimizer.model()

            # Evaluate the maximum distance traveled by any courier
            current_max_dist = max(model.eval(total_distance[c]).as_long() for c in Couriers)

            if current_best_max_dist is None or current_max_dist < current_best_max_dist:
                current_best_max_dist = current_max_dist

                # Print the current best solution
                print("\nCurrent best solution:")
                paths = []
                for c in Couriers:
                    path_length_c = model.eval(path_length[c]).as_long()
                    path_values = []
                    for j in range(2, path_length_c):
                        evaluated_value = model.evaluate(path[c][j]).as_long()
                        if evaluated_value != 0:
                            path_values.append(evaluated_value-1)
                    paths.append(path_values)
                    print(f'Courier {c}: {path_values}')
                print(f"Current best max distance: {current_best_max_dist}")
                
                 # Prepare JSON dictionary
                json_dict = {}
                json_dict['SMT'] = {}
                json_dict['SMT']['time'] = int(floor(time.time() - start_time))
                json_dict['SMT']['optimal'] = True if (time.time() - start_time < TIME_LIMIT) else False
                json_dict['SMT']['obj'] = int(current_best_max_dist) if current_best_max_dist is not None else None
                json_dict['SMT']['sol'] = paths

                # Write JSON to file
                with open(f'res/SMT/{str(int(instance_number))}.json', 'w') as outfile:
                    json.dump(json_dict, outfile)

            # Add a constraint to find a better solution in the next iteration
            optimizer.add(max_dist < current_best_max_dist)

        else:
            print("unsat")
            break