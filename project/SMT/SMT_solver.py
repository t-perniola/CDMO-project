from z3 import *
import utils
import time
import os

# NOTE: THEORIES USED
# - Boolean Logic
# - Arithmetic
# - EUF (uninterpreted functions)
# - Arrays

def SMT_model(instance_num, timeout):

    # Record start time
    start_time = time.time()

    # IMPORTING INSTANCE
    file_path = os.path.join('Instances', f'inst{instance_num}.dat')
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
    print(f"Instance {instance_num} chosen \n\n- Num couriers: {m} - Num items: {n}")

    # DECLARING VARIABLES USING SMT SORTS
    Z = IntSort()
    B = BoolSort()
    b_path = Array("b_path", Z, ArraySort(Z, B)) # create a boolean matrix
    load = Array('load', Z, Z)
    size = Array('size', Z, Z)
    path = Array('path', Z, ArraySort(Z, Z)) # create an integer matrix
    path_length = Array('path', Z, Z)
    total_distance = Array('total_distance', Z, Z)
    D_func = Function('D_func', Z, Z, Z) # takes two integer arguments (indices) and returns an integer (distance)

    # INITIALIZE the solver
    solver = Solver()
    solver.set(timeout=timeout)

    # HELPER FUNCTIONS
    # - all_different
    def distinct_except(values, forbidden_values):
        non_forbidden_values = [v for v in values if v not in forbidden_values]
        return Distinct(non_forbidden_values) # enforcing uniqueness

    # - lexicographic ordering
    def lexleq(a1, a2):
        if not a1: # if a1 is an empty list, then a1 precedes a2
            return True
        if not a2:
            return False 
        # if the first elem are different or if are the same recursively call the funct
        return Or(And(a1[0], Not(a2[0])), And(a1[0] == a2[0], lexleq(a1[1:], a2[1:])))

    # CONSTRAINTS
    # Mappings (due to 0-indexing): Z3 Arrays <-> Python arrays
    # - distance matrix
    for i in range(n+1):
        for j in range(n+1):
            solver.add(D_func(i, j) == D[i][j])
    # items' sizes
    for i in range(n):
        solver.add(size[i+1] == s[i])
    # Couriers' laod capacities
    for c in range(m):
        solver.add(load[c+1] == l[c])

    # Path should range between 0 and n+1        
    for c in Couriers:
        for j in range(1, MAX_ITEMS+1):
            solver.add(And(path[c][j] >= IntVal(0), path[c][j] <= IntVal(n+1)))

    # Boundaries of path_length's values
    for c in Couriers:
        solver.add(And(path_length[c] >= IntVal(3), path_length[c] <= IntVal(MAX_ITEMS)))

    # Define initial node and final node
    for c in Couriers:
        solver.add(path[c][1] == IntVal(n + 1)) # Initial node
        solver.add(path[c][path_length[c]] == IntVal(n + 1))  # Ending node

    # Set unvisited items to zero
    for c in Couriers:
        for i in range(1, MAX_ITEMS + 1):
            solver.add(Implies(i > path_length[c], path[c][i] == IntVal(0)))

    # No courier exceeds its load capacity
    for c in Couriers:    
        load_expr = Sum([If(b_path[c][j], size[j], 0) for j in Items])
        solver.add(load_expr <= load[c])

    # Exactly one item assignment to a courier
    for j in Items:
        solver.add(Sum([If(b_path[c][j], 1, 0) for c in Couriers]) == 1)

    # A courier cannot take more than MAX_ITEMS items
    for c in Couriers:
        solver.add(Sum([If(b_path[c][j], 1, 0) for j in Items]) <= MAX_ITEMS)

    # Couriers cannot visit same node twice (nor stay in the same node)
    for c in Couriers:
        solver.add(distinct_except([path[c][j] for j in range(1, MAX_ITEMS)], [0]))

    # Channeling: 
    # - b_path -> path
    for c in Couriers:
        for i in Items: # if b_bath[c][i] is true, then there must be true also at least one path[c][j] == i
            solver.add(Implies(b_path[c][i], Or([path[c][j] == i for j in range(1, MAX_ITEMS+1)])))
            # if b_bath[c][i] is false, then there won't be any path[c][j] == i
            solver.add(Implies(Not(b_path[c][i]), And([path[c][j] != i for j in range(1, MAX_ITEMS+1)])))

    # The items weight cannot exceed the load size
    for c in Couriers:
        solver.add(Sum([If(b_path[c][j], size[j], 0) for j in Items]) <= load[c])
        solver.add(Sum([If(j < path_length[c]-1, size[path[c][j]], 0) for j in range(2, MAX_ITEMS)]) <= load[c])
                
    # Distance computation
    for c in Couriers:
        # Sum for distances between two non-zero Items
        dist_expr = Sum([If(And(path[c][j] != 0, path[c][j+1] != 0), D_func(path[c][j]-IntVal(1), path[c][j+1]-IntVal(1)), 0)
                        for j in range(1, MAX_ITEMS)])
        
        # Sum for distances when there's a zero node between two non-zero items
        dist_expr += Sum([If(And(path[c][j] == 0, path[c][j-1] != 0, path[c][j+1] != 0), D_func(path[c][j-1]-IntVal(1), path[c][j+1]-IntVal(1)), 0)
                        for j in range(1, MAX_ITEMS)])
        
        solver.add(total_distance[c] == dist_expr)

    # Symmetry breaking: two couriers with the same load size
    for c1 in Couriers:
        for c2 in Couriers:
            if c1 < c2:  # Ensure c1 < c2 to avoid redundant comparisons
                # Add symmetry-breaking constraint if load sizes are equal
                sym_break_constraint = If(load[c1] == load[c2], lexleq([b_path[c1][j] for j in Items], [b_path[c2][j] for j in Items]), True)
                solver.add(sym_break_constraint)

    # OPTIMIZATION OBJECTIVE - Minimize the maximum distance traveled by any courier
    max_dist = Int('max_dist')
    solver.add([max_dist >= total_distance[c] for c in Couriers])

    return solver, max_dist

instance_num = "01"
timeout = 300000  # 5 minutes
solver, max_dist = SMT_model(instance_num, timeout)  # Unpack returned values
if solver.check() == sat:
    model = solver.model()
    max_distance_value = model[max_dist]
    print(f"Retrieved max distance: {max_distance_value}")
else:
    print("No solution found.")
