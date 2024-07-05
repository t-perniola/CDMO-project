from z3 import *
from utils import *
import time

# NOTE: THEORIES USED
# - Boolean Logic
# - Arithmetic
# - EUF (uninterpreted functions)
# - Arrays

# Record start time
start_time = time.time()

# IMPORTING INSTANCES
# Directory containing .dat files
directory = 'SMT\Instances'

# Choose the instance
NUM_INST = 10

# Read all .dat files and populate instances
instances = read_all_dat_files(directory)
instance = instances[NUM_INST-1] # extract the corresp. instance
m = instance['m']
n = instance['n']
l = instance['l']
s = instance['s']
D = instance['D']

# DECLARING CONSTANTS
MAX_ITEMS = (n // m) + 3
Couriers = range(1, m+1)
Items = range(1, n+1)
print(f"Instance {NUM_INST} chosen \nnum Couriers: {m}, num items: {n}")

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

# INITIALIZE the OPTIMIZER
optimizer = Solver()

# HELPER FUNCTIONS
# - all_different
def distinct_except(values, forbidden_values):
    non_forbidden_values = [v for v in values if v not in forbidden_values]
    return Distinct(non_forbidden_values) # enforcing uniqueness

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
        optimizer.add(And(path[c][j] >= 0, path[c][j] <= n+1))

# Boundaries of path_length's values
for c in Couriers:
    optimizer.add(And(path_length[c] >= 3, path_length[c] <= MAX_ITEMS))

# Define initial node and final node
for c in Couriers:
    optimizer.add(path[c][1] == n + 1) # Initial node
    optimizer.add(path[c][path_length[c]] == n + 1)  # Ending node

# Set unvisited Items to zero
for c in Couriers:
    for i in range(1, MAX_ITEMS + 1):
        optimizer.add(Implies(i > path_length[c], path[c][i] == 0))

# No courier exceeds its load capacity
for c in Couriers:    
    load_expr = Sum([If(b_path[c][j], size[j], 0) for j in Items])
    optimizer.add(load_expr <= load[c])

# Exactly one item assignment to a courier
for j in Items:
    optimizer.add(Sum([If(b_path[c][j], 1, 0) for c in Couriers]) == 1)

# A courier cannot take more than MAX_ITEMS items
for c in Couriers:
    optimizer.add(Sum([If(b_path[c][j], 1, 0) for j in Items]) <= MAX_ITEMS)

# Couriers cannot visit same node twice (nor stay in the same node)
for c in Couriers:
    optimizer.add(distinct_except([path[c][j] for j in range(1, MAX_ITEMS)], [0]))

# Channeling: 
# - b_path -> path
for c in Couriers:
    for i in Items: # if b_bath[c][i] is true, then there must be true also at least one path[c][j] == i
        optimizer.add(Implies(b_path[c][i], Or([path[c][j] == i for j in range(1, MAX_ITEMS+1)])))
        # if b_bath[c][i] is false, then there won't be any path[c][j] == i
        optimizer.add(Implies(Not(b_path[c][i]), And([path[c][j] != i for j in range(1, MAX_ITEMS+1)])))

# The items weight cannot exceed the load size (NOTE: redundant? since channeling is there...)
for c in Couriers:
    optimizer.add(Sum([If(b_path[c][j], size[j], 0) for j in Items]) <= load[c])
    optimizer.add(Sum([If(j < path_length[c]-1, size[path[c][j]], 0) for j in range(2, MAX_ITEMS)]) <= load[c])
            
# Distance computation
for c in Couriers:
    # Sum for distances between two non-zero Items
    dist_expr = Sum([If(And(path[c][j] != 0, path[c][j+1] != 0), D_func(path[c][j]-1, path[c][j+1]-1), 0)
                    for j in range(1, MAX_ITEMS)])
    
    # Sum for distances when there's a zero node between two non-zero items
    # FIXME: allowing zeros in the paths, we have to do this additional computation
    dist_expr += Sum([If(And(path[c][j] == 0, path[c][j-1] != 0, path[c][j+1] != 0), D_func(path[c][j-1]-1, path[c][j+1]-1), 0)
                      for j in range(1, MAX_ITEMS)])
    
    optimizer.add(total_distance[c] == dist_expr)

# OPTIMIZATION OBJECTIVE - Minimize the maximum distance traveled by any courier
max_dist = Int('max_dist')
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
    if optimizer.check() == sat:
        model = optimizer.model()

        # Evaluate the maximum distance traveled by any courier
        current_max_dist = max(model.eval(total_distance[c]).as_long() for c in Couriers)

        if current_best_max_dist is None or current_max_dist < current_best_max_dist:
            current_best_max_dist = current_max_dist

            # Print the current best solution
            print("\nCurrent best solution:")
            paths = {}
            for c in Couriers:
                path_length_c = model.eval(path_length[c]).as_long()
                path_values = []
                for j in range(1, path_length_c + 1):
                    evaluated_value = model.evaluate(path[c][j]).as_long()
                    if evaluated_value != 0:
                        path_values.append(evaluated_value)
                paths[c] = path_values
                print(f'Courier {c}: {path_values}')
            print(f"Current best max distance: {current_best_max_dist}")

        # Add a constraint to find a better solution in the next iteration
        optimizer.add(max_dist < current_best_max_dist)

    else:
        print("unsat")
        break

# Record end time
end_time = time.time()
# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"\nElapsed time: {elapsed_time} seconds")

# Draw the graph with each courier's path
#try: # 'paths' variable exists only exists if at least a solution is found 
#    draw_graph(num_items=n, Couriers=Couriers, paths=paths)
#except NameError:
    # print("\nNo solution found.")
#else: print(f"\nMax distance: {current_best_max_dist}")
