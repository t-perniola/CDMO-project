from z3 import *

# THEORIES USED
# - Boolean Logic
# - Arithmetic
# - EUF (uninterpreted functions)

# TODO dynamic data initializations
m = 2
n = 3

max_nodes = (n // m) + 3

COURIERS = range(m)
NODES = range(n)

l = [18, 30]
s = [20, 17, 6]

D = [[0, 21, 86, 99],
     [21, 0, 71, 80], 
     [92, 71, 0, 61], 
     [59, 80, 61, 0]]

# VARIABLES
# creates m x n variables that tell if the item j is assigned to courier i
b_path = [[Bool(f"x_{c}_{i}") for i in NODES] for c in COURIERS]
'''
Example of the matrix 'x':
x = [1, 0, 0, 1]  # Courier 1: items 1 and 4 are assigned (x_1_1, x_1_4 = True)
    [0, 1, 0, 0]  # Courier 2: item 2 is assigned
    [0, 0, 1, 0]  # Courier 3: item 3 is assigned
'''

# load of each item picked up by each courier
# - I have to sum only the variables assigned to True, namely only when x_ij = 1
load = [Sum([If(b_path[c][i], s[i], 0) for i in NODES]) for c in COURIERS]

# tour matrix: defines the order of the distribution points (where NODES are) visited
path = [[Int(f"p_{c}_{j}") for j in range(max_nodes)] for c in COURIERS]
print("Empty path: ", path)
'''
Example of the matrix 'path' (@ stands for the origin point):
path = [5, 3, 5]     # Courier 1 path:  origin --> 3 --------> origin
       [5, 2, 5]     # Courier 2 path:  origin --> 2 --------> origin
       [5, 1, 4, 5]  # Courier 3 path:  origin --> 1 --> 4 --> origin
'''

# max distance for each courier
total_distance = [Int(f"max_dist_{c}") for c in COURIERS]

# Define variables for the distance traveled by each courier
dist = [Int(f'dist_{c}') for c in COURIERS]

# path length for each courier
path_length = [Int(f"len_{c}") for c in COURIERS]

# HELPER functions
# - all_different
def distinct_except(values, forbidden_values):
    non_forbidden_values = [v for v in values if v not in forbidden_values]
    return Distinct(non_forbidden_values) # enforcing uniqueness

# - among
def among(value_set, value):
    return Or([value == v for v in value_set])

# OPTIMIZING SOLVER INIT
solver = Solver()

# CONSTRAINTS
# path should range between 0 and n+1
for c in COURIERS:
    for j in range(max_nodes):
        solver.add(And(path[c][j] >= 0, path[c][j] <= n+1))

# path length should range between 3 and max_nodes
for c in COURIERS:
    solver.add(And(path_length[c] >= 3, path_length[c] <= max_nodes))

# Couriers cannot visit same node twice (nor stay in the same node)
for c in COURIERS:
    solver.add(distinct_except([path[c][j] for j in range(1, max_nodes)], [0]))

# Define initial node and final node
for c in COURIERS:
    solver.add(path[c][0] == n+1)
    #solver.add(path[c][path_length[c]] == n + 1)

# Set to zero unvisited nodes
for c in COURIERS:
    for j in range(max_nodes):
        solver.add(If(j > path_length[c], path[c][j] == 0, True))

# If you have more load size than me, then your load must be greater than mine
for c1 in COURIERS:
    for c2 in COURIERS:
        print()
        solver.add(Sum([If(And(c2 > c1, l[c2] > l[c1]), b_path[c1][j]*s[j], 0) for j in NODES])
                <= Sum([b_path[c2][j]*s[j] for j in NODES]))

# All the couriers must visit different nodes
'''
for c in COURIERS:
    solver.add(distinct_except([path[c][j] for j in range(max_nodes)], [0, n+1]))
'''

for i in NODES:
    solver.add(Sum([If(b_path[c][i] == True, 1 , 0) for c in COURIERS]) == 1)

for j in NODES:
    solver.add(Or([b_path[c][j] == True for c in COURIERS]))

# The path length of the single courier, in order to be balanced, cannot be longer than n-m
for c in COURIERS:
    solver.add(path_length[c] <= max_nodes)

for c in COURIERS:
    solver.add(Sum([If(b_path[c][i] == True, 1, 0) for i in NODES]) <= max_nodes)

# CHECK SATISFIABILITY
if solver.check() == sat:
    model = solver.model()

 ############################################################################à

    # ADDITIONAL CONSTRAINTS unfeasible before
    # Define final node
    for c in COURIERS:
        path_length_value = model.eval(path_length[c])
        solver.add(path[c][path_length_value.as_long()] == n + 1)

    # Channeling constraint
    for c in COURIERS:
        path_length_value = model.eval(path_length[c])
        for i in NODES:
            solver.add(Implies(b_path[c][i], Or([path[c][j] == i for j in range(1, path_length_value.as_long() + 1)])))
            solver.add(Implies(Or([path[c][j] == i for j in range(1, path_length_value.as_long() + 1)]), b_path[c][i]))

    # Distance computation
    for c in COURIERS:
        path_length_value = model.eval(path_length[c]).as_long()
        total_distance_expr = Sum([D[model.eval(path[c][j]).as_long()-1][model.eval(path[c][j+1]).as_long()-1] for j in range(path_length_value)])
        solver.add(total_distance[c] == total_distance_expr)

    # All the adjacent values until path_length must be different from 0
    for c in COURIERS:
        path_length_value = model.eval(path_length[c])
        for j in range(1, path_length_value.as_long()):
            solver.add(And(model[path[c][j]] != 0, model[path[c][j+1]] != 0))

    '''
    # Load size constraint
    for c in COURIERS:
        path_length_value = model.eval(path_length[c])
        solver.add(Sum([s[model[path[c][j]].as_long()] for j in range(1, path_length_value.as_long())]) <= l[c])
    '''
    # Rerun the solver
    if solver.check() == sat:
        model = solver.model()

        path_values = [[model[path[c][j]] for j in range(max_nodes)] for c in COURIERS]
        print("\nPaths:")
        for c in COURIERS:
            print(f"Courier {c}: {path_values[c]}")

        print("\nItem Assignments (T/F):")
        for c in COURIERS:
            print(f"Courier {c}: {[model.eval(b_path[c][i]) for i in NODES]}")

        print("\nTotal distances:")
        for c in COURIERS:
            print(f"Courier {c}: {model.eval(total_distance[c])}")

    else:
        print("The model is unsatisfiable after adding new constraints.")

else:
    print("The model is unsatisfiable.")