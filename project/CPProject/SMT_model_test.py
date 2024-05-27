from z3 import *

# THEORIES USED
# - Boolean Logic
# - Arithmetic
# - EUF (uninterpreted functions)

# TODO dynamic data initializations
m = 2
n = 3

COURIERS = range(m)
ITEMS = range(n)
TIMESTEPS = range(n+2) # including starting & ending times (that would still coincide in the origin point)

l = [18, 30]
s = [20, 17, 6]

D = [[0, 21, 86, 99],
     [21, 0, 71, 80], 
     [92, 71, 0, 61], 
     [59, 80, 61, 0]]

# VARIABLES
# creates m x n variables that tell if the item j is assigned to courier i
x = [[Bool(f"x_{c}_{i}") for i in ITEMS] for c in COURIERS]
#print(x)
'''
Example of the matrix 'x':
x = [1, 0, 0, 1]  # Courier 1: items 1 and 4 are assigned (x_1_1, x_1_4 = True)
    [0, 1, 0, 0]  # Courier 2: items 2 is assigned
    [0, 0, 1, 0]  # Courier 3: item 3 is assigned
'''

# load of each item picked up by each courier
# - I have to sum only the variables assigned to True, namely only when x_ij = 1
load = [Sum([If(x[c][i], s[i], 0) for i in ITEMS]) for c in COURIERS]
#print(load)

# tour matrix: defines the order of the distribution points (where items are) visited
# path[c][t] indicates the item visited by courier c at timestep t
path = [[Int(f"p_{c}_{t}") for t in TIMESTEPS] for c in COURIERS]
#print("Empty path: ", path)
'''
Example of the matrix 'path' (@ stands for the origin point):
path = [@, 3, 0, 0, 0, @]  # Courier 1 path:  origin --> 3 --------> origin
       [@, 2, 0, 0, 0, @]  # Courier 2 path:  origin --> 2 --------> origin
       [@, 1, 4, 0, 0, @]  # Courier 3 path:  origin --> 1 --> 4 --> origin
'''

# max distance for each courier
max_distance = Int("max_dist")

# Define variables for the distance traveled by each courier
dist = [Int(f'dist_{c}') for c in COURIERS]

# OPTIMIZING SOLVER INIT
solver = Solver()

# CONSTRAINTS
# Each courier should not overload itself, namely it mustn't exceed its load capacity
'''
for c in COURIERS:
    solver.add(load[c] <= l[c]) # remember: solver.add('expression') the solution must satisfy the given expression
'''

# Each item should be assigned to exactly one courier, namely each item can be picked only once
# - what about using instead (Exactly <-> AtLeast and AtMost) ?
for i in ITEMS:
    solver.add(Sum([If(x[c][i], 1, 0) for c in COURIERS]) == 1)

# Each path must start and end at the origin point, represented by index n (origin)
for c in COURIERS:
    solver.add(path[c][0] == n+1)
    solver.add(path[c][n+1] == n+1)

'''
# Each courier should pick at least one item [redundant: it can be inferred from other constraints]
for c in COURIERS:
    #solver.add(Or([x[c][i]] for i in ITEMS)) # iterating along the rows of x
    solver.add(Sum([x[c][i] for i in ITEMS]) >= 1)
'''
'''
# All items should be picked-up
for i in ITEMS:
    #solver.add(Or([x[c][i]] for c in COURIERS)) # iterating along the columns of x
    solver.add(Sum([x[c][i] for c in COURIERS]) >= 1)
'''

# A courier cannot pick all the items, moreover it can pick at most (n-m+1) items
for c in COURIERS:
    solver.add(Sum([If(x[c][i], 1, 0) for i in ITEMS]) <= n-m+1)

'''
# A courier can visits only the given n distribution points, except for the boundaries
for c in COURIERS:
    for t in range(1, n+1):
        solver.add(path[c][t] >= 1)
        solver.add(path[c][t] <= n)
'''
'''
# A courier cannot take the same element twice in its tour
for c in COURIERS: # 'Distinct' serves the same purpose of 'alldifferent' in CP
    solver.add(Distinct([path[c][t] for t in range(1, n+1)]))
'''

# Channeling: If an item appears in any path in the 'tour' matrix, then it must be True in the 'x' matrix (and viceversa)
for c in COURIERS:
    for i in ITEMS: 
        solver.add(And(
            Implies(Or([path[c][t] == i for t in range(1, n+1)]), x[c][i] == True),
            Implies(x[c][i] == True, Or([path[c][t] == i for t in range(1, n+1)]))))
            # no built-in for iff (<->) then use let's use the construct: And(Implies(X), Implies(Y))

# No zeros between two numbers
#for c in COURIERS:
#    solver.add([(path[c][t], path[c][t+1]) for t in range(1, n+1)])

# Distances computation TODO
# 1st sum) from origin to the distribution point where x[c][i] is true, namely if the courier c has picked-up the i-th item
# 2nd sum) from each distribution point to the following one
# 3rd sum) from the last distribution point back to the origin

'''
for c in COURIERS: 
    print([path[c][t] for t in TIMESTEPS])
    print(path[c][1])
    distances = D[n][path[c][1]]
    print("1) ", distances)

    for t in range(2, n+1):
        distances += If(And(path[c][t] != 0, path[c][t+1]!=0, True), D[path[c][t]][path[c][t+1]], 0)
    print("2) ", distances)
    
    for t in reversed(range(2, n+1)):
        if path[c][t] != 0:
            last_item = path[c][t]
        break  # Exit the loop when the first non-zero value is found
    print("last ", last_item)

    distances += D[last_item][n]
    print("3) ", distances)
'''

#solver.minimize(max_distance)

if solver.check() == sat:
    print("\nThe model is satisfiable:")

    # a model represents an assignment of values to variables that satisfies all the constraints provided to the solver
    model = solver.model()

    # Print the path matrix
    path_values = [[model[path[c][t]] for t in TIMESTEPS] for c in COURIERS]
    print("\nPaths:")
    for c in COURIERS:
        print(f"Courier {c}: {path_values[c]}")

    # Print the x matrix with True/False values
    print("\nItem Assignments (True/False):")
    for c in COURIERS:
        print(f"Courier {c}: {[model.eval(x[c][i]) for i in ITEMS]}")
else:
    print("The model is unsatisfiable.")

