from z3 import Bool, Sum, Int, If, Solver, Or
# TODO 'import * (all)' does not work

# TODO dynamic data initializations
m = 2
n = 5

COURIERS = range(m)
ITEMS = range(n)

l = [18, 30]
s = [20, 17, 6]

D = [[0, 21, 86, 99],
     [21, 0, 71, 80], 
     [92, 71, 0, 61], 
     [59, 80, 61, 0]]

# VARIABLES
# creates m x n variables that tell if the item j is assigned to courier i
x = [[Bool(f"x_{c}_{i}") for c in COURIERS] for i in ITEMS]

# load of each item picked up by each courier
# - I have to sum only the variables assigned to True, namely only when x_ij = 1
load = [Sum([If(x[c][i], s[i], 0) for i in ITEMS]) for c in COURIERS]

# max distance for each courier
max_distance = Int("max_dist")

# SOLVER INIT
solver = Solver()

# CONSTRAINTS
# each courier should not overload itself, namely it mustn't exceed its load capacity
for c in COURIERS:
    solver.add(load[c] <= l[c]) # remember: solver.add('expression') the solution must satisfy the given expression

# each item should be assigned to exactly one courier, namely each item can be picked only once
for i in ITEMS:
    solver.add(Sum([If(x[c][i], 1, 0) for c in COURIERS]) == 1)

# each courier should pick at least one item (redundant)
for c in COURIERS:
    solver.add(Or([x[c][i]] for j in ITEMS))

# distances computation
for c in COURIERS:
    distances = [Sum(If(x[c][i], D[c][i]), 0) for i in ITEMS]
    solver.add(distances <= max_distance)

# OBJECTIVE
solver.minimize(max_distance)

