from z3 import Bool, Sum, Int, If, Solver
# TODO 'import * (all)' does not work

# TODO dynamic initializations
m = 2
n = 5

l = [18, 30]
s = [20, 17, 6]

D = [0, 21, 86, 99,
     21, 0, 71, 80, 
     92, 71, 0, 61, 
     59, 80, 61, 0]

# VARIABLES
# creates m x n variables that tell if the item j is assigned to courier i
x = [[Bool(f"x_{i}_{j}") for j in range(n)] for i in range(m)]

# load of each item picked up by each courier
# - I have to sum only the variables assigned to True, namely only when x_ij = 1
load = [Sum([If(x[i][j], s[j], 0) for j in range(n)]) for i in range(m)]

# max distance for each courier
max_distance = Int("max_dist")

# SOLVER INIT
solver = Solver()

# CONSTRAINTS
# each courier should not overload itself, namely it mustn't exceed its load capacity
for i in m:
    solver.add(load[i] <= l[i]) # remember: solver.add('expression') the solution must satisfy the given expression

# ...

# OBJECTIVE
solver.minimize(max_distance)

