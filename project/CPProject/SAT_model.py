from z3 import Bool, Sum, Int, If, Solver, Or
# TODO 'import * (all)' does not work

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
x = [[Bool(f"x_{c}_{i}") for c in COURIERS] for i in ITEMS]
'''
Example of the matrix 'x':
x = [1, 0, 0, 1]  # Courier 1: items 1 and 4 are assigned (x_1_1, x_1_4 = True)
    [0, 1, 0, 0]  # Courier 2: items 2 is assigned
    [0, 0, 1, 0]  # Courier 3: item 3 is assigned
'''

# load of each item picked up by each courier
# - I have to sum only the variables assigned to True, namely only when x_ij = 1
load = [Sum([If(x[c][i], s[i], 0) for c in COURIERS]) for i in ITEMS]

# tour matrix: defines the order of the distribution points (where items are) visited
path = [[Int(f"p_{c}_{t}") for t in TIMESTEPS] for c in COURIERS]
'''
Example of the matrix 'tour' (@ stands for the origin point):
tour = [@, 3, 0, 0, 0, @]  # Courier 1 path:  origin --> 3 --------> origin
       [@, 2, 0, 0, 0, @]  # Courier 2 path:  origin --> 2 --------> origin
       [@, 1, 4, 0, 0, @]  # Courier 3 path:  origin --> 1 --> 4 --> origin
'''

# max distance for each courier
max_distance = Int("max_dist")

# Define variables for the distance traveled by each courier
dist = [Int(f'dist_{i}') for i in COURIERS]

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
    solver.add(Or([x[c][i]] for i in ITEMS))

# distances computation TODO

# minimize max distance among all couriers
for c in COURIERS:
    solver.add(dist[i] <= max_distance)

# OBJECTIVE: minimize maximum distance
solver.minimize(max_distance)

