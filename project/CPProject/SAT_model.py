from z3 import *

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
# creates m x (n+2) x n matrix that tell if the item i is assigned to courier c at timestep t
x = [[[Bool("x_{c}_{t}_{i}") for i in ITEMS] for t in TIMESTEPS] for c in COURIERS]

# load of each item picked up by each courier
# - I have to sum only the variables assigned to True, namely only when x_ij = 1
load = [Sum([If(x[c][t][i], s[i], 0) for i in ITEMS for t in TIMESTEPS]) for c in COURIERS]

print(x)
print(load)

# SOLVER INIT
solver = Solver()

# CONSTRAINTS
# each courier should not overload itself, namely it mustn't exceed its load capacity
for c in COURIERS:
     solver.add(load[c] <= l[c])

# no zeros between two numbers

# distances computation TODO
# 1st sum) from origin to the distribution point where x[c][i] is true, namely if the courier c has picked-up the i-th item
# 2nd sum) from each distribution point to the following one
# 3rd sum) from the last distribution point back to the origin


        
