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
'''
Example of x matrix:

    [0,1,0,1] 
  [0,0,0,1]0]
[1,0,1,0]1,0]
[0,1,0,0] 0]
[0,0,0,1]0]
'''

# load of each item picked up by each courier (I want a 1D array)
# in SMT: load = [Sum([If(x[c][t][i], s[i], 0) for i in ITEMS for t in TIMESTEPS]) for c in COURIERS]
for c in COURIERS:
     for t in TIMESTEPS:
          for i in ITEMS:
               if x[c][t][i]:
                    ...

print(x)
#print(load)

# SOLVER INIT
solver = Solver()

# CONSTRAINTS
# each courier should not overload itself, namely it mustn't exceed its load capacity
#in SMT: for c in COURIERS:
     #solver.add(load[SSsc] <= l[c])

# all items should be picked-up
#for i in ITEMS:
#     solver.add() # n = len(ITEMS)

# each courier picks at most one item at each timestep
for c in COURIERS:
     for t in TIMESTEPS: # use * to unpack the single boolean variables
          solver.add(AtMost(*[x[c][t][i] for i in ITEMS], 1))
#equivalently...
for c in COURIERS:
     for t in TIMESTEPS:
          for i in range(n-1):
               for j in range(i+1, n):
                    solver.add(Or(Not(x[c][t][i]), Not(x[c][t][j])))

# each item must be picked up exaclty once at all
for i in ITEMS:
    solver.add()

# no zeros between two numbers

# distances computation TODO        
