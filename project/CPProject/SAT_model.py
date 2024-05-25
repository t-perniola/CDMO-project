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

# load of each item picked up by each courier (I want a 1D array) TODO
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
# Each courier should not overload itself, namely it mustn't exceed its load capacity TODO
for c in COURIERS:
    for t in TIMESTEPS:
         ...

# A courier must end its tour in the starting point
for c in COURIERS:
     for i in ITEMS:
          solver.add(x[c][1][i] == x[c][n+1][i])

# Each courier picks at most one item at each timestep
for c in COURIERS:
     for t in TIMESTEPS: # use * to unpack the single boolean variables 
          solver.add(AtMost(*[x[c][t][i] for i in ITEMS], 1))

#equivalently... Pairwise encoding
for c in COURIERS:
     for t in TIMESTEPS:
          for i in range(n-1):
               for j in range(i+1, n):
                    solver.add(Or(Not(x[c][t][i]), Not(x[c][t][j])))

#equivalently... Sequential Encoding
for c in COURIERS:
     for t in TIMESTEPS:
          y = [Bool(f"y_{i}") for i in ITEMS] # introduce new n variables y_i
          solver.add(And(Implies(x[c][t][1], y[1])))
          for i in range(2, n):
               solver.add(Or(Implies(Or(x[c][t][i], y[i-1]),y[i]), Implies(y[i-1], Not(x[c][t][i]))))
          solver.add(Implies(y[n-1], Not(x[c][t][i])))

#equivalently... Heule Encoding TODO
for c in COURIERS:
     for t in TIMESTEPS:
          if n <= 4: # apply pairwise encoding
               for i in range(n-1):
                    for j in range(i+1, n):
                         solver.add(Or(Not(x[c][t][i]), Not(x[c][t][j])))
          else: # n > 4
               y = Bool("y")
               ...


# Each item must be assigned to exactly one courier:
# first part: there exists at least one x[c][t][i] that is true, namely, all items should be picked-up
for i in ITEMS:
     for t in range(2, n):
          solver.add(Or([x[c][t][i] for c in COURIERS]))
# second part: if exists, no other x[k][t][i] where k != c can exists either
for i in ITEMS:
    for t in range(2, n):
         for c in COURIERS:
             for k in COURIERS:
                  if c != k:
                       solver.add(Implies(x[c][t][i], Not(x[k][t][i])))

# no zeros between two numbers
for c in COURIERS:
     for t in TIMESTEPS:
          for i in range(2, n):
               solver.add(Implies(x[c][t][i], x[c][t][i-1]))

# distances computation TODO        
