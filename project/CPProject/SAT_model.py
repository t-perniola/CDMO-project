from z3 import *

# TODO dynamic data initializations
m = 2
n = 3

COURIERS = range(m)
ITEMS = range(n)
TIMESTEPS = range(n+2) # including starting & ending times (that would still coincide in the origin point)

print

l = [18, 30]
s = [20, 17, 6]

D = [[0, 21, 86, 99],
     [21, 0, 71, 80], 
     [92, 71, 0, 61], 
     [59, 80, 61, 0]]

# VARIABLES
# creates m x n variables that tell if the item j is assigned to courier i
x = [[Bool(f"x_{c}_{i}") for i in ITEMS] for c in COURIERS]
print(x)
'''
Example of the matrix 'x':
x = [1, 0, 0, 1]  # Courier 1: items 1 and 4 are assigned (x_1_1, x_1_4 = True)
    [0, 1, 0, 0]  # Courier 2: items 2 is assigned
    [0, 0, 1, 0]  # Courier 3: item 3 is assigned
'''

# load of each item picked up by each courier
# - I have to sum only the variables assigned to True, namely only when x_ij = 1
load = [Sum([If(x[c][i], s[i], 0) for i in ITEMS]) for c in COURIERS]

# tour matrix: defines the order of the distribution points (where items are) visited
# path[c][t] indicates the item visited by courier c at timestep t
path = [[Int(f"p_{c}_{t}") for t in TIMESTEPS] for c in COURIERS]
print("EMPTY PATH: ", path)
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

# SOLVER INIT
solver = Solver()

# CONSTRAINTS
# each courier should not overload itself, namely it mustn't exceed its load capacity
for c in COURIERS:
    solver.add(load[c] <= l[c]) # remember: solver.add('expression') the solution must satisfy the given expression

# each item should be assigned to exactly one courier, namely each item can be picked only once
for i in ITEMS:
    solver.add(Sum([If(x[c][i], 1, 0) for c in COURIERS]) == 1)

# the boundaries (starting and ending point) must be equal to n+1
for c in COURIERS:
    solver.add(path[c][0] == n+1)
    solver.add(path[c][n+1] == n+1)
  
# each courier should pick at least one item [redundant: it can be inferred from other constraints]
for c in COURIERS:
    solver.add(Or([x[c][i]] for i in ITEMS)) # iterating along the rows of x

# all items should be picked-up
for c in ITEMS:
    solver.add(Or([x[c][i]] for c in COURIERS)) # iterating along the columns of x

# a courier cannot pick all the items, moreover it can pick at most (n-m+1) items [redundant wrt the objective function]
for c in COURIERS:
    solver.add(Sum([If(x[c][i], 1, 0) for i in ITEMS]) <= n-m+1)

# Channeling: If an item appears in any path in the 'tour' matrix, then it must be True in the 'x' matrix (and viceversa)
for c in COURIERS:
    for i in ITEMS: 
        solver.add(And(Implies(path[c][t]), Implies(x[c][i], ))) 
        # no built-in for iff (<->) then use let's use the construct: And(Implies(X), Implies(Y))


# distances computation TODO
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

        
# minimize max distance among all couriers
#for c in COURIERS:
#    solver.add(dist[c] == distances)

if solver.check() == sat:
  print("The model is satisfiable.")
else:
  print("The model is not satisfiable.")

# OBJECTIVE: minimize maximum distance

#This approach effectively transforms the optimization problem into a series of decision problems,
#leveraging the SAT solver to find the optimal solution, until the smallest feasible maximum distance is found. 
# Function to check feasibility with a given max distance
def is_feasible(max_dist):
    temp_solver = Solver()
    temp_solver.add(solver.assertions())
    for c in COURIERS:
        temp_solver.add(dist[c] <= max_dist)
    if temp_solver.check() == sat:
        return True, temp_solver.model()
    return False, None

'''
# Binary search to minimize the maximum distance
lower_bound = 0
upper_bound = sum(max(D[i]) for i in range(n + 1))  # Upper bound estimate
optimal_distance = upper_bound
optimal_model = None

while lower_bound <= upper_bound:
    mid = (lower_bound + upper_bound) // 2
    feasible, model = is_feasible(mid)
    if feasible:
        optimal_distance = mid
        optimal_model = model
        upper_bound = mid - 1
    else:
        lower_bound = mid + 1

# Display the optimal solution
if optimal_model:
    path_solution = [[optimal_model[path[c][t]].as_long() if optimal_model[path[c][t]] else n for t in TIMESTEPS] for c in COURIERS]
    print("Optimal max distance:", optimal_distance)
    print("Paths:", path_solution)
    for c in COURIERS:
        print(f"Courier {c} path: ", [optimal_model[path[c][t]] for t in TIMESTEPS])
else:
    print("No solution found")
'''
