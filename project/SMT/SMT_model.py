from z3 import *
from utils import *
import time

# THEORIES USED
# - Boolean Logic
# - Arithmetic
# - EUF (uninterpreted functions)
# - Arrays

# Record start time
start_time = time.time()

# IMPORTING INSTANCES
# Directory containing .dat files
directory = 'instances'

# Read all .dat files and populate instances
num_inst = 5 # choose the instance
instances = read_all_dat_files(directory)
instance = instances[num_inst-1] # extract the corresp. instance
m = instance['m']
n = instance['n']
l = instance['l']
s = instance['s']
D = instance['D']

# useful variables
max_nodes = (n // m) + 3
COURIERS = range(1, m+1)
NODES = range(1, n+1)
#print("max_nodes:", max_nodes)
print(f"Instance {num_inst} chosen \nnum couriers: {m}, num items: {n}")

# DECLARING VARIABLES USING SMT SORTS
# creates m x n variables that tell if the item j is assigned to courier i
Z = IntSort()
b_path = Array("b_path", IntSort(), ArraySort(IntSort(), BoolSort()))
load = Array('load', Z, Z)
size = Array('size', Z, Z)
path = Array('path', Z, ArraySort(Z, Z))
path_length = Array('path', Z, Z)
total_distance = Array('total_distance', Z, Z)
D_func = Function('D_func', Z, Z, Z) # takes two integer arguments (indices) and returns an integer (distance)

# OPTIMIZING optimizer INIT
optimizer = Optimize()

# HELPER FUNCTIONS
# - all_different
def distinct_except(values, forbidden_values):
    non_forbidden_values = [v for v in values if v not in forbidden_values]
    return Distinct(non_forbidden_values) # enforcing uniqueness

# - lexicographic ordering
def lexleq(a1, a2):
    if not a1: # if a1 is an empty list, then a1 precedes a2
        return True
    if not a2:
        return False 
    # if the first elem are different or if are the same recursively call the funct
    return Or(And(a1[0], Not(a2[0])), And(a1[0] == a2[0], lexleq(a1[1:], a2[1:])))

# CONSTRAINTS
# Mappings (due to 0-indexing): Z3 Arrays <-> Python arrays
for i in range(n+1):
    for j in range(n+1):
        optimizer.add(D_func(i, j) == D[i][j])

for i in range(n):
    optimizer.add(size[i+1] == s[i])

for c in range(m):
    optimizer.add(load[c+1] == l[c])

# Path should range between 0 and n+1        
for c in COURIERS:
    for j in range(1, max_nodes+1):
        optimizer.add(And(path[c][j] >= 0, path[c][j] <= n+1))

# Boundaries of path_length's values
for c in COURIERS:
    optimizer.add(And(path_length[c] >= 3, path_length[c] <= max_nodes))

# Define initial node and final node
for c in COURIERS:
    optimizer.add(path[c][1] == n + 1) # Initial node
    optimizer.add(path[c][path_length[c]] == n + 1)  # Ending node

# Set unvisited nodes to zero
for c in COURIERS:
    for i in range(1, max_nodes + 1):
        optimizer.add(Implies(i > path_length[c], path[c][i] == 0))

# No courier exceeds its load capacity
for c in COURIERS:    
    load_expr = Sum([If(b_path[c][j], size[j], 0) for j in NODES])
    optimizer.add(load_expr <= load[c])

# Exactly one item assignment to a courier
for j in NODES:
    optimizer.add(Sum([If(b_path[c][j], 1, 0) for c in COURIERS]) == 1)

# A courier cannot take more than max_nodes items
for c in COURIERS:
    optimizer.add(Sum([If(b_path[c][j], 1, 0) for j in NODES]) <= max_nodes)

# Couriers cannot visit same node twice (nor stay in the same node)
for c in COURIERS:
    optimizer.add(distinct_except([path[c][j] for j in range(1, max_nodes)], [0]))

'''
# All the nodes must be visited at least once (redundant)
for i in NODES:
    optimizer.add(Or([b_path[c][j] for c in COURIERS]))
'''
    
# Channeling: 
# - b_path -> path
for c in COURIERS:
    for i in NODES: # if b_bath[c][i] is true, then there must be true also at least one path[c][j] == i
        optimizer.add(Implies(b_path[c][i], Or([path[c][j] == i for j in range(1, max_nodes+1)])))
        # if b_bath[c][i] is false, then there won't be any path[c][j] == i
        optimizer.add(Implies(Not(b_path[c][i]), And([path[c][j] != i for j in range(1, max_nodes+1)])))

# - path -> b_path
for c in COURIERS:
    for j in range(max_nodes): # If node j is present in the path of courier c, set the corresponding item in b_path to True
        optimizer.add(Implies(path[c][j] != n+1, b_path[c][path[c][j]] == True))
        # If node j is not present in the path of courier c, set the corresponding item in b_path to False
        optimizer.add(Implies(path[c][j] == n+1, b_path[c][path[c][j]] == False))

# If you have more load size than me, then your load must be greater than mine
for c1 in COURIERS:
    for c2 in COURIERS:
        if c2 > c1:
            optimizer.add(If(load[c2] < load[c1], Sum([If(b_path[c1][j], size[j], 0) for j in NODES])
                        <= Sum([If(b_path[c2][j], size[j], 0) for j in NODES]), True))

# The items weight cannot exceed the load size (redundant? since channeling is there...)
for c in COURIERS:
    optimizer.add(Sum([If(b_path[c][j], size[j], 0) for j in NODES]) <= load[c])
    optimizer.add(Sum([If(j < path_length[c]-1, size[path[c][j]], 0) for j in range(2, max_nodes)]) <= load[c])
            
# Distance computation
for c in COURIERS:
    # Sum for distances between two non-zero nodes
    dist_expr = Sum([If(And(path[c][j] != 0, path[c][j+1] != 0), D_func(path[c][j]-1, path[c][j+1]-1), 0)
                    for j in range(1, max_nodes)])
    
    # Sum for distances when there's a zero node between two non-zero nodes
    dist_expr += Sum([If(And(path[c][j] == 0, path[c][j-1] != 0, path[c][j+1] != 0), D_func(path[c][j-1]-1, path[c][j+1]-1), 0)
                      for j in range(1, max_nodes)])
    
    optimizer.add(total_distance[c] == dist_expr)

'''
# No zeros in each path: IT DOES NOT WORK
for c in COURIERS:
    for i in range(2, max_nodes):
        optimizer.add(Implies(i < path_length[c]-1, path[c][i] != 0))
        optimizer.add(Implies(i < path_length[c]-1, path[c][i+1] != 0))
'''

# Symmetry breaking: two couriers with the same load size
for c1 in COURIERS:
    for c2 in COURIERS:
        if c1 < c2:  # Ensure c1 < c2 to avoid redundant comparisons
            # Add symmetry-breaking constraint if load sizes are equal
            sym_break_constraint = If(load[c1] == load[c2], lexleq([b_path[c1][j] for j in NODES], [b_path[c2][j] for j in NODES]), True)
            optimizer.add(sym_break_constraint)

# OPTIMIZATION OBJECTIVE - Minimize the maximum distance traveled by any courier
max_dist = Int('max_dist')
optimizer.add([max_dist >= total_distance[c] for c in COURIERS])
optimizer.minimize(max_dist)

# CHECK SATISFIABILITY
if optimizer.check() == sat:
    model = optimizer.model()

    # Print the path for each courier
    print("\nCouriers' paths")
    for c in COURIERS:
        path_values = [model.evaluate(path[c][j]) for j in range(1, model.eval(path_length[c]+1).as_long())]
        print(f'Courier {c}: {path_values}')
    
    '''
    # Print the item assignments in b_path
    for c in COURIERS:
        item_assignments = [model.evaluate(b_path[c][j]) for j in NODES]
        print(f'Courier {c} item assignments: {item_assignments}') 
    '''  

    #print("path length:", [model.eval(path_length[c]) for c in COURIERS]) 

    print("\nTotal distances")
    for c in COURIERS:
        print(f"Courier {c}: {model.eval(total_distance[c])}")
    
    print(f"\nMax distance: {model.eval(max_dist)}")

    # Record end time
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time} seconds")

else:
    print("unsat")