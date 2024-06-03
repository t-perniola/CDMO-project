from z3 import *
from utils import *
import matplotlib.pyplot as plt
import networkx as nx
import time

# Record start time
start_time = time.time()

# IMPORTING INSTANCES
# Directory containing .dat files
directory = 'instances'

# Choose the instance
NUM_INST = 13

# Read all .dat files and populate instances
instances = read_all_dat_files(directory)
instance = instances[NUM_INST-1] # extract the corresp. instance
m = instance['m']
n = instance['n']
l = instance['l']
s = instance['s']
D = instance['D']

# DECLARING CONSTANTS
MAX_ITEMS = (n // m) + 3
Couriers = range(1, m+1)
Items = range(1, n+1)
print(f"Instance {NUM_INST} chosen \nnum Couriers: {m}, num items: {n}")

# DECLARING VARIABLES USING SMT SORTS
Z = IntSort()
B = BoolSort()
b_path = Array("b_path", Z, ArraySort(Z, B)) # create a boolean matrix
load = Array('load', Z, Z)
size = Array('size', Z, Z)
path = Array('path', Z, ArraySort(Z, Z)) # create an integer matrix
path_length = Array('path', Z, Z)
total_distance = Array('total_distance', Z, Z)
D_func = Function('D_func', Z, Z, Z) # takes two integer arguments (indices) and returns an integer (distance)

# INITIALIZE the SOLVER
solver = Solver()

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
# - distance matrix
for i in range(n+1):
    for j in range(n+1):
        solver.add(D_func(i, j) == D[i][j])
# items' sizes
for i in range(n):
    solver.add(size[i+1] == s[i])
# Couriers' load capacities
for c in range(m):
    solver.add(load[c+1] == l[c])

# Path should range between 0 and n+1        
for c in Couriers:
    for j in range(1, MAX_ITEMS+1):
        solver.add(And(path[c][j] >= 0, path[c][j] <= n+1))

# Boundaries of path_length's values
for c in Couriers:
    solver.add(And(path_length[c] >= 3, path_length[c] <= MAX_ITEMS))

# Define initial node and final node
for c in Couriers:
    solver.add(path[c][1] == n + 1) # Initial node
    solver.add(path[c][path_length[c]] == n + 1)  # Ending node

# Set unvisited Items to zero
for c in Couriers:
    for i in range(1, MAX_ITEMS + 1):
        solver.add(Implies(i > path_length[c], path[c][i] == 0))

# No courier exceeds its load capacity
for c in Couriers:    
    load_expr = Sum([If(b_path[c][j], size[j], 0) for j in Items])
    solver.add(load_expr <= load[c])

# Exactly one item assignment to a courier
for j in Items:
    solver.add(Sum([If(b_path[c][j], 1, 0) for c in Couriers]) == 1)

# A courier cannot take more than MAX_ITEMS items
for c in Couriers:
    solver.add(Sum([If(b_path[c][j], 1, 0) for j in Items]) <= MAX_ITEMS)

# Couriers cannot visit same node twice (nor stay in the same node)
for c in Couriers:
    solver.add(distinct_except([path[c][j] for j in range(1, MAX_ITEMS)], [0]))

# Channeling: 
# - b_path -> path
for c in Couriers:
    for i in Items: # if b_bath[c][i] is true, then there must be true also at least one path[c][j] == i
        solver.add(Implies(b_path[c][i], Or([path[c][j] == i for j in range(1, MAX_ITEMS+1)])))
        # if b_bath[c][i] is false, then there won't be any path[c][j] == i
        solver.add(Implies(Not(b_path[c][i]), And([path[c][j] != i for j in range(1, MAX_ITEMS+1)])))

# - path -> b_path
for c in Couriers:
    for j in range(MAX_ITEMS): # If node j is present in the path of courier c, set the corresponding item in b_path to True
        solver.add(Implies(path[c][j] != n+1, b_path[c][path[c][j]] == True))
        # If node j is not present in the path of courier c, set the corresponding item in b_path to False
        solver.add(Implies(path[c][j] == n+1, b_path[c][path[c][j]] == False))

# If you have more load size than me, then your load must be greater than mine
for c1 in Couriers:
    for c2 in Couriers:
        if c2 > c1:
            solver.add(If(load[c2] < load[c1], Sum([If(b_path[c1][j], size[j], 0) for j in Items])
                        <= Sum([If(b_path[c2][j], size[j], 0) for j in Items]), True))

# The items weight cannot exceed the load size (NOTE: redundant? since channeling is there...)
for c in Couriers:
    solver.add(Sum([If(b_path[c][j], size[j], 0) for j in Items]) <= load[c])
    solver.add(Sum([If(j < path_length[c]-1, size[path[c][j]], 0) for j in range(2, MAX_ITEMS)]) <= load[c])

# Distance computation
for c in Couriers:
    # Sum for distances between two non-zero Items
    dist_expr = Sum([If(And(path[c][j] != 0, path[c][j+1] != 0), D_func(path[c][j]-1, path[c][j+1]-1), 0)
                    for j in range(1, MAX_ITEMS)])
    
    # Sum for distances when there's a zero node between two non-zero items
    dist_expr += Sum([If(And(path[c][j] == 0, path[c][j-1] != 0, path[c][j+1] != 0), D_func(path[c][j-1]-1, path[c][j+1]-1), 0)
                      for j in range(1, MAX_ITEMS)])
    
    solver.add(total_distance[c] == dist_expr)

# Symmetry breaking: two couriers with the same load size
for c1 in Couriers:
    for c2 in Couriers:
        if c1 < c2:  # Ensure c1 < c2 to avoid redundant comparisons
            # Add symmetry-breaking constraint if load sizes are equal
            sym_break_constraint = If(load[c1] == load[c2], lexleq([b_path[c1][j] for j in Items], [b_path[c2][j] for j in Items]), True)
            solver.add(sym_break_constraint)

# Initialize the variable to store the current best maximum distance
current_best_max_dist = None

while True:
    solver.push()

    # Check satisfiability
    if solver.check() == sat:
        model = solver.model()

        # Print the path for each courier (NOTE: without zeros)
        print("\nCouriers' paths")
        paths = {}
        for c in Couriers:
            path_length_c = model.eval(path_length[c]).as_long()
            path_values = []
            for j in range(1, path_length_c + 1):
                evaluated_value = model.evaluate(path[c][j]).as_long()
                if evaluated_value != 0:
                    path_values.append(evaluated_value)
            paths[c] = path_values
            print(f'Courier {c}: {path_values}')

        print("\nTotal distances")
        for c in Couriers:
            print(f"Courier {c}: {model.eval(total_distance[c])}")

        # Evaluate and find the current maximum distance
        current_max_dist = max(model.eval(total_distance[c]).as_long() for c in Couriers)
        print(f"\nCurrent max distance: {current_max_dist}")

        if current_best_max_dist is None or current_max_dist < current_best_max_dist:
            current_best_max_dist = current_max_dist

        # Add a constraint to find a better solution in the next iteration
        solver.add(Or([total_distance[c] < current_best_max_dist for c in Couriers]))
        
        solver.pop()
    else:
        solver.pop()
        break

# Record end time
end_time = time.time()
# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"\nElapsed time: {elapsed_time} seconds")

# Plot the paths using networkx
G = nx.Graph()

# Add nodes
for i in range(1, n + 2):
    G.add_node(i)

# Add edges for each courier's path
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors if needed
for c in Couriers:
    path_values = paths[c]
    edges = [(path_values[i], path_values[i + 1]) for i in range(len(path_values) - 1)]
    edges.append((path_values[-1], path_values[0]))  # To complete the loop
    G.add_edges_from(edges, color=colors[c % len(colors)], weight=2)

# Get edges and colors
edges = G.edges()
edge_colors = [G[u][v]['color'] for u, v in edges]
edge_weights = [G[u][v]['weight'] for u, v in edges]

# Draw the graph
pos = nx.circular_layout(G)
edges = [(u, v) for u, v in edges if u != v]
nx.draw(G, pos, edgelist=edges, edge_color=edge_colors, width=edge_weights, with_labels=True, node_size=500, node_color='lightblue')

plt.title('Paths taken by Couriers')
plt.show()
