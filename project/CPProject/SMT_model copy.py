from z3 import *

# THEORIES USED
# - Boolean Logic
# - Arithmetic
# - EUF (uninterpreted functions)
# - Arrays

# TODO dynamic data initializations

# INSTANCE 5
'''
m = 2
n = 3
l = [18, 30]
s = [20, 17, 6]
D = [[0, 21, 86, 99],
     [21, 0, 71, 80], 
     [92, 71, 0, 61], 
     [59, 80, 61, 0]]

# INSTANCE 2
m = 6
n = 9
l = [190, 185, 185, 190, 195, 185]
s = [11, 11, 23, 16, 2, 1, 24, 14, 20]
D = [[0, 199, 119, 28, 179, 77, 145, 61, 123, 87],
    [199, 0, 81, 206, 38, 122, 55, 138, 76, 113],
    [119, 81, 0, 126, 69, 121, 26, 117, 91, 32],
    [28, 206, 126, 0, 186, 84, 152, 68, 130, 94],
    [169, 38, 79, 176, 0, 92, 58, 108, 46, 98],
    [77, 122, 121, 84, 102, 0, 100, 16, 46, 96],
    [145, 55, 26, 152, 58, 100, 0, 91, 70, 58],
    [61, 138, 113, 68, 118, 16, 91, 0, 62, 87],
    [123, 76, 91, 130, 56, 46, 70, 62, 0, 66],
    [87, 113, 32, 94, 94, 96, 58, 87, 66, 0]]'''

# INSTANCE 3
m = 3
n = 7
l = [15, 10, 7]
s = [3, 2, 6, 8, 5, 4, 4]
D = [[0, 3, 3, 6, 5, 6, 6, 2],
    [3, 0, 6, 3, 4, 7, 7, 3],
    [3, 4, 0, 7, 6, 3, 5, 3],
    [6, 3, 7, 0, 5, 6, 7, 4],
    [5, 4, 6, 3, 0, 3, 3, 3],
    [6, 7, 3, 6, 3, 0, 2, 4],
    [6, 7, 5, 6, 3, 2, 0, 4],
    [2, 3, 3, 4, 3, 4, 4, 0]]

max_nodes = (n // m) + 3
COURIERS = range(m)
NODES = range(n)

print("\nmax_nodes:", max_nodes)

# VARIABLES
# creates m x n variables that tell if the item j is assigned to courier i
Z = IntSort()

b_path = Array('b_path', Z, BoolSort()) # is an array with int indices and boolean elements
load = Array('load', Z, Z)
path = Array('path', Z, Z)
indices = Array('ind', Z, Z)
total_distance = Array('total_distance', Z, Z)
D_func = Function('D_func', Z, Z, Z) # takes two integer arguments (indices) and returns an integer (distance)

# OPTIMIZING optimizer INIT
optimizer = Optimize()

# HELPER function
# - all_different
def distinct_except(values, forbidden_values):
    non_forbidden_values = [v for v in values if v not in forbidden_values]
    return Distinct(non_forbidden_values) # enforcing uniqueness

# CONSTRAINTS
# Path should range between 0 and n+1
for i in range(1, n+m+2):
    optimizer.add(And(Select(path, i) >= 1, Select(path, i) <= n + 1))

for i in range(m+1):
    optimizer.add(And(Select(indices, i) >= 1, Select(indices, i) <= m+n+1))

optimizer.add(Select(path, ))
        
# No courier exceeds its load capacity
for c in COURIERS:
    load_expr = Sum([If(Select(b_path, c * n + i), s[i], 0) for i in NODES])
    optimizer.add(load_expr <= l[c])
    optimizer.add(Select(load, c) == load_expr)

# A courier cannot take more than max_nodes items
for c in COURIERS:
    optimizer.add(Sum([If(Select(b_path, c * n + i), 1, 0) for i in NODES]) <= max_nodes)

# Couriers cannot visit same node twice (nor stay in the same node)
for c in COURIERS:
    optimizer.add(distinct_except([Select(path, c * max_nodes + j) for j in range(1, max_nodes)], [0]))

# Define initial node and final node
for c in COURIERS:
    optimizer.add(Select(path, c * max_nodes) == n + 1) # Initial node
    optimizer.add(Select(path, c * max_nodes - 1) == n + 1)  # Ending node (TODO correct indexing?)

# Mapping distance matrix to D_func
for i in range(n+1):
    for j in range(n+1):
        optimizer.add(D_func(i, j) == D[i][j])

# Channeling: 
# - b_path -> path
for c in COURIERS:
    for i in NODES: # if b_bath[c][i] is true, then there must be true also at least one path[c][j] == i
        optimizer.add(Implies(Select(b_path, c*n+i), Or([Select(path, c*max_nodes+j) == i for j in range(max_nodes)])))
        # if b_bath[c][i] is false, then there won't be any path[c][j] == i
        optimizer.add(Implies(Not(Select(b_path, c *n+i)), And([Select(path, c * max_nodes + j) != i for j in range(max_nodes)])))

# - path -> b_path
for c in COURIERS:
    for j in range(max_nodes): # If node j is present in the path of courier c, set the corresponding item in b_path to True
        optimizer.add(Implies(Select(path, c*max_nodes+j) != n+1, Select(b_path, c*n + Select(path, c*max_nodes+j)) == True))
        # If node j is not present in the path of courier c, set the corresponding item in b_path to False
        optimizer.add(Implies(Select(path, c * max_nodes + j) == n + 1, Select(b_path, c * n + Select(path, c * max_nodes + j)) == False))

# If you have more load size than me, then your load must be greater than mine
for c1 in COURIERS:
    for c2 in COURIERS:
        if c2 > c1 and l[c2] > l[c1]:
            optimizer.add(Sum([If(Select(b_path, c1 * n + j), s[j], 0) for j in NODES])
                        <= Sum([If(Select(b_path, c2 * n + j), s[j], 0) for j in NODES]))

# Exactly one item assignment to a courier
for i in NODES:
    optimizer.add(Sum([If(Select(b_path, c * n + i), 1, 0) for c in COURIERS]) == 1)

'''
# Distance computation
for c in COURIERS:
    dist_expr = Sum([And(If(Select(path, c*max_nodes+j) != 0), 
        D_func(Select(path, c * max_nodes + j) - 1, Select(path, c * max_nodes + j + 1) - 1), 0)
        for j in range(max_nodes - 1)])
    optimizer.add(Select(total_distance, c) == dist_expr)
'''

for c in COURIERS:
    dist_expr = Sum([If(And(Select(path, c * max_nodes + j) != 0, 
                             Select(path, c * max_nodes + j + 1) != 0), 
                        D_func(Select(path, c * max_nodes + j)-1, Select(path, c * max_nodes + j + 1)-1), 0) 
                    for j in range(max_nodes - 1)])
    optimizer.add(Select(total_distance, c) == dist_expr)

# Optimization objective - Minimize the maximum distance traveled by any courier
max_dist = Int('max_dist')
optimizer.add([max_dist >= Select(total_distance, c) for c in COURIERS])
optimizer.minimize(max_dist)

# Solve the optimization problem
if optimizer.check() == sat:
    model = optimizer.model()

    # Print the values of the distance function D_func
    print("Distance function D_func:")
    for i in range(n+1):
        for j in range(n+1):
            distance = model.eval(D_func(i, j))
            print(f"D_func({i}, {j}) = {distance}")
   
    # Print distances for each segment of the path and total distance for each courier
    for c in COURIERS:
        total_distance_courier = 0
        print(f"\nDistances for Courier {c}:")
        j = 0
        while j < max_nodes - 1:
            index1 = c * max_nodes + j
            index2 = c * max_nodes + j + 1
            path_index1 = model.eval(Select(path, index1)).as_long() - 1
            path_index2 = model.eval(Select(path, index2)).as_long() - 1
            
            # Skip zero values in the path
            if path_index1 == -1:
                j += 1
                continue

            # Handle zero values in the path
            if path_index2 == -1:
                path_index2 = path_index1
                while j < max_nodes - 1:
                    j += 1
                    index2 = c * max_nodes + j + 1
                    path_index2 = model.eval(Select(path, index2)).as_long() - 1
                    if path_index2 != -1:
                        break
                if path_index2 == -1:
                    break

            # Calculate distance
            distance = model.eval(D_func(path_index1, path_index2))
            total_distance_courier += distance
            print(f"Segment {j}: Nodes {path_index1 + 1} to {path_index2 + 1}, Distance: {distance}")
            j += 1

        print(f"Total distance for Courier {c}: {model.eval(total_distance_courier)}")

    # Print whole array path 
    print("\nwhole path array:", [model.eval(Select(path, c)) for c in range(2*n)])

    # Print paths for each courier
    print("\nPaths:")
    for c in COURIERS:
        path_list = []
        for j in range(max_nodes):
            node = model.eval(Select(path, c * max_nodes + j))
            path_list.append(node.as_long())
        print(f"Courier {c}: {path_list}")

    # Print b_path assignments for each courier
    print("\nItem assignments:")
    for c in COURIERS:
        b_path_list = []
        for i in NODES:
            assigned = model.eval(Select(b_path, c * n + i))
            b_path_list.append(assigned)
        print(f"Courier {c}: {b_path_list}")

    print("\nTotal distances:")
    for c in COURIERS:
        print(f"Courier {c}: {model.eval(total_distance[c])}")
    
    print(f"\nMax distance: {model.eval(max_dist)}")
else:
    print("No solution found")
