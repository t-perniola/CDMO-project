import os
import matplotlib.pyplot as plt
import networkx as nx

def read_dat_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Read m and n
    m = int(lines[0].strip())
    n = int(lines[1].strip())
    
    # Read l vector
    l = list(map(int, lines[2].strip().split()))
    
    # Read s vector
    s = list(map(int, lines[3].strip().split()))
    
    # Read D matrix
    D = []
    for line in lines[4:]:
        D.append(list(map(int, line.strip().split())))
    
    return {
        'm': m,
        'n': n,
        'l': l,
        's': s,
        'D': D
    }

def read_dat_file_2(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Read m and n
    m = int(lines[0].strip())
    n = int(lines[1].strip())
    
    # Read l vector
    l = list(map(int, lines[2].strip().split()))
    
    # Read s vector
    s = list(map(int, lines[3].strip().split()))
    
    # Read D matrix
    D = []
    for line in lines[4:]:
        D.append(list(map(int, line.strip().split())))
    
    maxD = float('-inf')
    minD = float('inf')

    for i in range(len(D)):
        for j in range(len(D[i])):
            if i != j:  # Exclude diagonal elements
                if D[i][j] > maxD:
                    maxD = D[i][j]
                if D[i][j] < minD:
                    minD = D[i][j]
    
    heuristic_number_of_nodes_per_courier = n//m +3

    lb = heuristic_number_of_nodes_per_courier * minD
    ub = heuristic_number_of_nodes_per_courier * maxD

    return {
        'm': m,
        'n': n,
        'l': l,
        's': s,
        'D': D,
        'lb': lb,
        'ub': ub
    }


def read_all_dat_files(directory):
    instances = []
    for filename in os.listdir(directory):
        if filename.endswith('.dat'):
            file_path = os.path.join(directory, filename)
            instance = read_dat_file(file_path)
            instances.append(instance)
    return instances

# Draw graph of each courier's path
def draw_graph(num_items, Couriers, paths):
    # Plot the paths using networkx
    G = nx.Graph()
    
    # Add nodes
    for i in range(1, num_items + 2):
        G.add_node(i)
    
    # Add edges for each courier's path
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for c in Couriers:
        path_values = paths[c]
        for i in range(len(path_values) - 1):
            G.add_edge(path_values[i], path_values[i + 1], color=colors[c % len(colors)], weight=2)

    # Get edges and colors
    edges = G.edges()
    edge_colors = [G[u][v]['color'] for u, v in edges]
    edge_weights = [G[u][v]['weight'] for u, v in edges]

    # Draw the graph
    pos = nx.spring_layout(G)

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos)

    # Draw directed edges with arrows following path order
    for c in Couriers:
        path_values = paths[c]
        path_edges = [(path_values[i], path_values[i + 1]) for i in range(len(path_values) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=colors[c % len(colors)], width=2, arrows=True, arrowstyle='-|>', arrowsize=20)

    plt.title('Couriers paths')
    plt.show()
