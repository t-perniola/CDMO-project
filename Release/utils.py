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

    # Read l array
    l = list(map(int, lines[2].strip().split()))

    # Read s array
    s = list(map(int, lines[3].strip().split()))

    # Read D matrix
    D = []
    for i in range(4, len(lines)):
        D.append(list(map(int, lines[i].strip().split())))
    
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


