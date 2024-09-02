import os

def read_dat_file(file_path):
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

    return m, n, l, s, D, lb, ub

def write_dzn_file(file_path, m, n, l, s, D, lb, ub):
    with open(file_path, 'w') as file:
        file.write(f'm = {m};\n')
        file.write(f'n = {n};\n')
        file.write(f'l = {l};\n')
        file.write(f's = {s};\n')
        file.write('D = [|\n')
        for row in D:
            file.write('     | ' + ', '.join(map(str, row)) + '\n')
        file.write('     |];\n')
        file.write(f'lb = {lb};\n')
        file.write(f'ub = {ub};\n')

def convert_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.dat'):
            dat_file_path = os.path.join(input_dir, file_name)
            dzn_file_path = os.path.join(output_dir, file_name.replace('.dat', '.dzn'))

            m, n, l, s, D, lb, ub = read_dat_file(dat_file_path)
            write_dzn_file(dzn_file_path, m, n, l, s, D, lb, ub)

def main():
    input_dir = 'Instances'  # Directory containing .dat files
    output_dir = 'Instances2'  # Directory to save .dzn files

    convert_files(input_dir, output_dir)

if __name__ == "__main__":
    path=os.path.join(os.getcwd(), 'Desktop/UNIBO AI/Combinatorial and DecisionMaking/ProjectWork')
    print(path)
    os.chdir(path)
    main()