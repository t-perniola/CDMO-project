import os

def read_dat_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    m = int(lines[0].strip())
    n = int(lines[1].strip())
    l = list(map(int, lines[2].strip().split()))
    s = list(map(int, lines[3].strip().split()))
    D = [list(map(int, line.strip().split())) for line in lines[4:]]

    # Trova maxD e minD ignorando la diagonale
    values = [D[i][j] for i in range(len(D)) for j in range(len(D[i])) if i != j]
    maxD = max(values)
    minD = min(values)

    # Calcolo dei limiti
    heuristic_number_of_nodes_per_courier = n // m + 3
    lb = heuristic_number_of_nodes_per_courier * minD
    ub = heuristic_number_of_nodes_per_courier * maxD

    return m, n, l, s, D, lb, ub, lines  # Restituisce anche le linee originali


def append_lb_ub_if_needed(file_path, lb, ub, original_lines):
    # Controlliamo se lb e ub sono già presenti alla fine del file
    if len(original_lines) >= 2 and original_lines[-2].strip().isdigit() and original_lines[-1].strip().isdigit():
        existing_lb = int(original_lines[-2].strip())
        existing_ub = int(original_lines[-1].strip())

        if existing_lb == lb and existing_ub == ub:
            return  # Se i valori sono già presenti, non facciamo nulla

    # Se lb e ub non sono presenti, li aggiungiamo in fondo
    with open(file_path, 'a') as file:
        file.write(f"{lb}\n")
        file.write(f"{ub}\n")


def convert_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        dat_file_path = os.path.join(input_dir, file_name)
        new_dat_file_path = os.path.join(output_dir, file_name)

        result = read_dat_file(dat_file_path)
        if result:
            m, n, l, s, D, lb, ub, original_lines = result
            if os.path.exists(new_dat_file_path):
                append_lb_ub_if_needed(new_dat_file_path, lb, ub, original_lines)
            else:
                with open(new_dat_file_path, 'w') as file:
                    file.writelines(original_lines)  # Scriviamo il file originale
                    file.write(f"{lb}\n")
                    file.write(f"{ub}\n")


def main():
    base_path = '/Users/armyy/Desktop/UNIBO AI/anno1/Combinatorial and DecisionMaking/ProjectWork'
    input_dir = os.path.join(base_path, 'Instances')
    output_dir = os.path.join(base_path, 'SAT', 'Instances')

    convert_files(input_dir, output_dir)


if __name__ == "__main__":
    path = '/Users/armyy/Desktop/UNIBO AI/anno1/Combinatorial and DecisionMaking/ProjectWork'
    os.chdir(path)
    main()
