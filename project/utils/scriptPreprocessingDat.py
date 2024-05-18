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

    return m, n, l, s, D

def write_dzn_file(file_path, m, n, l, s, D):
    with open(file_path, 'w') as file:
        file.write(f'm = {m};\n')
        file.write(f'n = {n};\n')
        file.write(f'l = {l};\n')
        file.write(f's = {s};\n')
        file.write('D = [|\n')
        for row in D:
            file.write('     | ' + ', '.join(map(str, row)) + '\n')
        file.write('     |];\n')

def convert_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.dat'):
            dat_file_path = os.path.join(input_dir, file_name)
            dzn_file_path = os.path.join(output_dir, file_name.replace('.dat', '.dzn'))

            m, n, l, s, D = read_dat_file(dat_file_path)
            write_dzn_file(dzn_file_path, m, n, l, s, D)

def main():
    input_dir = 'instances'  # Directory containing .dat files
    output_dir = 'instances2'  # Directory to save .dzn files

    convert_files(input_dir, output_dir)

if __name__ == "__main__":
    main()
