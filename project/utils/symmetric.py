import numpy as np
import os

def read_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = []
        is_matrix = False
        for line in lines:
            if line.startswith('D'):
                is_matrix = True
                continue
            if is_matrix:
                if '|];' in line:
                    row = list(map(int, filter(None, line.strip(' |];\n').split(','))))
                    matrix.append(row)
                    break
                elif '|' in line:
                    row = list(map(int, filter(None, line.strip(' |\n').split(','))))
                    matrix.append(row)
        
        # Debugging: Print the matrix to check row lengths
        for i, row in enumerate(matrix):
            print(f"Row {i} length {len(row)}: {row}")

        # Ensure all rows have the same length
        row_lengths = [len(row) for row in matrix]
        if len(set(row_lengths)) != 1:
            raise ValueError(f"Inconsistent row lengths found in the matrix: {row_lengths}")

        return np.array(matrix), lines

def check_symmetric(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        return False
    return np.array_equal(matrix, matrix.T)

def process_files(pre_dir, post_dir):
    for filename in os.listdir(pre_dir):
        if filename.endswith('.dat'):
            file_path = os.path.join(pre_dir, filename)
            matrix, lines = read_matrix_from_file(file_path)
            symmetry_flag = 1 if check_symmetric(matrix) else 0
            
            with open(os.path.join(post_dir, filename), 'w') as file:
                for line in lines:
                    file.write(line)
                file.write(f"symmetry = {symmetry_flag}\n")

# Define the directories
pre_dir = 'pre'
post_dir = 'post'

# Create the post directory if it does not exist
os.makedirs(post_dir, exist_ok=True)

# Process the files
process_files(pre_dir,post_dir)
