import os

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

def read_all_dat_files(directory):
    instances = []
    for filename in os.listdir(directory):
        if filename.endswith('.dat'):
            file_path = os.path.join(directory, filename)
            instance = read_dat_file(file_path)
            instances.append(instance)
    return instances
