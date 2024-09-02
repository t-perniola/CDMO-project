import os
from minizinc import Instance, Model, Solver
import utils
from datetime import timedelta
import json
import time
from math import floor

def convert_to_dat_format(data_tuple):
    m, n, l, s, D, lb, ub = data_tuple
    return {
        'm': m,
        'n': n,
        'l': l,
        's': s,
        'D': D,
        'lb': lb,
        'ub': ub,
    }

def json_fun(instance_number, dist, paths, start_time, TIME_LIMIT=10):
    file_path = f'res/CP/{str(int(instance_number))}.json'
    json_dict = {}
    json_dict['time'] = int(floor(time.time() - start_time))
    json_dict['optimal'] = True if (time.time() - start_time < TIME_LIMIT) else False
    json_dict['obj'] = int(dist) if dist is not None else None
    json_dict['sol'] = paths

    # Check if the file already exists
    if os.path.exists(file_path):
        # Read the existing JSON data
        with open(file_path, 'r') as infile:
            existing_data = json.load(infile)
    else:
        # If the file does not exist, start with an empty dictionary
        existing_data = {}

    # Add the new entry to the existing data
    model_type = 'CP_chuffed'  # Assuming model_type to be 'CP', adjust as needed
    existing_data[model_type] = json_dict

    # Write the updated data back to the file
    with open(file_path, 'w') as outfile:
        json.dump(existing_data, outfile, indent=4)

def reorder_path(start, path):
    ordered_path = []
    current = start  # Start from the start node

    while path[current] != start+1:
        next_node = path[current]  # Adjust for 0-based indexing
        print(next_node)
        ordered_path.append(next_node)
        current = next_node-1
    
    return ordered_path


def CP_chuffed(instance_number):
    # Load the MiniZinc model
    model = Model()
    model.add_file("M12.mzn")

    # IMPORTING INSTANCE
    try:
        file_path = os.path.join('Instances', f'inst{instance_number}.dat')
        inst = utils.read_dat_file_2(file_path)
        print("Parsed data from file:", inst)
    except Exception as e:
        print(f"Error reading the instance file: {e}")
        return None
    
    # Convert data to the format MiniZinc expects
    instt = convert_to_dat_format(inst)

    # Create a MiniZinc solver instance
    gecode = Solver.lookup("chuffed")

    # Create a MiniZinc instance with the model and data
    instance = Instance(gecode, model)
    
    # Pass the data as arguments to the MiniZinc model
    for key, value in instt.items():
        instance[key] = value

    # Set the timeout
    timeout = timedelta(seconds=10)  # 10 seconds timeout

    # Start timing
    start_time = time.time()

    # Solve the model with the timeout
    try:
        result = instance.solve(timeout=timeout)
    except Exception as e:
        print(f"Error solving the model: {e}")
        return None

    # Output the results
    if result:

        # Check and print contents of result.solution
        if hasattr(result, 'solution'):
            solution = result.solution
            print("Solution attributes:")
            
            # Initialize variables for JSON output
            paths = []
            max_distance = None

            # Print all available attributes with their values
            for attr in dir(solution):
                if not attr.startswith('__'):
                    try:
                        value = getattr(solution, attr)
                        print(f"{attr}: {value}")
                    except Exception as e:
                        print(f"Could not access {attr}: {e}")

            try:
                x = getattr(solution, 'x', None)
                max_distance = getattr(solution, 'objective', None)

                if x is None or max_distance is None:
                    print("One or more expected result variables not found.")
                else:
                    print("Paths:")
                    for i in range(instt['m']):
                        start_node = instt['n']  # n+1 is represented as 'n' since indexing starts from 0n
                        path = reorder_path(start_node, x[i])  # Reorder path starting from n+1
                        paths.append(path)  # Store path for JSON
                        print(f"Courier {i + 1}: {' -> '.join(map(str, path))}")
                    print("Maximum distance:", max_distance)

            except AttributeError as e:
                print(f"Error accessing result attributes: {e}")
            
            # Save results to JSON
            json_fun(instance_number, max_distance, paths, start_time)

        else:
            print("Solution attribute not found.")
    else:
        print("No solution found")

def main(instance_number):
    CP_chuffed(instance_number)

if __name__ == "__main__":
    print(os.getcwd())
    instance_number = "05"  # Example instance number
    main(instance_number)
