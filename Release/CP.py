import os
from minizinc import Instance, Model, Solver
import utils
from datetime import timedelta
import json

def convert_to_dat_format(data_tuple):
    m, n, l, s, D, lb, ub = data_tuple
    return {
        'm': m,
        'n': n,
        'l': l,
        's': s,
        'D': D,
        'lb': lb,
        'ub': ub
    }


def write_results_to_json(instance_number, solution, max_dist, time_taken):
    TIME_LIMIT = 300

    # Prepare the paths list based on the solution
    paths = []
    for courier_path in solution:
        path = [item for item in courier_path]
        paths.append(path)

    # Prepare JSON dictionary
    json_dict = {}
    json_dict['SAT'] = {}
    json_dict['SAT']['time'] = int(time_taken)
    json_dict['SAT']['optimal'] = True if (time_taken < TIME_LIMIT and opt_values[instance_number] == max_dist) else False
    json_dict['SAT']['obj'] = max_dist if max_dist is not None else None
    json_dict['SAT']['sol'] = paths

    # Write JSON to file
    with open(f'res/SAT/{str(int(instance_number))}.json', 'w') as outfile:
        json.dump(json_dict, outfile)

def CP(instance_number):
    # Load the MiniZinc model
    model = Model()
    model.add_file("M11.mzn")

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
    gecode = Solver.lookup("gecode")

    # Create a MiniZinc instance with the model and data
    instance = Instance(gecode, model)
    
    # Pass the data as arguments to the MiniZinc model
    for key, value in instt.items():
        instance[key] = value

    # Set the timeout
    timeout = timedelta(seconds=10)  # 10 seconds timeout

    # Solve the model with the timeout
    try:
        result = instance.solve(timeout=timeout)
    except Exception as e:
        print(f"Error solving the model: {e}")
        return None

    # Output the results
    if result:
        # Print available result attributes for debugging
        print("Available result attributes:")
        print(dir(result))  # List all attributes

        # Check and print contents of result.solution
        if hasattr(result, 'solution'):
            solution = result.solution
            print("Solution attributes:")
            print(dir(solution))  # List attributes of solution
            
            # Print all available attributes with their values
            for attr in dir(solution):
                if not attr.startswith('__'):
                    try:
                        value = getattr(solution, attr)
                        print(f"{attr}: {value}")
                    except Exception as e:
                        print(f"Could not access {attr}: {e}")

            # Try to access expected result variables
            try:
                # Example for accessing potential attributes (adjust as necessary)
                x = getattr(solution, 'x', None)
                max_distance = getattr(solution, 'max_distance', None)

                if x is None or max_distance is None:
                    print("One or more expected result variables not found.")
                else:
                    print("Paths:")
                    for i in range(instt['m']):
                        path = " -> ".join(map(str, x[i]))
                        print(f"Courier {i + 1}: {path}")
                    print("Maximum distance:", max_distance)

            except AttributeError as e:
                print(f"Error accessing result attributes: {e}")
        else:
            print("Solution attribute not found.")
    else:
        print("No solution found")


def main(instance_number):
    CP(instance_number)

if __name__ == "__main__":
    path = os.path.join(os.getcwd(), 'Desktop/UNIBO AI/Combinatorial and DecisionMaking/ProjectWork/Release')
    print(path)
    os.chdir(path)
    instance_number = "01"  # Example instance number
    main(instance_number)




    json_dict['time'] = int(floor(time.time() - start_time))
    json_dict['optimal'] = True if (time.time() - start_time < TIME_LIMIT) else False
    json_dict['obj'] = int(current_best_max_dist) if current_best_max_dist is not None else None
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
    existing_data[model_type] = json_dict

    # Write the updated data back to the file
    with open(file_path, 'w') as outfile:
        json.dump(existing_data, outfile, indent=4)

    break