import os
from minizinc import Instance, Model, Solver
import json
import time
from datetime import timedelta
from math import floor

def json_fun(instance_number, dist, paths, start_time, TIME_LIMIT, symm_break, chuffed):
    file_path = f'res/CP/{str(int(instance_number))}.json'
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    end_time = int(floor(time.time() - start_time))

    json_dict = {
        "time": TIME_LIMIT if end_time > TIME_LIMIT else end_time,
        "optimal": (time.time() - start_time) < TIME_LIMIT,
        "obj": int(dist) if dist is not None else None,
        "sol": paths
    }

    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as infile:
            existing_data = json.load(infile)
    else:
        existing_data = {}

    # Update the data
    model_type = "CP_gecode_noSB"
    if symm_break:
        model_type = "CP_gecode_SB"
    if chuffed:
        model_type = "CP_chuffed_SB"
    
    existing_data[model_type] = json_dict

    # Write updated data to file
    with open(file_path, 'w') as outfile:
        json.dump(existing_data, outfile, indent=4)

def reorder_path(start, path):
    ordered_path = []
    current = start

    while path[current] != start + 1:
        next_node = path[current]
        ordered_path.append(next_node)
        current = next_node - 1
    
    return ordered_path

def CP(instance_number, symm_break=True, chuffed=False):
    model_def = f"\nRunning CP model on instance {instance_number} with "

    # Select the appropriate model
    if chuffed:
        model_name = "models\CP\M12.mzn"
        model_def += "Chuffed as solver and SB constraints"
    elif symm_break:
        model_name = "models\CP\M11.mzn"
        model_def += "Gecode as solver and SB constraints"
    else:
        model_name = "models\CP\M10.mzn"
        model_def += "Gecode as solver without SB constraints"

    print(model_def+":")

    # Load the selected MiniZinc model
    model = Model(model_name)

    # Define the path to the `.dzn` file
    data_file = os.path.join("instances\Instances (.dzn)", f"inst{instance_number}.dzn")

    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found!")
        return None

    # Select the solver
    solver_name = "chuffed" if chuffed else "gecode"
    solver = Solver.lookup(solver_name)
    
    # Create a MiniZinc instance
    instance = Instance(solver, model)

    # Load the `.dzn` file as input data
    instance.add_file(data_file)

    # Set timeout
    timeout = timedelta(seconds=300)

    # Start timing
    start_time = time.time()

    # Solve the model
    try:
        result = instance.solve(timeout=timeout)
    except Exception as e:
        print(f"Error solving the model: {e}")
        return None

    # Output the results
    if result:
        if hasattr(result, "solution"):
            solution = result.solution
            #print("Solution attributes:")

            paths = []
            max_distance = None

            try:
                x = getattr(solution, "x", None)
                max_distance = getattr(solution, "objective", None)

                if x is None or max_distance is None:
                    print("One or more expected result variables not found.")
                else:
                    print("Objective value (max dist): {}\n".format(max_distance))

            except AttributeError as e:
                print(f"Error accessing result attributes: {e}")

            # Save results to JSON
            json_fun(instance_number, max_distance, paths, start_time, TIME_LIMIT=300, symm_break=symm_break, chuffed=chuffed)

        else:
            print("Solution attribute not found.")
    else:
        print("No solution found")
