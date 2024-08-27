from SMT_solver import SMT_model
from time import time as clock_time
import os
import numpy as np

def run_SMT(solver_type, path, pivot, lowerbound, m, time):
    smt_lib = ""
    try:
        with open(os.path.join(path), "r") as f:
            smt_lib = f.read()
            smt_lib = smt_lib.split("\n")
    except IOError as e:
            print(f"An error occurred: {e}")

    # Add the necessary assertions based on the provided pivot and lowerbound
    smt_lib.append(f"(assert (<= max_dist {pivot}))")
    smt_lib.append(f"(assert (>= max_dist {lowerbound}))")
    smt_lib.append("(check-sat)")
    
    # Add queries to get the values for the paths taken by the couriers
    for i in range(1, m + 1):
        for j in range(1, time + 1):
            smt_lib.append(f"(get-value ((select (select path {i}) {j})))")
    
    smt_lib = "\n".join(smt_lib)

    try:
        with open(os.path.join(path), "w") as f:
            f.write(smt_lib)
    except IOError as e:
            print(f"An error occurred: {e}")

    result = os.popen(f"{solver_type} {path}").read()
    print(result)
    result = result.split("\n")
    return result

def format_smtlib_output(instance, result):
    try:
        m = instance["m"]
        time = instance["time"]
        result = result[1:]  # Skip the "sat" or "unsat" line
        # Remove extra parentheses and extract values
        result = [x.replace("(", "").replace(")", "").split()[1] for x in result if x]
        courier_result = np.array(result, dtype=int)
        courier_result = np.reshape(courier_result, (m, time)).tolist()
        return courier_result
    except Exception as e:
        print(f"An error occurred while formatting output: {e}")
        return None

def run_binary_search(add_intermediate_result, file_path, min_dist, max_dist, solver, m, time, instance):
    solver_type = {
        "z3": "z3",
        "cvc4": "cvc4 --lang smt --produce-models --incremental",
        "cvc5": "cvc5 --produce-models --incremental"
    }.get(solver, None)
    
    if solver_type is None:
        raise Exception("Solver not supported")

    last_result = None
    lower = min_dist
    upper = max_dist
    pivot = None
    is_first = True

    while lower <= upper:
        pivot = (upper + lower) // 2
        try:
            with open(os.path.join(file_path), "r") as f:
                smt2 = f.read()
                smt2 = smt2.split("\n")
            if not is_first:
                smt2 = smt2[:-(m*time+3)]
            smt2 = "\n".join(smt2)
        except IOError as e:
            print(f"An error occurred: {e}")
        
        try:
            with open(os.path.join(file_path), "w") as f:
                f.write(smt2)
        except IOError as e:
            print(f"An error occurred: {e}")

        result = run_SMT(solver_type, file_path, pivot, lower, m, time)
        is_first = False

        if result[0] == "unsat":
            lower = pivot + 1
        else:
            formatted_result = format_smtlib_output(instance, result)
            if formatted_result is None:
                return None
            add_intermediate_result(formatted_result)
            last_result = formatted_result
            upper = pivot - 1

    return last_result

import re
def disambiguate_smtlib(smtlib_script):
    """
    Automatically add sort annotations to ambiguous references in the SMT-LIB script.
    This is tailored for cases where 'path' is an array sort of integers.
    :param smtlib_script: The original SMT-LIB script as a string.
    :return: The disambiguated SMT-LIB script.
    """
    # Replace occurrences of `path` with its qualified expression
    # This assumes `path` always refers to the array sort of integers
    return re.sub(r'\bpath\b', '(as path (Array Int (Array Int Int)))', smtlib_script)


def run_smt_lib(add_intermediate_result, instance, timeout, instance_number, solver):
    try:
        generation_start_time = clock_time()
        instance_number_string = f"0{instance_number}"
        model = SMT_model(instance_number_string, timeout)
        smt2 = model.sexpr()
        smt2 = disambiguate_smtlib(smt2)

        file_path = f"project/SMT/SMT_instances/smt2_{instance_number}.smt2"

        try:
            with open(os.path.join(file_path), "w") as f:
                f.write("(set-logic ALL)\n")
                smt2 = smt2.split("\n")[:-2]  # Removing (check-sat) and last empty line
                smt2 = "\n".join(smt2)
                f.write(smt2)
                f.write("\n")
        

        except IOError as e:
            print(f"An error occurred: {e}")

        generation_time = clock_time() - generation_start_time
        residual_time = timeout - generation_time

        running_start_time = clock_time()
        m = instance['m']
        time = instance['time']
        min_dist = instance['min_dist']
        max_dist = instance['max_dist']
        result = run_binary_search(add_intermediate_result, file_path, min_dist, max_dist, solver, m, time + 1, instance)

        elapsed = clock_time() - running_start_time

        if result is None:
            return None
        elif result == "unsat":
            return "unsat"
        else:
            return result, int(elapsed) < residual_time
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Sample instance for testing
instance = {
    "m": 2,  # Number of couriers
    "time": 3,  # Number of time slots
    "min_dist": 18,  # Minimum distance
    "max_dist": 30,  # Maximum distance
    "time_constraints": [20, 17, 6],  # Time constraints
    "distance_matrix": [
        [0, 21, 86, 99],
        [21, 0, 71, 80],
        [92, 71, 0, 61],
        [59, 80, 61, 0]
    ]  # Distance matrix
}

def add_intermediate_result(result):
    print("Intermediate Result:", result)

# Define solver and other parameters
solver_type = "z3"  # or "cvc4" or "cvc5"
timeout = 60  # Set your timeout here
instance_number = 5

result = run_smt_lib(add_intermediate_result, instance, timeout, instance_number, solver_type)
print("Final Result:", result)
