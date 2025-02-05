from itertools import combinations
from z3 import Bool, BoolVal, Xor, Or, And, Not
import math
import re
from collections import defaultdict

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

    lb = 1 if lb == 0 else lb

    return {
        'm': m,
        'n': n,
        'l': l,
        's': s,
        'D': D,
        'lb': lb,
        'ub': ub
    }

def number_of_bits(x):
    return math.floor(math.log2(x)) + 1

def int_to_bool_list(x, digits):
    bits = [(x % (2 ** (i + 1)) // (2 ** i)) == 1 for i in range(digits - 1, -1, -1)]
    return [BoolVal(bit) for bit in bits]

def bool_list_to_int(bool_list):
    value = 0
    for i, b in enumerate(bool_list):
        if b:
            value += 2 ** (len(bool_list) - 1 - i)
    return value

def pad_bool_list(bool_list, target_length):
    return [BoolVal(False)] * (target_length - len(bool_list)) + bool_list

def at_least_one(vars_list):
    return Or(vars_list)

def at_most_one(vars_list, name=""):
    return And([Not(And(pair[0], pair[1])) for pair in combinations(vars_list, 2)])

def exactly_one(vars_list, name=""):
    return And(at_least_one(vars_list), at_most_one(vars_list, name))

def full_adder_bit(x, y, carry_in, result_bit, carry_out):
    eq_res = result_bit == Xor(Xor(x, y), carry_in)
    eq_carry = carry_out == Or(And(Xor(x, y), carry_in), And(x, y))
    return And(eq_res, eq_carry)

def binary_adder(x_bits, y_bits, result_bits, name=""):
    max_len = max(len(x_bits), len(y_bits))
    x_bits = pad_bool_list(x_bits, max_len)
    y_bits = pad_bool_list(y_bits, max_len)
    carries = [Bool(f"carry_{name}_{i}") for i in range(max_len)] + [BoolVal(False)]
    constraints = []
    for i in range(max_len):
        constraints.append(
            full_adder_bit(
                x=x_bits[max_len - i - 1],
                y=y_bits[max_len - i - 1],
                carry_in=carries[max_len - i],
                result_bit=result_bits[max_len - i - 1],
                carry_out=carries[max_len - i - 1]
            )
        )
    constraints.append(Not(carries[0]))
    return And(constraints)

def mask_bool_list(bool_list, mask_value):
    return [And(bit, mask_value) for bit in bool_list]

def conditional_binary_sum(num_list, mask, result_bits, name=""):
    constraints = []
    temp_results = [[BoolVal(False) for _ in range(len(result_bits))]] + [
        [Bool(f"{name}_temp_{i}_{j}") for j in range(len(result_bits))]
        for i in range(len(num_list))
    ]
    for i in range(len(num_list)):
        constraints.append(
            binary_adder(
                x_bits=temp_results[i],
                y_bits=mask_bool_list(num_list[i], mask[i]),
                result_bits=temp_results[i + 1],
                name=f"{name}_{i}"
            )
        )
    constraints.append(equal_bool_lists(temp_results[-1], result_bits))
    return And(constraints)

def geq_bool_lists(x_bits, y_bits):
    if len(x_bits) != len(y_bits):
        max_len = max(len(x_bits), len(y_bits))
        x_bits = pad_bool_list(x_bits, max_len)
        y_bits = pad_bool_list(y_bits, max_len)
    if len(x_bits) == 1:
        return Or(x_bits[0] == y_bits[0], And(Not(y_bits[0]), x_bits[0]))
    else:
        return Or(And(Not(y_bits[0]), x_bits[0]), And(x_bits[0] == y_bits[0], geq_bool_lists(x_bits[1:], y_bits[1:])))

def equal_bool_lists(x_bits, y_bits):
    max_len = max(len(x_bits), len(y_bits))
    x_bits = pad_bool_list(x_bits, max_len)
    y_bits = pad_bool_list(y_bits, max_len)
    return And([x_bits[i] == y_bits[i] for i in range(max_len)])

def max_bool_variable(bool_vars_list, max_var_bits):
    equality_constraints = Or([equal_bool_lists(var_bits, max_var_bits) for var_bits in bool_vars_list])
    geq_constraints = And([geq_bool_lists(max_var_bits, var_bits) for var_bits in bool_vars_list])
    return And(equality_constraints, geq_constraints)

def lexicographical_less(v1, v2):
    if not v1:
        return BoolVal(True)
    if not v2:
        return BoolVal(False)
    return Or(And(Not(v1[0]), v2[0]), And(v1[0] == v2[0], lexicographical_less(v1[1:], v2[1:])))

def refine_solution(time_taken, obj_value, solution, search_method, depot, time_limit):
    time_taken = int(time_taken)
    time_limit_sec = time_limit / 1000  
    is_optimal = False
    courier_routes = defaultdict(list)
    package_pattern = re.compile(r'p_(\d+)_(\d+)_(\d+)')
    solution_dict = {str(var): solution[var] for var in solution}
    
    if time_taken != time_limit_sec:
        is_optimal = True

    for entry, value in solution_dict.items():
        match = package_pattern.match(entry)
        if match and value:
            courier, package, step = map(int, match.groups())
            if package != depot:
                package = package + 1  
                courier_routes[courier].append((step, package))
    
    refined_routes = [
        [pkg for _, pkg in sorted(routes, key=lambda x: x[0])]
        for _, routes in sorted(courier_routes.items())
    ]

    return {search_method: {
        "time": time_taken,
        "optimal": is_optimal,
        "obj": obj_value,
        "sol": refined_routes
    }}
