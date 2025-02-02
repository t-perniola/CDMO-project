from itertools import combinations
from z3 import *
import math
import re
from collections import defaultdict

def read_dat_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    m = int(lines[0].strip())
    n = int(lines[1].strip())
    
    l = list(map(int, lines[2].strip().split()))
    
    s = list(map(int, lines[3].strip().split()))
    
    D = []
    for line in lines[4:-2]:
        D.append(list(map(int, line.strip().split())))
    
    lb = int(lines[-2].strip())
    ub = int(lines[-1].strip())
    
    return {
        'm': m,
        'n': n,
        'l': l,
        's': s,
        'D': D,
        'lb': lb,
        'ub': ub
    }

def parse_value(value):
    return value.strip().strip(';')

def num_bits(x):
    """Compute the number of bits necessary to represent the integer number x"""
    return math.floor(math.log2(x)) + 1

def int_to_bin(x, digits):
    """Converts an integer x into a binary list of Z3 BoolVal (True/False)."""
    x_bin = [(x % (2 ** (i + 1)) // 2 ** i) == 1 for i in range(digits - 1, -1, -1)]
    return [BoolVal(b) for b in x_bin]


def bin_to_int(val):
    """
    :param val: array of binary variabled to be converted to an integer
    :return: integer value
    """
    number = 0
    for i in range(len(val)):
        if val[i]:
            number += 2 ** (len(val) - 1 - i)
    return number

def pad_bool(x, length):
    """Pad a binary list with False (0) to reach a specific length."""
    return [BoolVal(False)] * (length - len(x)) + x

# SAT Constraints for "At Least/At Most/Exactly One" Encodings

def at_least_one_np(bool_vars):
    return Or(bool_vars)

def at_most_one_np(bool_vars, name=""):
    return And([Not(And(pair[0], pair[1])) for pair in combinations(bool_vars, 2)])

def exactly_one_np(bool_vars, name=""):
    return And(at_least_one_np(bool_vars), at_most_one_np(bool_vars, name))

# Sequential Encoding

def at_least_one_seq(bool_vars):
    return at_least_one_np(bool_vars)

def at_most_one_seq(bool_vars, name):
    constraints = []
    n = len(bool_vars)
    s = [Bool(f"s_{name}_{i}") for i in range(n - 1)]
    constraints.append(Or(Not(bool_vars[0]), s[0]))
    constraints.append(Or(Not(bool_vars[n - 1]), Not(s[n - 2])))
    for i in range(1, n - 1):
        constraints.append(Or(Not(bool_vars[i]), s[i]))
        constraints.append(Or(Not(bool_vars[i]), Not(s[i - 1])))
        constraints.append(Or(Not(s[i - 1]), s[i]))
    return And(constraints)

def exactly_one_seq(bool_vars, name):
    return And(at_least_one_seq(bool_vars), at_most_one_seq(bool_vars, name))

# Bitwise Encoding

def toBinary(num, length=None):
    num_bin = bin(num).split("b")[-1]
    if length:
        return "0" * (length - len(num_bin)) + num_bin
    return num_bin

def at_least_one_bw(bool_vars):
    return at_least_one_np(bool_vars)

def at_most_one_bw(bool_vars, name):
    constraints = []
    n = len(bool_vars)
    m = math.ceil(math.log2(n))
    r = [Bool(f"r_{name}_{i}") for i in range(m)]
    binaries = [toBinary(i, m) for i in range(n)]
    for i in range(n):
        for j in range(m):
            phi = Not(r[j])
            if binaries[i][j] == "1":
                phi = r[j]
            constraints.append(Or(Not(bool_vars[i]), phi))
    return And(constraints)

def exactly_one_bw(bool_vars, name):
    return And(at_least_one_bw(bool_vars), at_most_one_bw(bool_vars, name))

# Heuristic Encoding

def at_least_one_he(bool_vars):
    return at_least_one_np(bool_vars)

def at_most_one_he(bool_vars, name):
    if len(bool_vars) <= 4:
        return And(at_most_one_np(bool_vars))
    y = Bool(f"y_{name}")
    return And(And(at_most_one_np(bool_vars[:3] + [y])), And(at_most_one_he(bool_vars[3:] + [Not(y)], name + "_")))

def exactly_one_he(bool_vars, name):
    return And(at_most_one_he(bool_vars, name), at_least_one_he(bool_vars))

# Constraints for "At Least/At Most/Exactly k"

def at_least_k_np(bool_vars, k):
    return at_most_k_np([Not(var) for var in bool_vars], len(bool_vars) - k)

def at_most_k_np(bool_vars, k):
    return And([Or([Not(x) for x in X]) for X in combinations(bool_vars, k + 1)])

def exactly_k_np(bool_vars, k):
    return And(at_most_k_np(bool_vars, k), at_least_k_np(bool_vars, k))

# Binary Operations

def geq(x, y):
    if len(x) != len(y):
        max_len = max(len(x), len(y))
        x = pad_bool(x, max_len)
        y = pad_bool(y, max_len)
    if len(x) == 1:
        return Or(x[0] == y[0], And(Not(y[0]), x[0]))
    else:
        return Or(And(Not(y[0]), x[0]), And(x[0] == y[0], geq(x[1:], y[1:])))

def sum_one_bit(x, y, c_in, res, c_res):
    c_1 = res == Xor(Xor(x, y), c_in)
    c_2 = c_res == Or(And(Xor(x, y), c_in), And(x, y))
    return And(c_1, c_2)

def sum_bin(x, y, res, name=""):
    max_len = max(len(x), len(y))
    x = pad_bool(x, max_len)
    y = pad_bool(y, max_len)
    c = [Bool(f"carry_{name}_{i}") for i in range(max_len)] + [BoolVal(False)]
    constr = []
    for i in range(max_len):
        constr.append(sum_one_bit(x=x[max_len - i - 1], y=y[max_len - i - 1], c_in=c[max_len - i], res=res[max_len - i - 1], c_res=c[max_len - i - 1]))
    constr.append(Not(c[0]))
    return And(constr)

def eq_bin(x, y):
    max_len = max(len(x), len(y))
    x = pad_bool(x, max_len)
    y = pad_bool(y, max_len)
    return And([x[i] == y[i] for i in range(max_len)])

def cond_sum_bin(num_list, mask, res, name=""):
    constr = []
    res_temp = [[BoolVal(False) for _ in range(len(res))]] + [[Bool(name + f"res_t_{i}_{j}") for j in range(len(res))] for i in range(len(num_list))]
    for i in range(len(num_list)):
        constr.append(sum_bin(res_temp[i], mask_bins(num_list[i], mask[i]), res_temp[i + 1], name + f"_{i}"))
    constr.append(eq_bin(res_temp[i + 1], res))
    return And(constr)

def mask_bins(list_bin, mask_value):
    return [And(i, mask_value) for i in list_bin]


def max_var(list_var_bits, max_var):
    
    equal_number = Or([eq_bin(var_bits, max_var) for var_bits in list_var_bits])
    geq_list = And([geq(max_var, var_bits) for var_bits in list_var_bits])

    return And(equal_number, geq_list)


def lex_less(v1, v2):
    if len(v1) == 0: return True
    if len(v2) == 0: return False
    return Or(And(Not(v1[0]), v2[0]), Or(((v1[0] == v2[0]), lex_less(v1[1:], v2[1:]))))


def refineSol(time_taken, obj_value, solution, search, depot, time_limit):
    time_taken = int(time_taken)
    time_limit = time_limit/1000
    opt = False
    courier_routes = defaultdict(list)  
    package_pattern = re.compile(r'p_(\d+)_(\d+)_(\d+)')
    solution_dict = {str(var): solution[var] for var in solution}
    
    if time_taken != time_limit:
        opt = True

    for entry, value in solution_dict.items():
        match = package_pattern.match(entry)
        if match and value:
            courier, package, step = map(int, match.groups())
            if package != depot:  
                package = package + 1  
                courier_routes[courier].append((step, package))  
    
    refined_sol = [
        [pkg for _, pkg in sorted(routes, key=lambda x: x[0])]  
        for _, routes in sorted(courier_routes.items())  
    ]

    result = {
        "time": time_taken,
        "optimal": opt,
        "obj": obj_value,
        "sol": refined_sol
    }

    return {search: result}

