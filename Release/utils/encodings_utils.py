from z3 import *
from itertools import combinations
import uuid

###############################################  READ INSTANCES  ########################################################

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

###############################################  ENCODINGS  ############################################################

###################################################################################################### - Naive encodings
# one
def at_least_one_np(bool_vars):
    return Or(bool_vars)

def at_most_one_np(bool_vars, name = ""):
    return And([Not(And(pair[0], pair[1])) for pair in combinations(bool_vars, 2)])

def exactly_one_np(bool_vars, name = ""):
    return And(at_least_one_np(bool_vars), at_most_one_np(bool_vars, name))

# K
def at_least_k_np(bool_vars, k, name = ""):
    return at_most_k_np([Not(var) for var in bool_vars], len(bool_vars)-k, name)

def at_most_k_np(bool_vars, k, name = ""):
    return And([Or([Not(x) for x in X]) for X in combinations(bool_vars, k + 1)])

def exactly_k_np(bool_vars, k, name = ""):
    return And(at_most_k_np(bool_vars, k, name), at_least_k_np(bool_vars, k, name))

###################################################################################################### - Bitwise encodings
# one   
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

###################################################################################################### - Sequential encodings
# one
def at_least_one_seq(bool_vars):
    return at_least_one_np(bool_vars)

def at_most_one_seq(bool_vars, name):
    constraints = []
    n = len(bool_vars)
    s = [Bool(f"s_{name}_{i}") for i in range(n - 1)]
    constraints.append(Or(Not(bool_vars[0]), s[0]))
    constraints.append(Or(Not(bool_vars[n-1]), Not(s[n-2])))
    for i in range(1, n - 1):
        constraints.append(Or(Not(bool_vars[i]), s[i]))
        constraints.append(Or(Not(bool_vars[i]), Not(s[i-1])))
        constraints.append(Or(Not(s[i-1]), s[i]))
    return And(constraints)

def exactly_one_seq(bool_vars, name):
    return And(at_least_one_seq(bool_vars), at_most_one_seq(bool_vars, name))

# K
def at_least_k_seq(bool_vars, k, name):
    return at_most_k_seq([Not(var) for var in bool_vars], len(bool_vars)-k, name)

def at_most_k_seq(bool_vars, k, name):
    constraints = []
    n = len(bool_vars)
    if n == 0:
        return True
    s = [[Bool(f"s_{name}_{i}_{j}") for j in range(k)] for i in range(max(1, n - 1))]
    constraints.append(Or(Not(bool_vars[0]), s[0][0]))
    if n == 1:
        return And(constraints)
    constraints += [Not(s[0][j]) for j in range(1, k)]
    for i in range(1, n-1):
        constraints.append(Or(Not(bool_vars[i]), s[i][0]))
        constraints.append(Or(Not(s[i-1][0]), s[i][0]))
        constraints.append(Or(Not(bool_vars[i]), Not(s[i-1][k-1])))
        for j in range(1, k):
            constraints.append(Or(Not(bool_vars[i]), Not(s[i-1][j-1]), s[i][j]))
            constraints.append(Or(Not(s[i-1][j]), s[i][j]))
    constraints.append(Or(Not(bool_vars[n-1]), Not(s[n-2][k-1])))
    return And(constraints)

def exactly_k_seq(bool_vars, k, name):
    return And(at_most_k_seq(bool_vars, k, name), at_least_k_seq(bool_vars, k, name))

###################################################################################################### - Heule encodings
# one
def at_least_one_he(bool_vars):
    return at_least_one_np(bool_vars)

def at_most_one_he(bool_vars, name):
    if len(bool_vars) <= 4: 
        return And(at_most_one_np(bool_vars))
    y = Bool(f"y_{name}") 
    return And(And(at_most_one_np(bool_vars[:3] + [y])), And(at_most_one_he(bool_vars[3:] + [Not(y)], name+"_")))

def exactly_one_he(bool_vars, name):
    return And(at_most_one_he(bool_vars, name), at_least_one_he(bool_vars))

###################################################################################################### - Auxiliary functions

def toBinary(num, length = None):
    num_bin = bin(num).split("b")[-1]
    if length:
        return "0"*(length - len(num_bin)) + num_bin
    return num_bin

def full_adder(A, B, Cin):
    # Compute sum and carry-out (Cout)
    Sum = A ^ B ^ Cin  # XOR operation gives the sum
    Cout = (A & B) | (Cin & (A ^ B))  # Compute carry-out
    return Sum, Cout

def convert(number: int, length: int) -> list[BoolRef]:
    num = []
    str_num = bin(number)[2:].zfill(length)  # Convert number to binary string and zero-pad to specified length
    for i in range(length):
        ni = Bool(str(uuid.uuid4()))  # Create a new Z3 boolean variable with a random name
        if str_num[i] == '1':
            num.append(ni)
            ni = True
        else:
            ni = False
    return num