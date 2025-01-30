from z3 import *
import os
import utils
import subprocess

def generate_smt_lib(m, n, l, s, D):
    smt_code = """(set-logic QF_AUFLIRA)
    
    ; Declare constants
    (declare-const m Int)
    (declare-const n Int)
    (assert (= m {}))
    (assert (= n {}))

    ; Declare paths (path[courier, position] = item or -1 if unused)
    (declare-fun path (Int Int) Int)

    ; Declare distance matrix (0-indexed, {} = depot)
    (declare-fun D_func (Int Int) Int)
    """.format(m, n, n)
    
    for i in range(n+1):
        for j in range(n+1):
            smt_code += "(assert (= (D_func {} {}) {}))\n".format(i, j, D[i][j])
    
    smt_code += "\n; Declare load capacities of couriers\n"
    smt_code += "(declare-fun load (Int) Int)\n"
    for c in range(m):
        smt_code += "(assert (= (load {}) {}))\n".format(c, l[c])
    
    smt_code += "\n; Declare item sizes\n"
    smt_code += "(declare-fun size (Int) Int)\n"
    for j in range(n):
        smt_code += "(assert (= (size {}) {}))\n".format(j, s[j])
    
    smt_code += """
    ; CONSTRAINTS

    ; Each item is assigned exactly once
    (assert (forall ((j Int))
        (=> (and (>= j 0) (< j n))
            (= (+ """
    for c in range(m):
        for k in range(2):
            smt_code += "(ite (= (path {} {}) j) 1 0) ".format(c, k)
    smt_code += ") 1))))\n"

    smt_code += """
    ; Each courier must start at the depot and return to the depot
    (assert (forall ((c Int))
        (=> (and (>= c 0) (< c m))
            (and (= (D_func n (path c 0)) (D_func (path c 0) n))
                 (>= (path c 0) 0) (< (path c 0) n)))))

    ; Each courier should take at least one item
    (assert (forall ((c Int))
        (=> (and (>= c 0) (< c m))
            (or (not (= (path c 0) -1)) (not (= (path c 1) -1))))))

    ; Ensure couriers only pick valid items or mark as unused (-1)
    (assert (forall ((c Int) (k Int))
        (=> (and (>= c 0) (< c m) (>= k 0) (< k 2))
            (or (= (path c k) -1) (and (>= (path c k) 0) (< (path c k) n))))))

    ; Compute total distance per courier
    (declare-fun total_distance (Int) Int)
    (assert (forall ((c Int))
        (=> (and (>= c 0) (< c m))
            (= (total_distance c)
               (+ (D_func n (path c 0))
                  (D_func (path c 0) (path c 1))
                  (D_func (ite (= (path c 1) -1) (path c 0) (path c 1)) n))))))

    ; Load constraint: total load of assigned items must not exceed courier capacity
    (assert (forall ((c Int))
        (=> (and (>= c 0) (< c m))
            (<= (+ (ite (= (path c 0) -1) 0 (size (path c 0)))
                   (ite (= (path c 1) -1) 0 (size (path c 1))))
                (load c)))))

    (assert (forall ((i Int) (j Int))
    (=> (or (< i 0) (> i n) (< j 0) (> j n))
        (= (D_func i j) 0))))  ; Or some invalid large value

    ; Minimize maximum distance among couriers
    (declare-const max_dist Int)
    (assert (forall ((c Int))
        (=> (and (>= c 0) (< c m))
            (<= (total_distance c) max_dist))))
    (minimize max_dist)

    ; Check satisfiability
    (check-sat)

    ; Retrieve paths, loads, and max_dist
    (get-value ("""
    for c in range(m):
        for k in range(2):
            smt_code += "(path {} {}) ".format(c, k)
    smt_code += "max_dist "
    for c in range(m):
        smt_code += "(total_distance {}) ".format(c)
    smt_code += "))\n"
        
    return smt_code

# Load and execute SMT-LIB model
solver = Solver()

# IMPORTING INSTANCE
instance_num = '07'
file_path = os.path.join('Instances', f'inst{instance_num}.dat')
instance = utils.read_dat_file(file_path)
m = instance['m']
n = instance['n']
l = instance['l']
s = instance['s']
D = instance['D']

# Generate SMT-LIB model
smt_model = generate_smt_lib(m, n, l, s, D)

# Write to file
smt_file = "test_smtlib.smt2"
with open(smt_file, "w") as file:
    file.write(smt_model)

# Run Z3 using Python API
solver = Solver()
solver.from_string(smt_model)

if solver.check() == sat:
    print("SAT (Z3)")
    model = solver.model()
    for d in model.decls():
        print(f"{d.name()} = {model[d]}")
else:
    print("UNSAT or UNKNOWN (Z3)")