import os
from minizinc import Instance, Model, Solver
import utils

def CP(instance_number):
    # Load the MiniZinc model
    model = Model()
    model.add_string("""
    include "globals.mzn";
    include "gecode.mzn"; 

    int: m; 
    int: n; 
    array[1..m] of int: l;     
    array[1..n] of int: s;     
    array[1..n+1, 1..n+1] of int: D;  

    array[1..m, 1..n+1] of var 1..n+1: x;
    int: total = sum(i in 1..n+1, j in 1..n+1)(D[i,j]);
    array[1..m] of var n+1..total: max_distance_per_courier;

    constraint
        forall(i in 1..m) (
            subcircuit([x[i, j] | j in 1..n+1])
        );
        
    constraint
        forall(j in 1..n) (
            count(i in 1..m)(x[i, j] != j) == 1
        );

    constraint forall(i in 1..m)(x[i,n+1] != n+1);
    constraint forall(i in 1..m)(count(j in 1..n)(x[i, j] == n+1) == 1);

    constraint
        forall(i in 1..m)(
            sum(j1 in 1..n)(
                if x[i, j1] == j1 then 0 else s[j1] endif
            ) <= l[i]
        );

    constraint
        forall(i in 1..m-1, z in i+1..m where z > i /\ l[i] >= l[z])(
            sum(j1 in 1..n)(if x[i, j1] == j1 then 0 else s[j1] endif) >= sum(j2 in 1..n)(if x[i, j2] == j2 then 0 else s[j2] endif)
        );

    constraint
        forall(i in 1..m-1, z in i+1..m where z>i /\ l[i] == l[z])(
            lex_lesseq([x[z,k1] | k1 in 1..n+1], [x[i,k] | k in 1..n+1])
        )::domain_propagation;

    constraint
        forall(i in 1..m) (
            max_distance_per_courier[i] = sum([D[j1, x[i, j1]] | j1 in 1..n+1])
        );
            
    var int: max_distance = max(i in 1..m)(max_distance_per_courier[i]);

    solve 
    :: int_search(x, dom_w_deg, indomain_random) 
    :: restart_luby(100) 
    :: relax_and_reconstruct(array1d(x), 83)
    minimize max_distance;

    output [
        "Paths:\n",
        concat([concat([show(x[i, j]) ++ " " | j in 1..n+1]) ++ "\n" | i in 1..m]),
        "Maximum distance: ", show(max_distance), "\n"
    ];
    """)

    # IMPORTING INSTANCE
    try:
        file_path = os.path.join('Instances', f'inst{instance_number}.dat')
        inst = utils.read_dat_file(file_path)
    except Exception as e:
        print(f"Error reading the instance file: {e}")
        return None

    # Create a MiniZinc instance
    gecode = Solver.lookup("gecode")
    instance = Instance(gecode, model)

    # Add all data from inst to the MiniZinc instance
    instance.add_data(inst)

    # Solve the model
    result = instance.solve()

    # Output the results
    if result:
        x = result["x"]
        max_distance = result["max_distance"]
        
        print("Paths:")
        for i in range(instance["m"]):
            print(f"Courier {i + 1}: {' -> '.join(map(str, x[i]))}")
        print("Maximum distance:", max_distance)
    else:
        print("No solution found")
