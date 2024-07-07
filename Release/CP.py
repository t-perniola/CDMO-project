import os
import time
import json
import datetime
from minizinc import Instance, Model, Solver, Status
import utils  # Assuming you have a utils module to read .dat files


def CP(instance_number):
    start_time = time.time()

    # IMPORTING INSTANCE
    file_path = os.path.join('../Instances', f'inst{instance_number}.dat')
    instance_data = utils.read_dat_file(file_path)

    m = instance_data['m']
    n = instance_data['n']
    l = instance_data['l']
    s = instance_data['s']
    D = instance_data['D']

    # MiniZinc model
    model_code = """
    include "all_different_except.mzn"; 
    include "gecode.mzn";

    % Data
    int: m; 
    int: n; 
    array[1..m] of int: l;     
    array[1..n] of int: s;     
    array[1..n+1, 1..n+1] of int: D;                  

    % Variables
    array[1..n+m+1] of var 1..n+1: y; 
    array[0..m] of var 1..m+n+1: ind; 
    array[1..m] of var n..ceil(n/m)*200: max_distance_per_courier; 

    % Constraints 
    constraint y[1] = n+1;
    constraint y[n+m+1] = n+1;
    constraint y[2] != n+1;
    constraint y[n+m] != n+1;
    constraint ind[0] = 1;
    constraint ind[m] = n+m+1;

    constraint forall(i in 1..m-1) (y[ind[i]] == n+1);
    constraint forall(i in 3..n+m-1)(y[i] == n+1 -> y[i-1] != n+1 /\\ y[i+1] != n+1);
    constraint forall(i in 0..m-1)(ind[i] + ceil(n/m) <= ind[i+1] /\\ ind[i+1]-ind[i]-1 <= n-m+1)::domain_propagation;

    constraint all_different_except(y,{n+1})::domain_propagation;

    constraint forall(i in 1..m)(sum(c in ind[i-1]+1..ind[i]-1)(s[y[c]]) <= l[i]);

    % Distance
    constraint forall(i in 1..m)(
        sum(c in ind[i-1]..ind[i]-1)(
            D[y[c],y[c+1]]
        ) = max_distance_per_courier[i]
    );

    var int: max_distance = max(i in 1..m)(max_distance_per_courier[i]);

    solve 
    :: int_search(y, dom_w_deg, indomain_random) 
    :: int_search(ind, first_fail, indomain_random)
    :: restart_luby(100) 
    :: relax_and_reconstruct(y, 83)
    minimize max_distance;
      
    % Output results
    output ["Paths: ", show(y), "\\n",
            "indexes: ", show(ind), "\\n",
            "Maximum distance for each courier: ", show(max_distance), "\\n",
            "ind: ", show(ind), "\\n"
    ];
    """

    # Save the model to a file (optional)
    model_path = "model.mzn"
    with open(model_path, 'w') as model_file:
        model_file.write(model_code)

    # Load the model
    model = Model(model_path)
    solver = Solver.lookup("gecode")

    # Create an instance of the model
    instance = Instance(solver, model)

    # Set the data for the instance
    instance["m"] = m
    instance["n"] = n
    instance["l"] = l
    instance["s"] = s
    instance["D"] = D

    # Solve the instance with a time limit of 5 minutes (300 seconds)
    timeout = datetime.timedelta(seconds=300)
    result = instance.solve(timeout=timeout)

    # Check if the result is optimal
    optimal = result.status == Status.OPTIMAL_SOLUTION

    # Extract paths
    paths = []
    if optimal:
        for i in range(m):
            path = []
            for j in range(result["ind"][i], result["ind"][i+1]):
                if result["y"][j] <= n:
                    path.append(result["y"][j] - 1)
            paths.append(path)

    # Prepare the result in JSON format
    result_json = {
        "Gecode": {
            "time": round(time.time() - start_time, 2),
            "optimal": optimal,
            "obj": result["max_distance"] if optimal else None,
            "sol": paths
        }
    }

    # Save the result in JSON format
    with open(f'res/CP/{str(int(instance_number))}.json', 'w') as outfile:
        json.dump(result_json, outfile, indent=4)

# Example usage
CP('01')
