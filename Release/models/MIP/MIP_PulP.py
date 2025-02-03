import pulp as lp
import numpy as np
import multiprocessing
from datetime import datetime
from utils.utils import read_dat_file
import json
import os
import copy

def find_path(adj_matrix, n):
    res = []
    source = n
    while True:
        for dest in range(n+1):
            if adj_matrix[source][dest] == 1:
                if dest == n:
                    return res
                res.append(dest+1)
                source = dest
                break

def order_solution(sol, m, fake_l, real_l):
    c = 0
    unordered_idxs = []
    for c in range(m):
        if fake_l[c] != real_l[c]:
            unordered_idxs.append(c)
    
    while len(unordered_idxs) > 1:
        idx1 = unordered_idxs[-1]
        
        for idx in unordered_idxs:
            if fake_l[idx1] == real_l[idx]:
                park = copy.deepcopy(sol[idx1])
                sol[idx1] = copy.deepcopy(sol[idx])
                sol[idx] = park

                unordered_idxs.remove(idx1)
                break

def solve_with_timeout(parsed_data, time_limit):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=MIP_thread, args=(parsed_data, queue))
    process.start()
    process.join(timeout=time_limit)  # Wait for at most `time_limit` seconds

    if process.is_alive():
        print("Terminating process due to timeout")
        process.terminate() 
        process.join()

    return queue.get() if not queue.empty() else None 

def MIP_thread(parsed_data, queue):
    model = lp.LpProblem("MultipleCouriersProblem", lp.LpMinimize)

    m, n, s, l, D = parsed_data

    maxD = np.max(D)
    minD = np.min(D)
    minL = np.min(s)
    heuristic_number_of_nodes_per_courier = np.ceil(n/m) +3


    max_total_dist = lp.LpVariable("MaxTotalDist", cat='Integer')

    load = lp.LpVariable.dicts("Load", range(m), lowBound=minL, cat='Integer')
    dist = lp.LpVariable.dicts("Dist", range(m), lowBound=minD, cat='Integer')

    paths = lp.LpVariable.dicts("Paths", ((i, j, k) for i in range(m) for j in range(n+1) for k in range(n+1)), cat='Binary')
    u = lp.LpVariable.dicts("u", ((i, j) for i in range(m) for j in range(n)), cat='Integer')

    model += max_total_dist, "Objective"

    #Objective Function Boundaries
    initial_upper_bound = heuristic_number_of_nodes_per_courier * maxD
    initial_lower_bound = heuristic_number_of_nodes_per_courier * (minD*3)

    model += max_total_dist <= initial_upper_bound
    model += max_total_dist >= initial_lower_bound

    for c in range(m):
        model += dist[c] <= max_total_dist

    for dest in range(n):
        model += lp.lpSum(paths[c,source,dest] for c in range(m) for source in range(n+1)) <= 1
        model += lp.lpSum(paths[c,source,dest] for c in range(m) for source in range(n+1)) >= 1

    for c in range(m):
        #Every courier must visit the depot as last node
        model += lp.lpSum(paths[c,source,n] for source in range(n)) == 1

        #Cumulative distance update
        model += dist[c] >= lp.lpSum(D[source][dest]*paths[c,source,dest] for source in range(n+1) for dest in range(n+1))

        #Cumulative load update
        model += load[c] >= lp.lpSum(s[dest]*paths[c,source,dest] for dest in range(n) for source in range(n+1))

        #Each courier must start from depot
        model += lp.lpSum(paths[c,n,dest] for dest in range(n)) >= 1
        model += lp.lpSum(paths[c,n,dest] for dest in range(n)) <= 1

        for source in range(n+1):
            #Couriers cannot stay still
            model += paths[c,source,source] <= 0.5

            for dest in range(n+1):
                #Path contiguity
                model += lp.lpSum(paths[c,source,j] for j in range(n+1)) == sum(paths[c,j,source] for j in range(n+1))

            #Just in order to speed up the search    
            model += lp.lpSum(paths[c,source,j] for j in range(n+1)) <= 1
            model += lp.lpSum(paths[c,j,source] for j in range(n+1)) <= 1

    for c in range(m):  # for each courier
        for source in range(n):
            for dest in range(n):
                if source != dest:
                    model += u[c, source] - u[c, dest] + n * paths[c, source, dest] <= n - 1

    for c in range(m-1):
        model +=  lp.lpSum(paths[c,source,dest] for source in range(n) for dest in range(n)) >= \
                        lp.lpSum(paths[c+1,source,dest] for source in range(n) for dest in range(n)) 


    #Maximum load
    for c in range(m):
        model += load[c] <= l[c]


    model.solve(lp.PULP_CBC_CMD(msg=False))
    queue.put((lp.value(model.objective), lp.LpStatus[model.status], paths))  # Store result
    
def MIP(instance_number):
    file_path = os.path.join('instances\Instances (.dat)', str(f'inst{instance_number}.dat'))
    parsed_data = read_dat_file(file_path)

    print(f"\nRunning MIP model with PulP as solver on instance {instance_number}")

    m = parsed_data['m']
    n = parsed_data['n']
    s = parsed_data['s']
    l = sorted(parsed_data['l'], reverse=True)
    D = parsed_data['D']

    starting_time = datetime.now()
    time_limit = 5*60
    safe_bound = 5
    
    thread_solution = solve_with_timeout((m, n, s, l, D), time_limit-safe_bound)
    
    end_time = datetime.now() - starting_time
    status = "Undefined"
    sol = []

    if thread_solution is not None:
        objective, status, paths = thread_solution
        print(f"Objective value (max dist) = {objective}")

        if str(objective).split('.')[-1] != '0':
            status = "Undefined"
        
        if status == "Optimal" or status == "Feasible":
            for c in range(m):
                matrix_values = [[0 for _ in range(n+1)] for _ in range(n+1)]

                for i in range(n+1):
                    for j in range(n+1):
                        matrix_values[i][j] = paths[c,i,j].varValue

                sol.append(find_path(matrix_values, n))

            order_solution(sol,m,l,parsed_data['l'])

    json_dict = {}
    json_dict['MIP'] = {}
    json_dict['MIP']['time'] = end_time.seconds if (end_time.seconds+safe_bound<time_limit) else time_limit
    json_dict['MIP']['optimal'] = True if (end_time.seconds+safe_bound<time_limit) else False
    json_dict['MIP']['obj'] = int(objective) if (status == "Optimal" or status == "Feasible") else None
    json_dict['MIP']['sol'] = sol

    with open(f'res/MIP/{str(int(instance_number))}.json', 'w') as outfile:
        json.dump(json_dict, outfile)

