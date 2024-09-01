import gurobipy as gp
from gurobipy import GRB
import numpy as np
from datetime import datetime
from math import floor
import json
import os
import utils
import copy

def find_path(adj_matrix, courier, n):
    res = []
    source = n
    while True:
        for dest in range(n+1):
            if adj_matrix[courier, source, dest].X == 1:
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


def MIP(instance_number):
    starting_time = datetime.now()
    time_limit = 5*60

    # Create the model
    model = gp.Model()
    file_path = os.path.join('Instances', str(f'inst{instance_number}.dat'))
    parsed_data = utils.read_dat_file(file_path)

    m = parsed_data['m']
    n = parsed_data['n']
    s = parsed_data['s']
    l = sorted(parsed_data['l'], reverse=True)
    D = parsed_data['D']

    maxD = np.max(D)
    minD = np.min(D)
    heuristic_number_of_nodes_per_courier = np.ceil(n/m) +3

    #Callback function for dynamic boundaries
    def update_upper_bound(model, where):
        if where == GRB.Callback.MIPSOL:
            # Get the current value of the objective function
            current_obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            
            # Replace the upper bound with the new objective value
            model.cbLazy(max_total_dist <= current_obj_val)

    # Define the variables
    max_total_dist = model.addVar(vtype=gp.GRB.INTEGER, name='MaxTotalDist')

    load = model.addVars(m, lb=0, vtype=gp.GRB.INTEGER, name='Load')
    dist = model.addVars(m, lb=0, vtype=gp.GRB.INTEGER, name='Dist')

    paths = model.addVars(m, n+1, n+1, vtype=gp.GRB.BINARY, name='Paths')

    u = model.addVars(m, n, vtype=gp.GRB.INTEGER, name='u')

    #Objective Function Boundaries
    initial_upper_bound = heuristic_number_of_nodes_per_courier * maxD
    initial_lower_bound = heuristic_number_of_nodes_per_courier * (minD*3)

    model.addConstr( max_total_dist <= initial_upper_bound)
    model.addConstr( max_total_dist >= initial_lower_bound)

    model.Params.lazyConstraints = 1

    # Define Objective function
    model.setObjective(max_total_dist, gp.GRB.MINIMIZE)

    # Define Constraints

    #Maximum total distance is equal to the maximum total distance computed in the final depot "slot" for each courier
    model.addConstrs( dist[c] <= max_total_dist for c in range(m))

    #Every node must be visited exactly once (except for the depot n+1)
    for dest in range(n):
        model.addConstr(gp.quicksum(paths[c,source,dest] for c in range(m) for source in range(n+1)) <= 1)
        model.addConstr(gp.quicksum(paths[c,source,dest] for c in range(m) for source in range(n+1)) >= 1)

    for c in range(m):
        #Every courier must visit the depot as last node
        model.addConstr(gp.quicksum(paths[c,source,n] for source in range(n)) == 1)

        #Cumulative distance update
        model.addConstr(dist[c] >= (gp.quicksum(D[source][dest]*paths[c,source,dest] for source in range(n+1) for dest in range(n+1)) ))

        #test
        model.addConstr(dist[c] >= initial_lower_bound)
        model.addConstr(dist[c] <= initial_upper_bound)

        #Cumulative load update
        model.addConstr(load[c] >= (gp.quicksum(s[dest]*paths[c,source,dest] for dest in range(n) for source in range(n+1))) ) #npe

        #Each courier must start from depot
        model.addConstr(gp.quicksum(paths[c,n,dest] for dest in range(n)) >= 1)
        model.addConstr(gp.quicksum(paths[c,n,dest] for dest in range(n)) <= 1)

        for source in range(n+1):
            #Couriers cannot stay still
            model.addConstr(paths[c,source,source] <= 0.5) #npe

            for dest in range(n+1):
                #Path contiguity
                model.addConstr(gp.quicksum(paths[c,source,j] for j in range(n+1)) == (gp.quicksum(paths[c,j,source] for j in range(n+1))))

            #Just in order to speed up the search    
            model.addConstr(gp.quicksum(paths[c,source,j] for j in range(n+1)) <= 1)
            model.addConstr(gp.quicksum(paths[c,j,source] for j in range(n+1)) <= 1)

    #Loop Avoidance 
    for c in range(m):  
        for source in range(n):
            for dest in range(n):
                if source != dest:
                    model.addConstr(u[c, source] - u[c, dest] + n * paths[c, source, dest] <= n - 1)

    #Symmetry breaking constraints
    for c in range(m-1):
        model.addConstr( gp.quicksum(paths[c,source,dest] for source in range(n) for dest in range(n)) >= \
                        gp.quicksum(paths[c+1,source,dest] for source in range(n) for dest in range(n)) )

    '''
    for c in range(m-1):
        model.addConstr(gp.quicksum(paths[c,source,dest] for source in range(n) for dest in range(n)) <= heuristic_number_of_nodes_per_courier )
    '''
        
    '''
    for c1 in range(m):
        for c2 in range(c1+1,m):
            model.addConstr(dist[c1]-dist[c2] <= 2*maxD )

    for c in range(m-1):
        model.addConstr(dist[c] >= dist[c+1])
    '''

    '''
    for c in range(m - 1):
        model.addConstr(
            gp.quicksum(paths[c, source, dest] * (source + dest) for source in range(n) for dest in range(n)) <= \
            gp.quicksum(paths[c + 1, source, dest] * (source + dest) for source in range(n) for dest in range(n)) )
    '''

    '''
    for k in range(m - 1):
        for i in range(n):
            for j in range(i + 1, n):
                model.addConstr(paths[k, n, i] + paths[k + 1, n, j] <= 1)
    '''

    #Maximum load
    for c in range(m):
        model.addConstr(load[c] <= l[c])

    preprocessing_time = datetime.now() - starting_time
    safe_bound = 5
    model.setParam('TimeLimit', time_limit - preprocessing_time.seconds - safe_bound)

    model.optimize(update_upper_bound)

    sol = []
    if model.SolCount > 0:
        for c in range(m):
            sol.append(find_path(paths,c,n))

    order_solution(sol,m,l,parsed_data['l'])

    json_dict = {}
    json_dict['MIP'] = {}
    json_dict['MIP']['time'] = int(floor(model.Runtime + preprocessing_time.seconds)) if model.SolCount > 0 else time_limit
    json_dict['MIP']['optimal'] = True if (model.Runtime + preprocessing_time.seconds + safe_bound < time_limit) else False
    json_dict['MIP']['obj'] = int(model.ObjVal) if model.SolCount > 0 else None
    json_dict['MIP']['sol'] = sol

    with open(f'res/MIP/{str(int(instance_number))}.json', 'w') as outfile:
        json.dump(json_dict, outfile)


MIP('09')