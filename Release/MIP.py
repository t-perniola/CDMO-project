import utils
import gurobipy as gp
from datetime import datetime
from math import floor
import json
import os

def find_path(adj_matrix, courier, n):
    res = []
    source = n
    while True:
        for dest in range(n+1):
            if adj_matrix[courier, source, dest].X == 1:
                if dest == n:
                    return res
                res.append(dest)
                source = dest
                break

def MIP(instance_number):
    starting_time = datetime.now()
    time_limit = 5*60
    
    model = gp.Model()

    file_path = os.path.join('Instances', str(f'inst{instance_number}.dat'))
    parsed_data = utils.read_dat_file(file_path)

    m = parsed_data['m']
    n = parsed_data['n']
    s = parsed_data['s']
    #l = sorted(parsed_data['l'], reverse=True)
    l = parsed_data['l']
    D = parsed_data['D']

    max_total_dist = model.addVar(vtype=gp.GRB.CONTINUOUS, name='MaxTotalDist')

    cumulative_dist = model.addVars(m, n+1, vtype=gp.GRB.INTEGER, name='CumulativeDist')
    cumulative_load = model.addVars(m, n, lb=0, ub=n*max(s), vtype=gp.GRB.INTEGER, name='CumulativeLoad')

    paths = model.addVars(m, n+1, n+1, vtype=gp.GRB.BINARY, name='Paths')

    model.setObjective(max_total_dist, gp.GRB.MINIMIZE)

    #Maximum total distance is equal to the maximum total distance computed in the final depot "slot" for each courier
    model.addConstrs( gp.quicksum(cumulative_dist[c,j] for j in range(n+1)) <= max_total_dist for c in range(m))

    #Every node must be visited exactly once (except for the depot n+1)
    for dest in range(n):
        model.addConstr(gp.quicksum(paths[c,source,dest] for c in range(m) for source in range(n+1)) <= 1)
        model.addConstr(gp.quicksum(paths[c,source,dest] for c in range(m) for source in range(n+1)) >= 1)
    
    for c in range(m):
        #Every courier must visit the depot as last node
        model.addConstr(gp.quicksum(paths[c,source,n] for source in range(n)) == 1)

        #Each courier must start from depot
        model.addConstr(gp.quicksum(paths[c,n,dest] for dest in range(n)) == 1)

        for source in range(n+1):
            #Couriers cannot stay still
            model.addConstr(paths[c,source,source] <= 0.5) #npe

            for dest in range(n+1):
                #Cumulative distance update
                model.addConstr((paths[c,source,dest] == 1) >> (cumulative_dist[c,dest] == D[source][dest])) #npe

                #Path contiguity
                model.addConstr((paths[c,source,dest] == 1) >> (gp.quicksum(paths[c,j,source] for j in range(n+1)) >= 1))
                model.addConstr((paths[c,source,dest] == 1) >> (gp.quicksum(paths[c,dest,j] for j in range(n+1)) >= 1))

                #Load update
                if dest < n:
                    model.addConstr((paths[c,source,dest] == 1) >> (cumulative_load[c,dest] == s[dest])) #npe

            #Just in order to speed up the search    
            model.addConstr(gp.quicksum(paths[c,source,j] for j in range(n+1)) <= 1)
    
    #Loop avoidance
    for c in range(m):
        for source in range(n):
            for dest in range(source+1):
                model.addConstr( (gp.quicksum(paths[c,dest,j] for j in range(source+1)) + paths[c,source,dest]) <= 1)

    #Symmetry breaking constraints
    '''
    for c in range(m-1):
        model.addConstr( gp.quicksum(paths[c,source,dest] for source in range(n) for dest in range(n)) >= \
                        gp.quicksum(paths[c+1,source,dest] for source in range(n) for dest in range(n)) )
    '''
    
    #Maximum load
    for c in range(m):
        model.addConstr(gp.quicksum(cumulative_load[c,j] for j in range(n)) <= l[c])

    preprocessing_time = datetime.now() - starting_time
    safe_bound = 5
    model.setParam('TimeLimit', time_limit - preprocessing_time.seconds - safe_bound)

    model.optimize()

    sol = []
    if model.SolCount > 0:
        for c in range(m):
            sol.append(find_path(paths,c,n))    
    
    json_dict = {}
    json_dict['Gurobi'] = {}
    json_dict['Gurobi']['time'] = int(floor(model.Runtime + preprocessing_time.seconds)) if model.SolCount > 0 else time_limit
    json_dict['Gurobi']['optimal'] = True if (model.Runtime + preprocessing_time.seconds + safe_bound < time_limit) else False
    json_dict['Gurobi']['obj'] = int(model.ObjVal) if model.SolCount > 0 else None
    json_dict['Gurobi']['sol'] = sol

    with open(f'res/MIP/{str(int(instance_number))}.json', 'w') as outfile:
        json.dump(json_dict, outfile)