import pulp as lp
import numpy as np
import multiprocessing
from datetime import datetime
import utils
import json
import os
import copy
from pyscipopt import Model, quicksum
from pyscipopt import scip
import itertools
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def find_path(adj_matrix, courier, n):
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

def solve_with_timeout(parsed_data, time_limit, seed):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=MIP_thread, args=(parsed_data, queue, seed))
    process.start()
    process.join(timeout=time_limit)  # Wait for at most `time_limit` seconds

    if process.is_alive():
        process.terminate() 
        process.join()

    return queue.get() if not queue.empty() else None 

def print_info(n, courier_number, route, paths, load, dist):
    print(f'Courier: {courier_number}')
    print(route)
    print(f'Load={load}')
    print(f'Dist={dist}')
    print('Path:')
    for i in range(n+1):
        for j in range(n+1):
            if i == j:
                print(0, end=' ')
            else:
                print(paths[courier_number-1,i,j].start, end=' ')
        print()

def MIP_thread(parsed_data, queue, seed):
    m, n, s, l, D = parsed_data
    D = np.array(D)

    model = Model("MCP_SOTA")
    ARCLIM = np.infty

    if seed:
        UB_seed = max(                 # longest tour length in the Clarke-Wright seed
            sum(D[i_prev][i_curr] for i_prev, i_curr in zip([n] + tour, tour + [n]))
            for tour in seed
        )
        ARCLIM = UB_seed               # any distance ≥ this cannot be in an optimal min-max

    edges = [
            (i, j) for i in range(n + 1) for j in range(n + 1)
            if i != j and (i == n or j == n or D[i][j] < ARCLIM)
        ]

    # Variables
    x = {}  # Arc decision variables
    for c in range(m):
        for (i,j) in edges:
            x[c, i, j] = model.addVar(vtype='B', name=f"x_{c}_{i}_{j}")

    f = {}                                                              # NEW
    for c in range(m):
        for i, j in edges:
            f[c, i, j] = model.addVar(lb=0, ub=l[c], name=f"f_{c}_{i}_{j}")

    y = [model.addVar(vtype='B', name=f"use_{c}") for c in range(m)]

    load = [model.addVar(lb=0, name=f"load_{c}") for c in range(m)]
    dist = [model.addVar(lb=0, name=f"dist_{c}") for c in range(m)]

    max_dist = model.addVar(lb=0, name="max_dist")

    # Objective: minimize maximum distance
    model.setObjective(max_dist, sense="minimize")

    # -----------------------------------------
    #  STATIC CAPACITY SUBSET CUTS (|S|<=3)
    # -----------------------------------------
    max_cap = max(l)
    V = list(range(n))   # just the customer nodes (0..n-1)
    for r in (2,3):
        for S in itertools.combinations(V, r):
            if sum(s[i] for i in S) <= max_cap:
                continue

            cross_arcs = [
                (c, i, j)
                for c in range(m)
                for i in S
                for j in range(n + 1)
                if j not in S and (i, j) in x          # (i,j) survived filtering
            ]

            # ☛ NEW GUARD — only add the cut if at least one arc survived
            if not cross_arcs:
                continue

            lhs = quicksum(x[idx] for idx in cross_arcs)
            model.addCons(lhs >= 1)
    # -----------------------------------------

    # Constraints

    # Each customer visited exactly once
    for j in range(n):
        model.addCons(quicksum(x[c, i, j] for c in range(m) for i in range(n+1) if (i,j) in edges) == 1)

    '''
    # Each courier starts and ends at depot
    for c in range(m):
        model.addCons(quicksum(x[c, n, j] for j in range(n)) == 1)
        model.addCons(quicksum(x[c, i, n] for i in range(n)) == 1)
    '''
    for c in range(m):
        # leave depot at most once – but *only* if y_c == 1
        model.addCons(
            quicksum(x[c, n, j] for j in range(n)) == y[c]
        )
        # come back at most once
        model.addCons(
            quicksum(x[c, i, n] for i in range(n)) == y[c]
        )

    BIGM = sum(l)                     # any safe upper bound
    for c in range(m):
        model.addCons(load[c] <= l[c] * y[c])        # capacity only if used
        model.addCons(dist[c] <= BIGM * y[c])        # distance 0 when parked

    # Flow conservation
    for c in range(m):
        for h in range(n):
            model.addCons(quicksum(x[c, i, h] for i in range(n+1) if (i,h) in edges) ==
                          quicksum(x[c, h, j] for j in range(n+1) if (h,j) in edges))

    for c in range(m):
        for (i,j) in edges:
            model.addCons(f[c,i,j] <= l[c]*x[c,i,j])

        # flow balance
        for j in range(n):
            model.addCons(
                quicksum(f[c,i,j] for i in range(n+1) if (i,j) in edges) -
                quicksum(f[c,j,k] for k in range(n+1) if (j,k) in edges)
                == s[j] * quicksum(x[c,i,j] for i in range(n+1) if (i,j) in edges)
            )

        # outgoing flow from depot equals courier load
        model.addCons(quicksum(f[c, n, j] for j in range(n) if (n, j) in edges) == load[c])

        # no load on the return arcs to the depot
        for i in range(n):
            if (i, n) in edges:
                model.addCons(f[c, i, n] == 0)

    # Load constraints
    for c in range(m):
        model.addCons(load[c] == quicksum(s[j] * quicksum(x[c, i, j] for i in range(n+1) if (i,j) in edges) for j in range(n)))
        model.addCons(load[c] <= l[c])

    # Distance constraints
    for c in range(m):
        model.addCons(dist[c] == quicksum(D[i][j] * x[c, i, j] for i in range(n+1) for j in range(n+1) if (i,j) in edges))
        model.addCons(dist[c] <= max_dist)

    # Symmetry breaking: Load ordering
    for c in range(m - 1):
        model.addCons(load[c] >= load[c + 1])

    depot   = n
    radii   = [2 * D[depot][i] for i in range(n)]
    lb_split = split_lb(s, radii, max(l))           # provable lower bound
    model.addCons(max_dist >= lb_split)             # always valid

    # Warm-start (optional)
    if seed:
        init_sol = model.createSol()
        dist_c_values = []

        for c, tour_nodes in enumerate(seed):
            if not tour_nodes:          # parked truck – no arcs to set
                model.setSolVal(init_sol, y[c], 0)
                continue
            
            model.setSolVal(init_sol, y[c], 1)
            tour_nodes = [t - 1 for t in seed[c]]
            prev = n
            dist_c = 0

            remain = load_c = sum(s[node] for node in tour_nodes)

            # flow on arc depot→first
            first = tour_nodes[0]
            model.setSolVal(init_sol, f[c, n, first], remain)


            for node in tour_nodes:
                model.setSolVal(init_sol, x[c, prev, node], 1)

                if prev != n:                    # we already set depot→first
                    model.setSolVal(init_sol, f[c, prev, node], remain)
                remain -= s[node]

                dist_c += D[prev][node]
                prev = node

            dist_c += D[prev][n]
            dist_c_values.append(dist_c)

            model.setSolVal(init_sol, x[c, prev, n], 1)
            #model.setSolVal(init_sol, f[c, prev, n], 0)

            model.setSolVal(init_sol, load[c], load_c)  
            model.setSolVal(init_sol, dist[c], dist_c)

            remain = l[c]
            # at the depot you start with full capacity
            prev = n
            for node in tour_nodes:
                # take the load on at `node`
                remain -= s[node]
                prev = node

            # Explicitly set unused arcs and MTZ explicitly to zero
            for i in range(n + 1):
                for j in range(n + 1):
                    if i != j and (i, j) not in zip([n] + tour_nodes, tour_nodes + [n]):
                        model.setSolVal(init_sol, x[c, i, j], 0)

        max_d = max(dist_c_values)          # dist_c_values = list of the m distances

        if max_d >= lb_split:
            model.setObjlimit(max_d)
            for c in range(m):
                model.addCons(dist[c] <= max_d)

        model.setSolVal(init_sol, max_dist, float(max_d))
        # model.addSol(init_sol, free=True)
        model.addSol(init_sol, free=False)     # keep it as incumbent 


    # Solve
    model.setParam("limits/time", 290)
    model.setParam('presolving/maxrounds', 20)
    model.setParam('heuristics/undercover/freq', 10)   # good on CVRP
    model.setParam("parallel/maxnthreads", 8)

    # getting a tigher dual bound
    model.setParam("separating/maxcutsroot", 1000)
    model.setParam("separating/maxcuts", 200)	
    #model.setParam("branching/scorefunc", "pscost")

    # getting a better primal bound
    model.setParam("heuristics/localbranching/freq", 1)	
    model.setParam("heuristics/rins/freq", 5)

    model.setParam("limits/presolve", 60.0)

    model.optimize()

    print(model.getObjVal())
    return

    queue.put((lp.value(model.objective), lp.LpStatus[model.status], paths))  # Store result

def split_lb(s, r, Q):
    """
    s  demand  (list of length n)
    r  radii   (same length, r[i] = 2*D[depot][i])
    Q  the largest vehicle capacity
    """
    items = sorted(zip(r, s), reverse=True)      # long radii first
    bins  = []                                   # each entry = remaining capacity

    for rad, dem in items:
        placed = False
        for b in bins:                           # try to reuse a bin
            if b['load'] + dem <= Q:
                b['load'] += dem
                b['radius'] = max(b['radius'], rad)
                placed = True
                break
        if not placed:
            bins.append({'load': dem, 'radius': rad})

    return max(b['radius'] for b in bins)

def ortools_seed(n, m, demand, capacity, dist, depot=None, seconds=8):
    depot   = depot if depot is not None else n
    manager = pywrapcp.RoutingIndexManager(n + 1, m, depot)
    routing = pywrapcp.RoutingModel(manager)

    # distance matrix --------------------------------------------------------
    def dcb(from_idx, to_idx):
        i, j = manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)
        return int(dist[i][j])
    transit = routing.RegisterTransitCallback(dcb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    # capacity ---------------------------------------------------------------
    def qcb(idx):
        node = manager.IndexToNode(idx)
        return 0 if node == depot else demand[node]
    qidx = routing.RegisterUnaryTransitCallback(qcb)
    routing.AddDimensionWithVehicleCapacity(qidx, 0, capacity, True, "Load")

    # distance *dimension* so we can penalise the longest route --------------
    routing.AddDimension(transit, 0, sum(map(max, dist)), True, "RouteLen")
    route_len = routing.GetDimensionOrDie("RouteLen")
    # minimise the maximum “RouteLen” over all vehicles
    route_len.SetGlobalSpanCostCoefficient(1000)

    # allow vehicles to stay idle  -------------------------------------------
    # (drop Visited≥1 & fixed costs – that helps min-max objectives)
    # ------------------------------------------------------------------------
    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search.time_limit.FromSeconds(seconds)

    sol = routing.SolveWithParameters(search)
    if sol is None:
        raise RuntimeError("OR-Tools could not find a seed.")

    # extract routes ----------------------------------------------------------
    tours = []
    for v in range(m):
        idx, tour = routing.Start(v), []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != depot:
                tour.append(node + 1)          # 1-based
            idx = sol.Value(routing.NextVar(idx))
        if tour:                               # keep only non-empty tours
            tours.append(tour)

    # symmetry-breaking order: load ↓
    tours.sort(key=lambda r: -sum(demand[i-1] for i in r))
    return tours


def clarke_wright_seed(m, n, s, l, D):
    """Returns `seed[c]` (1-based customers) with  len(seed) == m  and
       ⋆ no empty tour
       ⋆ every customer appears once
       ⋆ tours ordered to satisfy  load[c]≥load[c+1],  …"""
    depot     = n
    max_cap   = max(l)

    # ── 1. Clarke-Wright merges on 0-based customers ────────────
    routes     = [[i] for i in range(n)]
    route_load = [s[i] for i in range(n)]
    route_of   = list(range(n))

    savings = [(D[i][depot] + D[depot][j] - D[i][j], i, j)
               for i in range(n) for j in range(i+1, n)]
    savings.sort(reverse=True)

    for _, i, j in savings:
        ri, rj = route_of[i], route_of[j]
        if ri == rj:                 # already same tour
            continue
        if route_load[ri] + route_load[rj] > max_cap:
            continue
        # endpoints?
        if (routes[ri][0] in (i, j) or routes[ri][-1] in (i, j)) and \
           (routes[rj][0] in (i, j) or routes[rj][-1] in (i, j)):
            if routes[ri][-1] != i:
                routes[ri].reverse()
            if routes[rj][0] != j:
                routes[rj].reverse()
            routes[ri].extend(routes[rj])
            route_load[ri] += route_load[rj]
            for cust in routes[rj]:
                route_of[cust] = ri
            routes[rj] = []

    routes = [r for r in routes if r]          # drop empty bins

    # ── 2. split until we have exactly m tours ──────────────────
    while len(routes) < m:
        # take last customer from heaviest tour → new singleton tour
        idx = max(range(len(routes)), key=lambda r: sum(s[c] for c in routes[r]))
        cust = routes[idx].pop()               # never empties because len< m ≤ n
        routes.append([cust])

    # (if after CW we had > m routes, merge light pairs until len==m)
    while len(routes) > m:
        routes.sort(key=lambda r: sum(s[c] for c in r))   # lightest first
        a = routes.pop(0);  b = routes.pop(0)
        if sum(s[c] for c in a+b) <= max_cap:
            routes.append(a+b)
        else:                         # cannot merge → keep both; break loop
            routes.extend([a, b]); break

    # after the splitting loop and before you sort routes
    for idx, r in enumerate(routes):
        changed = True
    while changed:
        changed = False
        for idx, r in enumerate(routes[:m]):                   # only the first m tours matter
            over = sum(s[i] for i in r) - l[idx]
            if over <= 0:
                continue                                       # already fits
            # try to move ONE customer out of r
            # pick the heaviest customer whose weight ≤ max spare capacity
            spare = [l[j] - sum(s[i] for i in routes[j]) for j in range(m)]
            best_tgt = max((j for j in range(m) if spare[j] >= 1),
                        key=lambda j: spare[j], default=None)
            if best_tgt is None:
                raise ValueError("total capacity < total demand ‒ infeasible instance")
            # choose customer to move: heaviest that fits in best_tgt
            cand = max((c for c in r if s[c] <= spare[best_tgt]),
                    key=lambda c: s[c], default=None)
            if cand is None:                                    # nothing fits -> give up
                raise ValueError("cannot rebalance routes within capacities")
            r.remove(cand)
            routes[best_tgt].append(cand)
            changed = True

    # ── 3. order tours to respect symmetry-breaking rows ─────────
    def route_key(route):
        load   = sum(s[i] for i in route)
        dist   = sum(D[a][b] for a,b in zip([depot]+route, route+[depot]))
        first  = route[0]
        return (-load, -dist, first)          # load↓, dist↓, firstIdx↑
    
    routes.sort(key=route_key)

    # ── 4. final 1-based seed  ───────────────────────────────────
    seed = [[cust + 1 for cust in r] for r in routes[:m]]
    # optional safety check
    assert len(seed) == m
    assert sorted(c for tour in seed for c in tour) == list(range(1, n+1))
    return seed


def MIP(instance_number):
    file_path = os.path.join('Instances', str(f'inst{instance_number}.dat'))
    parsed_data = utils.read_dat_file(file_path)

    m = parsed_data['m']
    n = parsed_data['n']
    s = parsed_data['s']
    l = sorted(parsed_data['l'], reverse=True)
    D = parsed_data['D']

    starting_time = datetime.now()
    time_limit = 5*60
    safe_bound = 5
    
    #######################################################
    #######################################################
    #######################################################

    #seed = clarke_wright_seed(m,n,s,l,D)       
    try:
        seed = ortools_seed(n, m, s, l, D)
        print('Seed found using or-tools!')
    except RuntimeError:
        print("OR-Tools had no seed ➜ using Clarke-Wright.")
        seed = clarke_wright_seed(m, n, s, l, D)
        print('Seed found using Clarke-Wright!')

    thread_solution = solve_with_timeout((m, n, s, l, D), time_limit-safe_bound, seed)

    #######################################################
    #######################################################
    #######################################################
    
    end_time = datetime.now() - starting_time
    status = "Undefined"
    sol = []

    if thread_solution is not None:
        objective, status, paths = thread_solution
        print(f"Objective value = {objective}")

        if str(objective).split('.')[-1] != '0':
            status = "Undefined"
        
        if status == "Optimal" or status == "Feasible":
            for c in range(m):
                print(f'Courier: {c+1}')
                matrix_values = [[0 for _ in range(n+1)] for _ in range(n+1)]

                for i in range(n+1):
                    for j in range(n+1):
                        if i != j:
                            matrix_values[i][j] = int(paths[c,i,j].varValue)
                        print(matrix_values[i][j], end=' ')
                    print()

                sol.append(find_path(matrix_values,c,n))

            order_solution(sol,m,l,parsed_data['l'])

    json_dict = {}
    json_dict['MIP'] = {}
    json_dict['MIP']['time'] = end_time.seconds if (end_time.seconds+safe_bound<time_limit) else time_limit
    json_dict['MIP']['optimal'] = True if (end_time.seconds+safe_bound<time_limit) else False
    json_dict['MIP']['obj'] = int(objective) if (status == "Optimal" or status == "Feasible") else None
    json_dict['MIP']['sol'] = sol

    with open(f'res/MIP/{str(int(instance_number))}.json', 'w') as outfile:
        json.dump(json_dict, outfile)

if __name__ == "__main__":
    MIP('21')