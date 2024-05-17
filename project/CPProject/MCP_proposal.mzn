include "globals.mzn"; 
% include "lex_lesseq.mzn";

int: m;
int: n;

% when we refer to items we're referring to their positions in the cartesian plane;
% namely, we're assuming a one-by-one correspondence between the two concepts.
set of int: Couriers = 1..m;
set of int: Items = 1..n; % the i-th item is in the i-th distribution point
set of int: Timesteps = 0..n+1; % defines the order of the path that each courier follows

array[Couriers] of int: l;  % Each courier i has a maximum load size li
array[Items] of int: s;  % items' sizes
array[1..n+1, 1..n+1] of int: D; % array of the distances

int: total = sum(i in 1..n+1, j in 1..n+1)(D[i,j]); % total distance

% Maximum distance traveled by each courier
array[Couriers] of var 0..total: max_distance_per_courier;

% array of m Couriers and (n+1) timesteps:
% - the elements of the array are the picked-up items (0..n items) by the m-th courier at the n-th timestep
% Use 0 as a placeholder for no item
array[Couriers, Timesteps] of var 0..n: tour;

% starting point = ending point for each courier
constraint forall(c in Couriers)(
    tour[c,0] = 0 /\ tour[c,n+1] = 0
);

% each timestep is different for each courier (excluding 0 placeholders)
constraint forall(c in Couriers)(
    alldifferent([tour[c,t] | t in 1..n where tour[c,t] != 0]) 
);

% one item can be picked up only once; this means that one item can be assigned only to one courier
constraint forall(i in Items)(
    sum(c in Couriers, t in 1..n)(tour[c,t] = i) = 1
);

% load size of each courier cannot be exceeded by the sum of the picked-up items' sizes
constraint forall(c in Couriers)(
    sum(t in 1..n)(s[tour[c, t]]) <= l[c]
);

% maximum distance is respected for each courier
constraint forall(c in Couriers)(
    sum(t in 1..n)(
        D[tour[c,t], tour[c,t+1]]
    ) <= max_distance_per_courier[c]
);

% Optimization target: Minimize the maximum distance traveled by any courier
var int: max_distance = max(i in 1..m)(max_distance_per_courier[i]);
solve minimize max_distance;

% Output results
output [
    "Maximum distance by any courier = ", show(max_distance), "\n",
    "Paths:\n",
    concat([show([tour[c,t] | t in Timesteps]) ++ "\n" | c in Couriers]),
    "Maximum distance for each courier:\n",
    concat([show(max_distance_per_courier[c]) ++ "\n" | c in Couriers])
];
