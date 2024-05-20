include "globals.mzn"; 
% include "lex_lesseq.mzn";

int: n;

% when we refer to items we're referring to their positions in the cartesian plane;
% namely, we're assuming a one-by-one correspondence between the two concepts.
set of int: Items = 1..n; % the i-th item is in the i-th distribution point
set of int: Timesteps = 0..n+1; % defines the order of the path that each courier follows

int: l;
array[Items] of int: s;  % items' sizes
array[1..n+1, 1..n+1] of int: D; % array of the distances

int: total = sum(i in 1..n+1, j in 1..n+1)(D[i,j]); % total distance

% Maximum distance traveled by each courier
var n+1..total: max_distance_per_courier;

% array of m Couriers and (n+1) timesteps:
% - the elements of the array are the picked-up items (0..n items) by the m-th courier at the n-th timestep
% Use 0 as a placeholder for no item
array[Timesteps] of var 0..n: tour;

% CONSTRAINTS
% starting point = ending point = n+1 for each courier
constraint tour[0] = 0;

constraint tour[n+1] = 0;

/*
% each timestep is different for each courier (excluding 0 placeholders)
constraint forall(c in Couriers)(
    alldifferent([tour[c,t] | t in 1..n where tour[c,t] != 0])
);*/

% one item can be picked up only once; this means that one item can be assigned only to one courier
constraint forall(i in Items)(
    sum(t in 1..n)(tour[t] = i) = 1
);

% each tour must start at timestep 1 and it cannot stop until it returns to the origin
constraint tour[1] != 0;

constraint forall(t in 2..n)(
                if tour[t] = 0 then tour[t+1] = 0 endif
); % No 0 between two consecutive items


% load size of each courier cannot be exceeded by the sum of the picked-up items' sizes
constraint sum(t in 1..n)(s[tour[t]]) <= l;

/*
% maximum distance is respected for each courier
constraint forall(c in Couriers)(
    sum(t in 1..n-1)(
      if t = 0 then % if it starts at the origin namely at timestep 0 ..
        if t+1 = 0 then
           D[tour[c,n+1], tour[c,n+1]] % it always be zero
        else 
           D[tour[c,n+1], tour[c,t+1]] % the distance matrix has as origin points the n+1 rows/columns
        endif
        
      elseif t+1 = 0 then % if t != 0 but the next one is 0, namely it returns to the origin  ..
        D[tour[c,t], tour[c,n+1]]
      else
          D[tour[c,t], tour[c,t+1]] % Distance between two consecutive points
      endif
    ) <= max_distance_per_courier[c]
);
*/

% Optimization target: Minimize the maximum distance traveled by any courier
solve minimize max_distance_per_courier;

% Output results
output [
    "Maximum distance by any courier = ", show(max_distance_per_courier), "\n",
    "Paths:\n",
    concat([show([tour[t] | t in Timesteps]) ++ "\n"]),
    "Maximum distance for each courier:\n",
    concat([show(max_distance_per_courier) ++ "\n"]),
    %"Distances between objects:\n",
    %concat([show([(tour[c,t], tour[c, t+1]) | t in 1..n-1]) ++ "\n" | c in Couriers]),
    %concat([show([D[tour[c,t], tour[c, t+1]] | t in 1..n-1]) ++ "\n" | c in Couriers]),
];
