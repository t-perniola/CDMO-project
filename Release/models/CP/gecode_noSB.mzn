include "globals.mzn";
include "gecode.mzn"; 

% This model uses the predicate subcircuit(array[$$E] of var $$E: x)
% So it is a **successor-based routing model**.
% - Successor model: uses the `subcircuit` predicate to ensure each courier's path is a valid cycle 
%   (with possible self-loops for unvisited nodes).
%
% - x[i,j] = k means courier i goes from node j to node k.
%   If x[i,j] == j, then node j is not visited by courier i (it's a self-loop).

% Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int: m; 
int: n; 
array[1..m] of int: l;     
array[1..n] of int: s;     
array[1..n+1, 1..n+1] of int: D;  
int: lb; 
int: ub;

% Variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
array[1..m, 1..n+1] of var 1..n+1: x;
int: total = sum(i in 1..n+1, j in 1..n+1)(D[i,j]);
array[1..m] of var n+1..total: max_distance_per_courier; % Maximum distance traveled by each courier




% Constraints %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
constraint
    forall(i in 1..m) (
        subcircuit([x[i, j] | j in 1..n+1])
);
    
% each package is delivered exactly once
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
 
   
   
   
%%%%%%%% OBJECTIVE %%%%%%%%%%
constraint
    forall(i in 1..m) (
        max_distance_per_courier[i] = sum([D[j1, x[i, j1]] | j1 in 1..n+1])
    );
        
                  
var lb..ub: max_distance = max(i in 1..m)(max_distance_per_courier[i]);


% Search strategy configuration
solve 
% 1)
%:: int_search(x, dom_w_deg, indomain_min)

% 2)
:: int_search(x, dom_w_deg, indomain_random)  
:: restart_luby(100) 
:: relax_and_reconstruct(array1d(x), 83)
minimize max_distance;
  
% Output results
output [
    "Paths:\n",
    concat([concat([show(x[i, j]) ++ " " | j in 1..n+1]) ++ "\n" | i in 1..m]),
    "Maximum distance: ", show(max_distance), "\n"
];
