%% CHUFFED MODEL %%%%%%%%%%%%%%
include "globals.mzn";
include "chuffed.mzn";

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


%%%%%%%%% SYMMETRY BREAKING %%%%%%%%%%
constraint
    forall(i in 1..m-1, z in i+1..m where z > i /\ l[i] >= l[z])(
        sum(j1 in 1..n)(if x[i, j1] == j1 then 0 else s[j1] endif) >= sum(j2 in 1..n)(if x[i, j2] == j2 then 0 else s[j2] endif)
    );


constraint
    forall(i in 1..m-1, z in i+1..m where z>i /\ l[i] == l[z])(
        lex_lesseq([x[z,k1] | k1 in 1..n+1], [x[i,k] | k in 1..n+1])
    )::domain_propagation;
   
%%%%%%%% OBJECTIVE %%%%%%%%%%
constraint
    forall(i in 1..m) (
        max_distance_per_courier[i] = sum([D[j1, x[i, j1]] | j1 in 1..n+1])
    );
        
                  
var lb..ub: max_distance = max(i in 1..m)(max_distance_per_courier[i]);


% Best configuration found empirically
solve 
:: int_search(x, random_order, indomain_min)
:: restart_luby(100) 
minimize max_distance;
  
% Output results
output [
    "Paths:\n",
    concat([concat([show(x[i, j]) ++ " " | j in 1..n+1]) ++ "\n" | i in 1..m]),
    "Maximum distance: ", show(max_distance), "\n"
];
