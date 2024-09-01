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
array[1..m] of var lb..ub: max_distance_per_courier; 




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

constraint alldifferent([x[i, n+1] | i in 1..m])::domain_propagation;
constraint forall(i in 1..m)(x[i,n+1] != n+1);
constraint forall(i in 1..m)(count(j in 1..n)(x[i, j] == n+1) == 1);

% Pruning
constraint forall(i in 1..m)(count(j in 1..n)(x[i, j] != j) <= ceil(n/m)+1);  


constraint
    forall(i in 1..m)(
        sum(j1 in 1..n)(
            if x[i, j1] == j1 then 0 else s[j1] endif
        ) <= l[i]
    );




%%%%%%%%% SYMMETRY BREAKING %%%%%%%%%%

constraint
    forall(i in 1..m-1 where l[i] == l[i+1])(
        lex_lesseq([x[i,k1] | k1 in 1..n+1], [x[i+1,k] | k in 1..n+1])
    )::domain_propagation;
    


   
%%%%%%%% OBJECTIVE %%%%%%%%%%
constraint
    forall(i in 1..m) (
        max_distance_per_courier[i] = sum([D[j1, x[i, j1]] | j1 in 1..n+1])
    );
        
                  
var int: max_distance = max(i in 1..m)(max_distance_per_courier[i]);

solve 
%:: int_search(x, first_fail, indomain_min)
:: int_search(x, random_order, indomain_min)
:: restart_luby(100) 
minimize max_distance;
  
% Output results
output [
    "Paths:\n",
    concat([concat([show(x[i, j]) ++ " " | j in 1..n+1]) ++ "\n" | i in 1..m]),
    "Maximum distance: ", show(max_distance), "\n"
];
