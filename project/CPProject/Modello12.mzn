%% CHUFFED MODEL %%%%%%%%%%%%%%
include "globals.mzn";
include "chuffed.mzn";

% predicate subcircuit(array [$$E] of var $$E: x)
% bin_packing_load

% Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int: m; 
int: n; 
array[1..m] of int: l;     
array[1..n] of int: s;     
array[1..n+1, 1..n+1] of int: D;  

% Variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
array[1..m, 1..n+1] of var 1..n+1: x;

% symmetry breaking variable
%array[1..m, 1..n] of var 0..1: b_y;

% aggiustare la maxdistance -> fare un preprocessing
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


    
% aggiustare i bounds per i load -> fare un preprocessing  ----  bin_packing_load
constraint
    forall(i in 1..m)(
        sum(j1 in 1..n)(
            if x[i, j1] == j1 then 0 else s[j1] endif
        ) <= l[i]
    );


% Channeling 
%constraint forall(i in 1..m, j in 1..n)(x[i, j] != j <-> b_y[i, j] = 1)::domain_propagation;

  
% Symmetry Breaking constraints %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ensuring exactly one 1 per column in b_y for the first n elements

% More size to bigger courier
%constraint symmetry_breaking_constraint(forall(c1 in 1..m-1, c2 in c1+1..m where c2>c1 /\ l[c1]>=l[c2])(sum(j in 1..n)(b_y[c1,j]*s[j]) >= sum(j in 1..n)(b_y[c2,j]*s[j])));

% lex_lesseq
%constraint symmetry_breaking_constraint(forall(c1 in 1..m-1, c2 in c1+1..m where l[c1]==l[c2] /\ c2<c1)(lex_lesseq([b_y[c2,j] | j in 1..n],[b_y[c1,j1] | j1 in 1..n])));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
