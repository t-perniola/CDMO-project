include "globals.mzn";
include "gecode.mzn"; 

% predicate subcircuit(array [$$E] of var $$E: x)
% bin_packing_load

% Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int: m; 
int: n; 
array[1..m] of int: l;     
array[1..n] of int: s;     
array[1..n+1, 1..n+1] of int: D;  

% Variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x[i, j1] = j2 means that the ith courier will go from the j1th node to the j2th node
% if x[i, k] = k then the courier i is not going to visit the node k
array[1..m, 1..n+1] of var 1..n+1: x;

% symmetry breaking variable
array[1..m, 1..n] of var 0..1: b_y;

% aggiustare la maxdistance -> fare un preprocessing
int: total = sum(i in 1..n+1, j in 1..n+1)(D[i,j]);
array[1..m] of var n+1..total: max_distance_per_courier; % Maximum distance traveled by each courier

% array[1..m] of var min_load..max_load: load;
array[1..m] of var int: load;
array[1..n] of var 1..m : bins;
array[1..m] of var int: numpacks;

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


/*
% aggiustare i bounds per i load -> fare un preprocessing  ----  bin_packing_load
constraint
    forall(i in 1..m)(
        sum(j1 in 1..n)(
            if x[i, j1] == j1 then 0 else s[j1] endif
        ) <= l[i]
    );
*/

constraint bin_packing_load(load, bins, s);

% link bins and x
constraint 
    forall(j in 1..n)(
        x[bins[j], j] != j
    );

% constraint on bins
constraint
    forall(i in 1..m)(
        count_eq(bins, i, numpacks[i])
    );

constraint
    forall(i in 1..m)(
        load[i] == sum([(if x[i, j1] == j1 then 0 else s[j1] endif) | j1 in 1..n]) /\
        load[i] <= l[i] 
    );

% Channeling 
%constraint forall(i in 1..m, j in 1..n)(x[i, j] != j <-> b_y[i, j] = 1)::domain_propagation;

  
% Symmetry Breaking constraints %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ensuring exactly one 1 per column in b_y for the first n elements
/*
constraint forall(j in 1..n)(
    sum(i in 1..m)(b_y[i, j]) = 1
);
*/

% More size to bigger courier
constraint symmetry_breaking_constraint(forall(c1,c2 in 1..m where c2>c1 /\ l[c2]>=l[c1])(sum(j in 1..n)(b_y[c1,j]*s[j]) <= sum(j in 1..n)(b_y[c2,j]*s[j])));

% lex_lesseq
constraint symmetry_breaking_constraint(forall(c1,c2 in 1..m where l[c1]==l[c2] /\ c1<c2)(lex_lesseq([b_y[c1,j] | j in 1..n],[b_y[c2,j] | j in 1..n])));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%% SYMMETRY BREAKING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
/*
constraint
    forall(i in 1..m-1, z in i+1..m where z > i /\ l[i] >= l[z])(
        sum(j1 in 1..n)(if x[i, j1] == j1 then 0 else s[j1] endif) >= sum(j2 in 1..n)(if x[i, j2] == j2 then 0 else s[j2] endif)
    );
*/  

% aggiungere bins
constraint
    forall(i in 1..m-1, z in i+1..m where l[i] >= l[z])(
        load[i] >= load[z]
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
:: int_search(x, dom_w_deg, indomain_random) 
%:: int_search(x, first_fail, indomain_min)
:: int_search(b_y, first_fail, indomain_min)
:: restart_luby(100) 
:: relax_and_reconstruct(array1d(x), 83)
minimize max_distance;
  
% Output results
output [
    "Paths:\n",
    concat([concat([show(x[i, j]) ++ " " | j in 1..n+1]) ++ "\n" | i in 1..m]),
    "Maximum distance: ", show(max_distance), "\n"
];
