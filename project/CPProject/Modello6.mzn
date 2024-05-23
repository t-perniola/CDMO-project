include "all_different_except.mzn"; 
include "gecode.mzn";

% Data
int: m; 
int: n; 
array[1..m] of int: l;     
array[1..n] of int: s;     
array[1..n+1, 1..n+1] of int: D;                  

% Variabili
array[1..n+m+1] of var 1..n+1: y; 
array[0..m] of var 1..m+n+1: ind; 
int: total = sum(i in 1..n+1, j in 1..n+1)(D[i,j]);
array[1..m] of var n+1..total-n: max_distance_per_courier; 

% Constraints 
constraint y[1] = n+1;
constraint y[n+m+1] = n+1;
constraint y[2] != n+1;
constraint y[n+m] != n+1;
constraint ind[0] = 1;
constraint ind[m] = n+m+1;

constraint forall(i in 1..m-1) (y[ind[i]] == n+1);
constraint forall(i in 3..n+m-1)(y[i] = n+1 -> y[i-1] != n+1 /\ y[i+1] != n+1);
constraint forall(i in 0..m-1)(ind[i] + ceil(n/m) <= ind[i+1] /\ ind[i+1]-ind[i]-1 <= n-m+1)::domain_propagation;

constraint all_different_except(y,{n+1})::domain_propagation;

constraint forall(i in 1..m)(sum(c in ind[i-1]+1..ind[i]-1)(s[y[c]]) <= l[i]);

% distance
constraint forall(i in 1..m)(
    sum(c in ind[i-1]..ind[i]-1)(
         D[y[c],y[c+1]] 
    ) = max_distance_per_courier[i]
);

var int: max_distance = max(i in 1..m)(max_distance_per_courier[i]);

solve 
%:: int_search(y, first_fail, indomain_min) 
:: int_search(y, dom_w_deg, indomain_random) 
:: restart_luby(10000)
:: relax_and_reconstruct(y, 83)
minimize max_distance;
  
% Output results
output ["Paths: ", show(y), "\n",
        "indexes: ", show(ind), "\n",
        "Maximum distance for each courier: ", show(max_distance), "\n"
];
