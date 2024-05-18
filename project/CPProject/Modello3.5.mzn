include "globals.mzn"; 
include "gecode.mzn";

% Data
int: m; 
int: n; 
array[1..m] of int: l;     
array[1..n] of int: s;     
array[1..n+1, 1..n+1] of int: D;                  

% Variabili
int: total = sum(i in 1..n+1, j in 1..n+1)(D[i,j]);
array[1..n] of var 1..m: x;
array[1..n+m+1] of var 1..n+1: y;  
% array[0..m] of var 1..n+m+1: cum_cont;
array[0..m] of var 1..m+n+1: ind; % array[1..m-1] of var 3..m+n-1: ind;
array[1..m] of var n+1..total-n: max_distance_per_courier; 

constraint ind[0] = 1;
constraint ind[m] = n+m+1;
constraint forall(i in 1..m-1) (y[ind[i]] == n+1);
% constraint forall(i in 1..m-1)(ind[i] = cum_cont[i]);
% constraint cum_cont[m] = n+m+1;
% constraint cum_cont[0] = 1;

constraint all_different_except(y,{n+1});

constraint y[1] = n+1;
constraint y[n+m+1] = n+1;
constraint y[2] != n+1;
constraint y[n+m] != n+1;
constraint count([y[j] | j in 3..n+m-1], n+1) == m-1;

constraint forall(i in 1..m, j in 1..n)(
    x[j] = i <-> exists(c in ind[i-1]+1..ind[i]-1)(y[c] = j)
);

constraint bin_packing_capa(l, x, s);

% distance
constraint forall(i in 1..m)(
    sum(c in ind[i-1]..ind[i]-1)(
         D[y[c],y[c+1]] 
    ) = max_distance_per_courier[i]
);

var int: max_distance = max(i in 1..m)(max_distance_per_courier[i]);

ann: search_ann;
search_ann = int_search(y, input_order, indomain_min); 

solve 
  ::search_ann 
minimize max_distance;


% Output results
output ["Assignments (courier:item) = ", show(x), "\n",
        "Maximum distance by any courier = ", show(max_distance), "\n",
        "Paths: ", show(y), "\n",
        "Maximum distance for each courier: ", show(max_distance_per_courier), "\n",
        "cum_counts: ", show(ind)];