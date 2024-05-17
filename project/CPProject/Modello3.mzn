include "globals.mzn"; 
include "gecode.mzn";

% Data
int: m; 
int: n; 
array[1..m] of int: l;     
array[1..n] of int: s;     
array[1..n+1, 1..n+1] of int: D;                  

% Variabili
array[1..n] of var 1..m: x;
array[1..n+m+1] of var 1..n+1: y;  
int: total = sum(i in 1..n+1, j in 1..n+1)(D[i,j]);
array[1..m] of var n+1..total: max_distance_per_courier; 

/*Trucco usato*/
/*n+1,1,2,3,n+1,2,3,n+1,5,n+1*/
/*  3,2,1*/
/*0,3,5,6*/

/*non so se tenerlo*/
array[1..m] of var 1..n-m+1: cont;
array[0..m] of var 0..n: cum_cont;
constraint forall(i in 1..m)(count_eq(x, i, cont[i]));
constraint cum_cont[m] = n;
constraint cum_cont[0] = 0;
constraint forall(i in 1..m-1)(cum_cont[i] = cum_cont[i-1] + cont[i]);

constraint all_different_except(y,{n+1});

constraint y[1] = n+1;
constraint y[n+m+1] = n+1;
constraint y[2] != n+1;
constraint y[n+m] != n+1;
constraint count([y[j] == n+1 | j in 3..n+m-1], true) == m-1;

constraint forall(i in 1..m, j in 1..n)(
    (x[j] = i) <-> exists(c in cum_cont[i-1]+1+i..cum_cont[i]+i+1)(y[c] = j)
);

constraint bin_packing_capa(l, x, s);

% distance
constraint forall(i in 1..m)(
    sum(c in cum_cont[i-1]+i..cum_cont[i]+i)(
         D[y[c],y[c+1]] 
    ) = max_distance_per_courier[i]
);

var int: max_distance = max(i in 1..m)(max_distance_per_courier[i]);

% search_annotations
ann: search_ann;
% search_ann = int_search(y, input_order, indomain_min); % migliore attualmente
% search_ann = int_search(y, input_order, indomain_random); 
% search_ann = int_search(y, first_fail, indomain_min);
% search_ann = int_search(y, first_fail, indomain_random);
search_ann = int_search(y, dom_w_deg, indomain_min);
% search_ann = int_search(y, dom_w_deg, indomain_random);

% solve minimize max_distance; 
solve 
  ::search_ann 
minimize max_distance;


% Output results
output ["Assignments (courier:item) = ", show(x), "\n",
        "Maximum distance by any courier = ", show(max_distance), "\n",
        "Paths: ", show(y), "\n",
        "Maximum distance for each courier:\n", show(max_distance_per_courier), "\n"];