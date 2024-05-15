include "globals.mzn"; 
% include "lex_lesseq.mzn";

% Data
int: m; 
int: n; 
array[1..m] of int: l;     
array[1..n] of int: s;     
array[1..n+1, 1..n+1] of int: D;             


% decision and optimization variables and domains 
array[1..n] of var 1..m: x;  
int: total = sum(i in 1..n+1, j in 1..n+1)(D[i,j]);
array[1..m] of var n+1..total: max_distance_per_courier; 
array[1..m,1..n+1] of var 0..n: path; % path mxn dove n in questo caso è l'odine, 0 si è arrivati all'origine
% path[1,1] = 3 dice che corriere 1 ha preso come primo l'oggetto 3

% soluzione del tipo:
% 146777
% 235555

constraint forall(i in 1..m)(path[i,n+1]=0);

constraint forall(j in 1..n)(all_different_except_0([path[i,j] | i in 1..m]));
constraint forall(i in 1..m)(all_different_except_0([path[i,j] | j in 1..n]));


% Collegamento tra x e i path
constraint forall(j in 1..n, i in 1..m)(
          if path[i,j] != 0 then x[path[i,j]] = i endif
);


constraint bin_packing_capa(l, x, s);


% Caso calcolo path maximum GENERALE, prendendo il minimo
constraint forall(i in 1..m)(
    sum(j in 1..n where path[i,j] != 0)(
        if j == 1 then 
            D[n+1,path[i,j]]
        elseif j == n then
            D[path[i,j], n+1]
        else 
            D[path[i,j-1],path[i,j]] 
        endif 
    ) = max_distance_per_courier[i]
);


var int: max_distance = max(i in 1..m)(max_distance_per_courier[i]);

solve minimize max_distance; 


% Output results
output ["Assignments (courier:item) = \(x)\n",
        "Maximum distance by any courier = \(max_distance)\n"];
output ["Paths:\n"];
output [show(path[i,j]) ++ " " ++ if j == n then "\n" else "" endif | i in 1..m, j in 1..n+1];
output ["Maximum distance for each courier:\n"];
output [show(max_distance_per_courier[i]) ++ "\n" | i in 1..m];
output show(total);