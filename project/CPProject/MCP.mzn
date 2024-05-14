include "globals.mzn"; % Packing constraints da vedere!! 

% Data
% n+1 nella matrice corrisponde all'origin point o
int: m; % m couriers
int: n; % n items
array[1..m] of int: l;     % Each courier i has a maximum load size li
array[1..n] of int: s;     % sj items sizes
array[1..n+1, 1..n+1] of int: D;  % Dij distance between distribution point i and distribution point j                 

% I couriers fanno un path tornano in origine e fanno altri path e così via
% oppure i couriers fanno un unico path e tornano in origine?
% Secondo me i couriers possono avere più paths
% Però per semplicità al momento si oconsidererà che ogni courier fa un path da 0 a 0 senza continuare

% Come constraint 

% Variables
% decision and optimization variables and domains 
array[1..n] of var 1..m: x;  % Decision variable: which rider picks which object
int: total = 2*sum(i in 1..n+1, j in 1..n+1)(D[i,j]);
array[1..m] of var 0..total: max_distance_per_courier; % Maximum distance traveled by each courier

% sto creando una matrice dove ho 0 se un courier non va su un oggetto e ho gli interi che mi certificano
% l'ordine in cui i courier vanno da n+1 agli items a n+1
% così poi mi calcolo i path
array[1..m,0..n+1] of var 0..n+1: path; % path


constraint 
    forall(i in 1..m)(
        path[i,0] = 1 /\ path[i,n+1] != 0 /\
        forall(j in 1..n)(
            path[i,j] < path[i,n+1] /\
            if path[i,j] > 0 then path[i,j] > path[i,0] endif
        )
);


% constraint che tra courier devono essere sempre diversi, se i ha 1 in colonna j allora il resto delle righe è 0
constraint forall(j in 1..n)(
    forall(i in 1..m)(
        (path[i,j] != 0) -> (forall(k in 1..m where k != i)(
            path[k,j] = 0
        ))
    )
);
/*
Traduzione con constraint globale
constraint forall(j in 1..n)(
    count([path[i,j] | i in 1..m], 0) <= m - 1
);
*/


% constraint at least sulle colonne
constraint
    forall(j in 1..n+1) (
        exists(i in 1..m) (
            path[i,j] != 0
        )
);


% all different constraint sulle righe
% constraint che tra i valori diversi da 0 su una riga questi devono essere tutti diversi
constraint forall(i in 1..m)(
    forall(j in 1..n, k in 1..n+1 where path[i,j] != 0 /\ j<k /\ path[i,k]!=0)(
            path[i,j] != path[i,k]
    )
);
/*
% Traduzione con constraint globale
constraint forall(i in 1..m)(
    all_different_except_0([path[i,j] | j in 1..n])
);
*/

% Collegamento tra x e i path
constraint forall(i in 1..m)(
    forall(j in 1..n where path[i,j] != 0)(
        x[j] = i
    )
);

% constraint bin_packing_capa(l, x, s);
% Constraints
% Load constraints for each courier
constraint forall(i in 1..m) (
    sum(j in 1..n)(if x[j] == i then s[j] else 0 endif) <= l[i]
);

% successività
constraint forall(i in 1..m)(
    forall(j in 0..n)(
        if path[i,j] != 0 then
            exists(k in 1..n+1 where k > j /\ path[i,k] != 0)(
                abs(path[i,j] - path[i,k]) = 1
            )
        else
            true % Non impongo alcuna restrizione se path[i,j] = 0
        endif
    )
);



% valore che mi calcola la lunghezza dei path
constraint 
    forall(i in 1..m)(
          sum(j in 0..n, k in 1..n+1 where path[i,j] != 0 /\ path[i,k] != 0 /\ j < k /\ abs(path[i,k]-path[i,j])==1)(
                if j == 0 then D[n+1,k] else
              D[k,j] endif) = max_distance_per_courier[i]
);


% Optimization target: Minimize the maximum distance traveled by any courier
var int: max_distance = max(i in 1..m)(max_distance_per_courier[i]);

solve minimize max_distance; 
% solve :: int_search(x, input_order, indomain_min)  minimize distance;

% Output results
output ["Assignments (courier:item) = \(x)\n",
        "Maximum distance by any courier = \(max_distance)\n"];
output ["Paths:\n"];
output [show(path[i,j]) ++ " " ++ if j == n+1 then "\n" else "" endif | i in 1..m, j in 0..n+1];
output ["Maximum distance for each courier:\n"];
output [show(max_distance_per_courier[i]) ++ "\n" | i in 1..m];
output show(total);