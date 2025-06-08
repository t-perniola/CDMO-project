include "globals.mzn";
include "gecode.mzn";

int: m; % numero di corrieri
int: n; % numero di clienti
array[1..m] of int: l; % capacità di ciascun corriere
array[1..n] of int: s; % domanda dei clienti
array[1..n+1, 1..n+1] of int: D; % matrice delle distanze (n+1 = deposito)
int: lb; 
int: ub;

% Variabili
array[1..m, 1..n+1, 1..n+1] of var 0..1: arc_used;
array[1..m, 1..n+1, 1..n+1] of var 0..sum(s): flow;

% Variabile ausiliaria: corriere assegnato a ciascun cliente
array[1..n] of var 1..m: courier_of_client;

% Nessun arco su sé stesso
constraint forall(i in 1..m, u in 1..n+1) (
  arc_used[i,u,u] = 0
);

% Vincolo principale: per ogni cliente, arco entrante e uscente devono essere usati dallo stesso corriere
constraint forall(j in 1..n)(
  forall(i in 1..m)(
    (courier_of_client[j] = i) <->
    (
      sum(u in 1..n+1 where u != j)(arc_used[i,u,j]) = 1 /\
      sum(v in 1..n+1 where v != j)(arc_used[i,j,v]) = 1
    )
  )
);

% Ogni corriere parte e torna al deposito una sola volta
constraint forall(i in 1..m)(
  sum(v in 1..n)(arc_used[i,n+1,v]) = 1 /\
  sum(u in 1..n)(arc_used[i,u,n+1]) = 1
);

% Capacità rispettata: ogni corriere trasporta al più la sua capacità
constraint forall(i in 1..m)(
  sum(j in 1..n)(
    s[j] * (if courier_of_client[j] = i then 1 else 0 endif)
  ) <= l[i]
);

% Conservazione del flusso e domanda per ogni cliente e corriere
constraint forall(i in 1..m)(
  forall(v in 1..n)(
    sum(u in 1..n+1 where u != v)(flow[i,u,v]) -
    sum(w in 1..n+1 where w != v)(flow[i,v,w]) =
    s[v] * (if courier_of_client[v] = i then 1 else 0 endif)
  )
);

constraint forall(i in 1..m)(
  sum(v in 1..n)(flow[i,n+1,v]) =
  sum(j in 1..n)(s[j] * (if courier_of_client[j] = i then 1 else 0 endif))
);

constraint forall(i in 1..m, u in 1..n+1, v in 1..n+1)(
  flow[i,u,v] <= sum(s) * arc_used[i,u,v]
);

array[1..m] of var 0..sum(D): max_distance_per_courier;
constraint forall(i in 1..m)(
  max_distance_per_courier[i] = sum(u in 1..n+1, v in 1..n+1)(
    D[u,v] * arc_used[i,u,v]
  )
);

var lb..ub: max_distance = max(i in 1..m)(max_distance_per_courier[i]);


solve minimize max_distance;

% Output
output [
  "Percorsi per corriere:\n"
] ++
[
  "Corriere " ++ show(i) ++ ": " ++
  concat([
    if fix(arc_used[i,u,v]) = 1 then
      "[" ++ show(u) ++ "→" ++ show(v) ++ "] "
    else
      ""
    endif
  | u in 1..n+1, v in 1..n+1]) ++ "\n"
| i in 1..m
] ++ [
  "Distanza massima: ", show(max_distance), "\n"
];