include "globals.mzn";
include "gecode.mzn";

%**********START REGION VARIABLES**********
%Instance variables

int: m; %couriers
int: n; %items === nodes

set of int: COURIERS = 1..m;
set of int: NODES = 1..n;

array[COURIERS] of int: l; %single courier load size
array[NODES] of int: s; %weights of the items

array[1..n+1, 1..n+1] of int: D;

int: avg_D = sum(D) div (n*n);
int: min_D = min(D);
int: max_nodes = (n div m)+3;

%Decision variables
array[COURIERS, 1..max_nodes] of var 0..n+1: path;
array[COURIERS, NODES] of var 0..1: b_path;
array[COURIERS] of var int: total_distance;
array[COURIERS] of var 3..max_nodes: path_length;
%**********END REGION VARIABLES**********

%**********START REGION CONSTRAINTS******
%Couriers cannot visit same node twice (nor stay in the same node).
constraint forall(c in COURIERS)( all_different_except_0([path[c,i] | i in 2..max_nodes-1]) )::domain_propagation;

%Define initial node and final node
constraint forall(c in COURIERS) (path[c,1] = n+1);
constraint forall(c in COURIERS) (path[c,path_length[c]] = n+1);

%Set to zero unvisited nodes
constraint forall(c in COURIERS)(
              forall(i in path_length[c]+1..max_nodes)(path[c,i] = 0)
           )::domain_propagation;
           

%The items weight cannot exceed the load size
%constraint forall(c in COURIERS) (sum(j in NODES)(b_path[c,j]*s[j]) <= l[c])::domain_propagation;
constraint forall(c in COURIERS)( sum(j in 2..path_length[c]-1)(s[path[c,j]]) <= l[c])::domain_propagation;

%If you have more load size than me, then your load must be greater than mine
constraint forall(c1,c2 in COURIERS where c2>c1 /\ l[c2]>=l[c1])( sum(j in NODES)(b_path[c1,j]*s[j]) <= sum(j in NODES)(b_path[c2,j]*s[j]) )::bounds_propagation;

%All the couriers must visit different nodes
constraint all_different_except(path, {0,n+1})::domain_propagation;
constraint forall(j in NODES)(count(c in COURIERS)(b_path[c,j]=1) = 1)::domain_propagation;

constraint forall(c in COURIERS)(
              total_distance[c] = sum(i in 1..path_length[c]-1) (D[path[c,i], path[c,i+1]])
           )::bounds_propagation;
    
    
%All the adjacent values until path_length must be different from 0
constraint forall(c in COURIERS)(
              forall(i in 2..path_length[c]-1)(path[c,i] != 0 /\ path[c,i+1] != 0)
           )::bounds_propagation;
%constraint forall(c in COURIERS) (among(0,[path[c,j] | j in 1..path_length[c]-1], {0}));
                     
%All the nodes must be visited
constraint among(n, path, NODES)::domain_propagation;


%The path length of the single courier, in order to be balanced, cannot be longer than n-m
constraint forall(c in COURIERS)(path_length[c] <= max_nodes)::domain_propagation;
constraint forall(c in COURIERS)(count(j in NODES)(b_path[c,j]=1) <= max_nodes)::domain_propagation; 

%Channeling constraint
constraint forall(c in COURIERS, j in NODES) (b_path[c,j] = 1 <-> exists(j2 in 2..path_length[c])(path[c,j2]=j) )::domain_propagation;

%symmetry breaking constraint
constraint forall(c1, c2 in COURIERS where l[c1]==l[c2] /\ c1<c2)(lex_lesseq([b_path[c1,j] | j in NODES],[b_path[c2,j] | j in NODES]))::domain_propagation;

%constraint forall(c in COURIERS, j in 2..path_length[c]-1)(if D[path[c,j],path[c,j+1]]==D[path[c,j+1],path[c,j]] then path[c,j]<path[c,j+1] endif)::domain_propagation; %ROTTO

%constraint forall(e1,e2 in NODES where s[e1]==s[e2] /\ e1<e2)() ?

%Objective function bounds
%constraint forall(c in COURIERS)(total_distance[c] <= max(total_distance))::domain_propagation;
%constraint forall(c in COURIERS)(total_distance[c] < ((n div m)+3)*avg_D /\ total_distance[c] > min_D)::domain_propagation;
%constraint forall(c1,c2 in COURIERS where c1<c2)(total_distance[c1] < total_distance[c2] + avg_D \/ total_distance[c1] > total_distance[c2] - avg_D)::domain_propagation;

%**********END REGION CONSTRAINTS********

ann: search_ann;
ann: bool_search_ann;
search_ann = int_search(path, dom_w_deg, indomain_random); 
bool_search_ann = int_search(b_path, dom_w_deg, indomain_min);

solve 
  :: search_ann 
  :: bool_search_ann
  :: int_search(path_length, first_fail, indomain_min)
  %:: restart_geometric(100,50000)
  :: restart_luby(10000) 
  :: relax_and_reconstruct(array1d(path), 50)
  minimize max(total_distance); % + 10*sum(c in COURIERS) (l[c] - cur_load[c]);
  %minimize max(c in COURIERS)( sum(j in 1..path_length[c]-1)(D[path[c,j], path[c,j+1]]) );

output ["distance = \(max(total_distance)))))\n"] ++
       [ show_int(1,path[c,j]) ++ " " ++ 
         if j == max_nodes then "\n" else "" endif|
         c in COURIERS, j in 1..max_nodes ];