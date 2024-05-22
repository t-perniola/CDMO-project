include "globals.mzn";
include "gecode.mzn";

%**********START REGION VARIABLES**********
%Instance variables

int: m; %couriers
int: n; %items === nodes

set of int: COURIERS = 1..m;
set of int: NODES = 1..n;
int: SUM_OF_NODES = sum(i in NODES)(i);

array[COURIERS] of int: l; %single courier load size
array[NODES] of int: s; %weights of the items

array[1..n+1, 1..n+1] of int: D;

int: avg_D = sum(D) div (n*n);
int: min_D = min(D);

%Decision variables
array[COURIERS, 1..n+2] of var 0..n+1: path;
array[COURIERS, NODES] of var 0..1: b_path;
array[COURIERS] of var int: total_distance;
array[COURIERS] of var 3..n-m+3: path_length;
%**********END REGION VARIABLES**********

%**********START REGION CONSTRAINTS******
%Couriers cannot visit same node twice (nor stay in the same node).
constraint forall(c in COURIERS)( all_different_except_0([path[c,i] | i in 2..n+2]) )::domain_propagation;

%Define initial node and final node
constraint forall(c in COURIERS) (path[c,1] = n+1);
constraint forall(c in COURIERS) (path[c,path_length[c]] = n+1);

%Set to zero unvisited nodes
constraint forall(c in COURIERS)(
              forall(i in path_length[c]+1..n+2)(path[c,i] = 0)
           );
           

%The items weight cannot exceed the load size
%constraint forall(c in COURIERS) (cur_load[c] <= l[c]);
constraint forall(c in COURIERS) (sum(j in NODES)(b_path[c,j]*s[j]) <= l[c]);

%All the couriers must visit different nodes
%constraint forall(j in NODES)(all_different_except_0([b_path[c,j] | c in COURIERS]))::domain_propagation;
%constraint forall(c1,c2 in COURIERS, j1,j2 in 2..n+2 where c1<c2 /\ j1<j2 /\ path[c1,j1] != 0 /\ path[c1,j1] != n+1 /\ path[c2,j2] != 0 /\ path[c2,j2] != n+1)(path[c1,j1] != path[c2,j2]);
constraint all_different_except(array1d(path), {0,n+1})::domain_propagation;

constraint forall(c in COURIERS)(
              total_distance[c] = sum(i in 1..n+1 where path[c,i] != 0 /\ path[c,i+1] != 0) (D[path[c,i], path[c,i+1]])
           );
    
    
%All the adjacent values until path_length must be different than 0
constraint forall(c in COURIERS)(
              forall(i in 1..path_length[c]-1)(path[c,i] != 0 /\ path[c,i+1] != 0)
           );
           
%All the nodes must be visited
constraint among(n, path, NODES)::domain_propagation;

%The path length of the single courier, in order to be balanced, cannot be longer than n-m
constraint forall(c in COURIERS)(path_length[c] <= n-m+3);

%Channeling constraint
constraint forall(c in COURIERS, j in NODES) (b_path[c,j] = 1 <-> exists(j2 in 2..path_length[c])(path[c,j2]=j) );

%symmetry breaking constraint
constraint forall(c1 in 1..m, c2 in 1..m where c1<c2)(if l[c1] == l[c2] then lex_lesseq([path[c1,j] | j in 1..n+2], [path[c2,j] | j in 1..n+2]) endif)::domain_propagation;
%constraint forall(c1, c2 in COURIERS where l[c1]==l[c2] /\ c1<c2)(lex_lesseq([b_path[c1,j] | j in NODES],[b_path[c2,j] | j in NODES]))::domain_propagation;
%constraint forall(c1, c2 in COURIERS where l[c1]==l[c2] /\ c1<c2)(lex_lesseq([b_path[c1,j] | j in NODES],[b_path[c2,j1] | j1 in NODES]))::domain_propagation; 
constraint forall(c in COURIERS, j in 2..path_length[c])(if D[path[c,j],path[c,j+1]]==D[path[c,j+1],path[c,j]] then path[c,j]<path[c,j+1] endif);

constraint forall(c in COURIERS)(total_distance[c] < (n-m+3)*avg_D /\ total_distance[c] > min_D);

constraint forall(c1,c2 in COURIERS where c1<c2)(total_distance[c1] < total_distance[c2] + avg_D \/ total_distance[c1] > total_distance[c2] - avg_D);

%**********END REGION CONSTRAINTS********

ann: search_ann;
search_ann = int_search(path, dom_w_deg, indomain_random); 

solve 
  :: search_ann 
  %:: restart_geometric(10,10000)
  :: int_search(path_length, first_fail, indomain_random)
  %:: int_search(cur_load, first_fail, indomain_min)
  %:: restart_luby(250) 
  %:: relax_and_reconstruct(array1d(path), 85) 
  minimize max(total_distance); % + 10*sum(c in COURIERS) (l[c] - cur_load[c]);

output ["distance = \(max(total_distance)))))\n"] ++
       [ show_int(1,path[c,j]) ++ " " ++ 
         if j == n+2 then "\n" else "" endif|
         c in COURIERS, j in 1..n+2 ];