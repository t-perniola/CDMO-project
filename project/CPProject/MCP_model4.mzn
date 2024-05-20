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

%Decision variables
array[COURIERS, 1..n+2] of var 0..n+1: path;
array[COURIERS] of var min(s)..sum(s): cur_load;
array[COURIERS] of var min(D)..sum(D): total_distance;
array[COURIERS] of var 3..n-m+3: path_length;
%**********END REGION VARIABLES**********

%**********START REGION CONSTRAINTS******
%Just for training. This won't be true in the project
constraint forall(c in COURIERS)( all_different_except_0([path[c,i] | i in 2..n]) )::domain_propagation;

%Define initial node and final node
constraint forall(c in COURIERS) (path[c,1] = n+1);
constraint forall(c in COURIERS, i in 1..n+1)(if path[c,i] != 0 then path[c,i] != path[c,i+1] endif);
%constraint exactly(2*m, path, n+1)::domain_propagation;
constraint forall(c in COURIERS) (path[c,path_length[c]] = n+1);

%Set to zero unvisited nodes
constraint forall(c in COURIERS)(
              forall(i in path_length[c]+1..n+2)(path[c,i] = 0)
           );
           

%The items weight cannot exceed the load size
constraint forall(c in COURIERS) (cur_load[c] <= l[c]);

%All the couriers must visit different nodes
constraint all_different_except(path, {0,n+1})::domain_propagation;

constraint forall(c in COURIERS)(
              total_distance[c] = sum(i in 1..n+1 where path[c,i] != 0 /\ path[c,i+1] != 0) (D[path[c,i], path[c,i+1]])
           );
    
constraint forall(c in COURIERS)(
                cur_load[c] = sum(i in 2..n+2 where path[c,i] != 0 /\ path[c,i] != n+1) (s[path[c,i]])
           );
    
%All the adjacent values until path_lenth must be different than 0
constraint forall(c in COURIERS)(
              forall(i in 1..path_length[c]-1)(path[c,i] != 0 /\ path[c,i+1] != 0)
           );
           
%All the nodes must be visited
%constraint sum(path) - 2*m*(n+1) = SUM_OF_NODES;
%constraint sum([1 | c in COURIERS, i in 2..n+1 where path[c, i] != 0 /\ path[c, i] != n+1]) = n;
constraint among(n, path, NODES)::domain_propagation;

%The path length of the single courier, in order to be balanced,
%cannot be longer than n-m
%constraint forall(c in COURIERS)(path_length[c] <= n-m+3);

%symmetry breaking constraint
constraint forall(c1 in 1..m, c2 in 1..m where c1<c2)(if l[c1] == l[c2] then lex_lesseq([path[c1,j] | j in 1..n+2], [path[c2,j] | j in 1..n+2]) endif)::domain_propagation;

%**********END REGION CONSTRAINTS********

ann: search_ann;
search_ann = int_search(path, dom_w_deg, indomain_min); 

solve 
  :: search_ann 
  :: int_search(path_length, first_fail, indomain_min)
  %:: restart_luby(250) 
  %:: relax_and_reconstruct(array1d(path), 85) 
  minimize max(total_distance); % + 10*sum(c in COURIERS) (l[c] - cur_load[c]);

output ["distance = \(max(total_distance)))))\n"] ++
       [ show_int(1,path[c,j]) ++ " " ++ 
         if j == n+2 then "\n" else "" endif|
         c in COURIERS, j in 1..n+2 ];