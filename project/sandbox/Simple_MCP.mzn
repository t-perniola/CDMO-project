include "globals.mzn";

%**********START REGION VARIABLES**********
%Instance variables

int: n; %items === nodes

set of int: NODES = 1..n;

int: l; %single courier load size
array[NODES] of int: s; %weights of the items

array[1..n+1, 1..n+1] of int: D;

%Decision variables
array[1..n+2] of var 0..n+1: path;
var int: cur_load;
var int: total_distance;
var int: path_length;
%**********END REGION VARIABLES**********

%**********START REGION CONSTRAINTS******
%Just for training. This won't be true in the project
constraint all_different_except_0([path[i] | i in 2..n]);

%Define initial node and final node
constraint path[1] = n+1;
constraint path[path_length] = n+1;

%Set to zero unvisited nodes (in our problem there will be unvisited nodes)
constraint forall(i in path_length+1..n+2)(path[i] = 0);

%The items weight cannot exceed the load size
constraint cur_load <= l;

constraint
    total_distance = sum(i in 1..n+1 where path[i] != 0 /\ path[i+1] != 0) (
        D[path[i], path[i+1]]
    );
    
constraint
    cur_load = sum(i in 1..n+2 where path[i] != 0 /\ path[i] != n+1) (
        s[path[i]]
    );
    
%All the adjacent values until path_lenth must be different than 0
constraint forall(i in 1..path_length-1)(path[i] != 0 /\ path[i+1] != 0);

%**********END REGION CONSTRAINTS********

solve minimize total_distance + (l - cur_load);

output ["distance = \(total_distance))\n"] ++
       [ show_int(1,path[i]) ++ " " |
         i in 1..n+2 ];