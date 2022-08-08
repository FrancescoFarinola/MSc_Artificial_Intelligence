include "globals.mzn";

% ****** Defining the problem ******

% *** Inputs ***
int: plate_w; %Width of the plate
int: n; % Number of total circuits
set of int: CIRCUITS = 1..n; % Enumerate circuits
array[CIRCUITS] of int: width;  % Widths of the circuits
array[CIRCUITS] of int: height; % Heights of the circuits

% *** Additional variables *** - Plate height (variable to minimize)
% Plate height is between the lower and upper bound which are respectively the max circuit height and the sum of all heights
var max(height)..sum(height): plate_h; 
array[CIRCUITS] of var bool : rotation; 
array[CIRCUITS] of var int: real_width = [if rotation[i] then height[i] else width[i] endif | i in CIRCUITS];
array[CIRCUITS] of var int: real_height = [if rotation[i] then width[i] else height[i] endif | i in CIRCUITS];

% *** Outputs *** - Coordinates (x,y) for each circuit
array[CIRCUITS] of var 0..plate_w - min(width): c_x;  
array[CIRCUITS] of var 0..sum(height) - min(height): c_y;

% ****** Constraints ******
% *** Reducing domain - Limit each circuit to appear in the board - w/o going outside***
constraint forall(i in CIRCUITS) (c_x[i] + real_width[i] <= plate_w /\
                                  c_y[i] + real_height[i] <= plate_h);

% *** Globals for no overlapping ***
constraint diffn(c_x,c_y,real_width,real_height);
constraint cumulative(c_x, real_width, real_height, plate_h);
constraint cumulative(c_y, real_height, real_width, plate_w);

% Give precedence to circuits with same size
constraint symmetry_breaking_constraint(
  forall (i in CIRCUITS) (
    let {
      array[int] of int: identical = [j | j in i+1..n where width[j] = width[i] /\ 
                                                            height[j] = height[i]]
    } in if length(identical)>1 /\ min(identical)=i then
      forall (j in 1..length(identical)-1) (lex_less([c_y[identical[j-1]], c_x[identical[j-1]]],
                                                     [c_y[identical[j]],   c_x[identical[j]] ]))
      else true endif
  )
);

% Sort circuits by decreasing area
array[CIRCUITS] of int : ordered_circuits = sort_by(CIRCUITS, [ - height[i]*width[i] | i in CIRCUITS]);
% Always place the biggest circuit at the bottom-left corner
constraint symmetry_breaking_constraint(
  let { 
    int: i=ordered_circuits[1], int: j=ordered_circuits[2]
  } 
  in 
  lex_less([c_y[i],c_x[i]], [c_y[j],c_x[j]])
);

% Forces circuits to be adjacent
constraint forall(i in CIRCUITS)(
    member([0] ++ [real_width[j] + c_x[j] | j in CIRCUITS where j != i], c_x[i]) /\
    member([0] ++ [real_height[j] + c_y[j] | j in CIRCUITS where j != i], c_y[i])
);

% Do not rotate circuits whose height is greater than the actual maximum plate width
constraint forall(i in CIRCUITS)(
    height[i] > plate_w -> rotation[i] = 0
);

% Do not rotate squares
constraint symmetry_breaking_constraint(
  forall (i in CIRCUITS) (height[i] = width[i] -> rotation[i] = 0)
);


% Do not rotate both circuits if rotated sizes are the same of another circuit
constraint forall(i in CIRCUITS)(
    forall(j in CIRCUITS where j!=i)(
        if width[i] = height[j] /\ height[i] = width[j]
        then (rotation[i] = 0 /\ rotation[j] = 0) \/
             (rotation[i] = 1 /\ rotation[j] = 0) \/
             (rotation[i] = 0 /\ rotation[j] = 1)
        endif
    )
);


solve
% *** Search integer values with different strategies ***
%  :: int_search([plate_h], input_order, indomain_min)
  :: int_search([plate_h], first_fail, indomain_min)
%  :: int_search([plate_h], dom_w_deg, indomain_min)
%  :: int_search([plate_h], input_order, indomain_random)
%  :: int_search([plate_h], first_fail, indomain_random)
%  :: int_search([plate_h], dom_w_deg, indomain_random)
  % *** Different restart strategies ***
%  :: restart_constant(750)
%  :: restart_linear(20)
%  :: restart_geometric(1.5, 700)
%  :: restart_luby(500)

  minimize plate_h;
