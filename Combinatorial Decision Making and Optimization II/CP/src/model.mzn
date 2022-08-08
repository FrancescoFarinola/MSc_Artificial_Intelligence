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

% *** Outputs *** - Coordinates (x,y) for each circuit
array[CIRCUITS] of var 0..plate_w - min(width): c_x;  
array[CIRCUITS] of var 0..sum(height) - min(height): c_y;  

% ****** Constraints ******
% *** Implied constraint - Limit each circuit to appear in the board - w/o going outside***
constraint forall(i in CIRCUITS) (c_x[i] + width[i] <= plate_w /\
                                  c_y[i] + height[i]<= plate_h );

% *** Globals for no overlapping and redundant (for propagation)***
constraint diffn(c_x,c_y,width,height);
constraint cumulative(c_x, width, height, plate_h);
constraint cumulative(c_y, height, width, plate_w);

% *** Give precedence to circuits with same size ***
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
%  *** Always place the biggest circuit at the bottom-left corner ***
constraint symmetry_breaking_constraint(
  let { 
    int: i=ordered_circuits[1], int: j=ordered_circuits[2]
  } 
  in 
  lex_less([c_y[i],c_x[i]], [c_y[j],c_x[j]])
);

% *** Forces circuits to be adjacent to each other or to the borders [0] ***
constraint forall(i in CIRCUITS)(
    member([0] ++ [width[j] + c_x[j] | j in CIRCUITS where j != i], c_x[i]) /\
    member([0] ++ [height[j] + c_y[j] | j in CIRCUITS where j != i], c_y[i])
);

solve
%  :: int_search(c_x, first_fail, indomain_min)
%  :: int_search(c_y, first_fail, indomain_min)
% *** Search integer values with different strategies ***
%  :: int_search([plate_h], input_order, indomain_min)
%  :: int_search([plate_h], first_fail, indomain_min)
%  :: int_search([plate_h], dom_w_deg, indomain_min)
%  :: int_search([plate_h], input_order, indomain_random)
  :: int_search([plate_h], first_fail, indomain_random)
%  :: int_search([plate_h], dom_w_deg, indomain_random)
  % *** Different restart strategies - NB: only for indomain_random ***
%  :: restart_constant(500)
%  :: restart_linear(200)
  :: restart_geometric(1.5, 700)
%  :: restart_luby(500)
  minimize plate_h;
