% CS F407 - Logical Planning Lab, Optional Extension (Task 8)
%
% Run with SWI-Prolog:
%   swipl road.pl
%   ?- reduce_speed.         % true

wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.
