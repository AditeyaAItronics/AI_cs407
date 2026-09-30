% CS F407 - Logical Planning Lab, Optional Extension (Tasks 6 and 7)
% Warehouse knowledge base used to verify moves proposed by the Python planner.
%
% Run with SWI-Prolog:
%   swipl planner.pl
%   ?- can_move(a,b).        % true
%   ?- can_move(a,c).        % false
%   ?- valid_move(a,b).      % true
%   ?- valid_move(b,c).      % true
%   ?- valid_move(a,c).      % false
%   ?- valid_plan([a,b,c]).  % true   (Move(a,b), Move(b,c))
%   ?- valid_plan([a,c]).    % false  (Move(a,c) is not supported)

connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

% Task 6: Connected(X,Y) -> CanMove(X,Y)
can_move(X,Y) :-
    connected(X,Y).

% Task 7: a single proposed move is valid if the locations are connected
valid_move(X,Y) :-
    connected(X,Y).

% Extension: a sequence of visited locations is a valid route if every
% consecutive pair is a valid move.
valid_plan([_]).
valid_plan([X,Y|Rest]) :-
    valid_move(X,Y),
    valid_plan([Y|Rest]).
