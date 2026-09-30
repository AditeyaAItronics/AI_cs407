"""
CS F407 - Laboratory: Search and A*  (Warehouse Robot Navigation)

Search problem P = (S, A, T, s0, G, c):
  S  : free grid cells (row, col)
  A  : Up, Down, Left, Right
  T  : move one cell in the chosen direction if the target cell is not '#'
  s0 : the cell marked 'S'
  G  : the cell marked 'G'
  c  : 1 per move

Runs every experiment in the lab and prints the evidence needed for the report:
  Task 3 - tests (original map, trivial, no solution, alternative paths)
  Task 5 - BFS vs A* comparison
  Task 6 - heuristic investigation (h = 0, Manhattan, Euclidean, 2 x Manhattan)

Usage:  python warehouse_search.py
"""

import heapq
import math
from collections import deque

WAREHOUSE = """\
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################"""

TRIVIAL = """\
#####
#SG##
#####"""

NO_SOLUTION = """\
#######
#S....#
###.###
#...#G#
#######"""

# Two routes from S to G: along the top (4 moves) and around the bottom (8 moves).
ALTERNATIVE = """\
#######
#S...G#
#.###.#
#.....#
#######"""

# A second, more open warehouse (found by a randomised search over small maps)
# on which an inadmissible heuristic returns a longer-than-optimal path.
SECOND_MAP = """\
##########
#S#....#.#
#..#.#...#
##.......#
#......#.#
##.####.G#
##########"""

ACTIONS = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}


# ---------------------------------------------------------------------------
# Problem representation
# ---------------------------------------------------------------------------
class Warehouse:
    def __init__(self, text):
        self.grid = [list(row) for row in text.splitlines()]
        self.start = self._find("S")
        self.goal = self._find("G")

    def _find(self, symbol):
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == symbol:
                    return (r, c)
        raise ValueError(f"map has no {symbol!r}")

    def is_free(self, cell):
        r, c = cell
        return 0 <= r < len(self.grid) and 0 <= c < len(self.grid[r]) and self.grid[r][c] != "#"

    def successors(self, state):
        """Transition function T: yields (action, next_state) for every valid action."""
        r, c = state
        for action, (dr, dc) in ACTIONS.items():
            nxt = (r + dr, c + dc)
            if self.is_free(nxt):          # an action is invalid if it enters an obstacle
                yield action, nxt

    def is_goal(self, state):
        return state == self.goal

    def render(self, path):
        grid = [row[:] for row in self.grid]
        for cell in path[1:-1]:
            grid[cell[0]][cell[1]] = "*"
        return "\n".join("".join(row) for row in grid)


# ---------------------------------------------------------------------------
# Heuristics  h(n)
# ---------------------------------------------------------------------------
def h_zero(state, goal):
    return 0


def h_manhattan(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def h_euclidean(state, goal):
    return math.hypot(state[0] - goal[0], state[1] - goal[1])


def h_double_manhattan(state, goal):
    return 2 * h_manhattan(state, goal)


# ---------------------------------------------------------------------------
# Search algorithms
# ---------------------------------------------------------------------------
def reconstruct(parent, state):
    path = [state]
    while parent[state] is not None:
        state = parent[state]
        path.append(state)
    return path[::-1]


def astar(problem, h=h_manhattan):
    """A* graph search. Returns (path or None, number of states expanded)."""
    start, goal = problem.start, problem.goal
    counter = 0                                   # tie-breaker: FIFO among equal f
    frontier = [(h(start, goal), counter, start)] # priority queue ordered by f(n)
    g = {start: 0}                                # best known cost from start
    parent = {start: None}
    expanded = set()                              # closed set (visited states)

    while frontier:
        f, _, state = heapq.heappop(frontier)     # state with the lowest f(n)
        if state in expanded:                     # stale queue entry, skip
            continue
        expanded.add(state)
        if problem.is_goal(state):                # goal test on expansion
            return reconstruct(parent, state), len(expanded)
        for _action, nxt in problem.successors(state):
            new_g = g[state] + 1                  # step cost c = 1
            if nxt not in g or new_g < g[nxt]:
                g[nxt] = new_g
                parent[nxt] = state
                counter += 1
                f_next = new_g + h(nxt, goal)     # f(n) = g(n) + h(n)
                heapq.heappush(frontier, (f_next, counter, nxt))
    return None, len(expanded)


def bfs(problem):
    """Breadth-first graph search. Goal test on expansion, matching astar()."""
    start = problem.start
    frontier = deque([start])                     # FIFO queue
    parent = {start: None}                        # doubles as the reached set
    expanded = 0
    while frontier:
        state = frontier.popleft()
        expanded += 1
        if problem.is_goal(state):
            return reconstruct(parent, state), expanded
        for _action, nxt in problem.successors(state):
            if nxt not in parent:
                parent[nxt] = state
                frontier.append(nxt)
    return None, expanded


# ---------------------------------------------------------------------------
# Independent validation of a returned path
# ---------------------------------------------------------------------------
def validate_path(problem, path):
    """Checks the path starts at S, ends at G, uses only free cells and unit moves."""
    if path is None:
        return False
    ok = path[0] == problem.start and path[-1] == problem.goal
    ok &= all(problem.is_free(cell) for cell in path)
    ok &= all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:]))
    return ok


def moves(path):
    names = {v: k for k, v in ACTIONS.items()}
    return [names[(b[0] - a[0], b[1] - a[1])] for a, b in zip(path, path[1:])]


def report(title, problem, path, expanded, show_map=True):
    print(f"\n--- {title} ---")
    if path is None:
        print(f"solution found: No   (states expanded: {expanded})")
        return
    print(f"solution found: Yes   path length: {len(path) - 1}   states expanded: {expanded}"
          f"   valid path: {validate_path(problem, path)}")
    print("path (row, col):", " -> ".join(map(str, path)))
    print("actions:", ", ".join(moves(path)))
    if show_map:
        print(problem.render(path))


def header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


if __name__ == "__main__":
    header("Task 3: tests of the A* agent (Manhattan heuristic)")
    for name, text in [("Test 1: original warehouse", WAREHOUSE),
                       ("Test 2: trivial case", TRIVIAL),
                       ("Test 3: no solution", NO_SOLUTION),
                       ("Test 4: alternative paths", ALTERNATIVE)]:
        problem = Warehouse(text)
        path, expanded = astar(problem)
        report(name, problem, path, expanded)
        bfs_path, _ = bfs(problem)
        if path is not None:
            print(f"shortest possible length (BFS): {len(bfs_path) - 1}  -> "
                  f"A* optimal: {len(path) == len(bfs_path)}")

    header("Task 5: BFS vs A* on the original warehouse")
    problem = Warehouse(WAREHOUSE)
    rows = []
    for name, (path, expanded) in [("BFS", bfs(problem)), ("A*", astar(problem))]:
        rows.append((name, path is not None, len(path) - 1 if path else None, expanded))
    print(f"{'measure':<16}{'BFS':>8}{'A*':>8}")
    print(f"{'solution found':<16}{str(rows[0][1]):>8}{str(rows[1][1]):>8}")
    print(f"{'path length':<16}{rows[0][2]:>8}{rows[1][2]:>8}")
    print(f"{'states expanded':<16}{rows[0][3]:>8}{rows[1][3]:>8}")
    free = sum(ch != "#" for row in problem.grid for ch in row)
    print(f"(free cells in the warehouse: {free})")

    header("Task 6: heuristic investigation")
    heuristics = [("h = 0", h_zero), ("Manhattan", h_manhattan),
                  ("Euclidean", h_euclidean), ("2 x Manhattan", h_double_manhattan)]
    for map_name, text in [("original warehouse", WAREHOUSE), ("second map", SECOND_MAP)]:
        problem = Warehouse(text)
        optimal = len(bfs(problem)[0]) - 1
        print(f"\n{map_name} (optimal path length {optimal})")
        print(f"{'heuristic':<15}{'found':>7}{'length':>8}{'expanded':>10}{'optimal?':>10}")
        for name, h in heuristics:
            path, expanded = astar(problem, h)
            length = len(path) - 1
            print(f"{name:<15}{str(path is not None):>7}{length:>8}{expanded:>10}{str(length == optimal):>10}")

    print("\nsecond map, 2 x Manhattan path:")
    problem = Warehouse(SECOND_MAP)
    print(problem.render(astar(problem, h_double_manhattan)[0]))
    print("\nsecond map, Manhattan path:")
    print(problem.render(astar(problem, h_manhattan)[0]))
