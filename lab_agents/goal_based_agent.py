"""
CS F407 - Laboratory: Agents  (Goal-Based Warehouse Navigation Agent)

A goal-based agent for the warehouse navigation problem. The agent keeps an
internal model of the environment (the map), its current state (position) and
an explicit goal. Its decision-making component searches the model for a
sequence of actions (a plan) that reaches the goal, then executes that plan
step by step in the environment.

Search algorithm: breadth-first search (BFS). Every move costs the same, so BFS
returns a shortest collision-free path, and it is complete on a finite grid.

Usage:  python goal_based_agent.py
"""

import heapq
import time
from collections import deque

WAREHOUSE = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################"""

# Same warehouse with the two cells next to G turned into shelves.
BLOCKED = """\
#####################
#S....#...........#G#
#.##....##########.##
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################"""

MOVES = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class WarehouseEnvironment:
    """The world the agent acts in. It knows where the vehicle really is and
    refuses any move into a shelf, so collisions would be detected here."""

    def __init__(self, text):
        self.grid = [list(row) for row in text.splitlines()]
        self.start = self.find("S")
        self.goal = self.find("G")
        self.vehicle = self.start
        self.collisions = 0

    def find(self, symbol):
        for r, row in enumerate(self.grid):
            if symbol in row:
                return (r, row.index(symbol))
        raise ValueError(f"map has no {symbol!r}")

    def is_free(self, cell):
        r, c = cell
        return 0 <= r < len(self.grid) and 0 <= c < len(self.grid[r]) and self.grid[r][c] != "#"

    def percept(self):
        return self.vehicle

    def execute(self, action):
        dr, dc = MOVES[action]
        target = (self.vehicle[0] + dr, self.vehicle[1] + dc)
        if self.is_free(target):
            self.vehicle = target
        else:
            self.collisions += 1

    def render(self, path):
        grid = [row[:] for row in self.grid]
        for r, c in path[1:-1]:
            grid[r][c] = "*"
        return "\n".join("".join(row) for row in grid)


# ---------------------------------------------------------------------------
# Search (decision-making component)
# ---------------------------------------------------------------------------
def successors(model, cell):
    for action, (dr, dc) in MOVES.items():
        nxt = (cell[0] + dr, cell[1] + dc)
        if model.is_free(nxt):
            yield action, nxt


def bfs(model, start, goal):
    """Returns (list of actions or None, states expanded)."""
    frontier = deque([start])
    parent = {start: None}
    expanded = 0
    while frontier:
        cell = frontier.popleft()
        expanded += 1
        if cell == goal:
            actions = []
            while parent[cell] is not None:
                cell, action = parent[cell]
                actions.append(action)
            return actions[::-1], expanded
        for action, nxt in successors(model, cell):
            if nxt not in parent:
                parent[nxt] = (cell, action)
                frontier.append(nxt)
    return None, expanded


def astar(model, start, goal):
    """A* with the Manhattan heuristic, used only for the scaling experiment."""
    h = lambda c: abs(c[0] - goal[0]) + abs(c[1] - goal[1])
    frontier = [(h(start), 0, start)]
    g = {start: 0}
    parent = {start: None}
    closed = set()
    counter = 0
    while frontier:
        _, _, cell = heapq.heappop(frontier)
        if cell in closed:
            continue
        closed.add(cell)
        if cell == goal:
            actions = []
            while parent[cell] is not None:
                cell, action = parent[cell]
                actions.append(action)
            return actions[::-1], len(closed)
        for action, nxt in successors(model, cell):
            if nxt not in g or g[cell] + 1 < g[nxt]:
                g[nxt] = g[cell] + 1
                parent[nxt] = (cell, action)
                counter += 1
                heapq.heappush(frontier, (g[nxt] + h(nxt), counter, nxt))
    return None, len(closed)


# ---------------------------------------------------------------------------
# Goal-based agent
# ---------------------------------------------------------------------------
class GoalBasedAgent:
    def __init__(self, model, goal):
        self.model = model          # internal model of the warehouse
        self.goal = goal            # explicit goal
        self.state = None           # current state, updated from percepts
        self.plan = []

    def perceive(self, percept):
        self.state = percept

    def decide(self):
        """Choose the next action: plan with BFS when there is no plan left."""
        if self.state == self.goal:
            return None
        if not self.plan:
            plan, self.expanded = bfs(self.model, self.state, self.goal)
            if plan is None:
                raise RuntimeError("no collision-free path exists")
            self.plan = plan
        return self.plan.pop(0)


def run_agent(text, verbose=True):
    env = WarehouseEnvironment(text)
    agent = GoalBasedAgent(env, env.goal)
    trajectory = [env.percept()]
    try:
        while True:
            agent.perceive(env.percept())         # sense
            action = agent.decide()               # think
            if action is None:
                break
            env.execute(action)                   # act
            trajectory.append(env.percept())
    except RuntimeError as err:
        if verbose:
            print(f"No path: {err} (states expanded: {bfs(env, env.start, env.goal)[1]})")
        return env, None
    if verbose:
        print(f"Goal reached: {env.vehicle == env.goal}   moves: {len(trajectory) - 1}"
              f"   states expanded by BFS: {agent.expanded}   collisions: {env.collisions}")
        print(env.render(trajectory))
    return env, trajectory


def scaled(text, factor):
    """Enlarge a map by repeating every cell factor x factor times."""
    rows = []
    for row in text.splitlines():
        wide = "".join(ch * factor for ch in row)
        rows.extend([wide] * factor)
    grid = [list(r) for r in rows]
    # keep exactly one S and one G (top-left copy of each)
    for sym in "SG":
        seen = False
        for r, row in enumerate(grid):
            for c, ch in enumerate(row):
                if ch == sym:
                    if seen:
                        grid[r][c] = "."
                    seen = True
    return "\n".join("".join(r) for r in grid)


def header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


if __name__ == "__main__":
    header("Run 1: goal-based agent on the laboratory warehouse")
    env, trajectory = run_agent(WAREHOUSE)
    names = {v: k for k, v in MOVES.items()}
    acts = [names[(b[0] - a[0], b[1] - a[1])] for a, b in zip(trajectory, trajectory[1:])]
    print("path (row, col):", " -> ".join(map(str, trajectory)))
    print("actions:", ", ".join(acts))

    header("Run 2: validation of the returned path")
    ok_start = trajectory[0] == env.start
    ok_goal = trajectory[-1] == env.goal
    ok_free = all(env.is_free(c) for c in trajectory)
    ok_steps = all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(trajectory, trajectory[1:]))
    print(f"starts at S: {ok_start}  ends at G: {ok_goal}  only free cells: {ok_free}  unit moves: {ok_steps}")
    manhattan = abs(env.start[0] - env.goal[0]) + abs(env.start[1] - env.goal[1])
    print(f"Manhattan lower bound on path length: {manhattan}; path length: {len(trajectory) - 1}")

    header("Run 3: goal enclosed by shelves (no path exists)")
    print(BLOCKED)
    run_agent(BLOCKED)

    header("Think About It: a warehouse twice as large")
    print(f"{'map':<10}{'cells':>8}{'BFS expanded':>14}{'A* expanded':>13}{'path len':>10}{'BFS ms':>9}")
    for factor in (1, 2, 4, 8):
        text = scaled(WAREHOUSE, factor)
        env = WarehouseEnvironment(text)
        t0 = time.perf_counter()
        plan_b, exp_b = bfs(env, env.start, env.goal)
        ms = (time.perf_counter() - t0) * 1000
        plan_a, exp_a = astar(env, env.start, env.goal)
        assert len(plan_a) == len(plan_b)
        cells = sum(ch != "#" for row in env.grid for ch in row)
        print(f"{str(factor) + 'x':<10}{cells:>8}{exp_b:>14}{exp_a:>13}{len(plan_b):>10}{ms:>9.2f}")
