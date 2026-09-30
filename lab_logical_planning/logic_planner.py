"""
CS F407 - Laboratory: Logical Reasoning for Planning  (Warehouse Delivery Robot)

A STRIPS-style planner:
  * a state is a frozenset of ground propositions, e.g. "At(Robot,A)";
  * an action has positive/negative preconditions and positive/negative effects;
  * an action is applicable in S iff S |= Preconditions(a)          (logic)
  * Apply(S, a) = (S - negative effects) | positive effects
  * breadth-first search over applicable action sequences          (search)

A separate validator re-executes any proposed plan step by step, so plans can
be checked independently of the planner that produced them (Task 5).

Usage:  python logic_planner.py
"""

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    name: str
    pos_pre: frozenset = frozenset()
    neg_pre: frozenset = frozenset()
    add: frozenset = frozenset()
    delete: frozenset = frozenset()

    def applicable(self, state):
        """Logical check: S |= Preconditions(a)."""
        return self.pos_pre <= state and not (self.neg_pre & state)

    def apply(self, state):
        """Successor state: remove negative effects, then add positive effects."""
        return (state - self.delete) | self.add

    def missing(self, state):
        """Preconditions that are not satisfied in state (used for explanations)."""
        return sorted(self.pos_pre - state) + [f"not {p}" for p in sorted(self.neg_pre & state)]


def fs(*props):
    return frozenset(props)


# ---------------------------------------------------------------------------
# Domain: robot, package, locations A - B - C
# ---------------------------------------------------------------------------
LOCATIONS = ["A", "B", "C"]
CONNECTED = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]


def move(x, y):
    return Action(f"Move({x},{y})", pos_pre=fs(f"At(Robot,{x})"),
                  add=fs(f"At(Robot,{y})"), delete=fs(f"At(Robot,{x})"))


def pickup(x):
    return Action(f"PickUp(Package,{x})",
                  pos_pre=fs(f"At(Robot,{x})", f"At(Package,{x})"),
                  neg_pre=fs("Holding(Package)"),
                  add=fs("Holding(Package)"), delete=fs(f"At(Package,{x})"))


def drop(x):
    return Action(f"Drop(Package,{x})",
                  pos_pre=fs(f"At(Robot,{x})", "Holding(Package)"),
                  add=fs(f"At(Package,{x})"), delete=fs("Holding(Package)"))


def warehouse_actions(with_pickup=True, extra=()):
    actions = [move(x, y) for x, y in CONNECTED]
    if with_pickup:
        actions += [pickup(x) for x in LOCATIONS]
    actions += [drop(x) for x in LOCATIONS]
    return actions + list(extra)


INITIAL = fs("At(Robot,A)", "At(Package,A)")
GOAL = fs("At(Package,C)")


# ---------------------------------------------------------------------------
# Planner (BFS) and independent validator
# ---------------------------------------------------------------------------
def plan_bfs(initial, goal, actions):
    """Returns (plan as list of Actions or None, number of states expanded)."""
    frontier = deque([initial])
    parent = {initial: None}
    expanded = 0
    while frontier:
        state = frontier.popleft()
        expanded += 1
        if goal <= state:                                   # S |= G
            plan = []
            while parent[state] is not None:
                state, action = parent[state]
                plan.append(action)
            return plan[::-1], expanded
        for action in actions:
            if action.applicable(state):                    # logic: preconditions
                nxt = action.apply(state)                   # effects
                if nxt not in parent:                       # search: explore once
                    parent[nxt] = (state, action)
                    frontier.append(nxt)
    return None, expanded


def validate(initial, goal, plan, actions_by_name):
    """Independent verifier: re-executes the plan from the initial state,
    checking every precondition, and finally checks the goal."""
    state = initial
    trace = [("S0", None, state)]
    for i, name in enumerate(plan, 1):
        action = actions_by_name.get(name)
        if action is None:
            return False, trace, f"step {i}: {name} is not an available action"
        if not action.applicable(state):
            return False, trace, f"step {i}: {name} not applicable, unsatisfied: {action.missing(state)}"
        state = action.apply(state)
        trace.append((f"S{i}", name, state))
    if not goal <= state:
        return False, trace, f"goal {sorted(goal)} not satisfied in final state"
    return True, trace, "all preconditions satisfied and goal reached"


def show(state):
    return "{" + ", ".join(sorted(state)) + "}"


def run_test(title, initial, goal, actions):
    print(f"\n--- {title} ---")
    print("initial state:", show(initial))
    print("goal         :", show(goal))
    plan, expanded = plan_bfs(initial, goal, actions)
    if plan is None:
        print(f"No plan found  (states explored: {expanded})")
        return None
    names = [a.name for a in plan]
    print(f"plan found ({len(plan)} steps, {expanded} states explored):", ", ".join(names))
    ok, trace, msg = validate(initial, goal, names, {a.name: a for a in actions})
    for label, name, state in trace:
        print(f"  {label}: {'' if name is None else name + ' -> '}{show(state)}")
    print("independent validation:", ok, "-", msg)
    return names


def header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


if __name__ == "__main__":
    actions = warehouse_actions()
    by_name = {a.name: a for a in actions}

    header("Task 0: action applicability in the initial state")
    print("I =", show(INITIAL))
    for a in actions:
        status = "applicable" if a.applicable(INITIAL) else f"not applicable, unsatisfied: {a.missing(INITIAL)}"
        print(f"  {a.name:<20} {status}")

    header("Task 1: hand-constructed plan, checked by the validator")
    hand_plan = ["PickUp(Package,A)", "Move(A,B)", "Move(B,C)", "Drop(Package,C)"]
    ok, trace, msg = validate(INITIAL, GOAL, hand_plan, by_name)
    for label, name, state in trace:
        print(f"  {label}: {'' if name is None else name + ' -> '}{show(state)}")
    print("valid:", ok, "-", msg)

    print("\nsequence listed as an illustration in the handout:")
    handout = ["Move(A,B)", "PickUp(Package,B)", "Move(B,C)", "Drop(Package,C)"]
    ok, trace, msg = validate(INITIAL, GOAL, handout, by_name)
    for label, name, state in trace:
        print(f"  {label}: {'' if name is None else name + ' -> '}{show(state)}")
    print("valid:", ok, "-", msg)

    header("Task 3: tests of the planner")
    run_test("Test A: solvable warehouse problem", INITIAL, GOAL, actions)
    run_test("Test B: PickUp removed (impossible)", INITIAL, GOAL, warehouse_actions(with_pickup=False))

    # Irrelevant actions: a direct robot-only shortcut A -> C and a no-op.
    shortcut = Action("Move(A,C)", pos_pre=fs("At(Robot,A)"),
                      add=fs("At(Robot,C)"), delete=fs("At(Robot,A)"))
    wait = Action("Wait", pos_pre=fs())
    extra = warehouse_actions(extra=[shortcut, wait])
    run_test("Test C1: irrelevant actions Move(A,C) and Wait added", INITIAL, GOAL, extra)
    run_test("Test C2: irrelevant actions added, PickUp removed "
             "(robot can reach C, package cannot)", INITIAL, GOAL,
             warehouse_actions(with_pickup=False, extra=[shortcut, wait]))
    run_test("Test C3: goal At(Robot,C) instead of At(Package,C)", INITIAL, fs("At(Robot,C)"), actions)

    header("Task 5: validating a claimed plan independently")
    claimed = ["PickUp(Package,A)", "Move(A,C)", "Drop(Package,C)"]
    print("claimed plan:", ", ".join(claimed))
    ok, _, msg = validate(INITIAL, GOAL, claimed, by_name)
    print("valid:", ok, "-", msg)
