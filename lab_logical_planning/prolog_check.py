"""
Minimal backward-chaining evaluator for the ground/Datalog-style clauses in
planner.pl and road.pl. It answers the laboratory's Prolog queries in Python
so the results can be reproduced where SWI-Prolog is not installed; running
the .pl files in SWI-Prolog is the authoritative check.

Supports: facts, rules whose body is a conjunction of atoms, variables
(capitalised names), and queries. No lists, negation, or arithmetic.

Usage:  python prolog_check.py
"""

import itertools
import re

_counter = itertools.count()


def parse_atom(text):
    text = text.strip()
    m = re.fullmatch(r"(\w+)(?:\((.*)\))?", text)
    name, args = m.group(1), m.group(2)
    return (name, tuple(a.strip() for a in args.split(",")) if args else ())


def load(path):
    clauses = []
    source = re.sub(r"%.*", "", open(path, encoding="utf-8").read())
    for stmt in source.split("."):
        stmt = " ".join(stmt.split())
        if not stmt or "[" in stmt:          # skip list-based clauses
            continue
        if ":-" in stmt:
            head, body = stmt.split(":-")
            body_atoms = [parse_atom(b) for b in re.split(r",\s*(?![^()]*\))", body)]
            clauses.append((parse_atom(head), body_atoms))
        else:
            clauses.append((parse_atom(stmt), []))
    return clauses


def is_var(t):
    return t[:1].isupper() or t[:1] == "_"


def walk(t, s):
    while is_var(t) and t in s:
        t = s[t]
    return t


def unify(a, b, s):
    if a[0] != b[0] or len(a[1]) != len(b[1]):
        return None
    s = dict(s)
    for x, y in zip(a[1], b[1]):
        x, y = walk(x, s), walk(y, s)
        if x == y:
            continue
        if is_var(x):
            s[x] = y
        elif is_var(y):
            s[y] = x
        else:
            return None
    return s


def rename(clause):
    n = next(_counter)
    ren = lambda atom: (atom[0], tuple(f"{t}_{n}" if is_var(t) else t for t in atom[1]))
    head, body = clause
    return ren(head), [ren(b) for b in body]


def fmt(atom):
    return atom[0] + (f"({', '.join(atom[1])})" if atom[1] else "")


def solve(goals, clauses, s, depth=0, trace=None):
    """SLD resolution, depth-first, clauses tried in program order."""
    if not goals:
        yield s
        return
    first, rest = goals[0], goals[1:]
    for clause in clauses:
        head, body = rename(clause)
        s2 = unify(first, head, s)
        if s2 is not None:
            if trace is not None:
                goal = fmt((first[0], tuple(walk(t, s) for t in first[1])))
                rule = fmt(clause[0]) + (" :- " + ", ".join(map(fmt, clause[1])) if clause[1] else "")
                trace.append("  " * depth + f"{goal}  resolved with  {rule}")
            yield from solve(body + rest, clauses, s2, depth + 1, trace)


def query(clauses, text, show_trace=False):
    trace = [] if show_trace else None
    result = next(solve([parse_atom(text)], clauses, {}, trace=trace), None) is not None
    print(f"?- {text}.   {'true' if result else 'false'}")
    if show_trace and trace:
        print("\n".join(trace))
    return result


if __name__ == "__main__":
    kb = load("planner.pl")
    print("Task 6: planner.pl")
    query(kb, "can_move(a,b)", show_trace=True)
    query(kb, "can_move(a,c)")

    print("\nTask 7: checking moves proposed by the Python planner")
    for q in ("valid_move(a,b)", "valid_move(b,c)", "valid_move(a,c)"):
        query(kb, q)

    print("\nTask 8: road.pl")
    query(load("road.pl"), "reduce_speed", show_trace=True)
