# -*- coding: utf-8 -*-
"""
dsatur_coloring.py -- Exam-day assignment with DSATUR only
==========================================================

Algorithm 2 of the comparison study. Contains NOTHING but DSATUR: no genetic
algorithm, no local search, no Welsh-Powell.

DSATUR (Brelaz, 1979) -- "Degree of SATURation"
-----------------------------------------------
A greedy colouring, like Welsh-Powell, but with a DYNAMIC vertex order.

  saturation degree of a course = how many DISTINCT days its already-scheduled
                                  neighbours occupy

At every step the most constrained course is coloured first (highest
saturation degree; ties broken by plain degree). Welsh-Powell sorts once by
degree and never looks again; DSATUR recomputes after every assignment, so it
sees the half-built solution instead of only the raw graph structure.

The idea: colour a cornered course while free days still exist. Postpone it
and every day is taken, forcing a brand-new exam day to be opened -- and that
new day is the waste.

Properties
----------
  * Provably optimal on bipartite graphs, cycles and wheels.
  * NOT optimal in general -- graph colouring is NP-hard, this is a heuristic.
  * Complexity O(n^2) as written here (O((n+m) log n) with a priority queue).
    Irrelevant at this scale: the largest major has 119 courses.

A clique lower bound is also reported. When the result equals that bound the
colouring is PROVABLY OPTIMAL and no algorithm on earth can do better.

HARD CONSTRAINT: zero conflicts. DSATUR is proper by construction, and the
result is re-verified independently before anything is written.

INTERFACE -- deliberately identical to coloring3.py:

  input : input.csv  (col 1 = student id, remaining cols = courses)
  output: ETP_Final_Schedule.xlsx   (Day/Color, Course, Course_Students,
                                     Total_Unique_Students_In_Day)
          coloring_results.xlsx     (Course_Color / Color_Courses /
                                     Color_Student_Counts)
          coloring_results.txt
          dsatur_run_summary.json   (extra, for the comparison report)

The conflict graph is built and the results are exported by re-using the
``Graph`` class from ``coloring3.py``, so every output file is byte-compatible
with the greedy run. Nothing downstream needs to change
(COUNT-NUMBER-OF-CONFLICTS.py, allocate_rooms.py, compare_schedules.py,
calendar_builder.py).

Usage
-----
    python dsatur_coloring.py [input.csv] [--outdir DIR]

    --restarts N   run N times with RANDOMISED tie-breaking and keep the best
                   (default 1 = the classic deterministic DSATUR)
    --seed N       RNG seed, only used when --restarts > 1
    --tag NAME     label written into the JSON summary
"""

import argparse
import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Re-use coloring3.py's Graph so the graph and every output file are identical
# ---------------------------------------------------------------------------
def load_coloring3_module():
    path = os.path.join(SCRIPT_DIR, "coloring3.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "coloring3.py was not found next to dsatur_coloring.py.\n"
            "This script re-uses its Graph class so that all algorithms build\n"
            "the exact same conflict graph and write the exact same output\n"
            "format. Put both files in the same folder."
        )
    spec = importlib.util.spec_from_file_location("_coloring3_for_dsatur", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_edge_arrays(g):
    """Graph -> flat edge arrays (eu, ev) over course indices."""
    idx = {course: i for i, course in enumerate(g.courses)}
    seen = set()
    for course, neighbours in g.graph.items():
        if course not in idx:
            continue
        a = idx[course]
        for other in neighbours:
            if other not in idx:
                continue
            b = idx[other]
            if a == b:
                continue                       # ignore self-loops
            seen.add((a, b) if a < b else (b, a))
    if not seen:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    edges = np.array(sorted(seen), dtype=np.int32)
    return edges[:, 0], edges[:, 1]


def build_adjacency(n, eu, ev):
    buckets = [[] for _ in range(n)]
    for a, b in zip(eu.tolist(), ev.tolist()):
        buckets[a].append(b)
        buckets[b].append(a)
    return [np.array(sorted(b), dtype=np.int32) for b in buckets]


# ---------------------------------------------------------------------------
# Clique lower bound -- optimality certificate
# ---------------------------------------------------------------------------
def clique_lower_bound(n, adj, time_budget=15.0, exact_limit=400):
    """Lower bound on the number of days via the conflict graph's clique
    number. Every course in a clique needs its own day (all pairs conflict),
    so the clique size is always a valid lower bound.

    A cheap greedy multi-start clique search always runs first. If the
    instance is small enough, that bound is then refined with a
    time-bounded EXACT maximum-clique search (Bron-Kerbosch with pivoting
    and branch-and-bound pruning), so the reported bound is the true
    (provable) clique number whenever the search finishes in time -- not
    just a heuristic guess. This matters here specifically because
    ``n_colors == lb`` is how "provably optimal" is decided below, so a
    loose bound would under-report how often DSATUR is provably optimal.
    """
    if n == 0:
        return 0
    sets = [set(a.tolist()) for a in adj]

    # ---- 1. cheap greedy multi-start (always available) -------------------
    order = sorted(range(n), key=lambda v: len(sets[v]), reverse=True)
    best = 1
    for start in order[:min(n, 30)]:
        clique = [start]
        cand = set(sets[start])
        while cand:
            nxt = max(cand, key=lambda v: len(sets[v] & cand))
            clique.append(nxt)
            cand &= sets[nxt]
        best = max(best, len(clique))

    if n > exact_limit:
        return best

    # ---- 2. time-bounded EXACT refinement (Bron-Kerbosch + pivoting) ------
    deadline = time.time() + time_budget
    best_box = [best]
    timed_out = [False]

    def bronk(r_size, P, X):
        if timed_out[0]:
            return
        if time.time() > deadline:
            timed_out[0] = True
            return
        if not P and not X:
            if r_size > best_box[0]:
                best_box[0] = r_size
            return
        if r_size + len(P) <= best_box[0]:
            return
        pivot = max(P | X, key=lambda v: len(sets[v] & P))
        for v in list(P - sets[pivot]):
            if timed_out[0]:
                return
            bronk(r_size + 1, P & sets[v], X & sets[v])
            P = P - {v}
            X = X | {v}

    bronk(0, set(range(n)), set())
    return best_box[0]


# ---------------------------------------------------------------------------
# DSATUR
# ---------------------------------------------------------------------------
def dsatur(n, adj, rng=None):
    """DSATUR colouring.

    Returns (colors, k) with colors as 0-based ints.

    rng is None  -> classic deterministic DSATUR (ties broken by degree, then
                    by vertex index)
    rng given    -> ties broken randomly, so repeated runs explore different
                    orders and --restarts can keep the best one
    """
    if n == 0:
        return np.zeros(0, dtype=np.int32), 0

    color = np.full(n, -1, dtype=np.int32)
    neigh_colors = [set() for _ in range(n)]      # distinct colours seen
    degree = [int(a.size) for a in adj]
    uncolored = set(range(n))
    used = 0

    while uncolored:
        # --- pick the most saturated course -------------------------------
        best_key = None
        best = []
        for v in uncolored:
            key = (len(neigh_colors[v]), degree[v])
            if best_key is None or key > best_key:
                best_key, best = key, [v]
            elif key == best_key:
                best.append(v)

        if rng is not None and len(best) > 1:
            v = int(best[rng.integers(0, len(best))])
        else:
            v = min(best)                          # deterministic tie-break

        # --- smallest day not used by any neighbour -----------------------
        forbidden = neigh_colors[v]
        c = 0
        while c in forbidden:
            c += 1

        color[v] = c
        used = max(used, c + 1)
        uncolored.discard(v)

        # --- refresh the saturation of the neighbours ---------------------
        for u in adj[v].tolist():
            neigh_colors[u].add(c)

    return color, used


def compact_colors(raw, courses):
    """Relabel used colours to a contiguous 1..m range (day numbers)."""
    used = sorted(set(int(c) for c in raw))
    remap = {old: new for new, old in enumerate(used, start=1)}
    return {course: remap[int(raw[i])] for i, course in enumerate(courses)}


def verify_no_conflicts(colors, g):
    """Independent check -- never trust the algorithm's own claim."""
    bad = []
    for course, neighbours in g.graph.items():
        for other in neighbours:
            if course in colors and other in colors and colors[course] == colors[other]:
                pair = tuple(sorted((course, other)))
                bad.append((pair[0], pair[1], colors[course]))
    return sorted(set(bad))


# ---------------------------------------------------------------------------
# Input resolution -- same rules as coloring3.py
# ---------------------------------------------------------------------------
def resolve_input_filename(explicit):
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        raise FileNotFoundError(f"Specified file not found: {explicit}")
    default_path = os.path.join(SCRIPT_DIR, "input.csv")
    if os.path.isfile(default_path):
        return default_path
    csv_files = glob.glob(os.path.join(SCRIPT_DIR, "*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    if len(csv_files) > 1:
        names = ", ".join(os.path.basename(f) for f in csv_files)
        raise FileNotFoundError(
            f"Multiple CSV files found ({names}). "
            f"Please specify which one to use: python dsatur_coloring.py <file.csv>"
        )
    raise FileNotFoundError(
        "No CSV file found. Place 'input.csv' next to the script, "
        "or run: python dsatur_coloring.py <filename.csv>"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Exam-day assignment with DSATUR only.")
    ap.add_argument("input", nargs="?", default=None, help="input CSV file")
    ap.add_argument("--outdir", default=None,
                    help="folder for the output files (default: script folder)")
    ap.add_argument("--restarts", type=int, default=1,
                    help="runs with randomised tie-breaking (1 = deterministic)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default="DSATUR")
    args = ap.parse_args()

    try:
        filename = resolve_input_filename(args.input)
        print(f"Using input file: {filename}")

        outdir = args.outdir or SCRIPT_DIR
        os.makedirs(outdir, exist_ok=True)

        c3 = load_coloring3_module()
        g = c3.Graph()
        g.load_data_from_csv(filename)
        print(f"\nNumber of courses (vertices): {g.v}")

        eu, ev = build_edge_arrays(g)
        adj = build_adjacency(g.v, eu, ev)
        print(f"Number of conflict edges: {eu.size}")

        lb = clique_lower_bound(g.v, adj)
        print(f"Clique lower bound on days: {lb}")

        print(f"\nAlgorithm: {args.tag}")
        restarts = max(1, args.restarts)
        print("Mode: deterministic" if restarts == 1
              else f"Mode: randomised tie-breaking, {restarts} restarts")

        rng = np.random.default_rng(args.seed)
        start_time = time.time()

        best_raw, best_k = None, None
        for r in range(restarts):
            raw, k = dsatur(g.v, adj, rng=None if restarts == 1 else rng)
            if best_k is None or k < best_k:
                best_raw, best_k = raw, k
            if best_k == lb:                       # cannot do better
                break

        computation_time = time.time() - start_time

        colors = compact_colors(best_raw, g.courses)
        n_colors = len(set(colors.values()))
        residual = verify_no_conflicts(colors, g)

        print(f"\nComputation time: {computation_time:.4f} seconds")
        print(f"Number of colors used: {n_colors}")
        if residual:
            print(f"WARNING: {len(residual)} conflicting course pairs remain!")
        else:
            print("No conflicts detected! The coloring is valid.")
        if n_colors == lb:
            print(f"This matches the clique lower bound ({lb}) "
                  f"-- provably optimal.")
        else:
            print(f"Gap to the lower bound: {n_colors - lb} day(s) "
                  f"(may or may not be closable).")

        color_student_counts = g.compute_color_student_counts(colors)
        print("\nColor -> number of unique students:")
        for color in sorted(color_student_counts):
            print(f"Color {color}: {color_student_counts[color]} students")

        # ---- identical outputs to coloring3.py (same writers, same format)
        cwd = os.getcwd()
        try:
            os.chdir(outdir)
            g.export_schedule_table(colors, color_student_counts)
            g.export_text_report(colors, color_student_counts, computation_time,
                                 conflicts=residual, input_filename=filename)
            g.export_detailed_excel(colors, color_student_counts)
        finally:
            os.chdir(cwd)

        summary = {
            "algorithm": args.tag,
            "mode": "dsatur",
            "input_file": os.path.abspath(filename),
            "n_courses": g.v,
            "n_students": len(g.students),
            "n_edges": int(eu.size),
            "colors_used": n_colors,
            "conflicts": len(residual),
            "clique_lower_bound": lb,
            "optimal_proven": bool(n_colors == lb),
            "computation_time_sec": round(computation_time, 4),
            "parameters": {"restarts": restarts, "seed": args.seed},
        }
        with open(os.path.join(outdir, "dsatur_run_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("Run summary saved successfully: dsatur_run_summary.json")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
