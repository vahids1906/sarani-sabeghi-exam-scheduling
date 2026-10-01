# -*- coding: utf-8 -*-
"""
rlf_coloring.py -- Exam-day assignment with RLF only
======================================================

Another greedy baseline for the comparison study, alongside Welsh-Powell
and DSATUR. Contains NOTHING but RLF: no genetic algorithm, no local
search, no metaheuristic.

RLF -- Recursive Largest First (Leighton, 1979)
------------------------------------------------
Welsh-Powell and DSATUR both colour ONE course at a time, choosing which
course to colour next and then giving it the smallest free day. RLF works
differently: it builds ONE ENTIRE EXAM DAY (one full colour class) before
moving to the next day.

To build a day:
  1. Among the still-uncolored courses, pick the one with the most
     still-uncolored conflicting courses (the most "troublesome" course
     right now) and put it on this day.
  2. Every course that conflicts with something already on this day is
     temporarily set aside (it cannot join this day, but it is still
     uncolored -- it will be considered again for the NEXT day).
  3. Among the courses that remain eligible for this day, repeatedly add
     the one with the MOST conflicts against the set-aside pile (adding it
     removes the most future trouble) -- ties broken by the fewest
     conflicts among the still-eligible courses (the safest addition).
  4. When no more courses are eligible for this day, the day is closed,
     every course placed on it gets the next day-number, and the
     set-aside pile becomes the new pool of uncolored courses for the next
     day.

This "build one full day, then move on" strategy is why RLF is generally
the best-performing of the classic greedy heuristics for graph colouring in the
literature: it looks one level deeper than DSATUR before committing to a
day, at the cost of being slower (roughly O(n^3) versus O(n^2) for
DSATUR/Welsh-Powell as implemented here).

Properties
----------
  * NOT optimal in general -- graph colouring is NP-hard, this is a heuristic.
  * Generally produces the fewest days among simple greedy heuristics
    (Welsh-Powell, DSATUR, RLF), but is the slowest of the three.
  * Irrelevant at this scale: the largest major has 119 courses.

A clique lower bound is also reported (exact when the instance is small
enough -- see clique_lower_bound below). When the result equals that bound
the colouring is PROVABLY OPTIMAL and no algorithm on earth can do better.

HARD CONSTRAINT: zero conflicts. RLF is proper by construction, and the
result is re-verified independently before anything is written.

INTERFACE -- deliberately identical to dsatur_coloring.py / coloring3.py:

  input : input.csv  (col 1 = student id, remaining cols = courses)
  output: ETP_Final_Schedule.xlsx   (Day/Color, Course, Course_Students,
                                     Total_Unique_Students_In_Day)
          coloring_results.xlsx     (Course_Color / Color_Courses /
                                     Color_Student_Counts)
          coloring_results.txt
          rlf_run_summary.json      (extra, for the comparison report)

The conflict graph is built and the results are exported by re-using the
``Graph`` class from ``coloring3.py``, so every output file is byte-compatible
with every other algorithm's run. Nothing downstream needs to change.

Usage
-----
    python rlf_coloring.py [input.csv] [--outdir DIR]

    --restarts N   run N times with RANDOMISED tie-breaking and keep the best
                   (default 1 = the classic deterministic RLF)
    --seed N       RNG seed, only used when --restarts > 1; also accepted
                   (and recorded, unused) when --restarts == 1, for CLI
                   compatibility with the other algorithm scripts
    --tag NAME     label written into the JSON summary
    --quiet        suppress progress printing
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
            "coloring3.py was not found next to rlf_coloring.py.\n"
            "This script re-uses its Graph class so that all algorithms build\n"
            "the exact same conflict graph and write the exact same output\n"
            "format. Put both files in the same folder."
        )
    spec = importlib.util.spec_from_file_location("_coloring3_for_rlf", path)
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
    just a heuristic guess.
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
# RLF -- Recursive Largest First
# ---------------------------------------------------------------------------
def pick_best(candidates, key_fn, rng):
    """Return the candidate maximising key_fn, breaking ties deterministically
    (smallest index) when rng is None, or randomly when rng is given."""
    best_key = None
    best = []
    for v in candidates:
        k = key_fn(v)
        if best_key is None or k > best_key:
            best_key, best = k, [v]
        elif k == best_key:
            best.append(v)
    if rng is not None and len(best) > 1:
        return int(best[rng.integers(0, len(best))])
    return min(best)


def rlf(n, adj_sets, rng=None):
    """RLF colouring. Returns (colors, k) with colors as 0-based ints."""
    if n == 0:
        return np.zeros(0, dtype=np.int32), 0

    color = np.full(n, -1, dtype=np.int32)
    uncolored = set(range(n))
    c = 0

    while uncolored:
        U = set(uncolored)     # still eligible for the day being built
        W = set()               # excluded from this day, remain uncolored
        this_day = []

        # ---- start the day: the course with the most uncolored conflicts
        x = pick_best(U, lambda v: len(adj_sets[v] & U), rng)
        this_day.append(x)
        U.discard(x)
        pushed = adj_sets[x] & U
        U -= pushed
        W |= pushed

        # ---- fill the day: repeatedly add the course that clears away the
        # most future trouble (most conflicts with the excluded pile W),
        # tie-broken by the course that is safest to add right now (fewest
        # remaining conflicts within U)
        while U:
            y = pick_best(U, lambda v: (len(adj_sets[v] & W),
                                        -len(adj_sets[v] & U)), rng)
            this_day.append(y)
            U.discard(y)
            pushed = adj_sets[y] & U
            U -= pushed
            W |= pushed

        for v in this_day:
            color[v] = c
        uncolored -= set(this_day)
        c += 1

    return color, c


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
# Input resolution -- same rules as coloring3.py / dsatur_coloring.py
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
            f"Please specify which one to use: python rlf_coloring.py <file.csv>"
        )
    raise FileNotFoundError(
        "No CSV file found. Place 'input.csv' next to the script, "
        "or run: python rlf_coloring.py <filename.csv>"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Exam-day assignment with RLF (Recursive Largest First) only.")
    ap.add_argument("input", nargs="?", default=None, help="input CSV file")
    ap.add_argument("--outdir", default=None,
                    help="folder for the output files (default: script folder)")
    ap.add_argument("--restarts", type=int, default=1,
                    help="runs with randomised tie-breaking (1 = deterministic)")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed, used only when --restarts > 1; accepted "
                         "either way for CLI compatibility with the other "
                         "algorithm scripts")
    ap.add_argument("--tag", default="RLF")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress progress printing")
    args = ap.parse_args()

    def log(*a, **kw):
        if not args.quiet:
            print(*a, **kw)

    try:
        filename = resolve_input_filename(args.input)
        log(f"Using input file: {filename}")

        outdir = args.outdir or SCRIPT_DIR
        os.makedirs(outdir, exist_ok=True)

        c3 = load_coloring3_module()
        g = c3.Graph()
        g.load_data_from_csv(filename)
        log(f"\nNumber of courses (vertices): {g.v}")

        eu, ev = build_edge_arrays(g)
        adj = build_adjacency(g.v, eu, ev)
        log(f"Number of conflict edges: {eu.size}")
        adj_sets = [set(a.tolist()) for a in adj]

        lb = clique_lower_bound(g.v, adj)
        log(f"Clique lower bound on days: {lb}")

        log(f"\nAlgorithm: {args.tag}")
        restarts = max(1, args.restarts)
        log("Mode: deterministic" if restarts == 1
              else f"Mode: randomised tie-breaking, {restarts} restarts")

        rng = np.random.default_rng(args.seed)
        start_time = time.time()

        best_raw, best_k = None, None
        for r in range(restarts):
            raw, k = rlf(g.v, adj_sets, rng=None if restarts == 1 else rng)
            if best_k is None or k < best_k:
                best_raw, best_k = raw, k
            if best_k == lb:                       # cannot do better
                break

        computation_time = time.time() - start_time

        colors = compact_colors(best_raw, g.courses)
        n_colors = len(set(colors.values()))
        residual = verify_no_conflicts(colors, g)

        log(f"\nComputation time: {computation_time:.4f} seconds")
        log(f"Number of colors used: {n_colors}")
        if residual:
            log(f"WARNING: {len(residual)} conflicting course pairs remain!")
        else:
            log("No conflicts detected! The coloring is valid.")
        if n_colors == lb:
            log(f"This matches the clique lower bound ({lb}) "
                  f"-- provably optimal.")
        else:
            log(f"Gap to the lower bound: {n_colors - lb} day(s) "
                  f"(may or may not be closable).")

        color_student_counts = g.compute_color_student_counts(colors)
        log("\nColor -> number of unique students:")
        for color in sorted(color_student_counts):
            log(f"Color {color}: {color_student_counts[color]} students")

        # ---- identical outputs to coloring3.py / dsatur_coloring.py
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
            "mode": "rlf",
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
        with open(os.path.join(outdir, "rlf_run_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log("Run summary saved successfully: rlf_run_summary.json")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
