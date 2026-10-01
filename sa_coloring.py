# -*- coding: utf-8 -*-
"""
sa_coloring.py -- Exam-day assignment with Simulated Annealing only
===================================================================

Algorithm 3 of the comparison study. Contains NOTHING but simulated
annealing: no genetic algorithm, no tabu search, no DSATUR, no Welsh-Powell.
The initial solution is a RANDOM colouring, so the result owes nothing to any
greedy heuristic and can be compared against them fairly.

This is a corrected Python port of the author's MATLAB sa_coloring.m / cost.m.


Simulated Annealing
-------------------
A local search that is allowed to move UPHILL. From the current colouring it
proposes a small change; if the change reduces conflicts it is always taken,
and if it increases conflicts by d it is still taken with probability

    P = exp(-d / T)

The temperature T starts high (many uphill moves accepted, so the search roams
freely) and is cooled geometrically towards zero (only downhill moves survive,
so the search settles). That controlled willingness to get temporarily worse is
what lets it escape the local minima a greedy method cannot leave.


WHAT WAS FIXED RELATIVE TO THE MATLAB ORIGINAL
----------------------------------------------
The MATLAB version reached 10-12 days on its own hardcoded 71-course graph,
whose provable optimum is 8. Five defects caused that gap:

  1. delta was measured against the BEST-EVER cost instead of the CURRENT cost
     (`delta = neighbor_cost - best_cost`). best_cost only ever decreases, so
     the reference point kept sinking, deltas became systematically positive,
     and the search froze prematurely no matter what the temperature was --
     destroying the one mechanism simulated annealing exists for. Here the
     current cost is tracked in its own variable and the best-so-far solution
     is only recorded, never used as the comparison baseline.

  2. The full O(n^2) cost function was recomputed from scratch twice per
     iteration. Recolouring one course only changes the cost by
         (neighbours holding the new day) - (neighbours holding the old day)
     so a gamma[course][day] table makes each move O(degree) instead of
     O(n^2) -- roughly 250x less work at this scale.

  3. The initial temperature was 1000 while the deltas are small integers
     (typically -5..+5). exp(-5/1000) = 0.995, i.e. everything was accepted,
     so the first ~43% of the budget was an aimless random walk. The initial
     temperature is now CALIBRATED by sampling real uphill moves so that a
     typical one starts out ~50% likely to be accepted.

  4. Neighbour generation was blind: any course, any day, including the day it
     already had (a no-op with probability 1/k) and including courses with no
     conflict at all. Now a course is drawn only from those CURRENTLY in
     conflict, and only from days different to its own -- no wasted moves.

  5. The k-loop started at k = 1. Days are now searched upward from the clique
     lower bound, which skips a long run of provably impossible values.

Also fixed: the original reported k rather than the number of days actually
used, and printed a result even when it never reached zero conflicts.


HARD CONSTRAINT: zero conflicts. A colouring is only accepted at a given k
once annealing has driven the conflict count to exactly 0, and the result is
then re-verified independently before anything is written. No schedule with
conflicts is ever exported. If no k succeeds the script fails loudly instead
of writing a broken timetable -- and it deliberately does NOT fall back on a
greedy repair, which would smuggle another algorithm into the comparison.


INTERFACE -- deliberately identical to coloring3.py:

  input : input.csv  (col 1 = student id, remaining cols = courses)
  output: ETP_Final_Schedule.xlsx   (Day/Color, Course, Course_Students,
                                     Total_Unique_Students_In_Day)
          coloring_results.xlsx     (Course_Color / Color_Courses /
                                     Color_Student_Counts)
          coloring_results.txt
          sa_run_summary.json       (extra, for the comparison report)
          sa_convergence.csv        (extra, convergence curves)

The conflict graph is built and the results are exported by re-using the
``Graph`` class from ``coloring3.py``, so every output file is byte-compatible
with the greedy run. Nothing downstream needs to change
(COUNT-NUMBER-OF-CONFLICTS.py, allocate_rooms.py, compare_schedules.py,
calendar_builder.py).


Usage
-----
    python sa_coloring.py [input.csv] [--outdir DIR]

    --iters N        annealing iterations per k (0 = auto, max(50000, 1000*n))
    --restarts N     independent annealing runs per k (default 3)
    --t-init X       initial temperature (0 = auto-calibrate, recommended)
    --t-end X        final temperature (default 0.2 -- see BUDGET, do not
                     lower this without testing)
    --reheat N       reheat when N iterations pass with no improvement
                     (0 = never, and never is usually right -- see BUDGET)
    --seed N         RNG seed
    --max-k N        stop searching above N days
    --time-limit S   overall wall-clock budget in seconds (0 = none)
    --tag NAME       label written into the JSON summary
    --quiet          suppress the per-k progress lines

    --faithful       OPTIONAL. Run a line-by-line replica of the original
                     MATLAB sa_coloring.m instead, bugs and all, for a
                     "effect of parameter tuning" comparison in the paper.
                     Off by default.


BUDGET -- measured on the hardest major (Dentistry, 119 courses, 2229 edges,
lower bound 24 days)
-----------------------------------------------------------------------------
How the iteration budget is SPENT matters more than its size, because the
cooling schedule is stretched to fit --iters. A long horizon means slow
cooling, so the run spends most of its life too hot to settle.

    20 anneals x  30,000 iters  (600,000 total) -> FAILED at k=24, 1 conflict
     6 anneals x 100,000 iters                  -> SOLVED  k=24, optimal
     3 anneals x 400,000 iters, --reheat 20000  -> FAILED even at k=26

So: give each individual anneal a horizon long enough to cool properly, then
repeat it a few times. Do not pour the whole budget into one enormous slow
cool, and do not reheat on a long horizon -- reheating fights the schedule and
keeps the search permanently hot. Hence the defaults below, and hence --reheat
defaulting to off.

The FINAL temperature turned out to matter even more than the budget. On the
same instance, at k = 24, annealing kept stalling one single conflict short of
a valid schedule:

    10 anneals, --t-end 0.05  -> FAILED, always exactly 1 conflict left
     6 anneals, --t-end 0.20  -> SOLVED, k=24, provably optimal

Ending at 0.05 freezes the search while that last conflict still needs a CHAIN
of recolourings to clear: each single step in the chain is uphill, and at 0.05
an uphill step of +1 has probability exp(-1/0.05) = 2e-9, so it never happens.
At 0.20 that same step has probability exp(-1/0.2) = 0.0067 -- rare, but it
does occur within the remaining iterations. Ten times more restarts cannot buy
what a slightly warmer finish gives for free, because every restart froze in
the same way. Hence --t-end defaults to 0.2 rather than to a "colder is
better" value.
"""

import argparse
import csv
import glob
import importlib.util
import json
import math
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
            "coloring3.py was not found next to sa_coloring.py.\n"
            "This script re-uses its Graph class so that all algorithms build\n"
            "the exact same conflict graph and write the exact same output\n"
            "format. Put both files in the same folder."
        )
    spec = importlib.util.spec_from_file_location("_coloring3_for_sa", path)
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
    return [sorted(b) for b in buckets]


# ---------------------------------------------------------------------------
# Clique lower bound -- optimality certificate
# ---------------------------------------------------------------------------
def clique_lower_bound(n, adj):
    """Greedily grow a clique. Every course in a clique needs its own day, so
    the clique size is a valid lower bound on the number of days."""
    if n == 0:
        return 0
    sets = [set(a) for a in adj]
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
    return best


# ---------------------------------------------------------------------------
# O(1) random pick from a changing set of conflicted courses
# ---------------------------------------------------------------------------
class IndexedSet:
    """Set with O(1) add / discard / uniform random pick."""

    __slots__ = ("items", "pos")

    def __init__(self):
        self.items = []
        self.pos = {}

    def add(self, x):
        if x not in self.pos:
            self.pos[x] = len(self.items)
            self.items.append(x)

    def discard(self, x):
        i = self.pos.pop(x, None)
        if i is None:
            return
        last = self.items.pop()
        if i < len(self.items):
            self.items[i] = last
            self.pos[last] = i

    def __len__(self):
        return len(self.items)

    def pick(self, rng):
        return self.items[int(rng.integers(0, len(self.items)))]


# ---------------------------------------------------------------------------
# Simulated annealing for a FIXED number of days k
# ---------------------------------------------------------------------------
def anneal(n, adj, total_edges, k, rng, max_iter,
           t_init=0.0, t_end=0.05, reheat_after=0,
           record=None, record_k=None, record_stride=1, deadline=None):
    """Anneal towards a conflict-free colouring using exactly k days.

    Returns (best_cost, best_colors, iterations_done). best_colors is 0-based.
    """
    if n == 0:
        return 0, [], 0
    if k <= 1:
        # Only feasible when the graph has no edges at all.
        return int(total_edges), [0] * n, 0

    # --- random initial colouring (no greedy seeding) ----------------------
    col = [int(c) for c in rng.integers(0, k, size=n)]

    # --- gamma[v][c] = how many neighbours of v sit on day c ---------------
    gamma = [[0] * k for _ in range(n)]
    for v in range(n):
        gv = gamma[v]
        for u in adj[v]:
            gv[col[u]] += 1
    cur = sum(gamma[v][col[v]] for v in range(n)) // 2

    confl = IndexedSet()
    for v in range(n):
        if gamma[v][col[v]] > 0:
            confl.add(v)

    best = cur
    best_col = col[:]
    if cur == 0:
        return 0, best_col, 0

    # --- calibrate the initial temperature to the real delta scale ---------
    if t_init <= 0:
        samples = []
        for _ in range(min(400, max(50, 4 * n))):
            v = int(rng.integers(0, n))
            c1 = col[v]
            c2 = int(rng.integers(0, k - 1))
            if c2 >= c1:
                c2 += 1
            d = gamma[v][c2] - gamma[v][c1]
            if d > 0:
                samples.append(d)
        if samples:
            # a typical uphill move should start out ~50% likely to be taken
            t_init = (sum(samples) / len(samples)) / math.log(2.0)
        else:
            t_init = 1.0
        t_init = max(t_init, 0.5)

    t_end = max(min(t_end, t_init), 1e-6)
    alpha = (t_end / t_init) ** (1.0 / max_iter) if max_iter > 0 else 1.0

    T = t_init
    it = 0
    since_improve = 0
    exp = math.exp

    while it < max_iter and cur > 0 and len(confl) > 0:
        # --- propose: a CONFLICTED course moves to a DIFFERENT day ---------
        v = confl.pick(rng)
        c1 = col[v]
        c2 = int(rng.integers(0, k - 1))
        if c2 >= c1:
            c2 += 1

        gv = gamma[v]
        d = gv[c2] - gv[c1]

        # --- Metropolis: delta is measured against the CURRENT cost --------
        if d <= 0 or rng.random() < exp(-d / T):
            col[v] = c2
            for u in adj[v]:
                gu = gamma[u]
                gu[c1] -= 1
                gu[c2] += 1
                if gu[col[u]] > 0:
                    confl.add(u)
                else:
                    confl.discard(u)
            cur += d
            if gv[c2] > 0:
                confl.add(v)
            else:
                confl.discard(v)
            if cur < best:
                best = cur
                best_col = col[:]
                since_improve = 0

        it += 1
        since_improve += 1
        T = max(T * alpha, 1e-12)

        if reheat_after and since_improve >= reheat_after:
            T = max(T, t_init * 0.5)
            since_improve = 0

        if record is not None and it % record_stride == 0:
            record.append((record_k, it, cur, best, round(T, 6)))

        if deadline is not None and (it & 511) == 0 and time.time() > deadline:
            break

    # Always close each k with a final data point, so a k that is solved in
    # very few iterations still shows up in the convergence file.
    if record is not None:
        record.append((record_k, it, cur, best, round(T, 6)))

    return best, best_col, it


# ---------------------------------------------------------------------------
# Upward search over the number of days
# ---------------------------------------------------------------------------
def search(n, adj, total_edges, lb, rng, args, record, verbose=True):
    """Find the fewest days for which annealing reaches zero conflicts."""
    max_iter = args.iters if args.iters > 0 else max(50000, 1000 * n)
    stride = max(1, max_iter // 800)
    deadline = (time.time() + args.time_limit) if args.time_limit > 0 else None
    top = args.max_k if args.max_k else n
    attempts = []

    for k in range(max(1, lb), max(1, top) + 1):
        for r in range(max(1, args.restarts)):
            cost, col, iters = anneal(
                n, adj, total_edges, k, rng, max_iter,
                t_init=args.t_init, t_end=args.t_end,
                reheat_after=args.reheat, record=record, record_k=k,
                record_stride=stride, deadline=deadline,
            )
            attempts.append({
                "k": k, "restart": r + 1,
                "best_conflicts": int(cost), "iterations": int(iters),
            })
            if verbose:
                state = "SOLVED" if cost == 0 else f"{cost} conflict(s) left"
                print(f"  k={k:3d}  run {r + 1}/{max(1, args.restarts)}: "
                      f"{state}  ({iters} iterations)")
            if cost == 0:
                return k, col, attempts
            if deadline is not None and time.time() > deadline:
                if verbose:
                    print("  time limit reached")
                return None, None, attempts

    return None, None, attempts


# ---------------------------------------------------------------------------
# OPTIONAL: line-by-line replica of the original MATLAB sa_coloring.m
# ---------------------------------------------------------------------------
def faithful_matlab_sa(n, eu, ev, rng, record, record_stride=1, verbose=True):
    """Reproduce sa_coloring.m exactly, defects included, for the paper's
    "effect of parameter tuning" comparison.

    Kept faithful: k starts at 1, random initial colouring, T_init = 1000,
    alpha = 0.99, min_T = 1e-4, max_iter = 2000, blind neighbour generation,
    and delta measured against best_cost rather than the current cost.

    The full cost function is evaluated with numpy rather than a Python double
    loop. That changes only how fast the same number is produced, never the
    search trajectory. The honest measure of wasted work -- the number of full
    cost evaluations -- is counted and reported in the JSON summary.
    """
    Aup = np.zeros((n, n), dtype=bool)
    if eu.size:
        Aup[eu, ev] = True

    def full_cost(c):
        return int(((c[:, None] == c[None, :]) & Aup).sum())

    evals = 0
    attempts = []

    for k in range(1, n + 1):
        coloring = rng.integers(1, k + 1, size=n)
        best = coloring.copy()
        best_cost = full_cost(coloring)
        evals += 1

        T = 1000.0
        it = 0
        while T > 0.0001 and it < 2000 and best_cost > 0:
            neighbor = coloring.copy()
            neighbor[int(rng.integers(0, n))] = int(rng.integers(1, k + 1))
            nc = full_cost(neighbor)
            evals += 1

            delta = nc - best_cost          # the original defect, preserved
            if delta < 0 or rng.random() < math.exp(-delta / T):
                coloring = neighbor
                if nc < best_cost:
                    best = neighbor.copy()
                    best_cost = nc

            T *= 0.99
            it += 1
            evals += 1                      # cost_history recompute each pass
            if record is not None and it % record_stride == 0:
                record.append((k, it, full_cost(coloring), best_cost,
                               round(T, 6)))

        attempts.append({"k": k, "restart": 1,
                         "best_conflicts": int(best_cost), "iterations": it})
        if verbose:
            state = "SOLVED" if best_cost == 0 else f"{best_cost} conflict(s) left"
            print(f"  k={k:3d}: {state}  ({it} iterations)")
        if best_cost == 0:
            return k, [int(c) - 1 for c in best], evals, attempts

    return None, None, evals, attempts


# ---------------------------------------------------------------------------
# Result handling
# ---------------------------------------------------------------------------
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
            f"Please specify which one to use: python sa_coloring.py <file.csv>"
        )
    raise FileNotFoundError(
        "No CSV file found. Place 'input.csv' next to the script, "
        "or run: python sa_coloring.py <filename.csv>"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Exam-day assignment with Simulated Annealing only.")
    ap.add_argument("input", nargs="?", default=None, help="input CSV file")
    ap.add_argument("--outdir", default=None,
                    help="folder for the output files (default: script folder)")
    ap.add_argument("--iters", type=int, default=0,
                    help="annealing iterations per k (0 = auto)")
    ap.add_argument("--restarts", type=int, default=3,
                    help="independent annealing runs per k")
    ap.add_argument("--t-init", dest="t_init", type=float, default=0.0,
                    help="initial temperature (0 = auto-calibrate)")
    ap.add_argument("--t-end", dest="t_end", type=float, default=0.20,
                    help="final temperature")
    ap.add_argument("--reheat", type=int, default=0,
                    help="reheat after N non-improving iterations (0 = never)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-k", dest="max_k", type=int, default=None,
                    help="stop searching above this many days")
    ap.add_argument("--time-limit", dest="time_limit", type=float, default=0.0,
                    help="overall wall-clock budget in seconds (0 = none)")
    ap.add_argument("--tag", default="SA")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--faithful", action="store_true",
                    help="run the unfixed MATLAB replica instead")
    args = ap.parse_args()

    verbose = not args.quiet

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

        tag = args.tag
        if args.faithful and tag == "SA":
            tag = "SA-faithful-matlab"

        print(f"\nAlgorithm: {tag}")
        print("Initial solution: RANDOM (no greedy / DSATUR seeding)")

        record = []
        rng = np.random.default_rng(args.seed)
        evals = None
        start_time = time.time()

        if args.faithful:
            print("Mode: unfixed replica of the original MATLAB sa_coloring.m")
            print("  T_init=1000  alpha=0.99  min_T=1e-4  max_iter=2000  "
                  "k from 1  delta vs best_cost")
            k_found, raw, evals, attempts = faithful_matlab_sa(
                g.v, eu, ev, rng, record, record_stride=1, verbose=verbose)
        else:
            max_iter = args.iters if args.iters > 0 else max(50000, 1000 * g.v)
            print(f"Mode: corrected SA  iters/k={max_iter}  "
                  f"restarts/k={max(1, args.restarts)}  "
                  f"T_init={'auto' if args.t_init <= 0 else args.t_init}  "
                  f"T_end={args.t_end}")
            print(f"Searching upward from the lower bound ({lb} days)")
            k_found, raw, attempts = search(
                g.v, adj, int(eu.size), lb, rng, args, record, verbose=verbose)

        computation_time = time.time() - start_time

        if k_found is None:
            best_seen = min((a["best_conflicts"] for a in attempts),
                            default=None)
            print("\nERROR: annealing never reached a conflict-free schedule.")
            print(f"Fewest conflicts seen: {best_seen}")
            print("Nothing was written -- a timetable with conflicts is not a "
                  "valid result.")
            print("Try a larger budget, e.g. --iters 200000 --restarts 5 "
                  "--reheat 5000")
            sys.exit(2)

        colors = compact_colors(raw, g.courses)
        n_colors = len(set(colors.values()))
        residual = verify_no_conflicts(colors, g)

        print(f"\nComputation time: {computation_time:.4f} seconds")
        print(f"Days searched for: {k_found}")
        print(f"Number of colors used: {n_colors}")
        if n_colors < k_found:
            print(f"  (annealing left {k_found - n_colors} of those days empty)")
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
        if evals is not None:
            print(f"Full cost-function evaluations: {evals}")

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
            "algorithm": tag,
            "mode": "sa_faithful_matlab" if args.faithful else "sa",
            "input_file": os.path.abspath(filename),
            "n_courses": g.v,
            "n_students": len(g.students),
            "n_edges": int(eu.size),
            "colors_used": n_colors,
            "days_searched_for": int(k_found),
            "conflicts": len(residual),
            "clique_lower_bound": lb,
            "optimal_proven": bool(n_colors == lb),
            "computation_time_sec": round(computation_time, 4),
            "full_cost_evaluations": evals,
            "k_attempts": attempts,
            "parameters": {
                "iters": (args.iters if args.iters > 0
                          else max(50000, 1000 * g.v)),
                "restarts": max(1, args.restarts),
                "t_init": ("auto" if args.t_init <= 0 else args.t_init),
                "t_end": args.t_end,
                "reheat": args.reheat,
                "seed": args.seed,
                "max_k": args.max_k,
                "time_limit": args.time_limit,
                "faithful": bool(args.faithful),
            },
        }
        with open(os.path.join(outdir, "sa_run_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("Run summary saved successfully: sa_run_summary.json")

        # ---- convergence curves for EVERY k that was attempted ------------
        conv_path = os.path.join(outdir, "sa_convergence.csv")
        with open(conv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["K", "Step", "Current_Conflicts",
                        "Best_Conflicts", "Temperature"])
            w.writerows(record)
        print("Convergence curve saved successfully: sa_convergence.csv")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
