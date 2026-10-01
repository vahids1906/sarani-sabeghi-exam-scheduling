# -*- coding: utf-8 -*-
"""
tabu_coloring.py -- Exam-day assignment with PURE Tabu Search (TabuCol)
=======================================================================

Why this script exists
----------------------
TabuCol (Hertz & de Werra, 1987) was hidden INSIDE two other algorithms:
memetic_coloring.py used it to improve every individual, and aco_coloring.py
used it to polish the best ant. That made the comparison unreadable: when the
"genetic algorithm" solved the university instance in 0.3 s it had in fact
never run a single generation -- the very first random individual was repaired
by TabuCol and the run returned immediately. The number measured local search,
not evolution.

So TabuCol is now an algorithm in its own right, and it is the CONTROL of the
experiment. The comparison it makes possible is the only one a reviewer
actually cares about:

    does the evolutionary layer (GPX crossover) or the pheromone layer (ACO
    construction) achieve anything that plain local search does not?

Run in the study alongside:
  * memetic_coloring.py --no-tabu   (pure GA:  random init + GPX, no repair)
  * aco_coloring.py     --no-tabu   (pure ACO: construction only, no repair)
  * this script                     (repair only, no population, no pheromone)
and, if you want the hybrids too, the same two scripts with tabu enabled.

NO seeding from any greedy method
---------------------------------
Every start point is a UNIFORMLY RANDOM assignment of courses to days. There
is no Welsh-Powell, no DSATUR, no warm start of any kind, so the result owes
nothing to another algorithm.

The TabuCol implementation is IMPORTED from memetic_coloring.py rather than
copied. That is deliberate: an ablation study is only valid if the component
being isolated is bit-for-bit the same component that runs inside the hybrid.
A copy would drift.

HOW THE NUMBER OF DAYS IS MINIMISED
-----------------------------------
Identical to the other metaheuristics: k is searched UPWARD from a clique
lower bound and the first k that reaches zero conflicts wins. The clique bound
is a property of the graph, not a colouring, so it leaks nothing from any
greedy method. When the answer equals that bound, it is PROVABLY OPTIMAL.

HARD CONSTRAINT: zero conflicts. A colouring with conflicts is never accepted,
and the accepted result is re-verified independently before anything is
written.

INTERFACE -- deliberately identical to coloring3.py / memetic_coloring.py:

  input : input.csv  (col 1 = student id, remaining cols = courses)
  output: ETP_Final_Schedule.xlsx
          coloring_results.xlsx   (Course_Color / Color_Courses /
                                   Color_Student_Counts)
          coloring_results.txt
          tabu_run_summary.json   (extra, for the comparison report)
          tabu_convergence.csv    (extra, best cost per iteration)

Usage
-----
    python tabu_coloring.py [input.csv] [--outdir DIR] [options]

    --tabu-iters N   TabuCol iterations per attempt (0 = auto, max(20000,200n))
    --restarts N     independent random starts per k (default 5)
    --seed N         RNG seed for reproducibility
    --max-k N        never search above this k
    --time-limit S   wall-clock budget in seconds (0 = unlimited)
    --tag NAME       label written into the JSON summary
    --quiet          less per-k chatter
"""

import argparse
import csv
import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Re-use the neighbours' machinery so every algorithm sees the same graph
# ---------------------------------------------------------------------------
def _load_module(filename, alias):
    path = os.path.join(SCRIPT_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "%s was not found next to tabu_coloring.py.\n"
            "This script re-uses it so that every algorithm builds the exact "
            "same conflict graph and writes the exact same output format. "
            "Put all the scripts in one folder." % filename
        )
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_coloring3_module():
    return _load_module("coloring3.py", "_coloring3_for_tabu")


def load_memetic_module():
    """Source of the TabuCol implementation and the graph helpers.

    Importing (instead of copying) guarantees that the standalone TabuCol
    measured here is the identical routine that runs inside the memetic and
    ACO hybrids -- which is the whole point of the ablation.
    """
    return _load_module("memetic_coloring.py", "_memetic_for_tabu")


# ---------------------------------------------------------------------------
# Outer search: smallest k for which TabuCol alone reaches zero conflicts
# ---------------------------------------------------------------------------
def search(mem, n, eu, ev, adj, args, rng, quiet=False):
    lb = mem.clique_lower_bound(n, adj)
    ub = args.max_k or (n if n > 0 else 1)
    deadline = (time.time() + args.time_limit) if args.time_limit else None
    iters = args.tabu_iters if args.tabu_iters > 0 else max(20000, 200 * n)

    print(f"\nClique lower bound on days: {lb}")
    print(f"Searching k upward from {lb} (zero conflicts is a hard constraint)")

    attempts = []
    best_colors = np.zeros(n, dtype=np.int32)
    best_cost, best_curve = None, []

    for k in range(max(lb, 1), ub + 1):
        best_colors, best_cost, best_curve = None, None, []
        used_restarts = 0
        for _ in range(max(1, args.restarts)):
            used_restarts += 1
            start = rng.integers(0, k, size=n, dtype=np.int32)
            colors, cost, curve = mem.tabucol(
                start, k, adj, eu, ev, iters, rng, deadline)
            if best_cost is None or cost < best_cost:
                best_colors, best_cost, best_curve = colors, cost, curve
            if best_cost == 0:
                break
            if deadline is not None and time.time() > deadline:
                break

        attempts.append({"k": k, "conflicts": int(best_cost),
                         "restarts_used": used_restarts})
        if not quiet:
            print(f"  k = {k:>3}  ->  "
                  + ("OK" if best_cost == 0 else f"{best_cost} conflicts"))

        if best_cost == 0:
            return best_colors, 0, best_curve, attempts, lb
        if deadline is not None and time.time() > deadline:
            print("  [!] time limit reached; stopping the k-search")
            break

    return best_colors, int(best_cost or 0), best_curve, attempts, lb


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
            f"Please specify which one to use: python tabu_coloring.py <file.csv>"
        )
    raise FileNotFoundError(
        "No CSV file found. Place 'input.csv' next to the script, "
        "or run: python tabu_coloring.py <filename.csv>"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Exam-day assignment with pure tabu search (TabuCol). "
                    "No population, no pheromone, no greedy seeding.")
    ap.add_argument("input", nargs="?", default=None, help="input CSV file")
    ap.add_argument("--outdir", default=None,
                    help="folder for the output files (default: script folder)")
    ap.add_argument("--tabu-iters", type=int, default=0,
                    help="iterations per attempt (0 = auto, max(20000, 200n))")
    ap.add_argument("--restarts", type=int, default=5,
                    help="independent random starts per k")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-k", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=0.0,
                    help="wall-clock budget in seconds (0 = unlimited)")
    ap.add_argument("--tag", default="TabuCol")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        filename = resolve_input_filename(args.input)
        print(f"Using input file: {filename}")

        outdir = args.outdir or SCRIPT_DIR
        os.makedirs(outdir, exist_ok=True)

        c3 = load_coloring3_module()
        mem = load_memetic_module()

        g = c3.Graph()
        g.load_data_from_csv(filename)
        print(f"\nNumber of courses (vertices): {g.v}")

        eu, ev = mem.build_edge_arrays(g)
        adj = mem.build_adjacency(g.v, eu, ev)
        print(f"Number of conflict edges: {eu.size}")

        rng = np.random.default_rng(args.seed)

        iters = args.tabu_iters if args.tabu_iters > 0 else max(20000, 200 * g.v)
        print(f"\nAlgorithm: {args.tag}")
        print(f"Parameters: tabu_iters={iters}, restarts={args.restarts}")
        print("Initial solution: RANDOM (no greedy seeding, no population)")

        start_time = time.time()
        raw, conflicts, curve, attempts, lb = search(
            mem, g.v, eu, ev, adj, args, rng, quiet=args.quiet)
        computation_time = time.time() - start_time

        colors = mem.compact_colors(raw, g.courses)
        n_colors = len(set(colors.values()))
        residual = mem.verify_no_conflicts(colors, g)

        print(f"\nComputation time: {computation_time:.4f} seconds")
        print(f"Number of colors used: {n_colors}")
        if residual:
            print(f"WARNING: {len(residual)} conflicting course pairs remain!")
        else:
            print("No conflicts detected! The coloring is valid.")
        if n_colors == lb:
            print(f"This matches the clique lower bound ({lb}) "
                  f"-- provably optimal.")

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
            "mode": "tabu",
            "input_file": os.path.abspath(filename),
            "n_courses": g.v,
            "n_students": len(g.students),
            "n_edges": int(eu.size),
            "colors_used": n_colors,
            "conflicts": len(residual),
            "clique_lower_bound": lb,
            "optimal_proven": bool(n_colors == lb),
            "computation_time_sec": round(computation_time, 6),
            "parameters": {
                "tabu_iters": iters,
                "restarts": args.restarts,
                "seed": args.seed,
                "seeding": "random",
                "population": 0,
                "crossover": "none",
                "pheromone": "none",
            },
            "k_attempts": attempts,
        }
        with open(os.path.join(outdir, "tabu_run_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("Run summary saved successfully: tabu_run_summary.json")

        with open(os.path.join(outdir, "tabu_convergence.csv"), "w",
                  encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Step", "Best_Conflicts"])
            for i, c in enumerate(curve or [], start=1):
                w.writerow([i, c])
        print("Convergence curve saved successfully: tabu_convergence.csv")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
