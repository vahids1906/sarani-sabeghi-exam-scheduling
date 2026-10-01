# -*- coding: utf-8 -*-
"""
memetic_coloring.py -- Exam-day assignment with a Genetic Algorithm + Tabu Search
=================================================================================

Algorithm 3 of the comparison study: a hybrid (memetic) evolutionary
algorithm. Contains NO DSATUR and NO Welsh-Powell anywhere. Every starting
point is random, so the result owes nothing to any greedy heuristic -- this is
the honest measurement of what the metaheuristic achieves on its own.

This is the Galinier-Hao scheme, the reference method for graph colouring:

  GPX  (Greedy Partition Crossover)
       Builds a child by alternately stealing the LARGEST remaining colour
       class from each parent, then randomly filling whatever is left over.
       Colour labels are arbitrary -- "day 3" in one parent and "day 7" in
       the other may describe the same group of courses. Classic single-point
       crossover on a string of colour numbers destroys exactly that
       structure, which is why the pure GA fails on this problem. GPX crosses
       over PARTITIONS instead of positions.

  TabuCol (Hertz & de Werra)
       Local search applied to every child. One move = recolour one
       CONFLICTING course. The (course, day) pair just left behind is made
       tabu for a dynamic number of iterations, which stops the search from
       walking straight back. Move evaluation is incremental through the
       matrix
           gamma[v][c] = how many neighbours of course v currently sit on day c
       so each move costs O(degree) instead of a full re-evaluation, and an
       aspiration rule still admits a tabu move when it beats the best cost
       ever seen.

Selection/replacement follows the standard HEA loop: two random parents -> GPX
-> TabuCol -> the child replaces the worse of the two parents.

ONE COST CONVENTION EVERYWHERE
------------------------------
    cost = number of edges whose two endpoints share a day  (LOWER IS BETTER)

The MATLAB prototype mixed conventions and paid for it: calculate_fitness.m
returns a POSITIVE conflict count while its own comment claims the value is
negated for maximisation; coloring3.m then sorted 'descend' and treated pop(1)
as the best individual, i.e. it optimised towards the WORST solution; and
tournament_selection.m compares with '>=', contradicting coloring4.m's
ascending sort. None of that ambiguity survives here.

HOW THE NUMBER OF DAYS IS MINIMISED
-----------------------------------
k is searched UPWARD starting from a clique lower bound, and the first k that
reaches zero conflicts wins. The clique bound is a property of the graph
itself -- it is not a colouring and leaks nothing from any greedy method.
When the answer equals that bound the result is PROVABLY OPTIMAL.

HARD CONSTRAINT: zero conflicts. A colouring with conflicts is never accepted,
and the accepted result is re-verified independently before anything is
written.

INTERFACE -- deliberately identical to coloring3.py:

  input : input.csv  (col 1 = student id, remaining cols = courses)
  output: ETP_Final_Schedule.xlsx   (Day/Color, Course, Course_Students,
                                     Total_Unique_Students_In_Day)
          coloring_results.xlsx     (Course_Color / Color_Courses /
                                     Color_Student_Counts)
          coloring_results.txt
          memetic_run_summary.json  (extra, for the comparison report)
          memetic_convergence.csv   (extra, best cost per step)

The conflict graph is built and the results are exported by re-using the
``Graph`` class from ``coloring3.py``, so every output file is byte-compatible
with the greedy run and nothing downstream needs to change
(COUNT-NUMBER-OF-CONFLICTS.py, allocate_rooms.py, compare_schedules.py,
calendar_builder.py).

Usage
-----
    python memetic_coloring.py [input.csv] [--outdir DIR] [options]

    --pop-size N      population size            (default 10)
    --generations N   GPX children per k         (default 40)
    --tabu-iters N    TabuCol iterations/child   (default 4000)
    --restarts N      independent attempts per k (default 1)
    --seed N          RNG seed for reproducibility
    --max-k N         never search above this k
    --time-limit S    wall-clock budget in seconds (0 = unlimited)
    --tag NAME        label written into the JSON summary
    --quiet           less per-k chatter
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
BIG = 1 << 30


# ---------------------------------------------------------------------------
# Re-use coloring3.py's Graph so the graph and every output file are identical
# ---------------------------------------------------------------------------
def load_coloring3_module():
    path = os.path.join(SCRIPT_DIR, "coloring3.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "coloring3.py was not found next to memetic_coloring.py.\n"
            "This script re-uses its Graph class so that all algorithms build\n"
            "the exact same conflict graph and write the exact same output\n"
            "format. Put both files in the same folder."
        )
    spec = importlib.util.spec_from_file_location("_coloring3_for_memetic", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_edge_arrays(g):
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


def clique_lower_bound(n, adj):
    """Greedily grow a clique: a valid lower bound on the number of days."""
    if n == 0:
        return 0
    sets = [set(a.tolist()) for a in adj]
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
# TabuCol -- local search
# ---------------------------------------------------------------------------
def tabucol(color, k, adj, eu, ev, max_iters, rng, deadline=None):
    """Returns (best_color, best_cost, curve)."""
    n = color.shape[0]
    color = color.astype(np.int32).copy()
    if n == 0 or eu.size == 0:
        return color, 0, [0]

    # gamma[v][c] = number of neighbours of v currently on day c
    gamma = np.zeros((n, k), dtype=np.int32)
    np.add.at(gamma, (eu, color[ev]), 1)
    np.add.at(gamma, (ev, color[eu]), 1)

    rows = np.arange(n)
    cost = int(gamma[rows, color].sum() // 2)
    best_color, best_cost = color.copy(), cost
    curve = [cost]

    tabu = np.zeros((n, k), dtype=np.int64)      # move allowed once iter >= tabu

    for it in range(max_iters):
        if cost == 0:
            break
        if deadline is not None and (it & 255) == 0 and time.time() > deadline:
            break

        conf = np.nonzero(gamma[rows, color] > 0)[0]   # conflicting courses
        if conf.size == 0:
            break

        cur = gamma[conf, color[conf]][:, None]
        delta = (gamma[conf] - cur).astype(np.int64)
        delta[np.arange(conf.size), color[conf]] = BIG        # cannot stay put

        # aspiration: a tabu move is allowed if it beats the best cost so far
        allowed = (tabu[conf] <= it) | ((cost + delta) < best_cost)
        cand = np.where(allowed, delta, BIG)

        m = cand.min()
        if m >= BIG:                                          # everything tabu
            vi = int(rng.integers(0, conf.size))
            v = int(conf[vi])
            new_c = int(rng.integers(0, k))
            if new_c == color[v]:
                new_c = (new_c + 1) % k
            d = int(gamma[v, new_c] - gamma[v, color[v]])
        else:
            flat = np.nonzero(cand.ravel() == m)[0]
            choice = int(flat[rng.integers(0, flat.size)])
            vi, new_c = divmod(choice, k)
            v = int(conf[vi])
            d = int(m)

        old_c = int(color[v])
        nb = adj[v]
        if nb.size:
            gamma[nb, old_c] -= 1
            gamma[nb, new_c] += 1
        color[v] = new_c
        cost += d

        tenure = int(rng.integers(0, 10)) + int(0.6 * conf.size)
        tabu[v, old_c] = it + 1 + tenure

        if cost < best_cost:
            best_cost, best_color = cost, color.copy()
        curve.append(cost)

    return best_color, best_cost, curve


# ---------------------------------------------------------------------------
# GPX -- Greedy Partition Crossover
# ---------------------------------------------------------------------------
def gpx(p1, p2, k, rng):
    n = p1.shape[0]
    parents = (p1.astype(np.int32), p2.astype(np.int32))
    child = np.full(n, -1, dtype=np.int32)
    unassigned = np.ones(n, dtype=bool)

    for i in range(k):
        if not unassigned.any():
            break
        p = parents[i & 1]                       # alternate between parents
        counts = np.bincount(p[unassigned], minlength=k)
        c = int(counts.argmax())                 # largest remaining class
        if counts[c] == 0:
            continue
        members = np.nonzero(unassigned & (p == c))[0]
        child[members] = i
        unassigned[members] = False

    leftover = np.nonzero(unassigned)[0]
    if leftover.size:
        child[leftover] = rng.integers(0, k, size=leftover.size)
    return child


# ---------------------------------------------------------------------------
# Memetic run at a fixed number of days k
# ---------------------------------------------------------------------------
def memetic_run(n, adj, eu, ev, k, pop_size, generations, tabu_iters, rng,
                deadline=None):
    """Population starts RANDOM -- no greedy seeding of any kind."""
    if n == 0:
        return np.zeros(0, dtype=np.int32), 0, [0], 0

    pop, costs = [], []
    for _ in range(pop_size):
        start = rng.integers(0, k, size=n, dtype=np.int32)
        sol, cost, _ = tabucol(start, k, adj, eu, ev, tabu_iters, rng, deadline)
        pop.append(sol)
        costs.append(cost)
        if cost == 0:
            return sol, 0, [cost], 0
        if deadline is not None and time.time() > deadline:
            break

    costs = np.array(costs, dtype=np.int64)
    curve = [int(costs.min())]
    gens = 0

    for _ in range(generations):
        if costs.min() == 0 or len(pop) < 2:
            break
        if deadline is not None and time.time() > deadline:
            break
        gens += 1

        i1, i2 = rng.choice(len(pop), size=2, replace=False)
        child = gpx(pop[i1], pop[i2], k, rng)
        child, ccost, _ = tabucol(child, k, adj, eu, ev, tabu_iters, rng,
                                  deadline)

        worse = i1 if costs[i1] >= costs[i2] else i2   # HEA replacement
        if ccost <= costs[worse]:
            pop[worse] = child
            costs[worse] = ccost
        curve.append(int(costs.min()))

        if ccost == 0:
            return child, 0, curve, gens

    best_i = int(costs.argmin())
    return pop[best_i], int(costs[best_i]), curve, gens


# ---------------------------------------------------------------------------
# Outer search: smallest k that reaches zero conflicts
# ---------------------------------------------------------------------------
def search(n, eu, ev, adj, args, rng, quiet=False):
    lb = clique_lower_bound(n, adj)
    ub = args.max_k or (n if n > 0 else 1)
    deadline = (time.time() + args.time_limit) if args.time_limit else None

    print(f"\nClique lower bound on days: {lb}")
    print(f"Searching k upward from {lb} (zero conflicts is a hard constraint)")

    attempts = []
    best_colors = np.zeros(n, dtype=np.int32)
    best_cost, best_curve, k = None, [], max(lb, 1)

    for k in range(max(lb, 1), ub + 1):
        best_colors, best_cost, best_curve, gens = None, None, [], 0
        for _ in range(max(1, args.restarts)):
            colors, cost, curve, used = memetic_run(
                n, adj, eu, ev, k,
                pop_size=args.pop_size,
                generations=args.generations,
                tabu_iters=args.tabu_iters,
                rng=rng,
                deadline=deadline,
            )
            gens += used
            if best_cost is None or cost < best_cost:
                best_colors, best_cost, best_curve = colors, cost, curve
            if best_cost == 0:
                break

        attempts.append({"k": k, "conflicts": int(best_cost),
                         "generations": gens})
        if not quiet:
            print(f"  k = {k:>3}  ->  "
                  + ("OK" if best_cost == 0 else f"{best_cost} conflicts"))

        if best_cost == 0:
            return best_colors, 0, best_curve, attempts, lb
        if deadline is not None and time.time() > deadline:
            print("  [!] time limit reached; stopping the k-search")
            break

    return best_colors, int(best_cost or 0), best_curve, attempts, lb


def compact_colors(raw, courses):
    used = sorted(set(int(c) for c in raw))
    remap = {old: new for new, old in enumerate(used, start=1)}
    return {course: remap[int(raw[i])] for i, course in enumerate(courses)}


def verify_no_conflicts(colors, g):
    bad = []
    for course, neighbours in g.graph.items():
        for other in neighbours:
            if course in colors and other in colors and colors[course] == colors[other]:
                pair = tuple(sorted((course, other)))
                bad.append((pair[0], pair[1], colors[course]))
    return sorted(set(bad))


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
            f"Please specify which one to use: python memetic_coloring.py <file.csv>"
        )
    raise FileNotFoundError(
        "No CSV file found. Place 'input.csv' next to the script, "
        "or run: python memetic_coloring.py <filename.csv>"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Exam-day assignment with a genetic algorithm + tabu search "
                    "(no DSATUR, no Welsh-Powell).")
    ap.add_argument("input", nargs="?", default=None, help="input CSV file")
    ap.add_argument("--outdir", default=None,
                    help="folder for the output files (default: script folder)")
    ap.add_argument("--pop-size", type=int, default=10)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--tabu-iters", type=int, default=4000)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-k", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=0.0)
    ap.add_argument("--tag", default="Memetic-GA-Tabu")
    ap.add_argument("--quiet", action="store_true")
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

        rng = np.random.default_rng(args.seed)

        print(f"\nAlgorithm: {args.tag}")
        print(f"Parameters: pop={args.pop_size}, generations={args.generations}, "
              f"tabu_iters={args.tabu_iters}, restarts={args.restarts}")
        print("Initial population: RANDOM (no greedy seeding)")

        start_time = time.time()
        raw, conflicts, curve, attempts, lb = search(
            g.v, eu, ev, adj, args, rng, quiet=args.quiet)
        computation_time = time.time() - start_time

        colors = compact_colors(raw, g.courses)
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
            "mode": "memetic",
            "input_file": os.path.abspath(filename),
            "n_courses": g.v,
            "n_students": len(g.students),
            "n_edges": int(eu.size),
            "colors_used": n_colors,
            "conflicts": len(residual),
            "clique_lower_bound": lb,
            "optimal_proven": bool(n_colors == lb),
            "computation_time_sec": round(computation_time, 4),
            "parameters": {
                "pop_size": args.pop_size,
                "generations": args.generations,
                "tabu_iters": args.tabu_iters,
                "restarts": args.restarts,
                "seed": args.seed,
                "seeding": "random",
            },
            "k_attempts": attempts,
        }
        with open(os.path.join(outdir, "memetic_run_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("Run summary saved successfully: memetic_run_summary.json")

        with open(os.path.join(outdir, "memetic_convergence.csv"), "w",
                  encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Step", "Best_Conflicts"])
            for i, c in enumerate(curve or [], start=1):
                w.writerow([i, c])
        print("Convergence curve saved successfully: memetic_convergence.csv")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
