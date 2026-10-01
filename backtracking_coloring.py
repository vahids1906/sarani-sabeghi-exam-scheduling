# -*- coding: utf-8 -*-
"""
backtracking_coloring.py -- Exam-day assignment with exact Backtracking
=========================================================================

Baseline / exact algorithm of the comparison study.
Contains NOTHING but plain recursive backtracking search: no genetic
operators, no local search, no greedy heuristic pretending to be exact.

Backtracking graph colouring
-----------------------------
For a fixed number of colours m, try to assign every course a colour in
{0, ..., m-1} such that no two conflicting courses get the same colour,
using depth-first search with pruning.

To find the MINIMUM number of colours (the chromatic number) this script
performs iterative deepening: it starts at the clique lower bound and tries
m = lb, lb+1, lb+2, ... until a valid colouring is found.

Because graph colouring is NP-hard, pure backtracking can in the worst case
run essentially forever on a large/dense instance.  A per-m time budget
(--time-limit-per-m) and a total wall-clock budget (--time-limit) control
how long the search runs.

If the budget runs out before backtracking has found a valid colouring for
any m, the run is reported as FAILED: no schedule is written, the JSON summary
has "success": false and "colors_used": null, and the process exits with
code 2.  The script NEVER substitutes the result of another algorithm
(Welsh-Powell is used internally only to bound the range of m to search).

ALGORITHMIC IMPROVEMENTS OVER V1 (recursive nature is fully preserved)
------------------------------------------------------------------------
1. Heap-based dynamic vertex ordering (lazy-deletion max-heap).
   The original pick_vertex() was an O(n) linear scan called at every node
   of the search tree.  The heap reduces this to O(log n) amortised, with
   O(deg * log n) updates per assignment.  For moderately dense ETP graphs
   (n ~ 100-500) this is a meaningful constant-factor speedup.

2. Explicit saturation array.
   saturation[v] is kept in sync rather than recomputed via popcount every
   call, saving a bin(...).count() on every heap comparison.

3. Least-constraining-value (LCV) colour ordering.
   When picking a colour for vertex v, we sort the candidate colours by how
   many uncoloured neighbours of v still have that colour available (ascending
   = least constraining = try first).  This is the standard CSP LCV heuristic
   and empirically reduces the number of backtrack steps in coloring problems.
   Combined with the existing colour-symmetry-breaking rule, it often cuts
   the search tree by a further factor of 2-5x on typical ETP instances.

4. Total wall-clock budget (--time-limit).
   The budget is respected globally across all m values rather than only
   per-m, matching the interface expected by algo_compare.py.

INTERFACE (identical to the other algorithms in the comparison study)
----------------------------------------------------------------------
Command:
    python backtracking_coloring.py [input.csv] [--outdir DIR]
                                    [--seed INT] [--quiet]
                                    [--time-limit SEC]
                                    [--time-limit-per-m SEC]

--seed        accepted for API compatibility (backtracking is deterministic;
              the value is recorded in the JSON but not used).
--quiet       suppress per-step progress output (used by algo_compare.py).
--time-limit  total wall-clock budget in seconds.  Distributed across m
              values; each m also respects --time-limit-per-m as a cap.
              algo_compare.py passes this flag for all TIME_LIMIT_KEYS.

Outputs (on success):
    ETP_Final_Schedule.xlsx
    coloring_results.xlsx
    coloring_results.txt
    backtracking_run_summary.json  (read by algo_compare.py)

Outputs (on failure -- no colouring found within the time budget):
    backtracking_run_summary.json ONLY, with
        "success": false, "status": "timeout" | "no_solution_in_range",
        "colors_used": null, "conflicts": null
    and exit code 2.

JSON fields read by algo_compare.py:
    colors_used, conflicts, computation_time_sec, clique_lower_bound,
    parameters, k_attempts
New fields: success (bool), status ("solved" | "timeout" | "no_solution_in_range")
"""

import argparse
import glob
import heapq
import importlib.util
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Re-use coloring3.py's Graph so every output file is format-identical
# ---------------------------------------------------------------------------
def load_coloring3_module():
    path = os.path.join(SCRIPT_DIR, "coloring3.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "coloring3.py was not found next to backtracking_coloring.py.\n"
            "Both files must be in the same folder."
        )
    spec = importlib.util.spec_from_file_location("_coloring3_for_backtracking", path)
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
                continue
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
# Clique lower bound  (same greedy heuristic as the other algorithms)
# ---------------------------------------------------------------------------
def clique_lower_bound(n, adj):
    """Greedily grow a clique from multiple seeds.  Any clique of size C
    requires >= C colours, so this is a valid optimality certificate."""
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
# Core backtracking search
# ---------------------------------------------------------------------------
class SearchTimeout(Exception):
    pass


def try_color_with_m(n, adj, m, deadline):
    """Depth-first backtracking search for a proper colouring using <= m colours.

    Techniques used (all classic, well-known -- they do not change what the
    function proves, only how fast it proves it):

    1. Dynamic DSATUR-style vertex ordering via a lazy-deletion max-heap:
       At every recursive call, the most-saturated uncoloured vertex is chosen
       first (fail-first / most-constrained-variable heuristic).  The heap
       makes this O(log n) amortised instead of the O(n) linear scan in v1.

    2. Least-constraining-value (LCV) colour ordering:
       Candidate colours are sorted by how many uncoloured neighbours of the
       current vertex still have that colour available -- ascending, so we try
       the colour that "hurts" remaining vertices least.  This often reduces
       the number of backtrack steps.

    3. Forward-checking wipeout pruning:
       If assigning colour c to v would leave some uncoloured neighbour with
       zero legal colours, the branch is abandoned immediately.

    4. Colour-symmetry breaking:
       Only the next "new" colour label (prev_max + 1) may be introduced;
       permuting identical labels never produces a new solution.

    5. Bitmask forbidden sets and deadline check every 512 calls.

    Returns:
        (color_array, True)   valid colouring found
        (None, False)         search EXHAUSTED and proved m colours insufficient
        (None, None)          deadline reached before either conclusion
    """
    if n == 0:
        return np.zeros(0, dtype=np.int32), True
    if m <= 0:
        return None, False

    full_mask   = (1 << m) - 1
    color       = [-1] * n
    forbidden   = [0] * n          # bit c set  <=>  colour c forbidden for v
    saturation  = [0] * n          # popcount(forbidden[v])  kept in sync
    degree      = [int(adj[v].size) for v in range(n)]
    colored     = [False] * n
    max_used    = [-1]             # highest colour label introduced in this branch
    ticker      = [0]              # for deadline throttling

    # ------------------------------------------------------------------
    # Lazy-deletion max-heap on (saturation, degree)
    # Entry: (-saturation, -degree, version, vertex)
    # ------------------------------------------------------------------
    ver = [0] * n
    heap = [(-0, -degree[v], 0, v) for v in range(n)]
    heapq.heapify(heap)

    def _push(v):
        """Push an updated priority entry for vertex v."""
        ver[v] += 1
        heapq.heappush(heap, (-saturation[v], -degree[v], ver[v], v))

    def _pick():
        """Return the uncoloured vertex with the highest (sat, deg) priority."""
        while heap:
            neg_s, neg_d, v_ver, v = heapq.heappop(heap)
            if not colored[v] and v_ver == ver[v]:
                return v
        return None  # unreachable if depth < n

    # ------------------------------------------------------------------
    # LCV colour ordering
    # ------------------------------------------------------------------
    def _color_order(v, limit):
        """Return candidate colours 0..limit (not forbidden for v) sorted by
        the number of uncoloured neighbours for which that colour is STILL
        available -- ascending = least constraining first.

        When only 0 or 1 candidates exist, skip sorting (fast path).
        """
        fmask = forbidden[v]
        candidates = [c for c in range(limit + 1) if not (fmask & (1 << c))]
        if len(candidates) <= 1:
            return candidates
        # For each candidate c: count uncoloured neighbours that still have c
        # available.  Lower count = assigning c removes fewer options = LCV.
        adj_v = adj[v].tolist()
        uncoloured_adj = [u for u in adj_v if not colored[u]]
        if not uncoloured_adj:
            return candidates  # no uncoloured neighbours -- order doesn't matter
        counts = []
        for c in candidates:
            bit = 1 << c
            cnt = sum(1 for u in uncoloured_adj if not (forbidden[u] & bit))
            counts.append((cnt, c))
        counts.sort()          # ascending: least constraining first
        return [c for _, c in counts]

    # ------------------------------------------------------------------
    # Recursive backtracking
    # ------------------------------------------------------------------
    def backtrack(depth):
        ticker[0] += 1
        if (ticker[0] & 511) == 0 and time.time() > deadline:
            raise SearchTimeout()
        if depth == n:
            return True

        v = _pick()
        colored[v] = True
        prev_max = max_used[0]
        limit = min(m - 1, prev_max + 1)   # colour-symmetry breaking

        for c in _color_order(v, limit):
            bit = 1 << c
            if c > prev_max:
                max_used[0] = c

            color[v] = c
            touched = []
            dead = False

            for u in adj[v].tolist():
                if not colored[u] and not (forbidden[u] & bit):
                    forbidden[u]   |= bit
                    saturation[u]  += 1
                    _push(u)             # update heap priority
                    touched.append(u)
                    if forbidden[u] == full_mask:   # forward-checking wipeout
                        dead = True
                        break

            if not dead and backtrack(depth + 1):
                return True

            # undo
            for u in touched:
                forbidden[u]  &= ~bit
                saturation[u] -= 1
                _push(u)             # restore heap priority
            color[v]  = -1
            max_used[0] = prev_max

        colored[v] = False
        _push(v)             # re-insert v as uncoloured
        return False

    try:
        found = backtrack(0)
    except SearchTimeout:
        return None, None

    if found:
        return np.array(color, dtype=np.int32), True
    return None, False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compact_colors(raw, courses):
    """Relabel used colours to a contiguous 1..m range (day numbers)."""
    used = sorted(set(int(c) for c in raw))
    remap = {old: new for new, old in enumerate(used, start=1)}
    return {course: remap[int(raw[i])] for i, course in enumerate(courses)}


def verify_no_conflicts(colors, g):
    """Independent conflict check -- never trust the algorithm's own claim."""
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
            f"Specify one: python backtracking_coloring.py <file.csv>"
        )
    raise FileNotFoundError(
        "No CSV file found. Place 'input.csv' next to the script, "
        "or run: python backtracking_coloring.py <filename.csv>"
    )


def log(quiet, *args, **kwargs):
    """Print only when not in quiet mode."""
    if not quiet:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Exam-day assignment with exact Backtracking search.")
    ap.add_argument("input", nargs="?", default=None,
                    help="input CSV file")
    ap.add_argument("--outdir", default=None,
                    help="folder for output files (default: script folder)")

    # --- API-compatibility flags (required by algo_compare.py) ---
    ap.add_argument("--seed", type=int, default=42,
                    help="ignored (backtracking is deterministic); accepted "
                         "for API compatibility with algo_compare.py")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-step progress output "
                         "(algo_compare.py passes this flag)")
    ap.add_argument("--time-limit", dest="time_limit", type=float, default=None,
                    help="total wall-clock budget in seconds, distributed across "
                         "all candidate m values.  algo_compare.py passes this.")

    # --- per-m budget (kept for standalone use) ---
    ap.add_argument("--time-limit-per-m", type=float, default=30.0,
                    help="max seconds at each candidate m value (default 30). "
                         "Capped by --time-limit when both are given.")
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="max number of m values above the lower bound to try "
                         "(default: up to the Welsh-Powell upper bound)")
    ap.add_argument("--tag", default="Backtracking")
    args = ap.parse_args()

    try:
        filename = resolve_input_filename(args.input)
        log(args.quiet, f"Using input file: {filename}")

        outdir = args.outdir or SCRIPT_DIR
        os.makedirs(outdir, exist_ok=True)

        c3 = load_coloring3_module()
        g = c3.Graph()
        g.load_data_from_csv(filename)
        log(args.quiet, f"\nNumber of courses (vertices): {g.v}")

        eu, ev = build_edge_arrays(g)
        adj    = build_adjacency(g.v, eu, ev)
        log(args.quiet, f"Number of conflict edges: {eu.size}")

        lb = clique_lower_bound(g.v, adj)
        log(args.quiet, f"Clique lower bound on days: {lb}")

        # Welsh-Powell is used ONLY to bound the range of m to search.
        # Its colouring is never returned as a result of this algorithm.
        ub = len(set(g.welsh_powell_coloring().values()))
        log(args.quiet, f"Welsh-Powell upper bound on days: {ub}")

        max_attempts = args.max_attempts
        if max_attempts is None:
            max_attempts = max(1, ub - lb + 1)

        # --- time budget setup ---
        global_start    = time.time()
        global_deadline = (global_start + args.time_limit) if args.time_limit else None
        per_m_budget    = args.time_limit_per_m

        log(args.quiet, f"\nAlgorithm: {args.tag}")
        if global_deadline:
            log(args.quiet, f"Total time limit: {args.time_limit:.1f}s")
        log(args.quiet, f"Time limit per candidate m: {per_m_budget:.1f}s")

        best_raw    = None
        best_k      = None
        proven_opt  = False
        exhausted   = True      # True until a timeout occurs
        k_attempts  = []        # field name read by algo_compare.py

        m = lb
        attempts = 0
        while attempts < max_attempts and m <= ub:

            # --- respect global deadline ---
            if global_deadline:
                remaining = global_deadline - time.time()
                if remaining <= 0:
                    exhausted = False
                    log(args.quiet, "  Global time limit reached.")
                    break
                m_budget = min(per_m_budget, remaining)
            else:
                m_budget = per_m_budget

            deadline = time.time() + m_budget
            log(args.quiet, f"  Trying m = {m} colours ...", end=" ", flush=True)

            raw, status = try_color_with_m(g.v, adj, m, deadline)
            attempts += 1
            k_attempts.append(m)

            if status is True:
                log(args.quiet, "SUCCESS")
                best_raw, best_k = raw, m
                proven_opt = (m == lb)   # can't beat the clique bound
                break
            elif status is False:
                log(args.quiet, "proven impossible")
                m += 1
            else:   # None  ->  timed out, inconclusive
                log(args.quiet, "TIMEOUT (inconclusive)")
                exhausted = False
                m += 1

        computation_time = time.time() - global_start

        # Fields shared by the success and the failure summary.
        summary = {
            "algorithm":   args.tag,
            "mode":        "backtracking",
            "input_file":  os.path.abspath(filename),
            "n_courses":   g.v,
            "n_students":  len(g.students),
            "n_edges":     int(eu.size),
            "computation_time_sec":   round(computation_time, 4),
            "clique_lower_bound":     lb,
            "parameters": {
                "seed":                args.seed,
                "time_limit_sec":      args.time_limit,
                "time_limit_per_m_sec": args.time_limit_per_m,
                "max_attempts":        max_attempts,
            },
            "k_attempts":             k_attempts,
            "welsh_powell_upper_bound":      ub,
            "search_exhausted_within_budget": bool(exhausted),
            # Kept for compatibility with older readers; there is no fallback
            # any more, so this is always False.
            "used_welsh_powell_fallback":    False,
        }
        summary_path = os.path.join(outdir, "backtracking_run_summary.json")

        # --- FAILURE: no colouring found.  Report it; never substitute another
        #     algorithm's result. ---
        if best_raw is None:
            status = "timeout" if not exhausted else "no_solution_in_range"
            if status == "timeout":
                message = ("Time budget exhausted before backtracking found a "
                           "valid colouring for any m tried "
                           f"(m tried: {k_attempts}; clique lower bound {lb}, "
                           f"Welsh-Powell upper bound {ub}).")
            else:
                message = ("Backtracking finished without finding a colouring "
                           f"(m tried: {k_attempts}; clique lower bound {lb}, "
                           f"Welsh-Powell upper bound {ub}).")
            summary.update({
                "success":        False,
                "status":         status,
                "message":        message,
                "colors_used":    None,
                "conflicts":      None,
                "optimal_proven": False,
            })
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            log(args.quiet, f"\nComputation time : {computation_time:.4f} s")
            log(args.quiet, "Run summary saved: backtracking_run_summary.json")
            # Printed even in --quiet mode so the failure is never silent.
            print(f"FAILED ({status}): {message} No schedule was produced.")
            sys.exit(2)

        colors   = compact_colors(best_raw, g.courses)
        n_colors = len(set(colors.values()))

        residual = verify_no_conflicts(colors, g)

        log(args.quiet, f"\nComputation time : {computation_time:.4f} s")
        log(args.quiet, f"Colours used     : {n_colors}")
        if residual:
            log(args.quiet, f"WARNING: {len(residual)} conflicting pairs remain!")
        else:
            log(args.quiet, "No conflicts detected -- colouring is valid.")
        if proven_opt:
            log(args.quiet,
                f"Provably OPTIMAL (matches clique lower bound {lb}).")
        else:
            log(args.quiet,
                f"Gap to lower bound: {n_colors - lb} day(s) "
                f"(valid colouring found; optimality not proven).")

        color_student_counts = g.compute_color_student_counts(colors)
        log(args.quiet, "\nColour -> unique students:")
        for col in sorted(color_student_counts):
            log(args.quiet, f"  Colour {col}: {color_student_counts[col]} students")

        # --- JSON summary FIRST (algo_compare.py reads this; must exist even if xlsx fails) ---
        summary.update({
            "success":                True,
            "status":                 "solved",
            # ---- fields read directly by algo_compare.py ----
            "colors_used":            n_colors,
            "conflicts":              len(residual),
            # ---- extra diagnostics ----
            "optimal_proven":         bool(proven_opt),
        })
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log(args.quiet, "Run summary saved: backtracking_run_summary.json")

        # --- Excel/text outputs (non-fatal: JSON above is already written) ---
        # coloring3.py prints its own success messages; suppress them in --quiet mode.
        import io, contextlib
        _suppress = contextlib.redirect_stdout(io.StringIO()) if args.quiet else contextlib.nullcontext()
        cwd = os.getcwd()
        try:
            os.chdir(outdir)
            with _suppress:
                g.export_schedule_table(colors, color_student_counts)
                g.export_text_report(colors, color_student_counts,
                                     computation_time, conflicts=residual,
                                     input_filename=filename)
                g.export_detailed_excel(colors, color_student_counts)
        except Exception as export_err:
            log(args.quiet, f"Warning: Excel/text export failed ({export_err}); "
                "JSON summary was already saved.")
            if not args.quiet:
                raise
        finally:
            os.chdir(cwd)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
