# -*- coding: utf-8 -*-
"""Ant Colony Optimisation + Tabu Search exam-day scheduling.

Algorithm 4 of the comparison study, alongside:
    coloring3.py        Welsh-Powell greedy      (the existing pipeline)
    dsatur_coloring.py  DSATUR
    memetic_coloring.py GA + TabuCol
    sa_coloring.py      Simulated Annealing
    aco_coloring.py     ACO + TabuCol            (this file)

This is a Python port of the author's ant_colony1.m / tabu_search.m, with the
algorithmic defects repaired. It is a NEW, STANDALONE file: coloring3.py and
run_pipeline.py are not modified. The command-line interface, the input format
and the three output files are deliberately identical to coloring3.py, so the
four algorithms can be swapped and compared on exactly the same footing.

Zero exam conflicts is a HARD constraint. The number of days k is searched
UPWARD from a proven lower bound, and the first k that reaches zero conflicts
wins. The final colouring is re-verified independently against the conflict
graph before anything is written to disk.

Requires coloring3.py in the same folder (it reuses its data loading and its
exporters, so the outputs are byte-compatible with the existing pipeline).


WHAT WAS WRONG IN THE MATLAB, AND WHAT THIS FILE DOES INSTEAD
=============================================================================
Everything below was verified by replaying the original code on the author's
own 71-vertex instance (see /data/matlab/review_aco.py). That instance has a
clique lower bound of 8 and DSATUR colours it in 8, so 8 days is provably
optimal and any honest algorithm should reach it.

1. THE PHEROMONE WAS NEVER USED -- the headline defect.
   run_ant_coloring computes
           p = tau(v,:).^alpha;
   and then, two lines later, unconditionally overwrites it:
           p = ones(1, k) / k;
   So every ant chose every colour uniformly at random. tau was evaporated and
   deposited into on every iteration, but never influenced a single decision.
   PROOF: running the code with rho = 0.1 and with rho = 0.9 -- which produces
   completely different pheromone tables (tau range 1.06..1.80 versus
   0.01..0.34) -- gives byte-identical results, 23/26/26 conflicts on seeds
   1/2/3. The "ant colony" was 50 x 100 = 5000 uniform random colourings with
   a best-of kept. There was no learning of any kind.
   FIXED: p is proportional to tau[v][c]**alpha * eta[v][c]**beta, and the
   pheromone genuinely steers construction.

2. NO HEURISTIC INFORMATION. beta was declared as 20 and never used; the
   author's own comment says so ("بدون استفاده در کد"). An ant coloured vertex
   v without even looking at the colours of v's already-coloured neighbours.
   FIXED: eta[v][c] = 1 / (1 + number of already-coloured neighbours of v that
   already use colour c), so a colour that is still free for v is strongly
   preferred. Combined with fix 1 this is what does the real work: with proper
   tau and eta, ACO alone reaches 0-1 conflicts at k = 8 where the original
   reached 23-26.

3. FIXED VERTEX ORDER. Every ant coloured vertices 1..n in the same order, so
   the colony had far less diversity than it appears to.
   FIXED: each ant walks a fresh random permutation.
   (Superseded as the default by defect 15 below; the uniform random
   permutation is still available as --order random.)

4. EVERY ANT DEPOSITED PHEROMONE, including the worst one. The guard was
           if AntConflicts(ant) < inf
   which is always true, so it did nothing. The accompanying comment worries
   about "infinite pheromone if the cost is zero", but deposit = 1/(1+conf) is
   bounded by 1, so that concern was misplaced.
   FIXED: elitist deposit -- only the iteration-best ant and the global best
   reinforce, and tau is clamped to [tau_min, tau_max] as in MAX-MIN Ant
   System, which prevents both stagnation and unbounded growth.

5. RouletteWheelSelection COULD RETURN EMPTY.
           j = find(r <= c, 1, 'first');
   If floating-point rounding leaves cumsum(P) slightly below r, find returns
   [] and the assignment colors(v) = [] corrupts the solution vector instead
   of raising a clear error.
   FIXED: cumulative search with an explicit clamp to the last colour.

6. THE TABU CRITERION TESTED THE WRONG ATTRIBUTE.
           if tabu_list(i,1) == v && tabu_list(i,2) == current_color_v
   The move under evaluation is "recolour v to new_color", but the test looks
   at current_color_v, which is the same for every new_color in that loop. So
   the criterion could not distinguish one candidate move from another: it
   either banned ALL moves of vertex v or none of them.
   I expected this to make the tabu list inert. MEASURED, IT DOES NOT: the
   test blocked 1302 candidate moves in a single 63-move run, so it is
   heavily OVER-restrictive. Yet it still failed at the one job a tabu list
   exists for -- 17 of 62 consecutive move pairs were immediate reversals,
   the algorithm undoing its own previous move. So the criterion is wrong in
   both directions at once: it forbids far too much, and it still permits the
   2-cycles it was supposed to prevent.
   FIXED: proper TabuCol. The forbidden attribute is the (vertex, colour) pair
   the search just LEFT, so returning there is what gets banned, and nothing
   else is restricted.

7. TABU TENURE WAS A FIXED 15, STORED IN A CIRCULAR BUFFER that was rescanned
   linearly inside the innermost loop.
   FIXED: a tabu_until[v][c] iteration-stamp matrix, so the test is O(1), and
   the standard dynamic tenure 10 + 0.6 * (number of conflicting vertices),
   which adapts to how stuck the search currently is.

8. THE COST WAS ALWAYS RECOMPUTED FROM SCRATCH. tabu_search_coloring built a
   temporary colouring and called coloring_conflicts -- an O(n^2) double loop
   -- for every one of the n x k candidate moves, every iteration. For
   Dentistry (119 courses, k = 24) that is 119 x 24 x 2229 = 6.4 million edge
   tests per iteration, roughly 640 million for a 100-iteration run.
   FIXED: the gamma[vertex][colour] structure used by the memetic and SA
   files. The exact delta of a move is gamma[v][c2] - gamma[v][c1], read in
   constant time, and applying a move costs O(degree of v).

9. k STARTED AT 1 AND CLIMBED ONE AT A TIME. For Medicine, whose lower bound
   is 29, that is 28 complete ACO + tabu runs that cannot possibly succeed.
   FIXED: the search starts at a clique lower bound, which also lets the file
   PROVE optimality when it lands on that bound.

10. best_conflicts_found WAS NEVER RESET BETWEEN k VALUES. It was set to inf
    once, before the k loop, and only ever lowered. So the per-k progress line
    "(تعداد تضادها: %d)" reported the best count over all k tried so far, not
    the result for the k just attempted -- k = 12 could look better than it
    was because k = 11 had done well.
    FIXED: every k attempt is scored and reported independently, and all of
    them are recorded in the JSON summary.

11. IT REPORTED k, NOT THE NUMBER OF DAYS ACTUALLY USED. A colouring found at
    k = 12 may only use 10 distinct colours, so the original could understate
    its own result.
    FIXED: the answer is compacted and the distinct-day count is reported.

12. NO INDEPENDENT VERIFICATION. The answer was trusted from an incrementally
    maintained counter. Since the tabu phase updated that counter with
    best_move_delta, a single delta error would silently produce a schedule
    that is announced as conflict-free but is not.
    FIXED: the final colouring is re-checked edge by edge before writing, and
    the script refuses to write a conflicted schedule.

13. tabu_search.m (the separate file) IS DEAD AND BROKEN. ant_colony1.m never
    calls it -- it has its own internal tabu_search_coloring. And it could not
    work anyway: it keeps a move when new_fitness > best_fitness, but
    calculate_fitness.m returns a POSITIVE conflict count, so that comparison
    MAXIMISES conflicts. It is the same sign error found earlier in
    tournament_selection.m. It also tests only one column of its tabu table
    (~ismember(v, tabu_list(:,tabu_idx))) rather than the whole list.
    DECISION: discarded. The internal tabu_search_coloring was used as the
    statement of intent, and reimplemented as textbook TabuCol.

14. THE ADJACENCY MATRIX IS HARDCODED and slightly wrong. It is byte-identical
    to the matrix in sa_coloring.m (0 differing cells) but differs from the
    one in coloring3.m in 4 cells. It has 4 asymmetric pairs -- (55,66),
    (60,65), (60,66), (65,67) in 1-based indexing -- and because
    coloring_conflicts only ever reads A(i,j) with i < j, 3 real conflict
    pairs were silently invisible: 345 edges were counted where the honest
    symmetrised graph has 348. Any number computed from that matrix is
    slightly optimistic, so old MATLAB results should not be quoted in the
    paper without recomputation.
    FIXED: the graph is built from the real student data by coloring3.py, and
    is symmetric by construction.

15. "NO OUTPUT" ON LARGE INSTANCES  (found afterwards, and measured)
    Two separate causes, both in this file:

    (a) A SILENT FAILURE PATH. When the search ended without a conflict-free
        colouring -- time limit reached, or the k range exhausted -- main()
        printed a message and exited WITHOUT writing anything, not even the
        JSON summary or the convergence file that this docstring promises
        covers "where the k search failed". To a pipeline that reads
        aco_run_summary.json that is indistinguishable from a crash.
        FIXED: the summary (success=false, status, fewest_conflicts_seen,
        k_attempts) and aco_convergence.csv are written, no schedule is
        written (a conflicted schedule is still refused), exit code is 2.
        The JSON is also now written BEFORE the Excel / text exporters, so an
        exporter exception can no longer make a solved run look empty.

    (b) THE ANT VERTEX ORDER (defect 3's "fresh random permutation") does not
        scale to graphs with hub vertices. Measured on a 825-course, 8,364-edge
        conflict graph built from the real enrolment sheet (clique bound 17,
        greedy 18; 251 courses have degree >= 21, the largest has 378):
          random order : never conflict-free. 90 s budget -> best 3
                         conflicts; 260 s -> solved only at k = 20; even 1
                         restart/k up to k = 24 stayed at 1-2 conflicts.
                         The leftover conflicts sat on hubs (degree 234-350)
                         whose neighbours already used every one of the k
                         colours -- a hub coloured late has no free colour,
                         and single-vertex tabu moves cannot free one.
          hubs first   : (default now) k = 18, 19, 20 solved in ~4 s each; k = 17,
                         the clique bound, solved in ~4 s in 1 of 3 seeds
                         (and restarts cover the rest).
        Bookkeeping was checked and is correct: the tabu's reported best
        equals a from-scratch recount of the returned colouring.
        FIXED: --order degree (default) visits vertices by
        degree * (1 + noise * U(0,1)), so hubs are coloured while colours are
        plentiful and the noise still makes every ant different. Colours are
        still chosen by pheromone x heuristic roulette; no greedy / DSATUR
        colouring is used as a seed. --order random reproduces the previous
        behaviour exactly.
        NOTE FOR THE PAPER: this changes how the ant construction is described
        (hubs-first randomised order, as in ANTCOL-style methods) and results
        produced before this change used the random order.

IN FAIRNESS TO THE ORIGINAL: on that 71-vertex instance the hybrid did reach
0 conflicts at k = 8, the true optimum. Starting from a 23-conflict random
colouring, the buggy tabu phase still repaired it in 63 moves. So the design
intent was sound and the pipeline worked end to end; what was broken was the
efficiency and the scaling, not the outcome on a small easy graph. The gap
shows up as the instances grow, which is exactly where the exam data lives.


Usage
-----
    python aco_coloring.py [input.csv] [--outdir DIR]

    --ants N         ants per ACO iteration (default 20)
    --aco-iters N    ACO iterations per attempt (default 30)
    --alpha X        pheromone weight (default 1.0)
    --beta X         heuristic weight (default 3.0)
    --rho X          pheromone evaporation rate (default 0.1)
    --tabu-iters N   TabuCol iterations (0 = auto, max(20000, 200*n))
    --no-tabu        run pure ACO with no tabu phase, for ablation
    --order MODE     ant vertex order: degree (default, hubs first with noise)
                     or random (uniform permutation, the previous behaviour)
    --order-noise X  noise for --order degree (default 0.3)
    --restarts N     independent ACO+tabu attempts per k (default 6)

                     Do not lower this without testing. At the original value
                     of 2 the file lost a day to plain Welsh-Powell on Nursing
                     (15 days against 14, lower bound 14): the tabu phase kept
                     stalling at exactly 1 conflict. At 6 restarts the third
                     attempt reached 14, provably optimal, in 1.3 seconds. ACO
                     is a randomised CONSTRUCTION method, so its variance sits
                     between independent runs rather than inside one run, and
                     extra restarts are the cheap way to buy it down. This is
                     the opposite of the lesson learned in sa_coloring.py,
                     where extra restarts were useless and a warmer final
                     temperature was what mattered.
    --seed N         RNG seed
    --max-k N        stop searching above N days
    --time-limit S   overall wall-clock budget in seconds (0 = none)
    --tag NAME       label written into the JSON summary
    --quiet          suppress the per-k progress lines

    --faithful       OPTIONAL. Replay the original MATLAB, bugs and all, for
                     the "effect of parameter tuning" section of the paper.
                     Off by default. Use it on the 71-vertex matrix instance,
                     NOT on the real majors: it rebuilds the cost from scratch
                     for every candidate move, so on Dentistry one tabu
                     iteration alone is ~6.4 million edge tests. The script
                     warns and honours --time-limit if you try.

Exit codes: 0 = solved, 2 = search failed (no conflict-free schedule; only
aco_run_summary.json with "success": false and aco_convergence.csv are
written), 1 = crash / bad input / failed final verification.

Outputs (identical names to coloring3.py, plus two extras)
----------------------------------------------------------
    ETP_Final_Schedule.xlsx / .csv    the schedule
    coloring_results.txt              the text report
    coloring_results.xlsx / .csv      the detailed workbook
    aco_run_summary.json              machine-readable result + parameters
    aco_convergence.csv               K, Phase, Step, Current, Best

The convergence file covers EVERY k attempted, not just the successful one,
and tags each row with the phase (ACO or TABU). That way the paper can show
both the per-k curves and the handoff from construction to local search, and
can document where the k search failed before it succeeded.
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
# Reuse coloring3.py so the data loading and the exporters stay identical
# ---------------------------------------------------------------------------
def load_coloring3_module():
    """Import coloring3.py from this script's folder."""
    path = os.path.join(SCRIPT_DIR, "coloring3.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "coloring3.py was not found next to aco_coloring.py.\n"
            f"Expected it at: {path}\n"
            "This script reuses coloring3.py for reading the input file and "
            "for writing the reports, so the two must sit in the same folder.")
    spec = importlib.util.spec_from_file_location("_coloring3_for_aco", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------
def build_edge_arrays(g):
    """Return the conflict graph as two parallel index arrays (u < v)."""
    idx = {c: i for i, c in enumerate(g.courses)}
    eu, ev = [], []
    for a, nbrs in g.graph.items():
        ia = idx.get(a)
        if ia is None:
            continue
        for b in nbrs:
            ib = idx.get(b)
            if ib is not None and ia < ib:
                eu.append(ia)
                ev.append(ib)
    return np.asarray(eu, dtype=np.int32), np.asarray(ev, dtype=np.int32)


def build_adjacency(n, eu, ev):
    """Adjacency lists as sorted int32 arrays."""
    buckets = [[] for _ in range(n)]
    for a, b in zip(eu.tolist(), ev.tolist()):
        buckets[a].append(b)
        buckets[b].append(a)
    return [np.asarray(sorted(b), dtype=np.int32) for b in buckets]


def clique_lower_bound(n, adj):
    """Greedy clique search: any clique of size c forces at least c days."""
    if n == 0:
        return 0
    sets = [set(a.tolist()) for a in adj]
    order = sorted(range(n), key=lambda v: -len(sets[v]))
    best = 1
    for seed in order[:min(n, 30)]:
        clique = {seed}
        cand = set(sets[seed])
        while cand:
            nxt = max(cand, key=lambda v: len(sets[v] & cand))
            clique.add(nxt)
            cand &= sets[nxt]
        best = max(best, len(clique))
    return best


# ---------------------------------------------------------------------------
# ACO construction
# ---------------------------------------------------------------------------
def roulette(weights, total, rng):
    """Pick an index proportional to weights. Clamped, so it never fails."""
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1          # floating-point guard


def build_ant(n, adj_l, k, tau, alpha, beta, rng, deg=None, noise=0.0):
    """One ant builds a complete colouring, guided by pheromone + heuristic.

    Vertex order (see the ROOT-CAUSE NOTE in the module docstring):
      deg is None  -> a fresh uniformly random permutation (the original rule)
      deg given    -> hubs first: vertices sorted by  deg * (1 + noise * U(0,1)),
                      so high-degree courses are coloured while colours are
                      still plentiful, and the noise keeps every ant different.
    The colour of each vertex is still drawn by pheromone x heuristic roulette
    in both cases; no greedy / DSATUR colouring is used as a seed.
    """
    col = [-1] * n
    if deg is None:
        order = rng.permutation(n).tolist()
    else:
        order = np.argsort(-(deg * (1.0 + noise * rng.random(n)))).tolist()
    for v in order:
        # eta[c] = 1 / (1 + how many coloured neighbours of v already use c)
        used = [0] * k
        for u in adj_l[v]:
            cu = col[u]
            if cu >= 0:
                used[cu] += 1
        tv = tau[v]
        w = [0.0] * k
        total = 0.0
        for c in range(k):
            val = (tv[c] ** alpha) * ((1.0 / (1.0 + used[c])) ** beta)
            w[c] = val
            total += val
        col[v] = (roulette(w, total, rng) if total > 0.0
                  else int(rng.integers(0, k)))
    return col


def count_conflicts(col, eu_l, ev_l):
    return sum(1 for a, b in zip(eu_l, ev_l) if col[a] == col[b])


def aco_phase(n, adj_l, eu_l, ev_l, k, rng, n_ants, iters, alpha, beta, rho,
              record=None, record_k=None, deadline=None, deg=None, noise=0.0):
    """Run the colony for one k. Returns (best_conflicts, best_colouring)."""
    tau = np.ones((n, k), dtype=np.float64)
    tau_min, tau_max = 0.01, 10.0
    best, best_col = BIG, None

    for it in range(1, iters + 1):
        it_best, it_best_col = BIG, None
        for _ in range(n_ants):
            col = build_ant(n, adj_l, k, tau, alpha, beta, rng, deg, noise)
            c = count_conflicts(col, eu_l, ev_l)
            if c < it_best:
                it_best, it_best_col = c, col
            if c < best:
                best, best_col = c, col[:]
            if best == 0:
                break

        tau *= (1.0 - rho)
        # Elitist deposit: only the iteration best and the global best.
        for col, c in ((it_best_col, it_best), (best_col, best)):
            if col is None:
                continue
            dep = 1.0 / (1.0 + c)
            for v in range(n):
                tau[v, col[v]] += dep
        np.clip(tau, tau_min, tau_max, out=tau)

        if record is not None:
            record.append((record_k, "ACO", it, it_best, best))
        if best == 0:
            break
        if deadline is not None and time.time() > deadline:
            break

    return best, best_col


# ---------------------------------------------------------------------------
# TabuCol -- the local search the MATLAB was reaching for
# ---------------------------------------------------------------------------
def tabucol(n, adj_l, k, col, rng, max_iter, record=None, record_k=None,
            record_stride=1, deadline=None):
    """Proper tabu search on the (vertex, colour) neighbourhood.

    gamma[v][c] = number of neighbours of v currently coloured c, so the exact
    delta of recolouring v to c is gamma[v][c] - gamma[v][col[v]].
    """
    col = col[:]
    gamma = [[0] * k for _ in range(n)]
    for v in range(n):
        for u in adj_l[v]:
            gamma[v][col[u]] += 1
    cur = sum(gamma[v][col[v]] for v in range(n)) // 2
    best, best_col = cur, col[:]

    # tabu_until[v][c] = first iteration at which v may return to colour c
    tabu_until = [[0] * k for _ in range(n)]

    for it in range(1, max_iter + 1):
        if cur == 0:
            break
        confl = [v for v in range(n) if gamma[v][col[v]] > 0]
        if not confl:
            break

        bd, cands = BIG, []
        for v in confl:
            c1 = col[v]
            gv = gamma[v]
            base = gv[c1]
            tv = tabu_until[v]
            for c2 in range(k):
                if c2 == c1:
                    continue
                d = gv[c2] - base
                # allowed unless tabu, but aspiration overrides tabu when the
                # move would beat the best colouring seen so far
                if tv[c2] >= it and not (cur + d < best):
                    continue
                if d < bd:
                    bd, cands = d, [(v, c2)]
                elif d == bd:
                    cands.append((v, c2))

        if not cands:
            break

        v, c2 = cands[int(rng.integers(0, len(cands)))]
        c1 = col[v]
        col[v] = c2
        for u in adj_l[v]:
            gamma[u][c1] -= 1
            gamma[u][c2] += 1
        cur += bd

        # forbid going back to the colour we just left
        tabu_until[v][c1] = it + 10 + int(0.6 * len(confl))

        if cur < best:
            best, best_col = cur, col[:]

        if record is not None and it % record_stride == 0:
            record.append((record_k, "TABU", it, cur, best))
        if deadline is not None and (it & 511) == 0 and time.time() > deadline:
            break

    if record is not None:
        record.append((record_k, "TABU", it, cur, best))
    return best, best_col


# ---------------------------------------------------------------------------
# Upward search over the number of days
# ---------------------------------------------------------------------------
def search(n, adj_l, eu_l, ev_l, total_edges, lb, rng, args, record,
           verbose=True):
    """Find the fewest days for which ACO + tabu reaches zero conflicts.

    Returns (k, colouring, attempts, timed_out).  k / colouring are None when
    no conflict-free schedule was found; timed_out says whether the wall-clock
    budget (--time-limit) was what stopped the search.
    """
    deg = (None if args.order == "random"
           else np.fromiter((len(a) for a in adj_l), dtype=np.float64, count=n))
    noise = args.order_noise
    timed_out = False
    tabu_iters = (args.tabu_iters if args.tabu_iters > 0
                  else max(20000, 200 * n))
    stride = max(1, tabu_iters // 400)
    deadline = (time.time() + args.time_limit) if args.time_limit > 0 else None
    top = args.max_k if args.max_k else n
    attempts = []

    for k in range(max(1, lb), max(1, top) + 1):
        if k <= 1:
            # every course on one day: conflicts == every edge
            best, best_col = total_edges, [0] * n
            if total_edges == 0:
                attempts.append({"k": k, "best_conflicts": 0,
                                 "aco_conflicts": 0, "runs": 0})
                return k, best_col, attempts, False
            attempts.append({"k": k, "best_conflicts": int(best),
                             "aco_conflicts": int(best), "runs": 0})
            if verbose:
                print(f"  k={k:>3}  {best} conflict(s) left  (trivial bound)")
            continue

        k_best, k_best_col, k_aco = BIG, None, BIG
        for run in range(1, max(1, args.restarts) + 1):
            a_conf, a_col = aco_phase(
                n, adj_l, eu_l, ev_l, k, rng, max(1, args.ants),
                max(1, args.aco_iters), args.alpha, args.beta, args.rho,
                record=record, record_k=k, deadline=deadline,
                deg=deg, noise=noise)
            k_aco = min(k_aco, a_conf)

            if a_col is None:
                continue
            if args.no_tabu or a_conf == 0:
                t_conf, t_col = a_conf, a_col
            else:
                t_conf, t_col = tabucol(
                    n, adj_l, k, a_col, rng, tabu_iters, record=record,
                    record_k=k, record_stride=stride, deadline=deadline)

            if t_conf < k_best:
                k_best, k_best_col = t_conf, t_col

            if verbose:
                tail = ("SOLVED" if t_conf == 0
                        else f"{t_conf} conflict(s) left")
                print(f"  k={k:>3}  run {run}/{max(1, args.restarts)}: "
                      f"ACO {a_conf} -> tabu {tail}", flush=True)

            if t_conf == 0:
                break
            if deadline is not None and time.time() > deadline:
                timed_out = True
                break

        attempts.append({"k": k, "best_conflicts": int(k_best),
                         "aco_conflicts": int(k_aco),
                         "runs": run})          # runs actually executed

        if k_best == 0:
            return k, k_best_col, attempts, timed_out

        if deadline is not None and time.time() > deadline:
            timed_out = True
            if verbose:
                print("  time limit reached", flush=True)
            break

    return None, None, attempts, timed_out


# ---------------------------------------------------------------------------
# Faithful replay of the original MATLAB, bugs included
# ---------------------------------------------------------------------------
def faithful_matlab(n, eu_l, ev_l, rng, record, verbose=True, deadline=None):
    """Replay ant_colony1.m as written: uniform ants, unused pheromone, k from
    1, full cost recomputation, and the mismatched tabu attribute.

    Returns (k_reported, colouring, full_cost_evaluations, attempts).
    """
    evals = 0
    attempts = []
    # NOTE: not reset per k -- that is defect 10, preserved on purpose.
    best_conf_found = BIG
    best_col_found = None
    reported_k = None

    for k in range(1, n + 1):
        if verbose:
            print(f"  [faithful] testing k = {k}", flush=True)

        # ---- ACO exactly as coded: p is overwritten with a uniform vector --
        tau = np.ones((n, max(1, k)), dtype=np.float64)
        aco_best, aco_col = BIG, None
        for it in range(1, 101):
            cols, confs = [], []
            for _ in range(50):
                col = [int(rng.integers(0, k)) for _ in range(n)]
                c = count_conflicts(col, eu_l, ev_l)
                evals += 1
                cols.append(col)
                confs.append(c)
                if c < aco_best:
                    aco_best, aco_col = c, col[:]
            tau *= 0.9                      # rho = 0.1, and then never read
            for col, c in zip(cols, confs):
                dep = 1.0 / (1.0 + c)
                for v in range(n):
                    tau[v, col[v]] += dep
            record.append((k, "ACO-faithful", it, aco_best, aco_best))
            if aco_best == 0:
                break
            if deadline is not None and time.time() > deadline:
                break

        # ---- tabu_search_coloring exactly as coded -----------------------
        cur = aco_col[:] if aco_col is not None else [0] * n
        cur_c = count_conflicts(cur, eu_l, ev_l)
        evals += 1
        t_best, t_best_col = cur_c, cur[:]
        tabu = [(-1, -1)] * 15

        for it in range(1, 101):
            record.append((k, "TABU-faithful", it, cur_c, t_best))
            bd, bmv = BIG, None
            for v in range(n):
                cv = cur[v]
                for nc in range(k):
                    if nc == cv:
                        continue
                    tmp = cur[:]
                    tmp[v] = nc
                    tc = count_conflicts(tmp, eu_l, ev_l)
                    evals += 1
                    d = tc - cur_c
                    # the bug: compares the stored colour against cv, the
                    # colour v has NOW, not against nc, the move's target
                    is_tabu = any(t[0] == v and t[1] == cv for t in tabu)
                    if (not is_tabu or tc < t_best) and d < bd:
                        bd, bmv = d, (v, nc)
            if bmv is None:
                break
            v, nc = bmv
            old = cur[v]
            cur[v] = nc
            cur_c += bd
            tabu.pop(0)
            tabu.append((v, old))
            if cur_c < t_best:
                t_best, t_best_col = cur_c, cur[:]
            if t_best == 0:
                break
            if deadline is not None and time.time() > deadline:
                break

        attempts.append({"k": k, "aco_conflicts": int(aco_best),
                         "best_conflicts": int(t_best)})

        if t_best < best_conf_found:
            best_conf_found = t_best
            best_col_found = t_best_col
            if best_conf_found == 0:
                reported_k = k

        if best_conf_found == 0:
            break
        if deadline is not None and time.time() > deadline:
            if verbose:
                print("  [faithful] time limit reached", flush=True)
            break

    return reported_k, best_col_found, evals, attempts


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def compact_colors(raw, courses):
    """Map a 0-based colour array to contiguous 1..m day numbers."""
    seen = {}
    out = {}
    for i, course in enumerate(courses):
        c = int(raw[i])
        if c not in seen:
            seen[c] = len(seen) + 1
        out[course] = seen[c]
    return out


def verify_no_conflicts(colors, g):
    """Independently re-check the schedule against the conflict graph."""
    bad = 0
    for a, nbrs in g.graph.items():
        for b in nbrs:
            if a < b and colors.get(a) == colors.get(b):
                bad += 1
    return bad


def resolve_input_filename(explicit):
    """Same resolution order as coloring3.py."""
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        raise FileNotFoundError(f"Specified file not found: {explicit}")
    here = os.path.join(SCRIPT_DIR, "input.csv")
    if os.path.isfile(here):
        return here
    found = glob.glob(os.path.join(SCRIPT_DIR, "*.csv"))
    if len(found) == 1:
        return found[0]
    raise FileNotFoundError(
        "Could not decide which CSV to read. Pass one explicitly.\n"
        f"Candidates in {SCRIPT_DIR}: {[os.path.basename(f) for f in found]}")


def write_json(outdir, summary):
    with open(os.path.join(outdir, "aco_run_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)


def write_convergence(outdir, record):
    if not record:
        return
    with open(os.path.join(outdir, "aco_convergence.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["K", "Phase", "Step", "Current_Conflicts",
                    "Best_Conflicts"])
        w.writerows(record)


# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Ant Colony Optimisation + Tabu Search exam scheduling.")
    ap.add_argument("input", nargs="?", default=None,
                    help="input CSV (same format as coloring3.py)")
    ap.add_argument("--outdir", default=None,
                    help="folder for the output files (default: script folder)")
    ap.add_argument("--ants", type=int, default=20,
                    help="ants per ACO iteration")
    ap.add_argument("--aco-iters", dest="aco_iters", type=int, default=30,
                    help="ACO iterations per attempt")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="pheromone weight")
    ap.add_argument("--beta", type=float, default=3.0,
                    help="heuristic weight")
    ap.add_argument("--rho", type=float, default=0.1,
                    help="pheromone evaporation rate")
    ap.add_argument("--tabu-iters", dest="tabu_iters", type=int, default=0,
                    help="TabuCol iterations (0 = auto)")
    ap.add_argument("--no-tabu", dest="no_tabu", action="store_true",
                    help="pure ACO, no tabu phase (ablation)")
    ap.add_argument("--restarts", type=int, default=6,
                    help="independent ACO+tabu attempts per k")
    ap.add_argument("--order", choices=["degree", "random"], default="degree",
                    help="vertex order each ant walks: 'degree' = hubs first "
                         "with random noise (default), 'random' = fresh "
                         "uniform permutation (the previous behaviour)")
    ap.add_argument("--order-noise", dest="order_noise", type=float,
                    default=0.3,
                    help="noise for --order degree: key = degree * "
                         "(1 + noise * U(0,1)) (default 0.3)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-k", dest="max_k", type=int, default=None,
                    help="stop searching above this many days")
    ap.add_argument("--time-limit", dest="time_limit", type=float,
                    default=0.0, help="wall-clock budget in seconds")
    ap.add_argument("--tag", default=None,
                    help="label written into the JSON summary")
    ap.add_argument("--faithful", action="store_true",
                    help="replay the original MATLAB, bugs and all")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-k progress lines")
    return ap.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet
    g3 = load_coloring3_module()

    filename = resolve_input_filename(args.input)
    print(f"\nUsing input file: {filename}")

    g = g3.Graph()
    g.load_data_from_csv(filename)

    eu, ev = build_edge_arrays(g)
    n = g.v
    adj = build_adjacency(n, eu, ev)
    adj_l = [a.tolist() for a in adj]
    eu_l, ev_l = eu.tolist(), ev.tolist()
    lb = clique_lower_bound(n, adj)

    print(f"Number of courses (vertices): {n}")
    print(f"Number of conflict edges: {len(eu_l)}")
    print(f"Clique lower bound on days: {lb}")

    tag = args.tag or ("ACO-faithful-matlab" if args.faithful
                       else ("ACO" if args.no_tabu else "ACO-Tabu"))
    print(f"\nAlgorithm: {tag}")
    if args.faithful or args.order == "random":
        print("Initial solution: RANDOM ant construction "
              "(no greedy / DSATUR seeding)")
    else:
        print("Initial solution: pheromone-guided ant construction, vertices "
              f"visited hubs-first with noise {args.order_noise} "
              "(no greedy / DSATUR colouring is used as a seed)")

    rng = np.random.default_rng(args.seed)
    record = []
    evals = None
    t0 = time.time()

    if args.faithful:
        print("Mode: FAITHFUL replay of ant_colony1.m  "
              "(uniform ants, unused pheromone, k from 1,")
        print("      full cost recomputation, mismatched tabu attribute)")
        if n > 80:
            print(f"WARNING: n = {n}. Faithful mode rebuilds the cost for "
                  "every candidate move;")
            print("         this will be extremely slow. "
                  "Consider --time-limit, or a smaller instance.")
        deadline = ((t0 + args.time_limit) if args.time_limit > 0 else None)
        k_used, raw, evals, attempts = faithful_matlab(
            n, eu_l, ev_l, rng, record, verbose=verbose, deadline=deadline)
        timed_out = deadline is not None and time.time() > deadline
    else:
        tabu_iters = (args.tabu_iters if args.tabu_iters > 0
                      else max(20000, 200 * n))
        print(f"Mode: corrected ACO{'' if args.no_tabu else ' + TabuCol'}  "
              f"ants={max(1, args.ants)}  aco_iters={max(1, args.aco_iters)}  "
              f"alpha={args.alpha}  beta={args.beta}  rho={args.rho}")
        if not args.no_tabu:
            print(f"      tabu_iters={tabu_iters}  "
                  f"restarts/k={max(1, args.restarts)}")
        print(f"Searching upward from the lower bound ({max(1, lb)} days)")
        k_used, raw, attempts, timed_out = search(
            n, adj_l, eu_l, ev_l, len(eu_l), lb, rng, args, record,
            verbose=verbose)

    elapsed = time.time() - t0
    outdir = args.outdir or SCRIPT_DIR
    os.makedirs(outdir, exist_ok=True)

    # Fields shared by the success and the failure summary.
    summary = {
        "algorithm": tag,
        "mode": ("aco_faithful_matlab" if args.faithful
                 else ("aco" if args.no_tabu else "aco_tabu")),
        "input_file": os.path.abspath(filename),
        "n_courses": n,
        "n_students": len(g.students),
        "n_edges": len(eu_l),
        "clique_lower_bound": lb,
        "computation_time_sec": round(elapsed, 4),
        "full_cost_evaluations": evals,
        "k_attempts": attempts,
        "parameters": {
            "ants": max(1, args.ants),
            "aco_iters": max(1, args.aco_iters),
            "alpha": args.alpha,
            "beta": args.beta,
            "rho": args.rho,
            "tabu_iters": (args.tabu_iters if args.tabu_iters > 0
                           else max(20000, 200 * n)),
            "no_tabu": bool(args.no_tabu),
            "restarts": max(1, args.restarts),
            "order": args.order,
            "order_noise": args.order_noise,
            "seed": args.seed,
            "max_k": args.max_k,
            "time_limit": args.time_limit,
            "faithful": bool(args.faithful),
        },
    }

    # ---- FAILURE: no conflict-free schedule.  Never write a conflicted
    #      schedule -- but never fail SILENTLY either: the summary and the
    #      convergence curves are exactly what shows where the search gave up.
    if raw is None or k_used is None:
        fewest = min((a["best_conflicts"] for a in attempts), default=None)
        status = "timeout" if timed_out else "no_solution_in_range"
        print("\nERROR: the search never reached a conflict-free schedule.")
        if timed_out:
            print(f"Reason: the --time-limit budget ({args.time_limit:g} s) "
                  "ran out.")
        if fewest is not None:
            print(f"Fewest conflicts seen: {fewest}")
        summary.update({
            "success": False,
            "status": status,
            "colors_used": None,
            "conflicts": None,
            "days_searched_for": None,
            "optimal_proven": False,
            "fewest_conflicts_seen": fewest,
            "message": ("No conflict-free schedule found "
                        f"({'time limit reached' if timed_out else 'k range exhausted'}"
                        f"; fewest conflicts seen: {fewest})."),
        })
        write_json(outdir, summary)
        write_convergence(outdir, record)
        print("No schedule was written, because a schedule with conflicts is "
              "not a usable answer.")
        print("Failure summary saved: aco_run_summary.json"
              + (", aco_convergence.csv" if record else ""))
        print("Try a larger --time-limit, or raise --aco-iters / --tabu-iters "
              "/ --restarts, or --max-k if the day cap was the limit.")
        sys.exit(2)          # 2 = search failed; 1 stays "crash / bad input"

    colors = compact_colors(raw, g.courses)
    residual = verify_no_conflicts(colors, g)
    used = len(set(colors.values()))
    counts = g.compute_color_student_counts(colors)

    print(f"\nComputation time: {elapsed:.4f} seconds")
    print(f"Days searched for: {k_used}")
    print(f"Number of colors used: {used}")
    if residual == 0:
        print("No conflicts detected! The coloring is valid.")
    else:
        print(f"ERROR: {residual} conflict(s) survived verification.")
        print("Refusing to write a schedule that is not conflict-free.")
        summary.update({"success": False, "status": "verification_failed",
                        "colors_used": used, "conflicts": residual,
                        "days_searched_for": k_used, "optimal_proven": False})
        write_json(outdir, summary)
        write_convergence(outdir, record)
        sys.exit(1)

    if used == lb:
        print(f"This matches the clique lower bound ({lb}) "
              "-- provably optimal.")
    else:
        print(f"Gap to the lower bound: {used - lb} day(s) "
              "(may or may not be closable).")

    print("\nColor -> number of unique students:")
    for c in sorted(counts):
        print(f"Color {c}: {counts[c]} students")

    # ---- JSON summary FIRST: the pipeline reads it, so it must exist even if
    #      one of the Excel / text exporters below raises.
    summary.update({
        "success": True,
        "status": "solved",
        "colors_used": used,
        "days_searched_for": k_used,
        "conflicts": residual,
        "optimal_proven": bool(used == lb),
    })
    write_json(outdir, summary)
    print("Run summary saved successfully: aco_run_summary.json")
    write_convergence(outdir, record)
    if record:
        print("Convergence curve saved successfully: aco_convergence.csv")

    # ---- schedule / report exports (coloring3.py's own writers) ----
    import io, contextlib
    _suppress = (contextlib.redirect_stdout(io.StringIO()) if args.quiet
                 else contextlib.nullcontext())
    cwd = os.getcwd()
    export_error = None
    try:
        os.chdir(outdir)
        with _suppress:
            g.export_schedule_table(colors, counts)
            g.export_text_report(colors, counts, elapsed, conflicts=residual,
                                 input_filename=filename)
            g.export_detailed_excel(colors, counts)
    except Exception as err:
        export_error = err
    finally:
        os.chdir(cwd)

    summary["exports_ok"] = export_error is None
    if export_error is not None:
        summary["export_error"] = f"{type(export_error).__name__}: {export_error}"
        write_json(outdir, summary)
        print(f"WARNING: schedule/report export failed ({export_error}); "
              "the JSON summary and convergence file were already saved.")
        if not args.quiet:
            raise export_error
    else:
        write_json(outdir, summary)


if __name__ == "__main__":
    main()
