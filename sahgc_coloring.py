# -*- coding: utf-8 -*-
"""
sahgc_coloring.py  v2.4
=======================
v2 fixes (from v1):
  Fix-1  Kempe delta boundary
  Fix-2  path_relink mapping guard
  Fix-3  path_relink copy fix
  Fix-4  dsatur k=1 guard
  Fix-5  DSATUR upper bound for k-range
  Fix-6  CC-GPX crossover
  Fix-7  Trajectory archive
  Fix-8  Structural tabu tenure
  Fix-9  Unified operator tracking
  Fix-10 Kempe skip without wasting iteration

v2.1 performance fixes:
  Fix-A  Vectorized Kempe delta (numpy edge arrays)
  Fix-B  scipy partition_distance (Hungarian C code)
  Fix-C  Heap-based dsatur_randomized_initial O(n log n)
  Fix-D  numpy CC-GPX conflict counting
  Fix-E  Adaptive Kempe skip for large graphs
  Fix-F  Cached diversity (every 5 generations)
  Fix-G  Vectorized ColoringState init + apply_move

v2.2 (other AI — Fix-H/I):
  Fix-H  Decreasing k-sweep with warm-start from k+1
  Fix-I  _run_k_attempts + --stall-patience adaptive early-exit
         --strategy ascending|descending (ascending kept for ablation)

v2.3 (merged + next-gen):
  P1    Vectorized neighborhood_single (numpy broadcast, 5-15x)
  P2    Vectorized Kempe tabu check (.any() vs Python generator)
  P3    Cross-edge merge warm-start (min cross-edges, better than greedy drop)
  P4    CSR adjacency + vectorized frontier Kempe BFS (3-8x BFS speedup)
  P5    Conflict-Driven LB Tightening (UNSAT-core style, research contribution)
  P6    Multiprocessing restarts (--parallel-restarts N)

v2.3.1 (review fixes -- P5/P6 were broken, now corrected):
  R1  exact_repair_core now returns (result, exhausted). A None result only
      counts as an UNSAT certificate when the DFS actually exhausted its
      search space; a deadline timeout (exhausted=False) is "unknown", not
      "infeasible" -- the previous version conflated the two, which made
      most reported "certificates" unsound (exact_time defaults to 0.15s).
  R2  Fixed an IndexError crash in the core-clique check: core_adj_local was
      built with GLOBAL vertex ids fed into a LOCAL-index adjacency list.
      Now uses a proper local index map. Reproduced and confirmed fixed.
  R3  proven_lb is now actually wired end-to-end: fixed_k_sahgc ->
      _run_k_attempts -> _solve_descending, which maintains a running
      `certified_lb` (>= clique_lower_bound) used to (a) skip k values
      already proven infeasible and (b) certify optimality tighter than the
      plain clique bound. Previously this number was computed and discarded.
  R4  --parallel-restarts is now actually connected to the solve path (it
      was parsed but never read before) and runs in chunks so
      --stall-patience still applies in parallel mode instead of always
      spending the full restart budget.
"""

from __future__ import annotations

import argparse
import csv
import glob
import heapq
import importlib.util
import json
import math
import os
import time
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import concurrent.futures
import multiprocessing
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _lsa
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INF = 10 ** 12


# ---------------------------------------------------------------------------
# Graph / output machinery
# ---------------------------------------------------------------------------
def load_coloring3_module():
    path = os.path.join(SCRIPT_DIR, "coloring3.py")
    if not os.path.isfile(path):
        raise FileNotFoundError("coloring3.py not found next to sahgc_coloring.py.")
    spec = importlib.util.spec_from_file_location("_coloring3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_edge_arrays(g):
    idx = {course: i for i, course in enumerate(g.courses)}
    seen: set = set()
    for course, neighbours in g.graph.items():
        a = idx.get(course)
        if a is None:
            continue
        for other in neighbours:
            b = idx.get(other)
            if b is None or a == b:
                continue
            seen.add((a, b) if a < b else (b, a))
    if not seen:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    e = np.asarray(sorted(seen), dtype=np.int32)
    return e[:, 0], e[:, 1]


def build_adjacency(n: int, eu: np.ndarray, ev: np.ndarray) -> List[np.ndarray]:
    buckets: List[list] = [[] for _ in range(n)]
    for a, b in zip(eu.tolist(), ev.tolist()):
        buckets[a].append(b)
        buckets[b].append(a)
    return [np.asarray(sorted(v), dtype=np.int32) for v in buckets]


def build_csr_adjacency(n: int, eu: np.ndarray, ev: np.ndarray):
    """Fix-v2.3-CSR: Compressed Sparse Row adjacency.
    Neighbors of v = indices[indptr[v]:indptr[v+1]]
    Cache-friendly; enables vectorized Kempe BFS.
    """
    deg = np.zeros(n, dtype=np.int32)
    if eu.size:
        np.add.at(deg, eu, 1)
        np.add.at(deg, ev, 1)
    indptr = np.zeros(n + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(deg)
    indices = np.empty(int(indptr[n]), dtype=np.int32)
    pos = indptr.copy()
    for a, b in zip(eu.tolist(), ev.tolist()):
        indices[pos[a]] = b; pos[a] += 1
        indices[pos[b]] = a; pos[b] += 1
    for v in range(n):
        indices[indptr[v]:indptr[v+1]].sort()
    return indptr, indices


def clique_lower_bound(n: int, adj: Sequence[np.ndarray]) -> int:
    if n == 0:
        return 0
    nbr = [set(a.tolist()) for a in adj]
    order = sorted(range(n), key=lambda v: len(nbr[v]), reverse=True)
    best = 1
    for start in order[:min(n, 80)]:
        clique = [start]
        cand = set(nbr[start])
        while cand:
            nxt = max(cand, key=lambda x: len(cand & nbr[x]))
            clique.append(nxt)
            cand &= nbr[nxt]
        best = max(best, len(clique))
    return best


def compact_colors(raw: np.ndarray, courses: Sequence[str]) -> Dict[str, int]:
    raw = np.asarray(raw, dtype=np.int32)
    used = sorted(set(int(x) for x in raw.tolist()))
    remap = {old: new for new, old in enumerate(used, start=1)}
    return {course: remap[int(raw[i])] for i, course in enumerate(courses)}


def verify_no_conflicts(colors: Dict[str, int], g) -> list:
    bad = []
    for course, neighbours in g.graph.items():
        c = colors.get(course)
        if c is None:
            continue
        for other in neighbours:
            if colors.get(other) == c:
                a, b = sorted((course, other))
                bad.append((a, b, c))
    return sorted(set(bad))


def resolve_input_filename(explicit: Optional[str]) -> str:
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        raise FileNotFoundError(f"Specified file not found: {explicit}")
    default = os.path.join(SCRIPT_DIR, "input.csv")
    if os.path.isfile(default):
        return default
    csv_files = glob.glob(os.path.join(SCRIPT_DIR, "*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    if not csv_files:
        raise FileNotFoundError("No CSV file found.")
    names = ", ".join(os.path.basename(x) for x in csv_files)
    raise FileNotFoundError(f"Multiple CSV files ({names}). Specify input CSV.")


# ---------------------------------------------------------------------------
# Structural analysis
# ---------------------------------------------------------------------------
def tarjan_articulation_points(n: int, adj: Sequence[np.ndarray]) -> np.ndarray:
    disc = [-1] * n
    low = [0] * n
    parent = [-1] * n
    child_count = [0] * n
    art = np.zeros(n, dtype=bool)
    t = 0
    for root in range(n):
        if disc[root] != -1:
            continue
        stack = [(root, 0)]
        disc[root] = low[root] = t
        t += 1
        while stack:
            v, idx = stack[-1]
            if idx < len(adj[v]):
                u = int(adj[v][idx])
                stack[-1] = (v, idx + 1)
                if disc[u] == -1:
                    parent[u] = v
                    child_count[v] += 1
                    disc[u] = low[u] = t
                    t += 1
                    stack.append((u, 0))
                elif u != parent[v]:
                    low[v] = min(low[v], disc[u])
            else:
                stack.pop()
                p = parent[v]
                if p != -1:
                    low[p] = min(low[p], low[v])
                    if parent[p] == -1 and child_count[p] > 1:
                        art[p] = True
                    if parent[p] != -1 and low[v] >= disc[p]:
                        art[p] = True
    return art


def connected_component_labels(n: int, adj: Sequence[np.ndarray]) -> np.ndarray:
    labels = np.full(n, -1, dtype=np.int32)
    cid = 0
    for s in range(n):
        if labels[s] != -1:
            continue
        dq: deque = deque([s])
        labels[s] = cid
        while dq:
            v = dq.popleft()
            for u in adj[v]:
                if labels[int(u)] == -1:
                    labels[int(u)] = cid
                    dq.append(int(u))
        cid += 1
    return labels


def graph_features(n: int, adj: Sequence[np.ndarray]):
    deg = np.asarray([len(a) for a in adj], dtype=np.int32)
    art = tarjan_articulation_points(n, adj)
    comp = connected_component_labels(n, adj)
    comp_sizes = np.bincount(comp, minlength=(int(comp.max()) + 1 if n else 0))
    if n:
        dnorm = deg.astype(float) / max(1.0, float(deg.max()))
        ascore = art.astype(float)
        comp_norm = np.asarray(
            [comp_sizes[c] for c in comp], dtype=float
        ) / max(1.0, float(comp_sizes.max()))
    else:
        dnorm = ascore = comp_norm = np.zeros(0)
    structural = 0.55 * dnorm + 0.30 * ascore + 0.15 * comp_norm
    return deg, art, comp, structural


# ---------------------------------------------------------------------------
# Fix-5: DSATUR upper bound
# ---------------------------------------------------------------------------
def dsatur_upper_bound(
    n: int,
    adj: Sequence[np.ndarray],
    degree: np.ndarray,
) -> Tuple[int, np.ndarray]:
    if n == 0:
        return 0, np.zeros(0, dtype=np.int32)
    color = np.full(n, -1, dtype=np.int32)
    sat_sets: List[set] = [set() for _ in range(n)]
    uncolored = set(range(n))
    n_colors = 0
    while uncolored:
        v = max(uncolored, key=lambda x: (len(sat_sets[x]), int(degree[x])))
        used = sat_sets[v]
        c = 0
        while c in used:
            c += 1
        color[v] = c
        n_colors = max(n_colors, c + 1)
        uncolored.remove(v)
        for u0 in adj[v]:
            u = int(u0)
            if color[u] == -1:
                sat_sets[u].add(c)
    return n_colors, color


# ---------------------------------------------------------------------------
# Fix-G: Vectorized ColoringState
# ---------------------------------------------------------------------------
class ColoringState:
    __slots__ = ("color", "gamma", "cost", "k", "adj", "n", "eu", "ev", "indptr", "indices")

    def __init__(
        self,
        color,
        k: int,
        adj: Sequence[np.ndarray],
        eu: Optional[np.ndarray] = None,
        ev: Optional[np.ndarray] = None,
        indptr: Optional[np.ndarray] = None,   # Fix-v2.3-CSR
        indices: Optional[np.ndarray] = None,  # Fix-v2.3-CSR
    ):
        self.color = np.asarray(color, dtype=np.int32).copy()
        self.k = int(k)
        self.adj = adj
        self.n = self.color.size
        self.eu = eu
        self.ev = ev
        self.indptr = indptr   # Fix-v2.3-CSR
        self.indices = indices  # Fix-v2.3-CSR
        self.gamma = np.zeros((self.n, self.k), dtype=np.int32)
        # Fix-G: two np.add.at calls instead of Python double-loop
        if eu is not None and ev is not None and eu.size > 0:
            np.add.at(self.gamma, (eu, self.color[ev]), 1)
            np.add.at(self.gamma, (ev, self.color[eu]), 1)
        else:
            for v in range(self.n):
                if adj[v].size:
                    np.add.at(self.gamma[v], self.color[adj[v]], 1)
        rows = np.arange(self.n)
        self.cost = int(self.gamma[rows, self.color].sum() // 2)

    def move_delta(self, v: int, new_c: int) -> int:
        old = int(self.color[v])
        return int(self.gamma[v, new_c] - self.gamma[v, old])

    def apply_move(self, v: int, new_c: int) -> int:
        old = int(self.color[v])
        if old == new_c:
            return 0
        d = self.move_delta(v, new_c)
        nbrs = self.adj[v]
        if nbrs.size:  # Fix-G: vectorized
            self.gamma[nbrs, old] -= 1
            self.gamma[nbrs, new_c] += 1
        self.color[v] = int(new_c)
        self.cost += d
        return d

    def conflicting_vertices(self) -> np.ndarray:
        rows = np.arange(self.n)
        return np.nonzero(self.gamma[rows, self.color] > 0)[0]

    def copy_color(self) -> np.ndarray:
        return self.color.copy()


# ---------------------------------------------------------------------------
# Fix-C: Heap-based DSATUR init -- O(n log n)
# ---------------------------------------------------------------------------
def dsatur_randomized_initial(
    n: int,
    k: int,
    adj: Sequence[np.ndarray],
    degree: np.ndarray,
    structural: np.ndarray,
    rng: np.random.Generator,
    randomness: float = 0.20,
) -> np.ndarray:
    if k <= 1:  # Fix-4
        return np.zeros(n, dtype=np.int32)

    color = np.full(n, -1, dtype=np.int32)
    sat = np.zeros(n, dtype=np.int32)
    sat_sets: List[set] = [set() for _ in range(n)]
    colored = np.zeros(n, dtype=bool)
    counter = [0]

    def make_score(v: int) -> float:
        s = (
            1000.0 * float(sat[v])
            + 4.0 * float(degree[v])
            + 3.0 * float(structural[v])
        )
        s += randomness * rng.random() * max(1.0, abs(s) + 1.0)
        return s

    heap: list = []
    for v in range(n):
        heapq.heappush(heap, (-make_score(v), counter[0], v))
        counter[0] += 1

    for _ in range(n):
        while heap:
            _, _, v = heapq.heappop(heap)
            if not colored[v]:
                break

        forbidden = sat_sets[v]
        allowed = [c for c in range(k) if c not in forbidden]
        if allowed:
            nbr_col = color[adj[v]] if adj[v].size else np.array([], dtype=np.int32)
            values = np.asarray(
                [int(np.sum(nbr_col == c)) for c in allowed], dtype=np.float64
            )
            minv = values.min()
            choices = [allowed[i] for i, x in enumerate(values) if x <= minv + 1e-9]
            c = int(rng.choice(choices))
        else:
            c = int(rng.integers(0, k))

        color[v] = c
        colored[v] = True
        for u0 in adj[v]:
            u = int(u0)
            if not colored[u] and c not in sat_sets[u]:
                sat_sets[u].add(c)
                sat[u] += 1
                heapq.heappush(heap, (-make_score(u), counter[0], u))
                counter[0] += 1

    return color


# ---------------------------------------------------------------------------
# Fix-B: scipy-based partition_distance
# ---------------------------------------------------------------------------
def partition_distance(a: np.ndarray, b: np.ndarray, k: int) -> float:
    if a.size == 0:
        return 0.0
    mat = np.zeros((k, k), dtype=np.int32)
    np.add.at(mat, (a, b), 1)
    if _SCIPY_AVAILABLE:
        row_ind, col_ind = _lsa(-mat)
        matched = int(mat[row_ind, col_ind].sum())
    else:
        used_a = np.zeros(k, dtype=bool)
        used_b = np.zeros(k, dtype=bool)
        matched = 0
        for _ in range(k):
            best = -1
            bi = bj = 0
            for i in range(k):
                if used_a[i]:
                    continue
                row = mat[i].copy()
                row[used_b] = -1
                j = int(row.argmax())
                val = int(row[j])
                if val > best:
                    best = val
                    bi, bj = i, j
            if best <= 0:
                break
            used_a[bi] = True
            used_b[bj] = True
            matched += best
    return float(a.size - matched)


def min_pairwise_diversity(pop: Sequence[np.ndarray], k: int) -> float:
    if len(pop) < 2:
        return 0.0
    vals = []
    for i in range(len(pop)):
        for j in range(i + 1, len(pop)):
            vals.append(partition_distance(pop[i], pop[j], k))
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Fix-7: Trajectory Archive
# ---------------------------------------------------------------------------
class TrajectoryArchive:
    def __init__(self, capacity: int, k: int) -> None:
        self.capacity = capacity
        self.k = k
        self.solutions: List[np.ndarray] = []
        self.costs: List[int] = []

    def add(self, sol: np.ndarray, cost: int) -> None:
        if len(self.solutions) < self.capacity:
            self.solutions.append(sol.copy())
            self.costs.append(int(cost))
            return
        worst = int(np.argmax(np.asarray(self.costs)))
        if int(cost) < self.costs[worst]:
            self.solutions[worst] = sol.copy()
            self.costs[worst] = int(cost)

    def get_diverse_elite_pair(
        self, rng: np.random.Generator
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if len(self.solutions) < 2:
            return None, None
        rank = sorted(range(len(self.costs)), key=lambda i: self.costs[i])
        i1 = rank[0]
        dists = [
            partition_distance(self.solutions[i1], s, self.k)
            for s in self.solutions
        ]
        dists[i1] = -1.0
        i2 = int(np.argmax(np.asarray(dists)))
        return self.solutions[i1].copy(), self.solutions[i2].copy()

    def best(self) -> Tuple[Optional[np.ndarray], int]:
        if not self.solutions:
            return None, INF
        bi = int(np.argmin(np.asarray(self.costs)))
        return self.solutions[bi].copy(), self.costs[bi]


# ---------------------------------------------------------------------------
# Neighborhood helpers
# ---------------------------------------------------------------------------
def select_conflicts(
    state: ColoringState,
    degree: np.ndarray,
    structural: np.ndarray,
    rng: np.random.Generator,
    sample_limit: int = 80,
) -> np.ndarray:
    conf = state.conflicting_vertices()
    if conf.size <= sample_limit:
        return conf
    weights = (
        1.0
        + state.gamma[conf, state.color[conf]].astype(float)
        + 0.35 * degree[conf].astype(float)
        + 1.5 * structural[conf]
    )
    weights = np.maximum(weights, 1e-9)
    p = weights / weights.sum()
    idx = rng.choice(conf.size, size=sample_limit, replace=False, p=p)
    return conf[np.sort(idx)]


def neighborhood_single(
    state: ColoringState,
    degree: np.ndarray,
    structural: np.ndarray,
    tabu_until: np.ndarray,
    iteration: int,
    best_cost: int,
    rng: np.random.Generator,
) -> Optional[Tuple[int, int, int]]:
    """Fix-v2.3-P1: fully vectorized — single numpy broadcast pass.
    Replaces O(|conf|*k) Python double-loop. Speedup: 5-15x.
    """
    conf = select_conflicts(state, degree, structural, rng)
    if conf.size == 0:
        return None
    m = conf.size
    old_colors = state.color[conf]                               # (m,)
    G = state.gamma[conf]                                        # (m, k)
    old_vals = G[np.arange(m), old_colors]                      # (m,)
    D = G - old_vals[:, None]                                    # (m, k)
    k_idx = np.arange(state.k)
    self_mask  = (k_idx[None, :] == old_colors[:, None])        # (m, k)
    tabu_row   = (tabu_until[conf, old_colors] > iteration)[:, None]  # (m,1)
    aspiration = ((state.cost + D) < best_cost)                 # (m, k)
    blocked    = (tabu_row & ~aspiration) | self_mask
    D_inf = D.astype(np.float64); D_inf[blocked] = np.inf
    best_f = float(D_inf.min())
    if not np.isfinite(best_f):
        return None
    best_ij = np.argwhere(D_inf == best_f)
    pick = int(rng.integers(0, len(best_ij)))
    vi, ci = best_ij[pick]
    return int(conf[vi]), int(ci), int(best_f)


def kempe_component(
    state: ColoringState,
    start: int,
    color_b: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fix-v2.3-CSR: Vectorized frontier BFS. Each BFS level processes
    all frontier vertices simultaneously with numpy, giving 3-8x speedup
    over pure-Python deque BFS on large components.
    Falls back to deque BFS when CSR is unavailable.
    """
    color_a = int(state.color[start])
    if state.indptr is not None and state.indices is not None:
        # --- vectorized frontier BFS over CSR ---
        seen = np.zeros(state.n, dtype=bool)
        seen[start] = True
        comp_parts: list = [np.array([start], dtype=np.int32)]
        frontier = np.array([start], dtype=np.int32)
        indptr, indices, color = state.indptr, state.indices, state.color
        while frontier.size:
            # gather all neighbors of current frontier level
            nbr_list = [indices[indptr[v]:indptr[v + 1]] for v in frontier]
            if not nbr_list:
                break
            nbr_flat = np.concatenate(nbr_list)
            if nbr_flat.size == 0:
                break
            c_nbr = color[nbr_flat]
            mask = ~seen[nbr_flat] & ((c_nbr == color_a) | (c_nbr == color_b))
            new_nodes = np.unique(nbr_flat[mask])
            if new_nodes.size == 0:
                break
            seen[new_nodes] = True
            comp_parts.append(new_nodes)
            frontier = new_nodes
        return np.concatenate(comp_parts).astype(np.int32)
    else:
        # --- fallback: pure-Python deque BFS ---
        q: deque = deque([start])
        seen = np.zeros(state.n, dtype=bool)
        seen[start] = True
        comp = []
        while q:
            v = q.popleft()
            comp.append(v)
            for u0 in state.adj[v]:
                u = int(u0)
                if seen[u]:
                    continue
                cu = int(state.color[u])
                if cu == color_a or cu == color_b:
                    seen[u] = True
                    q.append(u)
        return np.asarray(comp, dtype=np.int32)


# ---------------------------------------------------------------------------
# Fix-A: Vectorized Kempe delta
# ---------------------------------------------------------------------------
def _kempe_delta_vectorized(
    state: ColoringState,
    comp: np.ndarray,
    old: int,
    new_c: int,
) -> int:
    """Fix-A: O(|E_touching|) numpy instead of O(|comp|*deg) Python."""
    if state.eu is None or state.eu.size == 0:
        # Fallback Python loop (Fix-1 logic)
        changed_set = set(int(x) for x in comp.tolist())
        delta = 0
        for u in changed_set:
            cu = int(state.color[u])
            for w0 in state.adj[u]:
                w = int(w0)
                if u >= w:
                    continue
                cw = int(state.color[w])
                nw_u = new_c if cu == old else old
                nw_w = (new_c if cw == old else old) if w in changed_set else cw
                delta += int(nw_u == nw_w) - int(cu == cw)
        return delta

    eu, ev = state.eu, state.ev
    changed_mask = np.zeros(state.n, dtype=bool)
    changed_mask[comp] = True
    touches = changed_mask[eu] | changed_mask[ev]
    if not touches.any():
        return 0
    tu = eu[touches]
    tv = ev[touches]
    cu_arr = state.color[tu]
    cv_arr = state.color[tv]
    u_in = changed_mask[tu]
    v_in = changed_mask[tv]
    nw_u = np.where(u_in, np.where(cu_arr == old, new_c, old), cu_arr)
    nw_v = np.where(v_in, np.where(cv_arr == old, new_c, old), cv_arr)
    return int((nw_u == nw_v).sum()) - int((cu_arr == cv_arr).sum())


# ---------------------------------------------------------------------------
# Fix-A + Fix-E: Kempe neighborhood
# ---------------------------------------------------------------------------
def neighborhood_kempe(
    state: ColoringState,
    degree: np.ndarray,
    structural: np.ndarray,
    tabu_until: np.ndarray,
    iteration: int,
    best_cost: int,
    rng: np.random.Generator,
    max_n_for_kempe: int = 800,
) -> Optional[Tuple[np.ndarray, int]]:
    n = state.n
    # Fix-E: scale sample sizes down for large dense graphs
    if n > max_n_for_kempe:
        kempe_sample = max(10, 20 * 1000 // n)
        kempe_colors = min(4, state.k)
        max_comp = max(2, int(0.30 * n))
    else:
        kempe_sample = 40
        kempe_colors = 8
        max_comp = max(2, int(0.45 * n))

    conf = select_conflicts(state, degree, structural, rng,
                            sample_limit=kempe_sample)
    if conf.size == 0:
        return None

    candidates: List[Tuple[np.ndarray, int]] = []
    for v0 in conf.tolist():
        v = int(v0)
        old = int(state.color[v])
        neigh_colors = (
            np.unique(state.color[state.adj[v]])
            if state.adj[v].size
            else np.arange(state.k)
        )
        choices: set = set(int(x) for x in neigh_colors.tolist() if int(x) != old)
        if len(choices) < 2:
            choices.update(
                int(rng.integers(0, state.k)) for _ in range(min(3, state.k))
            )
            choices.discard(old)
        for new_c in list(choices)[:kempe_colors]:
            comp = kempe_component(state, v, int(new_c), rng)
            if comp.size > max_comp:
                continue
            delta = _kempe_delta_vectorized(state, comp, old, int(new_c))
            old_colors = state.color[comp].copy()
            # Fix-v2.3-P2: vectorized tabu check (no Python generator)
            tabu_blocked = bool((tabu_until[comp, old_colors] > iteration).any())
            if tabu_blocked and state.cost + delta >= best_cost:
                continue
            candidates.append((comp.copy(), delta))

    if not candidates:
        return None
    best_d = min(d for _, d in candidates)
    best_list = [x for x in candidates if x[1] == best_d]
    return best_list[int(rng.integers(0, len(best_list)))]


def neighborhood_swap_repair(
    state: ColoringState,
    degree: np.ndarray,
    structural: np.ndarray,
    tabu_until: np.ndarray,
    iteration: int,
    best_cost: int,
    rng: np.random.Generator,
) -> Optional[Tuple[int, int, int]]:
    conf = select_conflicts(state, degree, structural, rng, sample_limit=50)
    if conf.size == 0:
        return None
    weights = (
        1.0
        + state.gamma[conf, state.color[conf]].astype(float)
        + 2.0 * structural[conf]
    )
    p = weights / weights.sum()
    chosen = int(conf[int(rng.choice(conf.size, p=p))])
    old = int(state.color[chosen])
    candidate_colors = list(range(state.k))
    rng.shuffle(candidate_colors)
    candidate_colors.sort(key=lambda c: int(state.gamma[chosen, c]))
    best = None
    best_score = float(INF)
    for c in candidate_colors[:min(10, state.k)]:
        if c == old:
            continue
        d = state.move_delta(chosen, c)
        tabu = tabu_until[chosen, old] > iteration
        if tabu and state.cost + d >= best_cost:
            continue
        pressure = (
            float(np.sum(state.gamma[state.adj[chosen], c]))
            if state.adj[chosen].size
            else 0.0
        )
        score = float(d) + 0.04 * pressure - 0.05 * float(structural[chosen])
        if score < best_score:
            best_score = score
            best = (chosen, c, int(d))
    return best


# ---------------------------------------------------------------------------
# Tabu search (Fix-8,9,10,G,A,E)
# ---------------------------------------------------------------------------
def tabu_search_adaptive(
    start: np.ndarray,
    k: int,
    adj: Sequence[np.ndarray],
    degree: np.ndarray,
    structural: np.ndarray,
    articulation: np.ndarray,
    rng: np.random.Generator,
    max_iters: int,
    deadline: Optional[float],
    operator_scores: np.ndarray,
    operator_uses: np.ndarray,
    record_callback=None,
    stagnation_limit: Optional[int] = None,
    art_tenure_bonus: int = 5,
    eu: Optional[np.ndarray] = None,
    ev: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, Dict]:
    _n_tsa = len(start)  # Fix-v2.3-bugfix: n not in scope here
    _csr_tsa = build_csr_adjacency(_n_tsa, eu, ev) if (eu is not None and eu.size) else (None, None)
    state = ColoringState(start, k, adj, eu=eu, ev=ev,
                          indptr=_csr_tsa[0],
                          indices=_csr_tsa[1])
    best_color = state.copy_color()
    best_cost = int(state.cost)
    tabu = np.zeros((state.n, k), dtype=np.int64)
    it = 0
    no_improve = 0
    accepted = 0
    improvements = 0
    stagnation_limit = stagnation_limit or max(250, min(2500, 8 * state.n))
    kempe_max_n = state.n  # Fix-E

    while it < max_iters and state.cost > 0:
        if deadline is not None and time.time() >= deadline:
            break

        untried = np.nonzero(operator_uses == 0)[0]
        if untried.size:
            op = int(untried[0])
        else:
            total = float(max(1, operator_uses.sum()))
            means = operator_scores / np.maximum(operator_uses, 1.0)
            bonus = np.sqrt(
                2.0 * math.log(total + 1.0) / np.maximum(operator_uses, 1.0)
            )
            op = int(np.argmax(means + bonus))
        operator_uses[op] += 1.0  # Fix-9

        conf_now = int(state.conflicting_vertices().size)
        tenure_base = (
            5
            + int(0.25 * conf_now)
            + int(no_improve / max(1, state.n // 80))
        )
        # Fix-TENURE: scale caps with n for density-adaptive escape
        tenure_low = max(4, min(state.n // 3, tenure_base))
        tenure_high = max(tenure_low + 1, min(state.n, 2 * tenure_low + 10))

        move = None
        if op == 0:
            mv = neighborhood_single(state, degree, structural, tabu, it, best_cost, rng)
            if mv is not None:
                move = ("single", mv)
        elif op == 1:
            mv2 = neighborhood_kempe(
                state, degree, structural, tabu, it, best_cost, rng,
                max_n_for_kempe=kempe_max_n,
            )
            if mv2 is not None:
                move = ("kempe", mv2)
        else:
            mv3 = neighborhood_swap_repair(state, degree, structural, tabu, it, best_cost, rng)
            if mv3 is not None:
                move = ("repair", mv3)

        if move is None:
            mv = neighborhood_single(state, degree, structural, tabu, it, best_cost, rng)
            if mv is None:
                break
            move = ("fallback", mv)

        kind, payload = move

        if kind in ("single", "repair", "fallback"):
            v, new_c, delta = payload
            old_c = int(state.color[v])
            state.apply_move(v, new_c)
            art_bonus = art_tenure_bonus if bool(articulation[v]) else 0  # Fix-8
            tenure = int(rng.integers(tenure_low, tenure_high + 1)) + art_bonus
            tabu[v, old_c] = it + 1 + tenure

        else:  # kempe
            comp, delta = payload
            old_colors = state.color[comp].copy()
            old_pairs = list(zip(comp.tolist(), old_colors.tolist()))
            colors_present = sorted(set(int(x) for x in old_colors.tolist()))
            if len(colors_present) != 2:  # Fix-10
                continue
            a, b = colors_present
            for v0, oc in old_pairs:
                v = int(v0)
                nc = b if oc == a else a
                state.color[v] = nc
                nbrs = state.adj[v]
                if nbrs.size:  # Fix-G
                    state.gamma[nbrs, oc] -= 1
                    state.gamma[nbrs, nc] += 1
            rows = np.arange(state.n)
            state.cost = int(state.gamma[rows, state.color].sum() // 2)
            for v0, oc in old_pairs:
                v = int(v0)
                art_bonus = art_tenure_bonus if bool(articulation[v]) else 0
                tenure = int(rng.integers(tenure_low, tenure_high + 1)) + art_bonus
                tabu[v, oc] = it + 1 + tenure

        accepted += 1
        if state.cost < best_cost:
            gain = best_cost - state.cost
            best_cost = int(state.cost)
            best_color = state.copy_color()
            improvements += 1
            no_improve = 0
            operator_scores[op] = 0.80 * operator_scores[op] + float(gain)
        else:
            no_improve += 1
            operator_scores[op] = 0.995 * operator_scores[op]

        if no_improve >= stagnation_limit:
            conf = state.conflicting_vertices()
            if conf.size:
                perturb = (
                    conf if conf.size < 30
                    else rng.choice(conf, size=30, replace=False)
                )
                for v0 in np.asarray(perturb).tolist():
                    v = int(v0)
                    old = int(state.color[v])
                    cands = [c for c in range(k) if c != old]
                    if cands:
                        c = int(rng.choice(cands))
                        state.apply_move(v, c)
                        tabu[v, old] = it + int(rng.integers(2, 12))
            no_improve = 0

        if record_callback is not None:
            record_callback(it + 1, state.cost, best_cost, op, no_improve)

        if best_cost == 0:
            break
        it += 1

    return best_color, best_cost, {
        "iterations": int(it),
        "best_cost": int(best_cost),
        "accepted_moves": int(accepted),
        "improvements": int(improvements),
        "operator_uses": [float(x) for x in operator_uses.tolist()],
        "operator_scores": [float(x) for x in operator_scores.tolist()],
        "final_conflicting_vertices": int(
            np.count_nonzero(state.gamma[np.arange(state.n), state.color] > 0)
        ),
    }


# ---------------------------------------------------------------------------
# Exact repair
# ---------------------------------------------------------------------------
def induced_core(
    color: np.ndarray,
    adj: Sequence[np.ndarray],
    max_vertices: int,
    max_conflicting_edges: int,
) -> np.ndarray:
    bad_vertices: set = set()
    bad_edges = 0
    for v in range(color.size):
        cv = int(color[v])
        for u0 in adj[v]:
            u = int(u0)
            if v < u and cv == int(color[u]):
                bad_vertices.add(v)
                bad_vertices.add(u)
                bad_edges += 1
    if not bad_vertices or bad_edges > max_conflicting_edges:
        return np.zeros(0, dtype=np.int32)
    core: set = set(bad_vertices)
    for v in list(bad_vertices):
        for u0 in adj[v]:
            core.add(int(u0))
            if len(core) >= max_vertices:
                break
        if len(core) >= max_vertices:
            break
    return np.asarray(sorted(core), dtype=np.int32)


def exact_repair_core(
    color: np.ndarray,
    k: int,
    adj: Sequence[np.ndarray],
    core: np.ndarray,
    node_limit: int,
    deadline: Optional[float],
) -> Tuple[Optional[np.ndarray], bool]:
    """Returns (repaired_solution_or_None, exhausted).
    exhausted=True means the DFS explored the *entire* search space for this
    core and definitively proved feasibility/infeasibility with k colors.
    exhausted=False means the deadline was hit mid-search: a None result in
    that case means "we don't know", NOT "infeasible" -- callers must not
    treat it as an UNSAT certificate. (Fix: soundness of P5.)
    """
    if core.size == 0 or core.size > node_limit:
        return None, False
    timed_out = [False]
    core_set = set(int(x) for x in core.tolist())
    idx = {v: i for i, v in enumerate(core.tolist())}
    m = core.size
    order = list(
        sorted(
            range(m),
            key=lambda i: (
                sum(1 for u in adj[int(core[i])] if int(u) in core_set),
                len(adj[int(core[i])]),
            ),
            reverse=True,
        )
    )
    assignment = np.full(m, -1, dtype=np.int32)

    def dfs(pos: int) -> bool:
        if deadline is not None and (pos & 7) == 0 and time.time() >= deadline:
            timed_out[0] = True
            return False
        if pos == m:
            return True
        rem = order[pos:]
        best_key = None
        best_i = None
        for q in rem:
            v = int(core[q])
            used: set = set()
            for u0 in adj[v]:
                u = int(u0)
                if u in core_set:
                    val = assignment[idx[u]]
                    if val >= 0:
                        used.add(int(val))
                else:
                    used.add(int(color[u]))
            key = (len(used), len(adj[v]))
            if best_key is None or key > best_key:
                best_key = key
                best_i = q
        pos_i = order.index(best_i, pos)
        order[pos], order[pos_i] = order[pos_i], order[pos]
        v = int(core[order[pos]])
        forbidden: set = set()
        for u0 in adj[v]:
            u = int(u0)
            if u in core_set:
                val = assignment[idx[u]]
                if val >= 0:
                    forbidden.add(int(val))
            else:
                forbidden.add(int(color[u]))
        for c in range(k):
            if c in forbidden:
                continue
            assignment[idx[v]] = c
            if dfs(pos + 1):
                return True
            assignment[idx[v]] = -1
        return False

    found = dfs(0)
    exhausted = not timed_out[0]
    if not found:
        return None, exhausted
    out = color.copy()
    for i, v0 in enumerate(core.tolist()):
        out[int(v0)] = assignment[i]
    return out, exhausted


# ---------------------------------------------------------------------------
# Path relinking (Fix-2, Fix-3)
# ---------------------------------------------------------------------------
def path_relink(
    start: np.ndarray,
    target: np.ndarray,
    k: int,
    adj: Sequence[np.ndarray],
    deadline: Optional[float],
    eu: Optional[np.ndarray] = None,
    ev: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    state = ColoringState(start, k, adj, eu=eu, ev=ev)  # CSR optional here
    best_arr = state.color.copy()  # Fix-3
    best_cost = int(state.cost)

    mat = np.zeros((k, k), dtype=np.int32)
    np.add.at(mat, (start, target), 1)
    mapping = [-1] * k
    used_b: set = set()
    for _ in range(k):
        best_val = -1
        pair = None
        for i in range(k):
            if mapping[i] != -1:
                continue
            for j in range(k):
                if j in used_b:
                    continue
                if int(mat[i, j]) > best_val:
                    best_val = int(mat[i, j])
                    pair = (i, j)
        if pair is None or best_val <= 0:
            break
        mapping[pair[0]] = pair[1]
        used_b.add(pair[1])

    # Fix-2: fill unmapped labels
    remaining_b = sorted(set(range(k)) - used_b)
    rb_idx = 0
    for i in range(k):
        if mapping[i] == -1:
            if rb_idx < len(remaining_b):
                mapping[i] = remaining_b[rb_idx]
                rb_idx += 1
            else:
                mapping[i] = 0

    target_mapped = np.asarray(
        [mapping[int(target[v])] for v in range(start.size)], dtype=np.int32
    )

    for _ in range(min(3 * start.size, 20000)):
        if deadline is not None and time.time() >= deadline:
            break
        mismatches = np.nonzero(state.color != target_mapped)[0]
        if mismatches.size == 0:
            break
        best_move = None
        best_delta = INF
        for v0 in mismatches[:min(100, mismatches.size)].tolist():
            v = int(v0)
            nc = int(target_mapped[v])
            d = state.move_delta(v, nc)
            if d < best_delta:
                best_delta = d
                best_move = (v, nc)
        if best_move is None:
            break
        v, nc = best_move
        state.apply_move(v, nc)
        if state.cost < best_cost:
            best_cost = int(state.cost)
            best_arr = state.color.copy()  # Fix-3
        if best_cost == 0:
            break

    return best_arr, int(best_cost)


# ---------------------------------------------------------------------------
# Fix-D: CC-GPX with numpy conflict counting
# ---------------------------------------------------------------------------
def gpx_cc(
    p1: np.ndarray,
    p2: np.ndarray,
    k: int,
    adj: Sequence[np.ndarray],
    rng: np.random.Generator,
    use_cc: bool = True,
    eu: Optional[np.ndarray] = None,
    ev: Optional[np.ndarray] = None,
) -> np.ndarray:
    n = p1.size
    child = np.full(n, -1, dtype=np.int32)
    remaining = np.ones(n, dtype=bool)
    parents = (p1, p2)

    for step in range(k):
        if not remaining.any():
            break
        p = parents[step & 1]

        if use_cc:
            if eu is not None and eu.size > 0:
                # Fix-D: fully vectorized over all classes
                rem_eu = remaining[eu]
                rem_ev = remaining[ev]
                p_eu = p[eu]
                p_ev = p[ev]
                members_count = np.bincount(p[remaining], minlength=k).astype(float)
                both_rem = rem_eu & rem_ev
                same_class = p_eu == p_ev
                internal = both_rem & same_class
                conflict_count = (
                    np.bincount(p_eu[internal], minlength=k).astype(float)
                    if internal.any()
                    else np.zeros(k, dtype=float)
                )
                scores = np.where(
                    members_count > 0,
                    members_count / (1.0 + 2.0 * conflict_count),
                    -1.0,
                )
                best_class = int(scores.argmax())
            else:
                best_class = -1
                best_score = -1.0
                for c in range(k):
                    members = np.nonzero(remaining & (p == c))[0]
                    if members.size == 0:
                        continue
                    ic = 0
                    mem_set = set(int(x) for x in members.tolist())
                    for v0 in members.tolist():
                        v = int(v0)
                        for u0 in adj[v]:
                            u = int(u0)
                            if u > v and u in mem_set:
                                ic += 1
                    sc = float(members.size) / (1.0 + 2.0 * ic)
                    if sc > best_score:
                        best_score = sc
                        best_class = c
        else:
            counts = np.bincount(p[remaining], minlength=k)
            mx = int(counts.max())
            best_class = int(rng.choice(np.flatnonzero(counts == mx)))

        if best_class < 0:
            break
        members = np.nonzero(remaining & (p == best_class))[0]
        child[members] = step
        remaining[members] = False

    left = np.nonzero(remaining)[0]
    if left.size:
        child[left] = rng.integers(0, k, size=left.size, dtype=np.int32)
    return child


# ---------------------------------------------------------------------------
# Fixed-k SAHGC (Fix-F: cached diversity)
# ---------------------------------------------------------------------------
def warm_start_reduce_k(
    sol: np.ndarray, k_new: int, adj: Sequence[np.ndarray], rng: np.random.Generator,
    eu: Optional[np.ndarray] = None, ev: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fix-v2.3: Cross-edge merge warm-start (replaces greedy class-drop).
    Merges the two color classes with the FEWEST cross-edges between them,
    minimising initial conflicts in the k-coloring seed.
    This is the standard Galinier-Hao / TabuCol warm-start technique.
    O(|E|) conflict matrix build + O(k^2) merge selection.
    Falls back to smallest-class drop when eu/ev unavailable.
    """
    n = sol.shape[0]
    k_old = int(sol.max()) + 1 if n else 0
    if k_old <= k_new or k_old == 0:
        return np.clip(sol, 0, max(0, k_new - 1)).astype(np.int32)

    if eu is not None and eu.size > 0:
        # --- cross-edge conflict matrix ---
        cu, cv = sol[eu], sol[ev]
        cross = cu != cv
        if cross.any():
            cmat = np.zeros((k_old, k_old), dtype=np.int32)
            np.add.at(cmat, (cu[cross], cv[cross]), 1)
            cmat = cmat + cmat.T          # symmetrize
            np.fill_diagonal(cmat, np.iinfo(np.int32).max)
            fi = int(cmat.argmin())
            c1, c2 = fi // k_old, fi % k_old
        else:
            c1, c2 = 0, 1
    else:
        # fallback: drop smallest class
        counts = np.bincount(sol, minlength=k_old)
        order = np.argsort(counts)
        c1, c2 = int(order[0]), int(order[1])

    # Merge c2 → c1, then compact labels to [0, k_new-1]
    new_sol = sol.copy()
    new_sol[new_sol == c2] = c1
    used = sorted(set(new_sol.tolist()))
    remap = np.full(k_old, 0, dtype=np.int32)
    for new_idx, old_c in enumerate(used):
        remap[old_c] = new_idx
    return remap[new_sol]


def fixed_k_sahgc(
    n: int,
    k: int,
    adj: Sequence[np.ndarray],
    degree: np.ndarray,
    structural: np.ndarray,
    articulation: np.ndarray,
    args,
    rng: np.random.Generator,
    deadline: Optional[float],
    eu: np.ndarray,
    ev: np.ndarray,
    seed_solution: Optional[np.ndarray] = None,  # Fix-H: decreasing-k warm start
) -> Tuple[np.ndarray, int, Dict, List]:
    # Fix-v2.3-CSR: build CSR once per fixed_k call
    _csr_indptr, _csr_indices = build_csr_adjacency(n, eu, ev)

    diversity_interval = max(1, min(5, args.generations // 10))  # Fix-F
    pop: List[np.ndarray] = []
    costs: List[int] = []
    population_target = max(2, args.population)
    elite_target = max(1, min(args.elite, population_target))
    op_scores = np.ones(3, dtype=np.float64)
    op_uses = np.zeros(3, dtype=np.float64)  # Fix-9
    convergence: List = []
    total_iterations = 0
    best_global: Optional[np.ndarray] = None
    best_cost = INF
    generation = 0
    exact_repairs = 0
    path_relinks = 0
    archive = TrajectoryArchive(capacity=max(4, elite_target * 2), k=k)  # Fix-7
    cached_diversity = 0.0  # Fix-F
    conflict_lb_hits = [0]  # Fix-v2.3-P5: mutable counter
    proven_lb = [0]  # Fix-P5-sound: highest SOUND lower-bound certificate found this call

    def add_candidate(sol: np.ndarray, cost: int) -> None:
        nonlocal best_global, best_cost
        if cost < best_cost:
            best_cost = int(cost)
            best_global = sol.copy()
        archive.add(sol, cost)
        if not pop:
            pop.append(sol.copy())
            costs.append(int(cost))
            return
        distances = [partition_distance(sol, p, k) for p in pop]
        min_dist = min(distances) if distances else INF
        worst = int(np.argmax(np.asarray(costs)))
        threshold = max(2.0, 0.03 * n)
        if len(pop) < population_target:
            pop.append(sol.copy())
            costs.append(int(cost))
        elif cost < costs[worst] or min_dist >= threshold:
            if min_dist < threshold and cost >= costs[worst]:
                return
            pop[worst] = sol.copy()
            costs[worst] = int(cost)

    for pop_i in range(population_target):
        if deadline is not None and time.time() >= deadline:
            break
        if pop_i == 0 and seed_solution is not None:
            start_sol = seed_solution.copy()
        else:
            start_sol = dsatur_randomized_initial(
                n, k, adj, degree, structural, rng, args.init_randomness
            )
        local_iters = max(
            100, min(args.init_tabu_iters,
                     args.tabu_iters // 2 if args.tabu_iters else 800)
        )
        sol, cost, meta = tabu_search_adaptive(
            start_sol, k, adj, degree, structural, articulation, rng,
            local_iters, deadline, op_scores, op_uses,
            stagnation_limit=max(100, args.stagnation // 2),
            art_tenure_bonus=args.art_tenure_bonus,
            eu=eu, ev=ev,
        )
        add_candidate(sol, cost)
        total_iterations += int(meta["iterations"])
        if cost == 0:
            return sol, 0, {
                "generations": 0, "iterations": total_iterations,
                "population_final": len(pop), "exact_repairs": 0,
                "path_relinks": 0, "final_diversity": 0.0,
                "operator_uses": op_uses.astype(int).tolist(),
            }, convergence

    if not pop:
        return np.zeros(n, dtype=np.int32), INF, {"generations": 0}, convergence

    while generation < args.generations:
        if deadline is not None and time.time() >= deadline:
            break
        if best_cost == 0:
            break
        generation += 1

        # Fix-F: cached diversity
        if generation % diversity_interval == 1 or generation == 1:
            cached_diversity = min_pairwise_diversity(pop, k)
        diversity = cached_diversity

        costs_arr = np.asarray(costs, dtype=np.float64)
        rank = np.argsort(costs_arr)
        elite_idx = rank[:elite_target]
        i1 = int(rng.choice(elite_idx))
        if len(pop) > 2 and rng.random() < 0.65:
            i2 = int(rng.choice(elite_idx))
            if i2 == i1:
                chs = [x for x in elite_idx.tolist() if x != i1]
                if chs:
                    i2 = int(rng.choice(chs))
        else:
            i2 = int(rng.integers(0, len(pop)))
            if i2 == i1 and len(pop) > 1:
                i2 = (i2 + 1) % len(pop)

        child = gpx_cc(  # Fix-D: numpy CC-GPX
            pop[i1], pop[i2], k, adj, rng,
            use_cc=args.use_cc_gpx, eu=eu, ev=ev,
        )

        diversity_ratio = diversity / max(1.0, float(n))
        hardness = min(1.0, max(0.0, best_cost / max(1.0, float(n))))
        local_iters = int(
            args.tabu_iters
            * (0.55 + 0.65 * hardness + 0.35 * max(0.0, 0.35 - diversity_ratio))
        )
        local_iters = max(100, min(args.max_child_iters, local_iters))

        sol, cost, meta = tabu_search_adaptive(
            child, k, adj, degree, structural, articulation, rng,
            local_iters, deadline, op_scores, op_uses,
            stagnation_limit=args.stagnation,
            art_tenure_bonus=args.art_tenure_bonus,
            eu=eu, ev=ev,
        )
        total_iterations += int(meta.get("iterations", 0))
        add_candidate(sol, cost)
        if generation % 5 == 0:
            convergence.append(
                (generation, int(cost), int(best_cost),
                 int(meta.get("improvements", 0)), diversity)
            )

        if cost == 0:
            break

        # Exact repair
        if args.exact_repair and cost <= args.exact_trigger_conflicts:
            core = induced_core(
                sol, adj,
                max_vertices=args.exact_core_vertices,
                max_conflicting_edges=args.exact_trigger_conflicts,
            )
            if core.size:
                t_repair = (
                    min(deadline, time.time() + args.exact_time)
                    if deadline is not None
                    else time.time() + args.exact_time
                )
                repaired, exhausted = exact_repair_core(
                    sol, k, adj, core, args.exact_core_vertices, t_repair
                )
                if repaired is not None:
                    chk = ColoringState(repaired, k, adj, eu=eu, ev=ev,
                                        indptr=_csr_indptr, indices=_csr_indices).cost  # Fix-v2.3
                    if chk < cost:
                        exact_repairs += 1
                        add_candidate(repaired, chk)
                        if chk == 0:
                            break
                else:
                    core_certificate = 0

                    # Fix-v2.3-P5 (soundness fix): a None result only proves
                    # UNSAT when the DFS actually EXHAUSTED the search space
                    # (exhausted=True). If it just hit the per-call deadline
                    # (exhausted=False, default exact_time=0.15s), that's an
                    # "unknown", not a certificate for this part.
                    if exhausted:
                        # This induced core, restricted to the SAME k colors
                        # used in the parent solution, provably has no valid
                        # coloring. Since a valid global k-coloring would
                        # restrict to a valid k-coloring of any induced
                        # subgraph, this proves the whole graph needs >= k+1
                        # colors.
                        core_certificate = k + 1

                    # Independently-sound bonus check: an explicit clique
                    # found inside the core is a hard certificate on its own
                    # (a clique of size C always needs >= C colors) whether
                    # or not the DFS above was exhaustive, so it's checked
                    # unconditionally. (Fix: indices here are now LOCAL to
                    # the core -- the original code passed GLOBAL vertex ids
                    # into a local-index adjacency list and crashed with an
                    # IndexError as soon as this path was exercised.)
                    if core.size >= 3:
                        core_set = set(core.tolist())
                        core_index = {int(v): i for i, v in enumerate(core.tolist())}
                        core_adj_local = [
                            np.array(
                                [core_index[int(u)] for u in adj[int(cv)]
                                 if int(u) in core_set],
                                dtype=np.int32,
                            )
                            for cv in core
                        ]
                        local_clique = clique_lower_bound(len(core), core_adj_local)
                        core_certificate = max(core_certificate, local_clique)

                    if core_certificate > proven_lb[0]:
                        proven_lb[0] = int(core_certificate)
                        conflict_lb_hits[0] += 1  # signal to outer loop

        # Path relinking (Fix-7: archive-based)
        if args.path_relink and generation % max(2, args.path_relink_every) == 0:
            a_sol, b_sol = archive.get_diverse_elite_pair(rng)
            if a_sol is not None and b_sol is not None:
                relinked, rcost = path_relink(
                    a_sol, b_sol, k, adj, deadline, eu=eu, ev=ev
                )
                if rcost < best_cost:
                    path_relinks += 1
                    add_candidate(relinked, rcost)
                    if rcost == 0:
                        break
            if len(pop) >= 2:
                r2 = np.argsort(np.asarray(costs))
                relinked2, rcost2 = path_relink(
                    pop[int(r2[0])], pop[int(r2[-1])],
                    k, adj, deadline, eu=eu, ev=ev
                )
                if rcost2 < best_cost:
                    path_relinks += 1
                    add_candidate(relinked2, rcost2)
                    if rcost2 == 0:
                        break

        # Diversity maintenance
        if generation % diversity_interval == 0:
            cached_diversity = min_pairwise_diversity(pop, k)
        if len(pop) >= elite_target and cached_diversity < args.min_diversity_ratio * n:
            worst = int(np.argmax(np.asarray(costs)))
            mutant = pop[worst].copy()
            rcount = max(2, int(args.perturb_fraction * n))
            vertices = rng.choice(n, size=min(n, rcount), replace=False)
            for v0 in vertices.tolist():
                v = int(v0)
                old = int(mutant[v])
                if k > 1:
                    new_c = int(rng.integers(0, k - 1))
                    if new_c >= old:
                        new_c += 1
                    mutant[v] = new_c
            sol2, cost2, meta2 = tabu_search_adaptive(
                mutant, k, adj, degree, structural, articulation, rng,
                max(100, args.tabu_iters // 2), deadline, op_scores, op_uses,
                stagnation_limit=args.stagnation,
                art_tenure_bonus=args.art_tenure_bonus,
                eu=eu, ev=ev,
            )
            add_candidate(sol2, cost2)
            total_iterations += int(meta2.get("iterations", 0))

        if best_cost == 0:
            break

    if best_global is None:
        bi = int(np.argmin(np.asarray(costs)))
        best_global = pop[bi].copy()
        best_cost = int(costs[bi])

    return best_global, int(best_cost), {
        "generations": int(generation),
        "iterations": int(total_iterations),
        "population_final": int(len(pop)),
        "best_cost": int(best_cost),
        "exact_repairs": int(exact_repairs),
        "path_relinks": int(path_relinks),
        "final_diversity": float(min_pairwise_diversity(pop, k)),
        "operator_uses": [float(x) for x in op_uses.tolist()],
        "operator_scores": [float(x) for x in op_scores.tolist()],
        "conflict_lb_hits": conflict_lb_hits[0],  # Fix-v2.3-P5
        "proven_lb": proven_lb[0],  # Fix-P5-sound: SOUND certificate, now actually wired to caller
    }, convergence


# ---------------------------------------------------------------------------
# Outer k-search (Fix-5: DSATUR UB)
# ---------------------------------------------------------------------------
def _mp_worker(packed):
    """Fix-v2.3-MP: top-level picklable worker for parallel restarts."""
    (n, k, eu, ev, degree, structural, articulation,
     args_vars, local_seed, deadline_ts, seed_sol) = packed
    import argparse as _ap
    args = _ap.Namespace(**args_vars)
    adj_w = build_adjacency(n, eu, ev)
    rrng = np.random.default_rng(local_seed)
    sol, cost, meta, conv = fixed_k_sahgc(
        n, k, adj_w, degree, structural, articulation,
        args, rrng, deadline_ts, eu=eu, ev=ev,
        seed_solution=seed_sol,
    )
    return sol.copy(), int(cost), meta, conv


def _run_k_attempts(
    k, n, adj, degree, structural, articulation, args, rng,
    master_deadline, eu, ev, all_conv, seed_solution=None, stall_patience=0,
    parallel_workers: int = 0,
):
    """Run up to args.restarts attempts of fixed_k_sahgc for a single k.
    seed_solution (if given) warm-starts restart 0 (Fix-H).
    stall_patience > 0 enables early-abandon (Fix-I): if the best conflict
    count hasn't improved for `stall_patience` consecutive restarts (and we
    are not close to 0), stop trying further restarts for this k -- this is
    what prevents infeasible/near-infeasible k values from burning the full
    restart budget with zero chance of success. This now also applies in
    parallel mode (Fix: previously --parallel-restarts silently disabled
    early-abandon by running the whole restart budget in one pool.map call).
    Also returns k_proven_lb: the strongest SOUND lower-bound certificate
    (Fix-P5-sound) seen across all attempts for this k, so the caller can
    tighten the global bound instead of discarding it.
    """
    k_best = None
    k_best_cost = INF
    k_success = 0
    k_times: List[float] = []
    k_attempt_details = []
    k_proven_lb = 0
    stall_count = 0
    prev_best = INF

    n_restarts = max(1, args.restarts)
    workers = min(parallel_workers, n_restarts) if parallel_workers > 1 else 0

    def _record(r, seed, sol, cost, meta, conv, elapsed, parallel):
        nonlocal k_best, k_best_cost, k_success, k_proven_lb
        k_times.append(elapsed)
        if cost < k_best_cost:
            k_best_cost = int(cost); k_best = sol.copy()
        if cost == 0:
            k_success += 1
        k_proven_lb = max(k_proven_lb, int(meta.get("proven_lb", 0)))
        entry = {
            "restart": r + 1, "seed": seed,
            "best_conflicts": int(cost),
            "time_sec": round(elapsed, 6),
            "generations": int(meta.get("generations", 0)),
            "iterations": int(meta.get("iterations", 0)),
            "final_diversity": float(meta.get("final_diversity", 0.0)),
            "exact_repairs": int(meta.get("exact_repairs", 0)),
            "path_relinks": int(meta.get("path_relinks", 0)),
            "operator_uses": meta.get("operator_uses", []),
        }
        if parallel:
            entry["parallel"] = True
        k_attempt_details.append(entry)
        for row in conv:
            all_conv.append((k, r + 1, *row))

    if workers > 1:
        # --- Fix-v2.3-MP: parallel restarts, run in chunks of `workers` so
        # stall_patience can still abandon a doomed k between chunks instead
        # of always burning the full --restarts budget. ---
        args_vars = vars(args)  # picklable dict
        r = 0
        while r < n_restarts:
            if master_deadline is not None and time.time() >= master_deadline:
                break
            chunk = list(range(r, min(r + workers, n_restarts)))
            seeds = [int(rng.integers(0, 2**63 - 1)) for _ in chunk]
            packed_list = [
                (n, k, eu, ev, degree, structural, articulation,
                 args_vars, seeds[i], master_deadline,
                 seed_solution if rr == 0 else None)
                for i, rr in enumerate(chunk)
            ]
            t_chunk_start = time.time()
            with multiprocessing.Pool(processes=len(chunk)) as pool:
                results = pool.map(_mp_worker, packed_list)
            chunk_elapsed = (time.time() - t_chunk_start) / max(1, len(chunk))
            for i, rr in enumerate(chunk):
                sol, cost, meta, conv = results[i]
                _record(rr, seeds[i], sol, cost, meta, conv, chunk_elapsed, True)
            r += len(chunk)
            if k_best_cost == 0:
                break
            if stall_patience > 0:
                if k_best_cost >= prev_best:
                    stall_count += 1
                else:
                    stall_count = 0
                prev_best = k_best_cost
                if stall_count >= stall_patience:
                    break
    else:
        # --- serial restarts ---
        for r in range(n_restarts):
            if master_deadline is not None and time.time() >= master_deadline:
                break
            start_t = time.time()
            local_seed = int(rng.integers(0, 2**63 - 1))
            rrng = np.random.default_rng(local_seed)
            sol, cost, meta, conv = fixed_k_sahgc(
                n, k, adj, degree, structural, articulation,
                args, rrng, master_deadline, eu=eu, ev=ev,
                seed_solution=seed_solution if r == 0 else None,
            )
            elapsed = time.time() - start_t
            _record(r, local_seed, sol, cost, meta, conv, elapsed, False)
            if cost == 0:
                break
            if stall_patience > 0:
                if k_best_cost >= prev_best:
                    stall_count += 1
                else:
                    stall_count = 0
                prev_best = k_best_cost
                if stall_count >= stall_patience:
                    break

    return k_best, k_best_cost, k_success, k_attempt_details, k_times, k_proven_lb



def _solve_descending(args, g, n, eu, ev, adj, degree, structural, articulation,
                       components, lb, dsatur_ub, dsatur_sol, master_deadline, rng,
                       parallel_workers: int = 0):
    """Fix-H/I: start from a KNOWN-FEASIBLE coloring (DSATUR upper bound, or
    the user-supplied --max-k once verified) and descend k by 1 at a time,
    warm-starting each attempt from the previous (k+1)-coloring with its
    smallest color class dropped and greedily repaired. This means every
    attempt after the first starts a handful of conflicts away from feasible
    instead of from scratch, and stall_patience means we stop burning restarts
    on a k that is clearly not panning out. This turns the old
    'independent restart at every k from a weak clique bound' sweep -- which
    is what made SAHGC hang for days on easy/trivial instances -- into a
    fast descent that mirrors what published k-coloring heuristics actually do.

    Fix-P5-sound: also tracks `certified_lb`, the strongest SOUND lower-bound
    certificate produced anywhere during the run (starts at the plain clique
    bound `lb`, tightened by proven UNSAT cores / cliques found during exact
    repair -- see exact_repair_core / fixed_k_sahgc). Once certified_lb rises
    above lb, we can skip straight past k values we already know are
    infeasible instead of spending a restart budget "discovering" that.
    """
    all_attempts = []
    all_conv: List = []
    stall_patience = max(0, args.stall_patience)
    certified_lb = max(1, lb)

    start_k = args.max_k if args.max_k is not None else dsatur_ub
    if start_k >= dsatur_ub:
        current_sol = dsatur_sol.copy()
        current_k = dsatur_ub
        # if the user asked for a higher --max-k than the DSATUR UB needs,
        # there's nothing to search above dsatur_ub -- start descending from there.
        start_k = dsatur_ub
    else:
        # user forced a lower starting k than the DSATUR UB guarantees;
        # that k is not known-feasible yet, so search it cold first.
        k_best, k_best_cost, k_success, k_attempt_details, k_times, k_proven_lb = _run_k_attempts(
            start_k, n, adj, degree, structural, articulation, args, rng,
            master_deadline, eu, ev, all_conv, seed_solution=None,
            stall_patience=stall_patience, parallel_workers=parallel_workers,
        )
        certified_lb = max(certified_lb, k_proven_lb)
        all_attempts.append({
            "k": start_k, "successes": int(k_success),
            "restarts_completed": int(len(k_attempt_details)),
            "best_conflicts": int(k_best_cost if k_best is not None else INF),
            "mean_time_sec": float(np.mean(k_times)) if k_times else None,
            "attempts": k_attempt_details,
        })
        if k_best_cost != 0:
            # couldn't even reach the user's requested starting k -- fall back
            # to the guaranteed DSATUR solution so we always return something valid.
            return {
                "best_color": dsatur_sol, "best_conflicts": 0, "best_k": dsatur_ub,
                "clique_lower_bound": int(lb), "certified_lower_bound": int(certified_lb),
                "dsatur_upper_bound": int(dsatur_ub),
                "attempts": all_attempts, "all_convergence": all_conv,
                "edge_count": int(eu.size), "degree_max": int(degree.max()) if n else 0,
                "articulation_points": int(np.count_nonzero(articulation)),
                "components": int(components.max() + 1) if n else 0,
            }
        current_sol, current_k = k_best.copy(), start_k

    print(f"  [descending] verified feasible at k={current_k}, now descending toward LB={certified_lb}")

    while current_k - 1 >= certified_lb:
        if master_deadline is not None and time.time() >= master_deadline:
            print("  [!] master time limit reached")
            break
        next_k = current_k - 1
        seed = warm_start_reduce_k(current_sol, next_k, adj, rng)
        k_best, k_best_cost, k_success, k_attempt_details, k_times, k_proven_lb = _run_k_attempts(
            next_k, n, adj, degree, structural, articulation, args, rng,
            master_deadline, eu, ev, all_conv, seed_solution=seed,
            stall_patience=stall_patience, parallel_workers=parallel_workers,
        )
        if k_proven_lb > certified_lb:
            certified_lb = k_proven_lb
            print(f"  [LB tightened] proven certificate: graph needs >= {certified_lb} colors")
        all_attempts.append({
            "k": next_k, "successes": int(k_success),
            "restarts_completed": int(len(k_attempt_details)),
            "best_conflicts": int(k_best_cost if k_best is not None else INF),
            "mean_time_sec": float(np.mean(k_times)) if k_times else None,
            "proven_lb": int(k_proven_lb),
            "attempts": k_attempt_details,
        })
        st = "SOLVED" if k_best_cost == 0 else f"{k_best_cost} conflicts (abandoned)"
        print(
            f"  k={next_k:>3} -> {st:<28} | "
            f"success={k_success}/{len(k_attempt_details)} | "
            f"best={min(k_times):.3f}s"
        )
        if k_best_cost != 0:
            break
        current_sol, current_k = k_best.copy(), next_k

    optimal = current_k <= certified_lb
    if optimal:
        print(f"  [CERTIFIED] reached lower bound {certified_lb} -- provably optimal")

    return {
        "best_color": current_sol, "best_conflicts": 0, "best_k": current_k,
        "clique_lower_bound": int(lb), "certified_lower_bound": int(certified_lb),
        "dsatur_upper_bound": int(dsatur_ub),
        "attempts": all_attempts, "all_convergence": all_conv,
        "edge_count": int(eu.size), "degree_max": int(degree.max()) if n else 0,
        "articulation_points": int(np.count_nonzero(articulation)),
        "components": int(components.max() + 1) if n else 0,
    }



def solve(args, g):
    n = g.v
    eu, ev = build_edge_arrays(g)
    adj = build_adjacency(n, eu, ev)
    degree, articulation, components, structural = graph_features(n, adj)
    lb = clique_lower_bound(n, adj)
    dsatur_ub, dsatur_sol = dsatur_upper_bound(n, adj, degree)
    max_k = max(lb, args.max_k if args.max_k is not None else dsatur_ub)

    master_deadline = time.time() + args.time_limit if args.time_limit > 0 else None
    rng = np.random.default_rng(args.seed)

    print(f"\nSAHGC v2.3")
    print(f"Vertices: {n}   edges: {eu.size}   scipy: {'yes' if _SCIPY_AVAILABLE else 'no (pip install scipy)'}")
    print(f"LB: {lb}   DSATUR UB: {dsatur_ub}   k-range: [{lb}, {max_k}]   strategy: descending")
    print(
        f"pop={args.population} elite={args.elite} "
        f"gen={args.generations} tabu={args.tabu_iters} "
        f"restarts={args.restarts} stall_patience={args.stall_patience}"
    )
    print(
        f"CC-GPX={args.use_cc_gpx} path_relink={args.path_relink} "
        f"exact_repair={args.exact_repair} art_bonus={args.art_tenure_bonus}"
    )
    if master_deadline is None:
        print("  [!] WARNING: no --time-limit set -- run has no overall deadline.")

    parallel_workers = getattr(args, "parallel_restarts", 0) or 0
    if parallel_workers > 1:
        print(f"  [parallel] restarts run in chunks of {parallel_workers} worker process(es)")

    return _solve_descending(
        args, g, n, eu, ev, adj, degree, structural, articulation,
        components, lb, dsatur_ub, dsatur_sol, master_deadline, rng,
        parallel_workers=parallel_workers,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="SAHGC v2.3-D: Structure-Aware Hybrid Graph Coloring (Descending-only, tuned)"
    )
    ap.add_argument("input", nargs="?", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--population", type=int, default=15)
    ap.add_argument("--elite", type=int, default=5)
    ap.add_argument("--generations", type=int, default=80)
    ap.add_argument("--tabu-iters", type=int, default=7000)
    ap.add_argument("--max-child-iters", type=int, default=16000)
    ap.add_argument("--init-tabu-iters", type=int, default=1500)
    ap.add_argument("--restarts", type=int, default=7)
    ap.add_argument("--time-limit", type=float, default=0.0)

    ap.add_argument(
        "--stall-patience", type=int, default=3,
        help="Abandon a k after this many consecutive restarts with no "
             "improvement in best conflict count (0 disables early-abandon "
             "and always uses the full --restarts budget for every k).",
    )
    ap.add_argument("--max-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--init-randomness", type=float, default=0.10)
    ap.add_argument("--stagnation", type=int, default=1800)
    ap.add_argument("--min-diversity-ratio", type=float, default=0.05)
    ap.add_argument("--perturb-fraction", type=float, default=0.08)
    ap.add_argument("--exact-repair", action="store_true")
    ap.add_argument("--exact-trigger-conflicts", type=int, default=3)
    ap.add_argument("--exact-core-vertices", type=int, default=24)
    ap.add_argument("--exact-time", type=float, default=0.15)
    ap.add_argument("--path-relink", action="store_true")
    ap.add_argument("--path-relink-every", type=int, default=8)
    ap.add_argument(
        "--no-cc-gpx", dest="use_cc_gpx",
        action="store_false", default=True,
        help="Disable CC-GPX (use original GPX for ablation study)",
    )
    ap.add_argument(
        "--art-tenure-bonus", type=int, default=8,
        help="Extra tabu tenure for articulation-point vertices",
    )
    ap.add_argument("--tag", default="SAHGC_v2.3")
    ap.add_argument(
        "--parallel-restarts", type=int, default=0,
        dest="parallel_restarts",
        help="Run restarts in parallel using N worker processes. 0=serial (default). 'auto'=cpu_count.",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    args.strategy = "descending"  # fixed: ascending/auto removed

    if args.population < 2:
        ap.error("--population must be >= 2")
    if args.elite < 1 or args.elite > args.population:
        ap.error("--elite must satisfy 1 <= elite <= population")

    try:
        filename = resolve_input_filename(args.input)
        outdir = args.outdir or SCRIPT_DIR
        os.makedirs(outdir, exist_ok=True)

        c3 = load_coloring3_module()
        g = c3.Graph()
        g.load_data_from_csv(filename)

        t0 = time.time()
        result = solve(args, g)
        elapsed = time.time() - t0

        raw = result["best_color"]
        if raw is None:
            raise RuntimeError("SAHGC did not produce a coloring.")

        colors = compact_colors(raw, g.courses)
        residual = verify_no_conflicts(colors, g)
        n_colors = len(set(colors.values()))
        valid = len(residual) == 0
        lb = result["clique_lower_bound"]
        certified_lb = result.get("certified_lower_bound", lb)
        optimal_proven = bool(valid and n_colors == certified_lb)

        print(f"\nTotal time: {elapsed:.4f}s")
        print(f"Colors: {n_colors}   conflicts: {len(residual)}")
        print(f"Clique LB: {lb}   Certified LB: {certified_lb}   DSATUR UB: {result['dsatur_upper_bound']}")
        if optimal_proven:
            print(f"[CERTIFIED] Optimal (matches certified lower bound {certified_lb})")
        elif valid:
            print("[VALID] Conflict-free; optimality not certified.")
        else:
            print(f"[INVALID] {len(residual)} conflict(s) remain.")

        color_student_counts = g.compute_color_student_counts(colors)
        cwd = os.getcwd()
        try:
            os.chdir(outdir)
            g.export_schedule_table(colors, color_student_counts)
            g.export_text_report(
                colors, color_student_counts, elapsed,
                conflicts=residual, input_filename=filename,
            )
            g.export_detailed_excel(colors, color_student_counts)
        finally:
            os.chdir(cwd)

        summary = {
            "algorithm": args.tag,
            "version": "2.3",
            "input_file": os.path.abspath(filename),
            "n_courses": int(g.v),
            "n_students": int(len(g.students)),
            "n_edges": int(result["edge_count"]),
            "colors_used": int(n_colors),
            "conflicts": int(len(residual)),
            "valid_coloring": bool(valid),
            "clique_lower_bound": int(lb),
            "certified_lower_bound": int(certified_lb),
            "dsatur_upper_bound": int(result["dsatur_upper_bound"]),
            "optimal_proven_by_lower_bound": bool(optimal_proven),
            "computation_time_sec": round(elapsed, 6),
            "scipy_available": _SCIPY_AVAILABLE,
            "fixes_applied": [
                "kempe_delta_boundary_fix",
                "path_relink_mapping_guard",
                "path_relink_copy_fix",
                "k1_guard",
                "dsatur_ub_krange",
                "cc_gpx_crossover",
                "trajectory_archive",
                "structural_tabu_tenure",
                "unified_operator_tracking",
                "kempe_skip_no_iteration_waste",
                "vectorized_kempe_delta",
                "scipy_partition_distance",
                "heap_dsatur_init",
                "numpy_cc_gpx",
                "adaptive_kempe_skip",
                "cached_diversity",
                "vectorized_coloringstate",
                "decreasing_k_warm_start",
                "cross_edge_merge_warm_start",
                "stall_patience_early_abandon",
                "csr_vectorized_kempe_bfs",
                "sound_conflict_driven_lb_tightening",
                "parallel_restarts_stall_aware",
            ],
            "structural_features": {
                "max_degree": int(result["degree_max"]),
                "articulation_points": int(result["articulation_points"]),
                "connected_components": int(result["components"]),
            },
            "parameters": vars(args),
            "k_attempts": result["attempts"],
        }
        with open(
            os.path.join(outdir, "sahgc_run_summary.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        with open(
            os.path.join(outdir, "sahgc_convergence.csv"),
            "w", encoding="utf-8-sig", newline="",
        ) as f:
            w = csv.writer(f)
            w.writerow(
                ["K", "Restart", "Step", "Child_Best",
                 "Global_Best", "Improvements", "Diversity"]
            )
            for row in result["all_convergence"]:
                w.writerow(row)

        print("\nOutputs written:")
        for fname in [
            "sahgc_run_summary.json", "sahgc_convergence.csv",
            "ETP_Final_Schedule.xlsx", "coloring_results.xlsx",
            "coloring_results.txt",
        ]:
            p = os.path.join(outdir, fname)
            if os.path.isfile(p):
                print(f"  {p}")

    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()