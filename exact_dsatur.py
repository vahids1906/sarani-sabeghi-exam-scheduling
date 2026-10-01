# -*- coding: utf-8 -*-
"""
exact_dsatur.py  v3 -- interface-identical to dsatur_coloring.py
================================================================
Exact DSATUR Branch-and-Bound for Minimum Graph Coloring (exam-day assignment).

What changed in v3 (alignment + bug fixes)
------------------------------------------
A-1  input argument is now OPTIONAL and resolved with exactly the same rules
     as dsatur_coloring.py / coloring3.py (input.csv, or the single *.csv in
     the script folder).
A-2  CLI accepts the union of both scripts' flags:
         --outdir --restarts --seed --time-limit --quiet --tag
     (--restarts is accepted for pipeline compatibility; an exact solver has
      nothing to restart, so it is recorded in the JSON and ignored.)
A-3  coloring_results.xlsx / coloring_results.txt / ETP_Final_Schedule.xlsx are
     written ONLY by coloring3.Graph's own exporters, so the 3-sheet layout
     (Course_Color / Color_Courses / Color_Student_Counts) is preserved and
     byte-compatible with the greedy and DSATUR runs.  v2 overwrote those two
     files with its own simplified format -- that is gone.
A-4  exporter calls are no longer wrapped in a silent "except: pass".
A-5  clique lower bound uses the SAME routine as dsatur_coloring.py (greedy
     multi-start + time-bounded exact Bron-Kerbosch), so both algorithms
     report the same bound and "provably optimal" is decided identically.
A-6  colours are compacted to a contiguous 1..m day range (compact_colors).
A-7  JSON summary: dsatur_run_summary.json-compatible key set
     (adds n_students and optimal_proven; keeps the extra B&B fields).
     Written as exact_dsatur_run_summary.json AND, for backwards
     compatibility with older algo_compare.py, exact_dsatur_summary.json.
A-8  terminal output mirrors dsatur_coloring.py line for line.
B-1  recursion depth: solver runs in a thread with a large stack and a raised
     recursion limit, so graphs with >1000 courses no longer crash.

Usage
-----
    python exact_dsatur.py [input.csv] [--outdir DIR] [--time-limit SEC]
                           [--restarts N] [--seed N] [--quiet] [--tag NAME]

    A DIMACS .col/.graph/.dimacs file may also be passed; in that case there
    are no student counts, so only the text report and the JSON summary are
    produced.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

try:
    import openpyxl
except Exception:  # pragma: no cover
    openpyxl = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIMACS_EXTS = (".col", ".graph", ".dimacs")


# ---------------------------------------------------------------------------
# Internal graph representation
# ---------------------------------------------------------------------------
class InternalGraph:
    """Minimal graph representation used by the B&B solver."""

    def __init__(self, names: List[str], adj: List[List[int]]):
        self.names = names
        self.adj = adj
        self.n = len(names)
        self.m = sum(len(x) for x in adj) // 2


# ---------------------------------------------------------------------------
# coloring3.py reuse (same graph + same output writers as every other algo)
# ---------------------------------------------------------------------------
def load_coloring3_module():
    path = os.path.join(SCRIPT_DIR, "coloring3.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "coloring3.py was not found next to exact_dsatur.py.\n"
            "This script re-uses its Graph class so that all algorithms build\n"
            "the exact same conflict graph and write the exact same output\n"
            "format. Put both files in the same folder."
        )
    spec = importlib.util.spec_from_file_location("_coloring3_for_exact", path)
    if spec is None or spec.loader is None:
        raise ImportError("coloring3.py could not be loaded.")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_edge_list(g) -> List[Tuple[int, int]]:
    """coloring3.Graph -> sorted list of unique undirected edges (a < b).

    Identical construction to dsatur_coloring.build_edge_arrays, so the
    reported edge count matches exactly.
    """
    idx = {course: i for i, course in enumerate(g.courses)}
    seen: Set[Tuple[int, int]] = set()
    for course, neighbours in g.graph.items():
        if course not in idx:
            continue
        a = idx[course]
        for other in neighbours:
            if other not in idx:
                continue
            b = idx[other]
            if a == b:
                continue  # ignore self-loops
            seen.add((a, b) if a < b else (b, a))
    return sorted(seen)


def build_adjacency(n: int, edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    buckets: List[Set[int]] = [set() for _ in range(n)]
    for a, b in edges:
        buckets[a].add(b)
        buckets[b].add(a)
    return [sorted(s) for s in buckets]


def load_dimacs_internal(path: str) -> InternalGraph:
    """Load an undirected DIMACS .col graph."""
    n_vertices: Optional[int] = None
    edges: List[Tuple[int, int]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("c"):
                continue
            parts = line.split()
            if parts[0] == "p" and len(parts) >= 3:
                n_vertices = int(parts[2])
            elif parts[0] == "e" and len(parts) >= 3:
                a, b = int(parts[1]) - 1, int(parts[2]) - 1
                if a != b:
                    edges.append((a, b))
    if n_vertices is None:
        raise ValueError("Invalid DIMACS file: missing 'p' line.")
    clean = [
        (min(a, b), max(a, b))
        for a, b in edges
        if 0 <= a < n_vertices and 0 <= b < n_vertices
    ]
    adj = build_adjacency(n_vertices, sorted(set(clean)))
    return InternalGraph([str(i + 1) for i in range(n_vertices)], adj)


# ---------------------------------------------------------------------------
# A-5: clique lower bound -- same routine as dsatur_coloring.py
# ---------------------------------------------------------------------------
def clique_lower_bound(
    n: int,
    adj: List[List[int]],
    time_budget: float = 15.0,
    exact_limit: int = 400,
) -> Tuple[int, List[int]]:
    """Lower bound on the number of days via the clique number.

    Greedy multi-start first (always available), then a time-bounded EXACT
    Bron-Kerbosch refinement for small instances. Returns (size, vertices).
    """
    if n == 0:
        return 0, []
    sets = [set(a) for a in adj]

    # ---- 1. cheap greedy multi-start ------------------------------------
    order = sorted(range(n), key=lambda v: len(sets[v]), reverse=True)
    best, best_vs = 1, [order[0]]
    for start in order[: min(n, 30)]:
        clique = [start]
        cand = set(sets[start])
        while cand:
            nxt = max(cand, key=lambda v: len(sets[v] & cand))
            clique.append(nxt)
            cand &= sets[nxt]
        if len(clique) > best:
            best, best_vs = len(clique), clique

    if n > exact_limit:
        return best, sorted(best_vs)

    # ---- 2. time-bounded exact refinement -------------------------------
    deadline = time.time() + time_budget
    box = [best, list(best_vs)]
    timed_out = [False]

    def bronk(R: List[int], P: Set[int], X: Set[int]) -> None:
        if timed_out[0]:
            return
        if time.time() > deadline:
            timed_out[0] = True
            return
        if not P and not X:
            if len(R) > box[0]:
                box[0], box[1] = len(R), R[:]
            return
        if len(R) + len(P) <= box[0]:
            return
        pivot = max(P | X, key=lambda v: len(sets[v] & P))
        for v in list(P - sets[pivot]):
            if timed_out[0]:
                return
            bronk(R + [v], P & sets[v], X & sets[v])
            P = P - {v}
            X = X | {v}

    bronk([], set(range(n)), set())
    return box[0], sorted(box[1])


# ---------------------------------------------------------------------------
# Deterministic DSATUR (used as the B&B upper bound; same rules as
# dsatur_coloring.dsatur with rng=None)
# ---------------------------------------------------------------------------
def dsatur(n: int, adj: List[List[int]]) -> Tuple[List[int], int]:
    if n == 0:
        return [], 0
    color = [-1] * n
    neigh_colors: List[Set[int]] = [set() for _ in range(n)]
    degree = [len(a) for a in adj]
    uncolored = set(range(n))
    used = 0
    while uncolored:
        best_key = None
        best: List[int] = []
        for v in uncolored:
            key = (len(neigh_colors[v]), degree[v])
            if best_key is None or key > best_key:
                best_key, best = key, [v]
            elif key == best_key:
                best.append(v)
        v = min(best)
        c = 0
        while c in neigh_colors[v]:
            c += 1
        color[v] = c
        used = max(used, c + 1)
        uncolored.discard(v)
        for u in adj[v]:
            neigh_colors[u].add(c)
    return color, used


# ---------------------------------------------------------------------------
# Exact DSATUR branch-and-bound solver
# ---------------------------------------------------------------------------
class ExactDSATUR:
    def __init__(
        self,
        graph: InternalGraph,
        time_limit: Optional[float] = None,
        lower_bound: int = 0,
        clique: Optional[List[int]] = None,
    ):
        self.g = graph
        self.n = graph.n
        self.time_limit = time_limit if time_limit and time_limit > 0 else None
        self.start = 0.0
        self.timed_out = False
        self.nodes = 0

        self.colors = [-1] * self.n
        self.degree = [len(x) for x in graph.adj]
        self.forbidden = [[0] * (self.n + 1) for _ in range(self.n)]
        self.used_colors = 0

        # greedy upper bound
        greedy, ub = dsatur(self.n, graph.adj)
        self.best_colors = greedy[:]
        self.best_k = ub if self.n else 0

        # clique lower bound (computed once, outside, and shared)
        self.lower_bound = lower_bound
        self.clique = list(clique or [])

    # ------ helpers ------
    def time_exceeded(self) -> bool:
        if self.time_limit is None:
            return False
        if time.perf_counter() - self.start >= self.time_limit:
            self.timed_out = True
            return True
        return False

    def select_vertex(self) -> int:
        best, best_sat, best_deg = -1, -1, -1
        for v in range(self.n):
            if self.colors[v] != -1:
                continue
            sat = sum(1 for c in range(self.used_colors) if self.forbidden[v][c] > 0)
            if sat > best_sat or (sat == best_sat and self.degree[v] > best_deg):
                best, best_sat, best_deg = v, sat, self.degree[v]
        return best

    def assign(self, v: int, c: int) -> None:
        self.colors[v] = c
        for u in self.g.adj[v]:
            if self.colors[u] == -1:
                self.forbidden[u][c] += 1

    def unassign(self, v: int, c: int) -> None:
        for u in self.g.adj[v]:
            if self.colors[u] == -1:
                self.forbidden[u][c] -= 1
        self.colors[v] = -1

    def search(self, colored: int) -> None:
        if self.time_exceeded():
            return
        self.nodes += 1

        if colored == self.n:
            if self.used_colors < self.best_k:
                self.best_k = self.used_colors
                self.best_colors = self.colors[:]
            return

        # prune: cannot beat the incumbent any more
        if max(self.used_colors, self.lower_bound) >= self.best_k:
            return

        v = self.select_vertex()

        # reuse existing colour labels first (colour symmetry breaking)
        for c in range(self.used_colors):
            if self.time_exceeded():
                return
            if self.forbidden[v][c] == 0:
                self.assign(v, c)
                self.search(colored + 1)
                self.unassign(v, c)

        # then open exactly ONE new colour label
        if self.used_colors + 1 < self.best_k:
            c = self.used_colors
            self.used_colors += 1
            self.assign(v, c)
            self.search(colored + 1)
            self.unassign(v, c)
            self.used_colors = c

    def solve(self) -> Dict[str, object]:
        self.start = time.perf_counter()
        run_deep(lambda: self.search(0), self.n)
        elapsed = time.perf_counter() - self.start
        return {
            "best_k": self.best_k,
            "colors": self.best_colors[:],
            "lower_bound": self.lower_bound,
            "clique": self.clique[:],
            "nodes": self.nodes,
            "runtime_sec": elapsed,
            "timed_out": self.timed_out,
            "optimality_proven": not self.timed_out,
        }


# ---------------------------------------------------------------------------
# B-1: run deep recursion in a thread with a big stack
# ---------------------------------------------------------------------------
def run_deep(fn, depth_hint: int) -> None:
    needed = max(20000, depth_hint * 12 + 2000)
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, needed))
    error: List[BaseException] = []

    def target():
        try:
            fn()
        except BaseException as exc:  # re-raised on the main thread
            error.append(exc)

    try:
        threading.stack_size(256 * 1024 * 1024)
    except (ValueError, RuntimeError):
        try:
            threading.stack_size(64 * 1024 * 1024)
        except (ValueError, RuntimeError):
            pass
    t = threading.Thread(target=target)
    t.start()
    t.join()
    sys.setrecursionlimit(old_limit)
    if error:
        raise error[0]


# ---------------------------------------------------------------------------
# A-6: colour compaction + verification (same semantics as dsatur_coloring)
# ---------------------------------------------------------------------------
def compact_colors(raw: Sequence[int], courses: Sequence[str]) -> Dict[str, int]:
    """Relabel used colours to a contiguous 1..m range (day numbers)."""
    used = sorted({int(c) for c in raw})
    remap = {old: new for new, old in enumerate(used, start=1)}
    return {course: remap[int(raw[i])] for i, course in enumerate(courses)}


def verify_no_conflicts(colors: Dict[str, int], g) -> List[Tuple[str, str, int]]:
    """Independent check against coloring3's own adjacency dict."""
    bad = []
    for course, neighbours in g.graph.items():
        for other in neighbours:
            if course in colors and other in colors and colors[course] == colors[other]:
                pair = tuple(sorted((course, other)))
                bad.append((pair[0], pair[1], colors[course]))
    return sorted(set(bad))


def verify_no_conflicts_indexed(
    graph: InternalGraph, colors: Dict[str, int]
) -> List[Tuple[str, str, int]]:
    bad = []
    for v in range(graph.n):
        nv = graph.names[v]
        for u in graph.adj[v]:
            if u > v and colors[nv] == colors[graph.names[u]]:
                bad.append((nv, graph.names[u], colors[nv]))
    return sorted(set(bad))


# ---------------------------------------------------------------------------
# A-1: input resolution -- same rules as dsatur_coloring.py / coloring3.py
# ---------------------------------------------------------------------------
def resolve_input_filename(explicit: Optional[str]) -> str:
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
            f"Please specify which one to use: python exact_dsatur.py <file.csv>"
        )
    raise FileNotFoundError(
        "No CSV file found. Place 'input.csv' next to the script, "
        "or run: python exact_dsatur.py <filename.csv>"
    )


# ---------------------------------------------------------------------------
# Fallback exporters -- ONLY used for DIMACS input (no coloring3.Graph there).
# For CSV input every file is written by coloring3's own exporters (A-3).
# ---------------------------------------------------------------------------
def export_text_report_fallback(
    graph: InternalGraph,
    colors: Dict[str, int],
    result: Dict,
    input_path: str,
    outdir: str,
) -> None:
    with open(os.path.join(outdir, "coloring_results.txt"), "w", encoding="utf-8") as f:
        f.write("Exact DSATUR Branch-and-Bound v3 (DIMACS mode)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Input: {input_path}\n")
        f.write(f"Vertices: {graph.n}\n")
        f.write(f"Edges: {graph.m}\n")
        f.write(f"Colors used: {len(set(colors.values()))}\n")
        f.write(f"Clique lower bound: {result['lower_bound']}\n")
        f.write(f"Search nodes: {result['nodes']}\n")
        f.write(f"Runtime (sec): {result['runtime_sec']:.6f}\n")
        f.write(f"Timed out: {result['timed_out']}\n")
        f.write(f"Optimality proven: {result['optimality_proven']}\n\n")
        f.write("Coloring:\n")
        for name in graph.names:
            f.write(f"{name}\t{colors[name]}\n")


def export_excel_fallback(
    graph: InternalGraph, colors: Dict[str, int], outdir: str
) -> None:
    if openpyxl is None:
        print("Note: openpyxl is not installed, skipping the Excel file.")
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Course_Color"
    ws.append(["Course", "Color"])
    for name in graph.names:
        ws.append([name, colors[name]])
    wb.save(os.path.join(outdir, "coloring_results.xlsx"))


# ---------------------------------------------------------------------------
# A-7: JSON summary, dsatur_run_summary.json-compatible
# ---------------------------------------------------------------------------
def build_summary(
    graph: InternalGraph,
    colors: Dict[str, int],
    result: Dict,
    residual: Sequence,
    input_path: str,
    n_students: Optional[int],
    args,
) -> Dict[str, object]:
    n_colors = len(set(colors.values()))
    valid = len(residual) == 0
    proven = bool(result["optimality_proven"] and valid)
    return {
        # ---- keys shared with dsatur_run_summary.json --------------------
        "algorithm": args.tag,
        "mode": "exact",
        "input_file": os.path.abspath(input_path),
        "n_courses": graph.n,
        "n_students": n_students,
        "n_edges": graph.m,
        "colors_used": n_colors,
        "conflicts": len(residual),
        "clique_lower_bound": int(result["lower_bound"]),
        "optimal_proven": bool(proven or n_colors == int(result["lower_bound"])),
        "computation_time_sec": round(float(result["runtime_sec"]), 4),
        "parameters": {
            "restarts": args.restarts,
            "seed": args.seed,
            "time_limit": args.time_limit,
            "tag": args.tag,
        },
        # ---- extra fields specific to the exact solver -------------------
        "version": "3",
        "valid_coloring": valid,
        "optimality_proven": proven,
        "optimal_proven_by_lower_bound": bool(
            proven and n_colors == int(result["lower_bound"])
        ),
        "search_nodes": int(result["nodes"]),
        "timed_out": bool(result["timed_out"]),
        "clique_lower_bound_vertices": [graph.names[v] for v in result["clique"]],
        "color_class_sizes": {
            str(c): sum(1 for x in colors.values() if x == c)
            for c in sorted(set(colors.values()))
        },
    }


def write_summary(summary: Dict[str, object], outdir: str) -> List[str]:
    written = []
    for fname in ("exact_dsatur_run_summary.json", "exact_dsatur_summary.json"):
        path = os.path.join(outdir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        written.append(fname)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Exam-day assignment with EXACT DSATUR branch-and-bound."
    )
    ap.add_argument("input", nargs="?", default=None, help="input CSV file")
    ap.add_argument(
        "--outdir", default=None, help="folder for the output files (default: script folder)"
    )
    ap.add_argument(
        "--time-limit",
        type=float,
        default=0.0,
        help="time limit in seconds for the B&B search; 0 = no limit",
    )
    ap.add_argument(
        "--restarts",
        type=int,
        default=1,
        help="accepted for pipeline compatibility (an exact search has nothing to restart)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="accepted for pipeline compatibility (the exact solver is deterministic)",
    )
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--tag", default="ExactDSATUR")
    args = ap.parse_args()

    def say(*a):
        if not args.quiet:
            print(*a)

    try:
        ext = os.path.splitext(args.input or "")[1].lower()
        is_dimacs = ext in DIMACS_EXTS
        filename = args.input if is_dimacs else resolve_input_filename(args.input)
        if is_dimacs and not os.path.isfile(filename):
            raise FileNotFoundError(f"Specified file not found: {filename}")
        say(f"Using input file: {filename}")

        outdir = args.outdir or SCRIPT_DIR
        os.makedirs(outdir, exist_ok=True)

        c3_graph = None
        n_students = None
        if is_dimacs:
            graph = load_dimacs_internal(filename)
        else:
            c3 = load_coloring3_module()
            c3_graph = c3.Graph()
            c3_graph.load_data_from_csv(filename)
            edges = build_edge_list(c3_graph)
            graph = InternalGraph(
                list(c3_graph.courses), build_adjacency(c3_graph.v, edges)
            )
            n_students = len(c3_graph.students)

        say(f"\nNumber of courses (vertices): {graph.n}")
        say(f"Number of conflict edges: {graph.m}")

        lb, clique = clique_lower_bound(graph.n, graph.adj)
        say(f"Clique lower bound on days: {lb}")

        say(f"\nAlgorithm: {args.tag}")
        say(
            "Mode: exact branch-and-bound, no time limit"
            if args.time_limit <= 0
            else f"Mode: exact branch-and-bound, time limit {args.time_limit:g}s"
        )
        if args.time_limit <= 0 and graph.n > 60:
            say(
                "Warning: exact search on a graph this large can take a very long "
                "time. Consider --time-limit."
            )
        if args.restarts != 1:
            say("Note: --restarts is ignored by the exact solver (recorded in JSON).")

        solver = ExactDSATUR(
            graph, time_limit=args.time_limit, lower_bound=lb, clique=clique
        )
        say(f"Greedy DSATUR upper bound: {solver.best_k}")

        result = solver.solve()

        colors = compact_colors(result["colors"], graph.names)
        n_colors = len(set(colors.values()))
        residual = (
            verify_no_conflicts(colors, c3_graph)
            if c3_graph is not None
            else verify_no_conflicts_indexed(graph, colors)
        )

        say(f"\nComputation time: {result['runtime_sec']:.4f} seconds")
        say(f"Number of colors used: {n_colors}")
        if residual:
            say(f"WARNING: {len(residual)} conflicting course pairs remain!")
        else:
            say("No conflicts detected! The coloring is valid.")
        say(f"Search nodes explored: {result['nodes']}")
        if result["timed_out"]:
            say("Time limit reached -- optimality NOT proven.")
        if n_colors == lb:
            say(f"This matches the clique lower bound ({lb}) -- provably optimal.")
        elif not result["timed_out"]:
            say("Proven optimal by exhaustive branch-and-bound.")
        else:
            say(f"Gap to the lower bound: {n_colors - lb} day(s).")

        # ---- outputs -----------------------------------------------------
        if c3_graph is not None:
            color_student_counts = c3_graph.compute_color_student_counts(colors)
            say("\nColor -> number of unique students:")
            for color in sorted(color_student_counts):
                say(f"Color {color}: {color_student_counts[color]} students")

            # A-3/A-4: identical writers, identical format, no silent failures
            cwd = os.getcwd()
            try:
                os.chdir(outdir)
                c3_graph.export_schedule_table(colors, color_student_counts)
                c3_graph.export_text_report(
                    colors,
                    color_student_counts,
                    result["runtime_sec"],
                    conflicts=residual,
                    input_filename=filename,
                )
                c3_graph.export_detailed_excel(colors, color_student_counts)
            finally:
                os.chdir(cwd)
        else:
            export_text_report_fallback(graph, colors, result, filename, outdir)
            export_excel_fallback(graph, colors, outdir)

        summary = build_summary(
            graph, colors, result, residual, filename, n_students, args
        )
        written = write_summary(summary, outdir)
        say("Run summary saved successfully: " + ", ".join(written))

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
