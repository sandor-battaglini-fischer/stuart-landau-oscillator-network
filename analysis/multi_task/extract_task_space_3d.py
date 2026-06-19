#!/usr/bin/env python3
# Usage: python extract_3d_data.py [--figures]
"""
Axis mapping:
  X: lambda ∈ [-0.5, 0.5] → [0, 1]   (oscillating vs damped)
  Y: log(N)/log(128)                   (dimensionality, N ∈ {1,2,4,9,16,25,50,128})
  Z: nonlinearity level — 4 discrete values:
       0.000 = completely linear            (DHO, no tanh)
       0.333 = linear oscillator + tanh     (DHO with tanh)
       0.667 = nonlinear oscillator, no tanh (SL w/o tanh)
       1.000 = fully nonlinear              (SL with tanh)

Inferred-grid approach, three orthogonal measurements —
  - grid search: acc(lambda) at fixed N_grid, Z=1
  - scaling:     acc(N)      at fixed lambda_default, Z=1
  - ablation:    acc(Z)      at fixed N=50, lambda_default

From these we infer the full 11×8×4 = 352-point grid via the additive model:
  acc(lambda, N, Z) ≈ acc_grid(lambda)
                     + [acc_scale(N) - acc_scale(N_grid)]
                     + [acc_ablation(Z) - acc_ablation(1.0)]
clipped to [0, 1].
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.plotting_utils.style import apply_style, thesis_red, thesis_blue, ifisc_green, mycmap
apply_style()


BASE_DIR = Path(__file__).parent

LAMBDA_MIN, LAMBDA_MAX = -0.5, 0.5
LAMBDA_VALS = [round(LAMBDA_MIN + 0.1 * i, 1) for i in range(11)]   # [-0.5, ..., 0.5]
N_VALS      = [1, 2, 4, 9, 16, 25, 50, 128]
N_LOG_MAX   = math.log(128)
Z_VALS      = [0.0, 1/3, 2/3, 1.0]

Z_LEVEL = {
    "SL":           1.000,
    "SL w/o tanh":  2 / 3,
    "DHO":          1 / 3,
    "DHO w/o tanh": 0.000,
}


def norm_lambda(lam: float) -> float:
    return (lam - LAMBDA_MIN) / (LAMBDA_MAX - LAMBDA_MIN)


def norm_n(n) -> float:
    return math.log(max(n, 1)) / N_LOG_MAX


def log_interp(n, table: dict) -> float:
    """Log-linear interpolation of a {N: acc} table."""
    if n in table:
        return table[n]
    keys = sorted(table.keys())
    log_n = math.log(max(n, 1))
    for i in range(len(keys) - 1):
        n0, n1 = keys[i], keys[i + 1]
        if n0 <= n <= n1:
            t = (log_n - math.log(n0)) / (math.log(n1) - math.log(n0))
            return table[n0] * (1 - t) + table[n1] * t
    return table[keys[0]] if n < keys[0] else table[keys[-1]]


def parse_log_params(log_path: Path) -> dict:
    params = {}
    if not log_path.exists():
        return params
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if re.match(r"epoch\s+\d+", line):
                break
            if ":" in line and not line.startswith("="):
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if key and val and " " not in key:
                    params[key] = val
    return params


def parse_classification_final_acc(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    last_test = None
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if re.match(r"epoch\s+\d+", line):
                for part in line.split(","):
                    if "test:" in part:
                        try:
                            last_test = float(part.split(":")[1].strip().split()[0])
                        except (ValueError, IndexError):
                            pass
    return last_test / 100.0 if last_test is not None else None


def parse_mg_final_r2(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        data = json.load(f)
    r2_vals = [d["test_r2"] for d in data if "test_r2" in d]
    return float(r2_vals[-1]) if r2_vals else None


def find_run_dir(run_name: str) -> Path | None:
    for parent in (BASE_DIR, BASE_DIR / "runs-old", BASE_DIR / "Nonlinearity Ablation runs"):
        candidate = parent / run_name
        if candidate.is_dir():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Per-task data tables
# ---------------------------------------------------------------------------

def fill_lambda_gaps(table: dict) -> dict:
    """Fill missing lambda values: mirror from -lambda first, then nearest neighbour."""
    filled = dict(table)
    for lam in LAMBDA_VALS:
        if lam in filled:
            continue
        mirror = round(-lam, 10)
        if mirror in filled:
            filled[lam] = filled[mirror]
        elif filled:
            nearest = min(filled.keys(), key=lambda k: abs(k - lam))
            filled[lam] = filled[nearest]
    return filled


def build_grid_lambda_table(grid_results: list, tol: float = 0.05) -> dict:
    """Best accuracy for each lambda ∈ LAMBDA_VALS from a grid-search result list."""
    table = {}
    for lam in LAMBDA_VALS:
        matches = [r for r in grid_results
                   if not r.get("numerical_error") and abs(r["lambda"] - lam) < tol]
        if not matches:
            continue
        accs = sorted([r.get("best_test_acc", 0) / 100.0 for r in matches], reverse=True)
        # mean of top-10 to smooth over gamma variation
        table[lam] = float(np.mean(accs[:10]))
    return table


def build_grid_lambda_table_r2(grid_results: list, tol: float = 0.05) -> dict:
    table = {}
    for lam in LAMBDA_VALS:
        matches = [r for r in grid_results
                   if not r.get("numerical_error") and abs(r["lambda"] - lam) < tol]
        if not matches:
            continue
        accs = sorted([float(max(r.get("best_test_r2", 0), 0)) for r in matches], reverse=True)
        table[lam] = float(np.mean(accs[:10]))
    return table


def compute_hull(pts: list) -> tuple[list, list]:
    """
    Convex hull of the point positions. Returns (vertices, face-index-triples).
    Vertices are the subset of pts that lie on the hull surface.
    """
    from scipy.spatial import ConvexHull, QhullError
    if len(pts) < 4:
        return [], []
    xyz = np.array([[p[0], p[1], p[2]] for p in pts])
    try:
        hull = ConvexHull(xyz)
        verts = xyz[hull.vertices].round(5).tolist()
        idx_map = {old: new for new, old in enumerate(hull.vertices)}
        faces = [[idx_map[v] for v in f] for f in hull.simplices]
        return verts, faces
    except (QhullError, Exception) as e:
        print(f"  hull failed: {e}")
        return [], []


def filter_normalize(pts: list, threshold: float) -> list:
    """
    Drop points below threshold, then normalize remaining acc to [0, 1]
    so the best run maps to 1.0 and the threshold maps to 0.0.
    """
    good = [p for p in pts if p[3] >= threshold]
    if not good:
        return good
    max_acc = max(p[3] for p in good)
    span = max_acc - threshold
    if span < 1e-6:
        return [[p[0], p[1], p[2], 1.0] for p in good]
    return [[p[0], p[1], p[2], round((p[3] - threshold) / span, 5)] for p in good]


def infer_grid(lambda_table: dict,
               scale_table: dict,
               ablation_table: dict,
               n_grid: int,
               jitter_rng) -> list:
    """
    Generate 352-point inferred 3D grid using the additive model.
    Small positional jitter so overlapping points don't render as one blob.
    """
    ref_scale  = log_interp(n_grid, scale_table)
    ref_ablation = ablation_table.get(1.0, max(ablation_table.values()))
    points = []
    for lam in LAMBDA_VALS:
        if lam not in lambda_table:
            continue
        base_acc = lambda_table[lam]
        x = norm_lambda(lam)
        for n in N_VALS:
            delta_n = log_interp(n, scale_table) - ref_scale
            y = norm_n(n)
            for z in Z_VALS:
                delta_z = ablation_table.get(z, ref_ablation) - ref_ablation
                acc = float(np.clip(base_acc + delta_n + delta_z, 0.0, 1.0))
                # tiny jitter so each (x,y,z) point is visually distinct
                xj = float(np.clip(x + jitter_rng.uniform(-0.015, 0.015), 0, 1))
                yj = float(np.clip(y + jitter_rng.uniform(-0.015, 0.015), 0, 1))
                zj = float(np.clip(z + jitter_rng.uniform(-0.010, 0.010), 0, 1))
                points.append([round(xj, 5), round(yj, 5), round(zj, 5), round(acc, 5)])
    return points


def plot_2d_projections(all_data: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    TASK_COLORS = {"imdb": "#D62728", "smnist": "#1F77B4", "mg": "#2CA02C"}
    TASK_NAMES  = {"imdb": "IMDb", "smnist": "sMNIST", "mg": "Mackey-Glass"}
    TASKS = ["imdb", "smnist", "mg"]
    BG = (0.02, 0.03, 0.05)

    def hex_to_rgb1(h):
        return (int(h[1:3],16)/255, int(h[3:5],16)/255, int(h[5:7],16)/255)

    def task_cmap(task):
        r, g, b = hex_to_rgb1(TASK_COLORS[task])
        return LinearSegmentedColormap.from_list(
            task, [BG, (r*0.6, g*0.6, b*0.6), (r, g, b), (1.0, 1.0, 1.0)]
        )

    N_TICKS = [1, 2, 4, 9, 16, 25, 50, 128]

    nl_pos  = [0.0, 1/3, 2/3, 1.0]
    nl_lbl  = ["linear", "+tanh", "SL", "SL+tanh"]
    lam_pos = [norm_lambda(v) for v in [-0.5, -0.25, 0.0, 0.25, 0.5]]
    lam_lbl = ["-0.5", "-0.25", "0", "+0.25", "+0.5"]
    n_pos   = [norm_n(n) for n in N_TICKS]
    n_lbl   = [str(n) for n in N_TICKS]

    PROJS = [
        (0, 1, r"$\lambda$",        r"$N$ (log scale)",  lam_pos, lam_lbl, n_pos,  n_lbl,  "lambda_N"),
        (0, 2, r"$\lambda$",        "Nonlinearity",       lam_pos, lam_lbl, nl_pos, nl_lbl, "lambda_nl"),
        (1, 2, r"$N$ (log scale)",  "Nonlinearity",       n_pos,   n_lbl,   nl_pos, nl_lbl, "N_nl"),
    ]

    for xi, yi, xlabel, ylabel, xticks, xlbls, yticks, ylbls, suffix in PROJS:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        fig.patch.set_facecolor(BG)

        for ax, task in zip(axes, TASKS):
            ax.set_facecolor(BG)
            pts  = all_data[task]["pts"]
            xs   = [p[xi] for p in pts]
            ys   = [p[yi] for p in pts]
            accs = [p[3]  for p in pts]

            sc = ax.scatter(xs, ys, c=accs, cmap=task_cmap(task),
                            vmin=0, vmax=1, s=22, alpha=0.85,
                            linewidths=0, rasterized=True)

            cb = fig.colorbar(sc, ax=ax, pad=0.03, fraction=0.046)
            cb.set_label("Norm. accuracy", color="white")
            cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
            cb.outline.set_edgecolor("none")

            ax.set_xlabel(xlabel, color="white")
            ax.set_ylabel(ylabel, color="white")
            ax.set_title(TASK_NAMES[task], color="white", pad=6)
            ax.set_xticks(xticks); ax.set_xticklabels(xlbls, color="white",
                rotation=30 if len(xticks) > 4 else 0, ha="right" if len(xticks) > 4 else "center")
            ax.set_yticks(yticks); ax.set_yticklabels(ylbls, color="white")
            ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.06, 1.06)
            for spine in ax.spines.values():
                spine.set_edgecolor("rgba(255,255,255,0.15)" if False else "#2a3a50")
            ax.tick_params(colors="white", length=3)

        fig.tight_layout(pad=1.2)
        out_path = out_dir / f"task_space_{suffix}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"Saved → {out_path}")


def main(args=None):
    rng = np.random.default_rng(42)

    ABLATION_RUNS = {
        "imdb":  {"SL": "results/imdb/20260307_215103",  "SL w/o tanh": "results/imdb/20260307_215433",
                  "DHO": "results/imdb/20260307_215834",  "DHO w/o tanh": "results/imdb/20260307_220047"},
        "smnist":{"SL": "results/smnist/20260307_215124","SL w/o tanh": "results/smnist/20260307_215454",
                  "DHO": "results/smnist/20260307_215854","DHO w/o tanh": "results/smnist/20260307_220026"},
        "mg":    {"SL": "results/mackey_glass/20260307_215156",    "SL w/o tanh": "results/mackey_glass/20260307_215517",
                  "DHO": "results/mackey_glass/20260307_215920",    "DHO w/o tanh": "results/mackey_glass/20260307_220052"},
    }

    # -----------------------------------------------------------------------
    # IMDb
    # -----------------------------------------------------------------------
    imdb_grid_path = BASE_DIR / "grid_search_results/grid_search_results_20251209_164804.json"
    with open(imdb_grid_path) as f:
        imdb_grid = json.load(f)
    n_grid_imdb = imdb_grid["config"]["num_hidden"]   # 4

    imdb_lambda = fill_lambda_gaps(build_grid_lambda_table(imdb_grid["results"]))

    imdb_scale = {}
    for n, dirname in {1:"results/imdb/20251207_231522",2:"results/imdb/20251207_231503",
                       4:"results/imdb/20251208_104915",9:"results/imdb/20251207_221754",
                       50:"results/imdb/20251207_221628",128:"results/imdb/20251210_172243"}.items():
        acc = parse_classification_final_acc(BASE_DIR / "runs-old" / dirname / "log.txt")
        if acc is not None:
            imdb_scale[n] = acc

    imdb_ablation = {}
    for label, run_name in ABLATION_RUNS["imdb"].items():
        run_dir = find_run_dir(run_name)
        if run_dir:
            acc = parse_classification_final_acc(run_dir / "log.txt")
            if acc:
                imdb_ablation[Z_LEVEL[label]] = acc

    # -----------------------------------------------------------------------
    # sMNIST
    # -----------------------------------------------------------------------
    smnist_grid_path = BASE_DIR / "grid_smnist_results/grid_smnist_results_20260120_174247_1739s.json"
    with open(smnist_grid_path) as f:
        smnist_grid = json.load(f)
    n_grid_smnist = smnist_grid["config"]["num_hidden"]  # 50

    smnist_lambda = fill_lambda_gaps(build_grid_lambda_table(smnist_grid["results"]))

    smnist_scale = {}
    for n, dirname in {1:"results/smnist/20251218_114219",2:"results/smnist/20251218_114204",
                       4:"results/smnist/20251218_114148",9:"results/smnist/20251218_114028",
                       16:"results/smnist/20260312_220150",25:"results/smnist/20260312_110818",
                       50:"results/smnist/20251218_115016",128:"results/smnist/20251218_120601"}.items():
        acc = parse_classification_final_acc(BASE_DIR / "runs-old" / dirname / "log.txt")
        if acc is not None:
            smnist_scale[n] = acc

    smnist_ablation = {}
    for label, run_name in ABLATION_RUNS["smnist"].items():
        run_dir = find_run_dir(run_name)
        if run_dir:
            acc = parse_classification_final_acc(run_dir / "log.txt")
            if acc:
                smnist_ablation[Z_LEVEL[label]] = acc

    # -----------------------------------------------------------------------
    # Mackey-Glass
    # -----------------------------------------------------------------------
    mg_grid_path = BASE_DIR / "grid_mg_results/grid_mg_completed_20260209_165042.json"
    with open(mg_grid_path) as f:
        mg_grid = json.load(f)
    n_grid_mg = mg_grid["config"]["num_hidden"]  # 50

    mg_lambda = fill_lambda_gaps(build_grid_lambda_table_r2(mg_grid["results"]))

    mg_comp_path = BASE_DIR / "results/mackey_glass/comparison_20260203_184809" / "full_results.json"
    with open(mg_comp_path) as f:
        mg_comp = json.load(f)
    mg_scale = {}
    for r in mg_comp["results"]:
        n = r["parameters"]["num_hidden"]
        r2_scores = r.get("test_r2_scores", [])
        if r2_scores:
            mg_scale[n] = float(np.clip(r2_scores[-1], 0, 1))

    mg_ablation = {}
    for label, run_name in ABLATION_RUNS["mg"].items():
        run_dir = find_run_dir(run_name)
        if run_dir:
            acc = parse_mg_final_r2(run_dir / "metrics.json") or 0.0
            mg_ablation[Z_LEVEL[label]] = acc

    # -----------------------------------------------------------------------
    # Print what we have before inference
    # -----------------------------------------------------------------------
    print("=== IMDb ===")
    print(f"  lambda table:   {dict(sorted(imdb_lambda.items()))}")
    print(f"  scale table:    {dict(sorted(imdb_scale.items()))}")
    print(f"  ablation table: {imdb_ablation}")
    print("=== sMNIST ===")
    print(f"  lambda table:   {dict(sorted(smnist_lambda.items()))}")
    print(f"  scale table:    {dict(sorted(smnist_scale.items()))}")
    print(f"  ablation table: {smnist_ablation}")
    print("=== MG ===")
    print(f"  lambda table:   {dict(sorted(mg_lambda.items()))}")
    print(f"  scale table:    {dict(sorted(mg_scale.items()))}")
    print(f"  ablation table: {mg_ablation}")

    # -----------------------------------------------------------------------
    # Infer full 3D grids, then keep only "well-performing" runs
    # and normalise within that range so the shader sees [0, 1].
    # -----------------------------------------------------------------------
    THRESHOLDS = {"imdb": 0.80, "smnist": 0.90, "mg": 0.96}
    INNER_THRESHOLDS = {"imdb": 0.830, "smnist": 0.960, "mg": 0.980}

    raw = {
        "imdb":  infer_grid(imdb_lambda,  imdb_scale,  imdb_ablation,  n_grid_imdb,  rng),
        "smnist":infer_grid(smnist_lambda,smnist_scale,smnist_ablation,n_grid_smnist,rng),
        "mg":    infer_grid(mg_lambda,    mg_scale,    mg_ablation,    n_grid_mg,    rng),
    }

    all_data = {}
    for task, pts in raw.items():
        thr_raw  = THRESHOLDS[task]
        ithr_raw = INNER_THRESHOLDS[task]
        mx = max(p[3] for p in pts)

        kept  = filter_normalize(pts, thr_raw)
        # inner threshold expressed in normalised units: (ithr_raw - thr_raw) / (mx - thr_raw)
        inner_norm_thr = (ithr_raw - thr_raw) / (mx - thr_raw) if mx > thr_raw else 0.5
        core  = [p for p in kept if p[3] >= inner_norm_thr]

        hv,  hf  = compute_hull(kept)
        hv2, hf2 = compute_hull(core)
        all_data[task] = {"pts": kept, "hv": hv, "hf": hf, "hv2": hv2, "hf2": hf2}
        print(f"\n{task}: {len(pts)} inferred → {len(kept)} outer / {len(core)} inner "
              f"hull: {len(hv)} verts / inner hull: {len(hv2)} verts")

    # -----------------------------------------------------------------------
    # Save JSON and embed into HTML
    # -----------------------------------------------------------------------
    out_json = BASE_DIR / "task_space_data.json"
    with open(out_json, "w") as f:
        json.dump(all_data, f)
    print(f"\nSaved → {out_json}")

    _embed_in_html(BASE_DIR / "task_space_3d.html", all_data)
    print("Embedded data into task_space_3d.html ✓")

    if args and args.figures:
        print("\nGenerating 2D projection figures…")
        plot_2d_projections(all_data, BASE_DIR)


def _embed_in_html(html_path: Path, data: dict):
    with open(html_path) as f:
        html = f.read()

    def fp(pts):
        return "[" + ",".join("[" + ",".join(f"{v:.4f}" for v in p) + "]" for p in pts) + "]"

    def fv(verts):
        return "[" + ",".join("[" + ",".join(f"{v:.4f}" for v in v) + "]" for v in verts) + "]"

    def ff(faces):
        return "[" + ",".join("[" + ",".join(str(i) for i in f) + "]" for f in faces) + "]"

    parts = []
    for task in ("imdb", "smnist", "mg"):
        d = data[task]
        parts.append(
            f'{task}:{{pts:{fp(d["pts"])},hv:{fv(d["hv"])},hf:{ff(d["hf"])}'
            f',hv2:{fv(d["hv2"])},hf2:{ff(d["hf2"])}}}'
        )
    js = "var TASK_DATA={" + ",".join(parts) + "};"

    html = re.sub(
        r'(<script id="task-data">).*?(</script>)',
        r'\g<1>' + js + r'\g<2>',
        html,
        flags=re.DOTALL,
        count=1,
    )
    with open(html_path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", action="store_true",
                        help="Generate 2D projection PNGs for publication (requires matplotlib)")
    args = parser.parse_args()
    main(args)
