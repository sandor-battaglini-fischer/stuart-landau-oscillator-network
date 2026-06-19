import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.plotting_utils.style import apply_style, thesis_red, thesis_blue, ifisc_green, mycmap
apply_style()


thesis_blue = (0, 0.38, 0.68)
ifisc_green = (0.73, 0.83, 0.01)

TASK_COLOR = {'imdb': thesis_blue, 'mg': thesis_red, 'smnist': ifisc_green}
TASK_LABEL = {'mg': 'Mackey-Glass', 'imdb': 'IMDB', 'smnist': 'sMNIST'}
TASK_MARKER = {'imdb': 'o', 'mg': 's', 'smnist': 'D'}


def load_results(json_path, fallback_paths=None):
    with open(json_path, 'r') as f:
        data = json.load(f)
    results = data.get('results', [])
    config = data.get('config', {})

    if fallback_paths:
        existing = set()
        for r in results:
            existing.add((r.get('omega'), r.get('lambda'),
                          r.get('gamma_real'), r.get('gamma_imag')))
        for fb in fallback_paths:
            if fb and Path(fb).exists():
                with open(fb, 'r') as f:
                    fb_results = json.load(f).get('results', [])
                for r in fb_results:
                    key = (r.get('omega'), r.get('lambda'),
                           r.get('gamma_real'), r.get('gamma_imag'))
                    if key not in existing:
                        results.append(r)
                        existing.add(key)
    return results, config


def metric_value(result, task_name):
    if result.get('numerical_error', False):
        return 0.0
    if task_name == 'mg':
        return max(0.0, result.get('best_test_r2', 0.0)) * 100.0
    acc = result.get('best_test_acc', 0.0)
    return acc if acc > 1.0 else acc * 100.0


def results_to_arrays(results, task_name):
    lam, gr, gi, om, perf = [], [], [], [], []
    for r in results:
        if r.get('lambda') is None or r.get('gamma_real') is None:
            continue
        lam.append(r['lambda'])
        gr.append(r['gamma_real'])
        gi.append(r.get('gamma_imag', 0.0))
        om.append(r.get('omega', 0.0))
        perf.append(metric_value(r, task_name))
    return (np.array(lam), np.array(gr), np.array(gi),
            np.array(om), np.array(perf))


# ---------------------------------------------------------------------------
# Figure (a): collapse onto the fading-memory timescale tau_mem = 1 / |lambda|
# ---------------------------------------------------------------------------
def collapse_curve(lam, perf, agg='max'):
    neg = lam < -1e-9
    lam_n, perf_n = lam[neg], perf[neg]
    tau, val = [], []
    for u in sorted(set(lam_n.tolist())):
        sel = np.isclose(lam_n, u)
        if not np.any(sel):
            continue
        p = perf_n[sel]
        tau.append(1.0 / abs(u))
        val.append(np.nanmax(p) if agg == 'max' else np.nanmean(p))
    order = np.argsort(tau)
    return np.array(tau)[order], np.array(val)[order]


def build_figure_a(task_arrays, output_path, agg='max'):
    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(15, 6))

    for task in ('imdb', 'mg'):
        lam, gr, gi, om, perf = task_arrays[task]
        tau, val = collapse_curve(lam, perf, agg=agg)
        if len(tau) == 0:
            continue
        c = TASK_COLOR[task]
        ax_raw.plot(tau, val, marker=TASK_MARKER[task], color=c,
                    label=TASK_LABEL[task], markersize=6)
        vmin, vmax = np.nanmin(val), np.nanmax(val)
        norm = (val - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(val)
        ax_norm.plot(tau, norm, marker=TASK_MARKER[task], color=c,
                     label=TASK_LABEL[task], markersize=6)

    if 'smnist' in task_arrays:
        lam, gr, gi, om, perf = task_arrays['smnist']
        tau, val = collapse_curve(lam, perf, agg=agg)
        if len(tau) > 0:
            vmin, vmax = np.nanmin(val), np.nanmax(val)
            norm = (val - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(val)
            ax_norm.plot(tau, norm, marker=TASK_MARKER['smnist'], color=ifisc_green,
                         linestyle='--', alpha=0.55, markersize=5,
                         label=r'sMNIST (optimum off-axis: $\lambda>0$)')

    for ax in (ax_raw, ax_norm):
        ax.set_xscale('log')
        ax.set_xlabel(r'fading-memory timescale $\tau_{\mathrm{mem}} = 1/|\lambda|$')

    ax_raw.set_ylabel(r'best performance ($R^2$ / accuracy, \%)')
    ax_raw.set_title(r'(a) raw performance vs.\ $\tau_{\mathrm{mem}}$')
    ax_raw.legend(loc='best', frameon=False)

    ax_norm.set_ylabel('normalized performance (fraction of task best)')
    ax_norm.set_title(r'(b) collapse: IMDB$\uparrow$, MG$\downarrow$ cross')
    ax_norm.set_ylim(-0.05, 1.08)
    ax_norm.legend(loc='best', frameon=False)

    fig.suptitle(r'Memory-timescale collapse of the damped tasks', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, transparent=True)
    plt.close(fig)
    print(f"Generated figure (a): {output_path}")


# ---------------------------------------------------------------------------
# Figure (b): task-blind reservoir diagnostics over the (lambda, gamma_real) plane
# ---------------------------------------------------------------------------
def _legendre(n, x):
    if n == 1:
        return x
    if n == 2:
        return 0.5 * (3.0 * x ** 2 - 1.0)
    if n == 3:
        return 0.5 * (5.0 * x ** 3 - 3.0 * x)
    raise ValueError(n)


def run_reservoir_states(lam, gamma_real, omega, gamma_imag, u,
                         num_nodes, alpha, h, seed):
    import torch
    from models import SLON as HORN

    torch.manual_seed(seed)
    model = SLON(1, num_nodes, 1, h, alpha, omega, 0.0,
                 lambda_param=lam, gamma_real=gamma_real, gamma_imag=gamma_imag)
    model.eval()
    with torch.no_grad():
        stim = torch.tensor(u, dtype=torch.float32).reshape(-1, 1, 1)
        out = model.forward(stim, record=True)
        zr = out['rec_z_real'][0].numpy()
        zi = out['rec_z_imag'][0].numpy()
    states = np.concatenate([zr, zi], axis=1)
    return states


def _capacity(G_factor, Xc, target, tt_eps=1e-12):
    from scipy.linalg import cho_solve
    t = target - target.mean()
    tt = float(t @ t)
    if tt < tt_eps:
        return 0.0
    b = Xc.T @ t
    w = cho_solve(G_factor, b)
    c = float(b @ w) / tt
    return min(max(c, 0.0), 1.0)


def reservoir_capacities(states, u, n_washout, cap_floor,
                         k_lin, k_quad, k_cubic, k_pair, pair_window):
    from scipy.linalg import cho_factor

    X = states[n_washout:]
    if not np.all(np.isfinite(X)):
        return np.nan, np.nan
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xc = (X - mu) / sd
    Xc = np.concatenate([Xc, np.ones((Xc.shape[0], 1))], axis=1)

    n_feat = Xc.shape[1]
    ridge = 1e-6 * np.trace(Xc.T @ Xc) / n_feat
    G = Xc.T @ Xc + ridge * np.eye(n_feat)
    try:
        G_factor = cho_factor(G, lower=True)
    except Exception:
        return np.nan, np.nan

    T = Xc.shape[0]
    u_full = u[n_washout:]

    def delayed(k):
        return np.concatenate([np.zeros(k), u_full[:-k]]) if k > 0 else u_full

    mc = 0.0
    for k in range(1, k_lin + 1):
        c = _capacity(G_factor, Xc, _legendre(1, delayed(k)))
        if c > cap_floor:
            mc += c

    ipc_nl = 0.0
    for k in range(1, k_quad + 1):
        c = _capacity(G_factor, Xc, _legendre(2, delayed(k)))
        if c > cap_floor:
            ipc_nl += c
    for k in range(1, k_cubic + 1):
        c = _capacity(G_factor, Xc, _legendre(3, delayed(k)))
        if c > cap_floor:
            ipc_nl += c
    for k1 in range(1, k_pair + 1):
        for k2 in range(k1 + 1, min(k1 + pair_window, k_pair) + 1):
            tgt = _legendre(1, delayed(k1)) * _legendre(1, delayed(k2))
            c = _capacity(G_factor, Xc, tgt)
            if c > cap_floor:
                ipc_nl += c

    return mc, ipc_nl


def compute_capacity_fields(lambda_vals, gamma_real_vals, omega, gamma_imag,
                            num_nodes, alpha, h, seq_len, n_washout, seed,
                            cap_floor, k_lin, k_quad, k_cubic, k_pair, pair_window):
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, size=seq_len + n_washout)

    mc_field = np.full((len(gamma_real_vals), len(lambda_vals)), np.nan)
    ipc_field = np.full((len(gamma_real_vals), len(lambda_vals)), np.nan)

    total = len(gamma_real_vals) * len(lambda_vals)
    done = 0
    for i, gr in enumerate(gamma_real_vals):
        for j, lam in enumerate(lambda_vals):
            states = run_reservoir_states(
                lam, gr, omega, gamma_imag, u, num_nodes, alpha, h, seed)
            mc, ipc = reservoir_capacities(
                states, u, n_washout, cap_floor,
                k_lin, k_quad, k_cubic, k_pair, pair_window)
            mc_field[i, j] = mc
            ipc_field[i, j] = ipc
            done += 1
        print(f"  capacity field: {done}/{total} points", end='\r')
    print()
    return mc_field, ipc_field


def task_optimum_lambda_gamma(arrays, lambda_range, gamma_range, agg='mean'):
    lam, gr, gi, om, perf = arrays
    sel = ((lam >= lambda_range[0] - 1e-9) & (lam <= lambda_range[1] + 1e-9) &
           (gr >= gamma_range[0] - 1e-9) & (gr <= gamma_range[1] + 1e-9))
    lam, gr, perf = lam[sel], gr[sel], perf[sel]
    if len(lam) == 0:
        return None
    cells = {}
    for l, g, p in zip(lam, gr, perf):
        cells.setdefault((round(l, 6), round(g, 6)), []).append(p)
    best_key, best_val = None, -np.inf
    for key, vals in cells.items():
        v = np.mean(vals) if agg == 'mean' else np.max(vals)
        if v > best_val:
            best_val, best_key = v, key
    return best_key[0], best_key[1], best_val


def _field_panel(ax, field, lambda_vals, gamma_real_vals, title, cbar_label):
    lam = np.array(lambda_vals, dtype=float)
    gr = np.array(gamma_real_vals, dtype=float)
    masked = np.ma.array(field, mask=~np.isfinite(field))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'field', [(0, 0, 0), thesis_blue, ifisc_green, thesis_red], N=256)
    cmap.set_bad((1.0, 1.0, 1.0, 1.0))

    def edges(v):
        v = np.asarray(v, dtype=float)
        mid = (v[:-1] + v[1:]) / 2.0
        return np.concatenate([[v[0] - (mid[0] - v[0])], mid,
                               [v[-1] + (v[-1] - mid[-1])]])

    im = ax.pcolormesh(edges(lam), edges(gr), masked, cmap=cmap, shading='flat')
    ax.axvline(0.0, color='white', linewidth=2.0, alpha=0.9)
    ax.axhline(0.0, color='white', linewidth=1.2, alpha=0.6, linestyle=':')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\gamma_{\mathrm{real}}$')
    ax.set_title(title)
    ax.grid(False)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    return im


def build_figure_b(mc_field, ipc_field, lambda_vals, gamma_real_vals,
                   task_optima, output_path):
    fig, (ax_mc, ax_ipc) = plt.subplots(1, 2, figsize=(16, 6.5))

    _field_panel(ax_mc, mc_field, lambda_vals, gamma_real_vals,
                 '(a) linear memory capacity', 'MC (total)')
    _field_panel(ax_ipc, ipc_field, lambda_vals, gamma_real_vals,
                 '(b) nonlinear processing capacity', r'IPC$_{\mathrm{nl}}$ (deg.\ $\geq 2$)')

    for ax in (ax_mc, ax_ipc):
        for task, opt in task_optima.items():
            if opt is None:
                continue
            l_opt, g_opt, _ = opt
            ax.scatter([l_opt], [g_opt], s=260, marker='*',
                       facecolor=TASK_COLOR[task], edgecolor='white',
                       linewidth=1.6, zorder=5, label=TASK_LABEL[task])

    handles, labels = ax_mc.get_legend_handles_labels()
    if handles:
        ax_mc.legend(handles, labels, loc='upper left', frameon=True,
                     facecolor='white', framealpha=0.7)

    fig.suptitle(r'Task-blind dynamical order parameters with task optima overlaid', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, transparent=True)
    plt.close(fig)
    print(f"Generated figure (b): {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Collapse multi-task performance onto a memory-timescale order parameter')
    parser.add_argument('--mg-json', default='grid_mg_results/grid_mg_completed_20260209_165042.json')
    parser.add_argument('--imdb-json', default='grid_search_results/grid_search_results_20251209_164804.json')
    parser.add_argument('--smnist-json', default='grid_smnist_results/grid_smnist_results_20260120_174247_1739s.json')
    parser.add_argument('--mg-fallback', nargs='*', default=['grid_mg_results/grid_mg_completed_20260203_123501.json'])
    parser.add_argument('--imdb-fallback', nargs='*', default=['grid_search_results/grid_search_results_20251215_155017.json'])
    parser.add_argument('--smnist-fallback', nargs='*', default=['grid_smnist_results/grid_smnist_results_20260216_173939.json'])
    parser.add_argument('--output-dir', default='memory_timescale_plots')
    parser.add_argument('--agg', choices=['max', 'mean'], default='mean',
                        help='how to marginalize over the non-lambda parameters in figure (a)')

    parser.add_argument('--skip-fields', action='store_true',
                        help='skip the (expensive) MC/IPC reservoir computation')
    parser.add_argument('--lambda-min', type=float, default=-0.5)
    parser.add_argument('--lambda-max', type=float, default=0.5)
    parser.add_argument('--lambda-steps', type=int, default=21)
    parser.add_argument('--gamma-real-min', type=float, default=-0.2)
    parser.add_argument('--gamma-real-max', type=float, default=0.2)
    parser.add_argument('--gamma-real-steps', type=int, default=17)
    parser.add_argument('--field-omega', type=float, default=0.15)
    parser.add_argument('--field-gamma-imag', type=float, default=0.0)
    parser.add_argument('--num-nodes', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.04)
    parser.add_argument('--h', type=float, default=1.0)
    parser.add_argument('--seq-len', type=int, default=2500)
    parser.add_argument('--washout', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cap-floor', type=float, default=0.01)
    parser.add_argument('--k-lin', type=int, default=40)
    parser.add_argument('--k-quad', type=int, default=25)
    parser.add_argument('--k-cubic', type=int, default=15)
    parser.add_argument('--k-pair', type=int, default=20)
    parser.add_argument('--pair-window', type=int, default=8)

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mg_res, _ = load_results(args.mg_json, args.mg_fallback)
    imdb_res, _ = load_results(args.imdb_json, args.imdb_fallback)
    smnist_res, _ = load_results(args.smnist_json, args.smnist_fallback)
    print(f"Loaded MG={len(mg_res)}, IMDB={len(imdb_res)}, sMNIST={len(smnist_res)} results")

    task_arrays = {
        'mg': results_to_arrays(mg_res, 'mg'),
        'imdb': results_to_arrays(imdb_res, 'imdb'),
        'smnist': results_to_arrays(smnist_res, 'smnist'),
    }

    build_figure_a(task_arrays, out_dir / 'figureA_memory_timescale_collapse.png', agg=args.agg)

    if args.skip_fields:
        print("Skipping MC/IPC field computation (--skip-fields).")
        return

    lambda_vals = np.linspace(args.lambda_min, args.lambda_max, args.lambda_steps)
    gamma_real_vals = np.linspace(args.gamma_real_min, args.gamma_real_max, args.gamma_real_steps)

    print("Computing task-blind capacity fields (MC, IPC)...")
    mc_field, ipc_field = compute_capacity_fields(
        lambda_vals, gamma_real_vals, args.field_omega, args.field_gamma_imag,
        args.num_nodes, args.alpha, args.h, args.seq_len, args.washout, args.seed,
        args.cap_floor, args.k_lin, args.k_quad, args.k_cubic, args.k_pair, args.pair_window)

    lambda_range = (args.lambda_min, args.lambda_max)
    gamma_range = (args.gamma_real_min, args.gamma_real_max)
    task_optima = {
        task: task_optimum_lambda_gamma(task_arrays[task], lambda_range, gamma_range, agg='mean')
        for task in ('imdb', 'mg', 'smnist')
    }
    for task, opt in task_optima.items():
        if opt is not None:
            print(f"  {TASK_LABEL[task]} optimum: lambda={opt[0]:.3f}, "
                  f"gamma_real={opt[1]:.3f}, perf={opt[2]:.2f}")

    np.savez(out_dir / 'capacity_fields.npz',
             mc_field=mc_field, ipc_field=ipc_field,
             lambda_vals=lambda_vals, gamma_real_vals=gamma_real_vals)

    build_figure_b(mc_field, ipc_field, lambda_vals, gamma_real_vals,
                   task_optima, out_dir / 'figureB_capacity_fields_with_optima.png')


if __name__ == '__main__':
    main()
