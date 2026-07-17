import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

FS_TITLE = 15
FS_LABEL = 16
FS_REGIME = 16
FS_BOUNDARY = 12
FS_TICK = 16

INSET_W_IN = 1.05
INSET_H_IN = 0.78

TEXT_Y_BOTTOM = -1.05
INSET_Y_BOTTOM = -2.08
TEXT_Y_TOP = 1.96
INSET_Y_TOP = 0.78

def simulate(lam, omega, gr, gi, r0, T=20, dt=0.005):
    n = int(T / dt)
    x = r0 + 0j
    t = np.linspace(0, T, n)
    xs = np.zeros(n, dtype=complex)
    for i in range(n):
        xs[i] = x
        x += dt * ((lam + 1j*omega)*x + (gr + 1j*gi)*abs(x)**2 * x)
        if abs(x) > 50:
            xs[i+1:] = np.nan
            break
    return t, xs

fig, ax = plt.subplots(figsize=(7.2, 6.0))

ax.axhspan(0, 3, xmin=0, xmax=0.5, color='0.95')
ax.axhspan(0, 3, xmin=0.5, xmax=1, color='0.6')
ax.axhspan(-3, 0, xmin=0, xmax=0.5, color='0.95')
ax.axhspan(-3, 0, xmin=0.5, xmax=1, color='0.8')

ax.axhline(0, color='k', lw=1.2)
ax.axvline(0, color='k', lw=1.2, ls='--')

ax.text(0.07, 2.75, r'$\lambda=0$', fontsize=FS_BOUNDARY, ha='left', va='top')
ax.text(1.85, 0.12, r'$\gamma_{r}=0$', fontsize=FS_BOUNDARY, ha='right', va='bottom')

text_kw = dict(ha='center', va='center', fontsize=FS_REGIME, linespacing=1.35)
ax.text(-1, TEXT_Y_BOTTOM, 'Stable origin,\nno limit cycle', **text_kw)
ax.text(1, TEXT_Y_BOTTOM, 'Supercritical,\nstable limit cycle', **text_kw)
ax.text(-1, TEXT_Y_TOP, 'Subcritical,\nunstable limit cycle', **text_kw)
ax.text(1, TEXT_Y_TOP, 'Blowup,\nno saturation', **text_kw)

ax.set_xlim(-2, 2)
ax.set_ylim(-3, 3)
ax.set_xlabel(r'$\lambda$', fontsize=FS_LABEL, labelpad=8)
ax.set_ylabel(r'$\gamma_{\mathrm{real}}$', fontsize=FS_LABEL, labelpad=8)
ax.set_title('Stuart–Landau regimes', fontsize=FS_TITLE, pad=14)
ax.tick_params(axis='both', labelsize=FS_TICK, length=5, width=0.8)

cases = [
    (-0.5, -1.0, 0.8, (-1.0, INSET_Y_BOTTOM)),
    (0.5, -1.0, 0.1, (1.0, INSET_Y_BOTTOM)),
    (-0.5, 1.0, 0.3, (-1.0, INSET_Y_TOP)),
    (0.1, 0.2, 0.3, (1.0, INSET_Y_TOP)),
]

omega, gi = 4.0, 0.0

def limit_cycle_radius(lam, gr, eps=1e-9):
    if lam * gr < -eps:
        return np.sqrt(-lam / gr)
    return None


for lam, gr, r0, anchor in cases:
    T = 18 if (lam > 0 and gr > 0) else 25
    _, xs = simulate(lam, omega, gr, gi, r0, T=T)
    inax = inset_axes(
        ax,
        INSET_W_IN,
        INSET_H_IN,
        loc='center',
        bbox_to_anchor=anchor,
        bbox_transform=ax.transData,
        borderpad=0,
    )
    finite = np.isfinite(xs.real) & np.isfinite(xs.imag)
    inax.plot(xs.real[finite], xs.imag[finite], 'k-', lw=0.7)
    inax.plot(xs.real[0], xs.imag[0], 'ko', ms=2.2, mew=0.5)
    inax.plot(0, 0, 'k+', ms=5, mew=0.6)

    r_lc = limit_cycle_radius(lam, gr)
    if r_lc is not None:
        theta = np.linspace(0, 2 * np.pi, 200)
        inax.plot(
            r_lc * np.cos(theta),
            r_lc * np.sin(theta),
            'k--',
            lw=0.45,
            alpha=0.55,
        )

    re = xs.real[finite]
    im = xs.imag[finite]
    if len(re) > 0:
        rmax = max(
            np.max(np.abs(re)),
            np.max(np.abs(im)),
            r_lc or 0.0,
            abs(r0),
            0.2,
        ) * 1.2
        inax.set_xlim(-rmax, rmax)
        inax.set_ylim(-rmax, rmax)
    inax.set_aspect('equal', adjustable='box')
    inax.set_xticks([])
    inax.set_yticks([])
    inax.set_facecolor('white')
    for sp in inax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color('0.4')
    inax.axhline(0, color='0.7', lw=0.45)
    inax.axvline(0, color='0.7', lw=0.45)

fig.subplots_adjust(left=0.11, right=0.97, top=0.90, bottom=0.10)
import os
save_path = 'results/dynamics/regime_diagram_state_space.png'
save_dir = os.path.dirname(save_path)
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
fig.savefig(save_path, dpi=400)
plt.show()