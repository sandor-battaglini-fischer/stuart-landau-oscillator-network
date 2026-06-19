r"""
Memory kernel of the linearized Stuart-Landau node.

Near the origin the driven dynamics are
        z_dot = (lambda + i*omega) z + f(t),
whose response kernel (weight of an input that arrived a time s ago) is
        K(s) = exp(lambda*s) * exp(i*omega*s),  s = t - t' >= 0
              \_____________/   \_____________/
               amplitude /        phase /
               forgetting           clock
- |K(s)| = exp(lambda*s)  -> sets the FADING-MEMORY timescale tau = 1/|lambda|
- arg K(s) = omega*s      -> pure rotation, |.|=1, carries info without forgetting

Interactive sliders: lambda, omega, S_max.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider

N = 2000
INIT_LAM = -0.08
INIT_OM = 0.30
INIT_S_MAX = 60.0

norm = Normalize(vmin=-0.32, vmax=0.06)
cmap = cm.coolwarm


def col(lam):
    return cmap(norm(lam))


def amp(lam, s):
    return np.exp(lam * s)


def kernel(lam, om, s):
    return np.exp(lam * s) * np.exp(1j * om * s)


def spiral_label(lam, om):
    if lam < 0:
        return "damped: inward spiral"
    if lam == 0:
        return r"critical $\lambda=0$: circle, no fade"
    return r"$\lambda>0$: outward (lin. unstable)"


fig, ax = plt.subplots(2, 2, figsize=(12, 9))
fig.subplots_adjust(bottom=0.22)
fig.suptitle(
    r"Memory kernel  $K(s)=e^{\lambda s}\,e^{i\omega s}$  of the linearized SL node",
    fontsize=14,
    y=0.98,
)

s = np.linspace(0, INIT_S_MAX, N)

a = ax[0, 0]
tau0 = 1.0 / abs(INIT_LAM)
amp_line, = a.plot(
    s, amp(INIT_LAM, s), color=col(INIT_LAM), lw=2,
    label=rf"$\lambda={INIT_LAM:+.2f}$  ($\tau={tau0:.1f}$)",
)
amp_marker, = a.plot([tau0], [np.exp(-1)], "o", color=col(INIT_LAM), ms=6)
a.axhline(np.exp(-1), color="k", ls=":", lw=0.8)
amp_annot = a.text(
    INIT_S_MAX * 0.62, np.exp(-1) + 0.03, r"$1/e$ : input influence faded", fontsize=8
)
a.set_xlim(0, INIT_S_MAX)
a.set_ylim(0, 1.6)
a.set_xlabel("lag  s  (steps since input)")
a.set_ylabel(r"$|K(s)| = e^{\lambda s}$")
amp_title = a.set_title(r"Amplitude channel: forgetting, $\tau_{\rm mem}=1/|\lambda|$")
amp_legend = a.legend(fontsize=8, loc="upper right")

b = ax[0, 1]
K0 = kernel(INIT_LAM, INIT_OM, s)
real_line, = b.plot(s, K0.real, lw=1.8, color="tab:blue")
env_pos, = b.plot(s, amp(INIT_LAM, s), "k--", lw=1, alpha=0.7, label=r"$\pm e^{\lambda s}$ envelope")
env_neg, = b.plot(s, -amp(INIT_LAM, s), "k--", lw=1, alpha=0.7)
b.set_xlim(0, INIT_S_MAX)
b.set_ylim(-1.1, 1.1)
b.set_xlabel("lag  s  (steps since input)")
b.set_ylabel(r"$\mathrm{Re}\,K(s)=e^{\lambda s}\cos(\omega s)$")
phase_title = b.set_title(rf"Phase channel under fixed decay ($\lambda={INIT_LAM:+.2f}$)")
b.legend(fontsize=8, loc="upper right")

c = ax[1, 0]
K_spiral = kernel(INIT_LAM, INIT_OM, s)
spiral_line, = c.plot(
    K_spiral.real, K_spiral.imag, color=col(INIT_LAM), lw=1.6,
    label=rf"$\lambda={INIT_LAM:+.2f}$, $\omega={INIT_OM:.2f}$: {spiral_label(INIT_LAM, INIT_OM)}",
)
c.plot(0, 0, "k+", ms=10)
c.plot(1, 0, "ko", ms=4)
c.text(1.02, 0.04, "s=0", fontsize=8)
c.set_aspect("equal")
c.set_xlim(-1.6, 1.6)
c.set_ylim(-1.6, 1.6)
c.set_xlabel(r"$\mathrm{Re}\,K(s)$")
c.set_ylabel(r"$\mathrm{Im}\,K(s)$")
spiral_title = c.set_title(r"Kernel in the complex plane (amplitude $\times$ phase)")
spiral_legend = c.legend(fontsize=8, loc="upper left")

d = ax[1, 1]
lam_grid = np.linspace(-0.30, -0.005, 400)
mem_cap = 1.0 / (2.0 * np.abs(lam_grid))
tau_grid = 1.0 / np.abs(lam_grid)
d.plot(lam_grid, tau_grid, lw=2, color="tab:blue", label=r"timescale $\tau=1/|\lambda|$")
d.plot(lam_grid, mem_cap, lw=2, color="tab:green", label=r"L2 memory $\int|K|^2 ds=\frac{1}{2|\lambda|}$")
lam_vline = d.axvline(INIT_LAM, color="tab:red", ls="-", lw=2)
tasks = {"IMDb": -0.08, "Mackey-Glass": -0.10}
for name, lam in tasks.items():
    d.axvline(lam, color="grey", ls=":", lw=1)
    d.text(lam, 105, name, rotation=90, va="top", ha="right", fontsize=8)
d.set_xlim(-0.30, 0)
d.set_ylim(0, 110)
d.set_xlabel(r"$\lambda$  (forgetting rate, damped side)")
d.set_ylabel("memory  (steps)")
d.set_title(r"Critical slowing down: memory $\to\infty$ as $\lambda\to0^-$")
d.legend(fontsize=8, loc="upper left")

axcolor = "lightgoldenrodyellow"
ax_lam = plt.axes([0.12, 0.14, 0.78, 0.02], facecolor=axcolor)
ax_om = plt.axes([0.12, 0.10, 0.78, 0.02], facecolor=axcolor)
ax_smax = plt.axes([0.12, 0.06, 0.78, 0.02], facecolor=axcolor)

s_lam = Slider(ax_lam, r"$\lambda$", -0.30, 0.06, valinit=INIT_LAM, valstep=0.01)
s_om = Slider(ax_om, r"$\omega$", 0.05, 1.0, valinit=INIT_OM, valstep=0.01)
s_smax = Slider(ax_smax, r"$s_{\max}$", 10.0, 120.0, valinit=INIT_S_MAX, valstep=1.0)


def update(_):
    lam = s_lam.val
    om = s_om.val
    s_max = s_smax.val
    s_new = np.linspace(0, s_max, N)
    color = col(lam)

    y_amp = amp(lam, s_new)
    amp_line.set_data(s_new, y_amp)
    amp_line.set_color(color)
    if lam < 0:
        tau = 1.0 / abs(lam)
        amp_marker.set_data([tau], [np.exp(-1)])
        amp_marker.set_color(color)
        amp_marker.set_visible(True)
        amp_legend.get_texts()[0].set_text(
            rf"$\lambda={lam:+.2f}$  ($\tau={tau:.1f}$)"
        )
    else:
        amp_marker.set_visible(False)
        amp_legend.get_texts()[0].set_text(rf"$\lambda={lam:+.2f}$")
    a.set_xlim(0, s_max)
    amp_annot.set_position((s_max * 0.62, np.exp(-1) + 0.03))

    K = kernel(lam, om, s_new)
    real_line.set_data(s_new, K.real)
    env_pos.set_data(s_new, amp(lam, s_new))
    env_neg.set_data(s_new, -amp(lam, s_new))
    period = 2 * np.pi / om
    phase_title.set_text(
        rf"Phase channel: $\lambda={lam:+.2f}$, $\omega={om:.2f}$ (period {period:.1f})"
    )
    b.set_xlim(0, s_max)

    s_spiral = np.linspace(0, min(s_max, 45), min(N, 1500))
    K_sp = kernel(lam, om, s_spiral)
    spiral_line.set_data(K_sp.real, K_sp.imag)
    spiral_line.set_color(color)
    spiral_title.set_text(r"Kernel in the complex plane (amplitude $\times$ phase)")
    spiral_legend.get_texts()[0].set_text(
        rf"$\lambda={lam:+.2f}$, $\omega={om:.2f}$: {spiral_label(lam, om)}"
    )

    lam_vline.set_xdata([lam, lam])

    fig.canvas.draw_idle()


s_lam.on_changed(update)
s_om.on_changed(update)
s_smax.on_changed(update)
update(None)

plt.show()
