import numpy as np


def as_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def finite_floats(values):
    return [as_float(v) for v in values if np.isfinite(as_float(v))]


def set_epoch_xlim(ax, n_points):
    if n_points <= 1:
        ax.set_xlim(-0.5, 0.5)
    else:
        ax.set_xlim(0, n_points - 1)


def set_value_ylim(ax, values, default=(0.0, 1.0), pad_frac=0.1, clamp_min=None, clamp_max=None):
    finite = finite_floats(values)
    if not finite:
        ax.set_ylim(*default)
        return

    vmin = min(finite)
    vmax = max(finite)
    if vmax <= vmin:
        padding = abs(vmax) * pad_frac if vmax != 0 else default[1] * pad_frac
    else:
        padding = (vmax - vmin) * pad_frac

    low = vmin - padding
    high = vmax + padding
    if clamp_min is not None:
        low = max(clamp_min, low)
    if clamp_max is not None:
        high = min(clamp_max, high)

    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        ax.set_ylim(*default)
    else:
        ax.set_ylim(low, high)
