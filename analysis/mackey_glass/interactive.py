import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
from scipy.signal import find_peaks
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.plotting_utils.style import apply_style, thesis_red, thesis_blue, ifisc_green, mycmap
apply_style()


def generate_mackey_glass(
    length,
    tau=17.0,
    delta_t=1.0,
    beta=0.2,
    gamma=0.1,
    n=10,
    x0=1.2,
):
    delay_steps = int(tau / delta_t)
    total_len = length + delay_steps + 1
    x = np.zeros(total_len, dtype=np.float64)
    x[: delay_steps + 1] = x0

    for t in range(delay_steps, total_len - 1):
        x_tau = x[t - delay_steps]
        dx = beta * x_tau / (1.0 + x_tau**n) - gamma * x[t]
        x[t + 1] = x[t] + delta_t * dx

    return x[delay_steps + 1 :]


def main():
    series_length = 3000

    init_tau = 17.0
    init_delta_t = 1.0
    init_beta = 0.2
    init_gamma = 0.1
    init_n = 10.0
    init_x0 = 1.2

    series = generate_mackey_glass(
        length=series_length,
        tau=init_tau,
        delta_t=init_delta_t,
        beta=init_beta,
        gamma=init_gamma,
        n=init_n,
        x0=init_x0,
    )
    t = np.arange(len(series)) * init_delta_t

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3, left=0.12, bottom=0.35, top=0.97)
    ax_time = fig.add_subplot(gs[0, :])
    ax_freq = fig.add_subplot(gs[1, 0])
    ax_phase = fig.add_subplot(gs[1, 1])
    ax_residual = fig.add_subplot(gs[2, :])

    time_line, = ax_time.plot(t, series, lw=1.5, color=thesis_blue)
    ax_time.set_xlabel("time")
    ax_time.set_ylabel(r"$x(t)$")
    ax_time.set_title("Mackey–Glass time series")

    delay_steps = int(init_tau / init_delta_t)
    if delay_steps < 1:
        phase_x = np.array([])
        phase_y = np.array([])
    else:
        phase_x = series[:-delay_steps]
        phase_y = series[delay_steps:]
    phase_line, = ax_phase.plot(phase_x, phase_y, lw=0.8, color=thesis_red)
    ax_phase.set_xlabel(r"$x(t-\tau)$")
    ax_phase.set_ylabel(r"$x(t)$")
    ax_phase.set_title("Delay-embedded phase space")
    ax_phase.set_aspect("equal", adjustable="box")

    fft_vals = np.fft.fft(series)
    freqs = np.fft.fftfreq(len(series), init_delta_t)
    power = np.abs(fft_vals) ** 2
    positive_freqs = freqs[: len(freqs) // 2]
    positive_power = power[: len(power) // 2]
    
    low_freq_mask = positive_freqs <= 0.2
    low_freq_freqs = positive_freqs[low_freq_mask]
    low_freq_power = positive_power[low_freq_mask]
    
    peaks, properties = find_peaks(low_freq_power, height=np.max(low_freq_power) * 0.05, distance=len(low_freq_power)//100)
    if len(peaks) >= 5:
        top_5_indices = peaks[np.argsort(low_freq_power[peaks])[-5:]]
    elif len(peaks) > 0:
        top_5_indices = peaks[np.argsort(low_freq_power[peaks])[-len(peaks):]]
        remaining_needed = 5 - len(top_5_indices)
        all_indices = np.arange(len(low_freq_power))
        available_indices = np.setdiff1d(all_indices, top_5_indices)
        if len(available_indices) > 0:
            additional_indices = available_indices[np.argsort(low_freq_power[available_indices])[-remaining_needed:]]
            top_5_indices = np.concatenate([top_5_indices, additional_indices])
    else:
        top_5_indices = np.argsort(low_freq_power)[-5:]
    
    top_5_freqs = low_freq_freqs[top_5_indices]
    top_5_power = low_freq_power[top_5_indices]
    
    freq_line, = ax_freq.plot(low_freq_freqs, low_freq_power, lw=1.5, color=ifisc_green)
    peak_markers = ax_freq.plot([], [], 'ro', markersize=8)[0]
    ax_freq.set_xlabel("frequency")
    ax_freq.set_ylabel("power")
    ax_freq.set_title("Frequency spectrum")
    ax_freq.set_xlim(0, 0.2)
    ax_freq.set_yscale('log')
    peak_texts = []
    peak_markers.set_data(top_5_freqs, top_5_power)
    y_max = np.max(low_freq_power)
    for i, freq in enumerate(top_5_freqs):
        ax_freq.axvline(freq, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
        text = ax_freq.text(freq, top_5_power[i] * 1.2, f'{freq:.4f}', 
                           ha='center', va='bottom', fontsize=7, color='r')
        peak_texts.append(text)
    
    legend_labels = ', '.join([f'{f:.4f}' for f in sorted(top_5_freqs)])
    ax_freq.legend([peak_markers], [f'Top 5: {legend_labels}'], fontsize=7)

    def remove_top_frequencies(series, delta_t, n_freqs=3):
        if n_freqs == 0:
            return series.copy()
        
        fft_vals = np.fft.fft(series)
        freqs = np.fft.fftfreq(len(series), delta_t)
        power = np.abs(fft_vals) ** 2
        positive_freqs = freqs[: len(freqs) // 2]
        positive_power = power[: len(power) // 2]
        
        low_freq_mask = positive_freqs <= 0.2
        low_freq_indices = np.where(low_freq_mask)[0]
        low_freq_power = positive_power[low_freq_mask]
        
        top_local_indices = np.argsort(low_freq_power)[-n_freqs:]
        top_global_indices = low_freq_indices[top_local_indices]
        
        filtered_fft = np.zeros_like(fft_vals)
        for idx in top_global_indices:
            filtered_fft[idx] = fft_vals[idx]
            if idx > 0:
                neg_idx = len(freqs) - idx
                if neg_idx < len(freqs):
                    filtered_fft[neg_idx] = fft_vals[neg_idx]
        
        filtered_series = np.real(np.fft.ifft(filtered_fft))
        residual = series - filtered_series
        return residual

    init_n_freqs_remove = 3
    residual_series = remove_top_frequencies(series, init_delta_t, n_freqs=init_n_freqs_remove)
    residual_line, = ax_residual.plot(t, residual_series, lw=1.5, color=thesis_blue)
    ax_residual.set_xlabel("time")
    ax_residual.set_ylabel(r"$x(t)$")
    residual_title = ax_residual.set_title(f"Residual time series (top {init_n_freqs_remove} frequencies removed)")

    axcolor = "lightgoldenrodyellow"
    ax_tau = plt.axes([0.12, 0.27, 0.78, 0.015], facecolor=axcolor)
    ax_beta = plt.axes([0.12, 0.24, 0.78, 0.015], facecolor=axcolor)
    ax_gamma = plt.axes([0.12, 0.21, 0.78, 0.015], facecolor=axcolor)
    ax_n = plt.axes([0.12, 0.18, 0.78, 0.015], facecolor=axcolor)
    ax_dt = plt.axes([0.12, 0.15, 0.78, 0.015], facecolor=axcolor)
    ax_x0 = plt.axes([0.12, 0.12, 0.78, 0.015], facecolor=axcolor)
    ax_n_freqs = plt.axes([0.12, 0.09, 0.78, 0.015], facecolor=axcolor)

    s_tau = Slider(ax_tau, "tau", 5.0, 30.0, valinit=init_tau, valstep=0.1)
    s_beta = Slider(ax_beta, "beta", 0.05, 1.0, valinit=init_beta, valstep=0.01)
    s_gamma = Slider(ax_gamma, "gamma", 0.01, 0.5, valinit=init_gamma, valstep=0.01)
    s_n = Slider(ax_n, "n", 5.0, 20.0, valinit=init_n, valstep=0.5)
    s_dt = Slider(ax_dt, "delta_t", 0.1, 2.0, valinit=init_delta_t, valstep=0.05)
    s_x0 = Slider(ax_x0, "x0", 0.1, 3.0, valinit=init_x0, valstep=0.05)
    s_n_freqs = Slider(ax_n_freqs, "remove freqs", 0, 20, valinit=3, valstep=1)

    def update(_):
        tau = s_tau.val
        beta = s_beta.val
        gamma = s_gamma.val
        n = s_n.val
        delta_t = s_dt.val
        x0 = s_x0.val
        n_freqs_remove = int(s_n_freqs.val)

        new_series = generate_mackey_glass(
            length=series_length,
            tau=tau,
            delta_t=delta_t,
            beta=beta,
            gamma=gamma,
            n=n,
            x0=x0,
        )
        new_t = np.arange(len(new_series)) * delta_t

        time_line.set_data(new_t, new_series)
        ax_time.relim()
        ax_time.autoscale_view()

        delay_steps_local = int(tau / delta_t)
        if delay_steps_local < 1 or delay_steps_local >= len(new_series):
            phase_line.set_data([], [])
        else:
            x_tau = new_series[:-delay_steps_local]
            x_t = new_series[delay_steps_local:]
            phase_line.set_data(x_tau, x_t)
        ax_phase.relim()
        ax_phase.autoscale_view()
        ax_phase.set_aspect("equal", adjustable="box")

        fft_vals = np.fft.fft(new_series)
        freqs = np.fft.fftfreq(len(new_series), delta_t)
        power = np.abs(fft_vals) ** 2
        positive_freqs = freqs[: len(freqs) // 2]
        positive_power = power[: len(power) // 2]
        
        low_freq_mask = positive_freqs <= 0.2
        low_freq_freqs = positive_freqs[low_freq_mask]
        low_freq_power = positive_power[low_freq_mask]
        
        peaks, properties = find_peaks(low_freq_power, height=np.max(low_freq_power) * 0.05, distance=len(low_freq_power)//100)
        if len(peaks) >= 5:
            top_5_indices = peaks[np.argsort(low_freq_power[peaks])[-5:]]
        elif len(peaks) > 0:
            top_5_indices = peaks[np.argsort(low_freq_power[peaks])[-len(peaks):]]
            remaining_needed = 5 - len(top_5_indices)
            all_indices = np.arange(len(low_freq_power))
            available_indices = np.setdiff1d(all_indices, top_5_indices)
            if len(available_indices) > 0:
                additional_indices = available_indices[np.argsort(low_freq_power[available_indices])[-remaining_needed:]]
                top_5_indices = np.concatenate([top_5_indices, additional_indices])
        else:
            top_5_indices = np.argsort(low_freq_power)[-5:]
        
        top_5_freqs = low_freq_freqs[top_5_indices]
        top_5_power = low_freq_power[top_5_indices]
        
        freq_line.set_data(low_freq_freqs, low_freq_power)
        ax_freq.relim()
        ax_freq.autoscale_view()
        ax_freq.set_xlim(0, 0.2)
        ax_freq.set_yscale('log')
        
        for line in ax_freq.lines:
            if line != freq_line and line != peak_markers:
                line.remove()
        
        for text in peak_texts:
            text.remove()
        peak_texts.clear()
        
        peak_markers.set_data(top_5_freqs, top_5_power)
        y_max = np.max(low_freq_power)
        for i, freq in enumerate(top_5_freqs):
            ax_freq.axvline(freq, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
            text = ax_freq.text(freq, top_5_power[i] * 1.2, f'{freq:.4f}', 
                               ha='center', va='bottom', fontsize=7, color='r')
            peak_texts.append(text)
        
        legend_labels = ', '.join([f'{f:.4f}' for f in sorted(top_5_freqs)])
        ax_freq.legend([peak_markers], [f'Top 5: {legend_labels}'], fontsize=7)

        residual_series = remove_top_frequencies(new_series, delta_t, n_freqs=n_freqs_remove)
        residual_line.set_data(new_t, residual_series)
        residual_title.set_text(f"Residual time series (top {n_freqs_remove} frequencies removed)")
        ax_residual.relim()
        ax_residual.autoscale_view()

        fig.canvas.draw_idle()

    for slider in (s_tau, s_beta, s_gamma, s_n, s_dt, s_x0, s_n_freqs):
        slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()

