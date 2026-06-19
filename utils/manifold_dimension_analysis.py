import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
try:
    from scipy.spatial.distance import pdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def collect_hidden_states(data_loader, model, dynamics_type, max_samples=10000, batch_size_test=64, is_imdb=False):
    model.eval()
    all_states = []
    all_timesteps = []
    sample_count = 0

    with torch.no_grad():
        for batch in data_loader:
            if sample_count >= max_samples:
                break

            if is_imdb:
                if len(batch) >= 2:
                    inputs = batch[0]
                else:
                    continue
                out = model(inputs, record=True)
            else:
                if len(batch) >= 2:
                    inputs = batch[0]
                else:
                    continue

                if len(inputs.shape) == 4:
                    b, c, h, w = inputs.shape
                    inputs = inputs.view(b, c, h * w)
                    inputs = inputs.permute(2, 0, 1)
                elif len(inputs.shape) == 3:
                    inputs = inputs.permute(1, 0, 2)
                elif len(inputs.shape) == 2:
                    inputs = inputs.unsqueeze(-1).permute(1, 0, 2)

                out = model(inputs, record=True)

            if dynamics_type == "dho":
                x_t = out.get('rec_x_t', None)
                y_t = out.get('rec_y_t', None)
                if x_t is not None and y_t is not None:
                    batch_size, num_timesteps, _ = x_t.shape
                    for b in range(batch_size):
                        for t in range(num_timesteps):
                            if sample_count >= max_samples:
                                break
                            state = torch.cat([x_t[b, t, :], y_t[b, t, :]], dim=0)
                            all_states.append(state.detach().cpu().numpy())
                            all_timesteps.append(t)
                            sample_count += 1
            else:
                z_real = out.get('rec_z_real', None)
                z_imag = out.get('rec_z_imag', None)
                if z_real is not None and z_imag is not None:
                    batch_size, num_timesteps, _ = z_real.shape
                    for b in range(batch_size):
                        for t in range(num_timesteps):
                            if sample_count >= max_samples:
                                break
                            state = torch.cat([z_real[b, t, :], z_imag[b, t, :]], dim=0)
                            all_states.append(state.detach().cpu().numpy())
                            all_timesteps.append(t)
                            sample_count += 1

            if sample_count >= max_samples:
                break

    if not all_states:
        return None, None

    return np.array(all_states), np.array(all_timesteps)


def compute_pca_dimension(hidden_states, variance_threshold=0.95):
    if hidden_states is None or len(hidden_states) == 0:
        return None, None, None

    if not HAS_SKLEARN:
        print("Warning: sklearn not available, cannot compute PCA dimension")
        return None, None, None

    pca = PCA()
    pca.fit(hidden_states)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    effective_dim = np.argmax(explained_variance >= variance_threshold) + 1
    return effective_dim, explained_variance, pca


def compute_correlation_dimension(hidden_states, r_min=None, r_max=None, n_r=50, n_samples_max=5000):
    if hidden_states is None or len(hidden_states) == 0:
        return None, None, None, None, None

    if not HAS_SCIPY:
        print("Warning: scipy not available, cannot compute correlation dimension")
        return None, None, None, None, None

    if len(hidden_states) > n_samples_max:
        indices = np.random.choice(len(hidden_states), n_samples_max, replace=False)
        hidden_states = hidden_states[indices]

    distances = pdist(hidden_states, metric='euclidean')

    if r_min is None:
        r_min = np.percentile(distances, 1)
    if r_max is None:
        r_max = np.percentile(distances, 50)

    r_values = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    c_r_values = []
    for r in tqdm(r_values, desc="Computing correlation dimension", leave=False):
        c_r = np.sum(distances < r) / (len(distances) + 1e-10)
        c_r_values.append(c_r)

    c_r_values = np.array(c_r_values)
    log_r = np.log(r_values + 1e-10)
    log_c_r = np.log(c_r_values + 1e-10)

    valid_idx = (log_c_r > -np.inf) & (log_r > -np.inf) & (c_r_values > 0)
    if np.sum(valid_idx) < 5:
        return None, r_values, c_r_values, log_r, log_c_r

    log_r_valid = log_r[valid_idx]
    log_c_r_valid = log_c_r[valid_idx]
    if len(log_r_valid) < 2:
        correlation_dim = None
    else:
        slope, _ = np.polyfit(log_r_valid, log_c_r_valid, 1)
        correlation_dim = slope

    return correlation_dim, r_values, c_r_values, log_r, log_c_r


def plot_pca_analysis(explained_variance, effective_dim, output_dir, epoch=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes

    n_components = len(explained_variance)
    components = np.arange(1, n_components + 1)

    ax1.plot(components, explained_variance, 'b-', linewidth=2, label='Cumulative explained variance')
    ax1.axhline(0.95, color='r', linestyle='--', linewidth=1.5, label='95% threshold')
    ax1.axvline(effective_dim, color='g', linestyle='--', linewidth=1.5,
                label=f'Effective dim: {effective_dim}')
    ax1.set_xlabel('Number of components')
    ax1.set_ylabel('Cumulative explained variance')
    ax1.set_title('PCA: Explained Variance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, min(n_components, 50))

    explained_variance_ratio = np.diff(np.concatenate([[0], explained_variance]))
    ax2.bar(components[:min(20, n_components)],
            explained_variance_ratio[:min(20, n_components)],
            alpha=0.7, color='steelblue')
    ax2.set_xlabel('Component number')
    ax2.set_ylabel('Explained variance ratio')
    ax2.set_title('PCA: Individual Component Contributions')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    filename = f"{output_dir}/pca_analysis_epoch{epoch:02d}.png" if epoch is not None else f"{output_dir}/pca_analysis.png"
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_state_space(hidden_states, pca, output_dir, epoch=None, max_points=5000, timesteps=None):
    if hidden_states is None or len(hidden_states) == 0:
        return

    data = hidden_states
    n_samples, state_dim = data.shape

    if n_samples > max_points:
        idx = np.random.choice(n_samples, max_points, replace=False)
        data = data[idx]
        if timesteps is not None:
            timesteps = timesteps[idx]

    if pca is not None and hasattr(pca, "components_"):
        projected = pca.transform(data)
        if projected.shape[1] >= 3:
            x, y, z = projected[:, 0], projected[:, 1], projected[:, 2]
            x_label, y_label, z_label = "PC1", "PC2", "PC3"
        elif projected.shape[1] >= 2:
            x, y, z = projected[:, 0], projected[:, 1], None
            x_label, y_label, z_label = "PC1", "PC2", None
        else:
            x, y, z = projected[:, 0], np.zeros_like(projected[:, 0]), None
            x_label, y_label, z_label = "PC1", "0", None
    else:
        if state_dim >= 3:
            x, y, z = data[:, 0], data[:, 1], data[:, 2]
            x_label, y_label, z_label = "state[0]", "state[1]", "state[2]"
        elif state_dim >= 2:
            x, y, z = data[:, 0], data[:, 1], None
            x_label, y_label, z_label = "state[0]", "state[1]", None
        else:
            x, y, z = data[:, 0], np.zeros_like(data[:, 0]), None
            x_label, y_label, z_label = "state[0]", "0", None

    colors = timesteps if timesteps is not None else None
    cmap = plt.cm.viridis if colors is not None else None

    fig, ax = plt.subplots(figsize=(8, 6))
    if colors is not None:
        scatter = ax.scatter(x, y, c=colors, s=5, alpha=0.6, edgecolors="none", cmap=cmap)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Timestep")
    else:
        ax.scatter(x, y, s=5, alpha=0.4, edgecolors="none")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("State Space 2D (PCA projection)" + (f" - Epoch {epoch}" if epoch is not None else ""))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    filename_2d = f"{output_dir}/state_space_2d_epoch{epoch:02d}.png" if epoch is not None else f"{output_dir}/state_space_2d.png"
    fig.savefig(filename_2d, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if z is not None:
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            if colors is not None:
                scatter = ax.scatter(x, y, z, c=colors, s=5, alpha=0.6, edgecolors="none", cmap=cmap)
                cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
                cbar.set_label("Timestep")
            else:
                ax.scatter(x, y, z, s=5, alpha=0.4, edgecolors="none")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel(z_label)
            ax.set_title("State Space 3D (PCA projection)" + (f" - Epoch {epoch}" if epoch is not None else ""))
            fig.tight_layout()
            filename_3d = f"{output_dir}/state_space_3d_epoch{epoch:02d}.png" if epoch is not None else f"{output_dir}/state_space_3d.png"
            fig.savefig(filename_3d, dpi=150, bbox_inches='tight')
            plt.close(fig)
        except ImportError:
            pass


def create_state_space_animation(hidden_states, pca, output_dir, timesteps=None, max_points=10000,
                                  frames=100, duration=0.05):
    if hidden_states is None or len(hidden_states) == 0:
        return False

    try:
        import imageio
        HAS_IMAGEIO = True
    except ImportError:
        try:
            from PIL import Image
            HAS_PIL = True
            HAS_IMAGEIO = False
        except ImportError:
            return False

    data = hidden_states
    n_samples = len(data)

    if n_samples > max_points:
        idx = np.random.choice(n_samples, max_points, replace=False)
        data = data[idx]
        if timesteps is not None:
            timesteps = timesteps[idx]
        n_samples = max_points

    if timesteps is not None:
        sort_indices = np.argsort(timesteps)
        data = data[sort_indices]
        timesteps = timesteps[sort_indices]

    if pca is not None and hasattr(pca, "components_"):
        projected = pca.transform(data)
        if projected.shape[1] >= 3:
            x, y, z = projected[:, 0], projected[:, 1], projected[:, 2]
            x_label, y_label, z_label = "PC1", "PC2", "PC3"
            has_3d = True
        elif projected.shape[1] >= 2:
            x, y, z = projected[:, 0], projected[:, 1], None
            x_label, y_label, z_label = "PC1", "PC2", None
            has_3d = False
        else:
            return False
    else:
        if data.shape[1] >= 3:
            x, y, z = data[:, 0], data[:, 1], data[:, 2]
            x_label, y_label, z_label = "state[0]", "state[1]", "state[2]"
            has_3d = True
        elif data.shape[1] >= 2:
            x, y, z = data[:, 0], data[:, 1], None
            x_label, y_label, z_label = "state[0]", "state[1]", None
            has_3d = False
        else:
            return False

    colors = timesteps if timesteps is not None else None
    cmap = plt.cm.viridis if colors is not None else None
    points_per_frame = max(1, n_samples // frames)
    frame_files = []

    print(f"Creating {frames} animation frames...")
    for frame_idx in tqdm(range(frames), desc="Generating animation frames", leave=False):
        n_points = min((frame_idx + 1) * points_per_frame, n_samples)
        if n_points == 0:
            continue

        x_frame = x[:n_points]
        y_frame = y[:n_points]
        colors_frame = colors[:n_points] if colors is not None else None

        fig, ax = plt.subplots(figsize=(8, 6))
        if colors_frame is not None:
            scatter = ax.scatter(x_frame, y_frame, c=colors_frame, s=5, alpha=0.6,
                               edgecolors="none", cmap=cmap, vmin=colors.min(), vmax=colors.max())
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Timestep")
        else:
            ax.scatter(x_frame, y_frame, s=5, alpha=0.6, edgecolors="none")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"State Space 2D - {n_points}/{n_samples} points")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        fig.tight_layout()

        frame_file = f"{output_dir}/state_space_anim_frame_{frame_idx:04d}.png"
        fig.savefig(frame_file, dpi=100, bbox_inches='tight')
        plt.close(fig)
        frame_files.append(frame_file)

    try:
        if HAS_IMAGEIO:
            try:
                import imageio.v2 as imageio_v2
                images = [imageio_v2.imread(f) for f in frame_files]
                imageio_v2.mimsave(f"{output_dir}/state_space_2d_animation.gif", images, duration=duration, loop=0)
            except ImportError:
                images = [imageio.imread(f) for f in frame_files]
                imageio.mimsave(f"{output_dir}/state_space_2d_animation.gif", images, duration=duration, loop=0)
        else:
            pil_images = [Image.open(f) for f in frame_files]
            pil_images[0].save(
                f"{output_dir}/state_space_2d_animation.gif",
                save_all=True,
                append_images=pil_images[1:],
                duration=int(duration * 1000),
                loop=0
            )
        print(f"Created 2D animation: {output_dir}/state_space_2d_animation.gif")
    except Exception as e:
        print(f"Warning: Failed to create 2D animation: {e}")

    for f in frame_files:
        try:
            os.remove(f)
        except OSError:
            pass

    if has_3d and z is not None:
        try:
            frame_files_3d = []
            for frame_idx in tqdm(range(frames), desc="Generating 3D animation frames", leave=False):
                n_points = min((frame_idx + 1) * points_per_frame, n_samples)
                if n_points == 0:
                    continue

                x_frame = x[:n_points]
                y_frame = y[:n_points]
                z_frame = z[:n_points]
                colors_frame = colors[:n_points] if colors is not None else None

                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                if colors_frame is not None:
                    scatter = ax.scatter(x_frame, y_frame, z_frame, c=colors_frame, s=5, alpha=0.6,
                                       edgecolors="none", cmap=cmap, vmin=colors.min(), vmax=colors.max())
                    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
                    cbar.set_label("Timestep")
                else:
                    ax.scatter(x_frame, y_frame, z_frame, s=5, alpha=0.6, edgecolors="none")
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_zlabel(z_label)
                ax.set_title(f"State Space 3D - {n_points}/{n_samples} points")
                ax.set_xlim(x.min(), x.max())
                ax.set_ylim(y.min(), y.max())
                ax.set_zlim(z.min(), z.max())
                fig.tight_layout()

                frame_file = f"{output_dir}/state_space_3d_anim_frame_{frame_idx:04d}.png"
                fig.savefig(frame_file, dpi=100, bbox_inches='tight')
                plt.close(fig)
                frame_files_3d.append(frame_file)

            try:
                if HAS_IMAGEIO:
                    try:
                        import imageio.v2 as imageio_v2
                        images_3d = [imageio_v2.imread(f) for f in frame_files_3d]
                        imageio_v2.mimsave(f"{output_dir}/state_space_3d_animation.gif", images_3d, duration=duration, loop=0)
                    except ImportError:
                        images_3d = [imageio.imread(f) for f in frame_files_3d]
                        imageio.mimsave(f"{output_dir}/state_space_3d_animation.gif", images_3d, duration=duration, loop=0)
                else:
                    pil_images_3d = [Image.open(f) for f in frame_files_3d]
                    pil_images_3d[0].save(
                        f"{output_dir}/state_space_3d_animation.gif",
                        save_all=True,
                        append_images=pil_images_3d[1:],
                        duration=int(duration * 1000),
                        loop=0
                    )
                print(f"Created 3D animation: {output_dir}/state_space_3d_animation.gif")
            except Exception as e:
                print(f"Warning: Failed to create 3D animation: {e}")

            for f in frame_files_3d:
                try:
                    os.remove(f)
                except OSError:
                    pass
        except ImportError:
            pass

    return True


def plot_correlation_dimension(log_r, log_c_r, correlation_dim, output_dir, epoch=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes

    valid_idx = (log_c_r > -np.inf) & (log_r > -np.inf)
    log_r_valid = log_r[valid_idx]
    log_c_r_valid = log_c_r[valid_idx]

    ax1.scatter(log_r_valid, log_c_r_valid, alpha=0.6, s=20, color='steelblue')
    if correlation_dim is not None and len(log_r_valid) >= 2:
        slope, intercept = np.polyfit(log_r_valid, log_c_r_valid, 1)
        fit_line = slope * log_r_valid + intercept
        ax1.plot(log_r_valid, fit_line, 'r--', linewidth=2,
                label=f'Fit: D = {correlation_dim:.3f}')
        ax1.legend()
    ax1.set_xlabel('log(r)')
    ax1.set_ylabel('log(C(r))')
    ax1.set_title('Correlation Dimension: log-log plot')
    ax1.grid(True, alpha=0.3)

    r_values = np.exp(log_r)
    c_r_values = np.exp(log_c_r)
    valid_idx_r = (c_r_values > 0) & (r_values > 0)
    ax2.loglog(r_values[valid_idx_r], c_r_values[valid_idx_r], 'o-',
               alpha=0.6, markersize=4, color='steelblue')
    ax2.set_xlabel('r (distance)')
    ax2.set_ylabel('C(r) (correlation function)')
    ax2.set_title('Correlation Dimension: log-log scale')
    ax2.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    filename = f"{output_dir}/correlation_dimension_epoch{epoch:02d}.png" if epoch is not None else f"{output_dir}/correlation_dimension.png"
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def compute_lyapunov_exponents(data_loader, model, dynamics_type, output_dir,
                                epoch=None, batch_size_test=64, n_trajectories=10,
                                n_steps=50, perturbation_size=1e-6, is_imdb=False):
    model.eval()
    reference_data = []
    sample_count = 0

    with torch.no_grad():
        for batch in data_loader:
            if sample_count >= n_trajectories:
                break

            if is_imdb:
                if len(batch) >= 2:
                    inputs = batch[0]
                else:
                    continue
            else:
                if len(batch) >= 2:
                    inputs = batch[0]
                else:
                    continue

                if len(inputs.shape) == 4:
                    b, c, h, w = inputs.shape
                    inputs = inputs.view(b, c, h * w)
                    inputs = inputs.permute(2, 0, 1)
                elif len(inputs.shape) == 3:
                    inputs = inputs.permute(1, 0, 2)
                elif len(inputs.shape) == 2:
                    inputs = inputs.unsqueeze(-1).permute(1, 0, 2)

            reference_data.append(inputs)
            sample_count += 1
            if sample_count >= n_trajectories:
                break

    if len(reference_data) == 0:
        print("Warning: No reference trajectories collected for Lyapunov analysis")
        return None, None

    print(f"Computing Lyapunov exponents from {len(reference_data)} trajectories...")
    all_lyap_exponents = []
    divergence_rates = []

    for ref_inputs in reference_data:
        with torch.no_grad():
            out_ref = model(ref_inputs, random_init=None, record=True)
            if dynamics_type == "dho":
                x_ref = out_ref.get('rec_x_t', None)
                y_ref = out_ref.get('rec_y_t', None)
                if x_ref is None or y_ref is None:
                    continue
                num_timesteps = min(n_steps, x_ref.shape[1])
            else:
                z_real_ref = out_ref.get('rec_z_real', None)
                z_imag_ref = out_ref.get('rec_z_imag', None)
                if z_real_ref is None or z_imag_ref is None:
                    continue
                num_timesteps = min(n_steps, z_real_ref.shape[1])

            out_pert = model(ref_inputs, random_init=perturbation_size, record=True)
            if dynamics_type == "dho":
                x_pert = out_pert.get('rec_x_t', None)
                y_pert = out_pert.get('rec_y_t', None)
                if x_pert is None or y_pert is None:
                    continue
            else:
                z_real_pert = out_pert.get('rec_z_real', None)
                z_imag_pert = out_pert.get('rec_z_imag', None)
                if z_real_pert is None or z_imag_pert is None:
                    continue

        if dynamics_type == "dho":
            ref_state_0 = torch.cat([x_ref[0, 0, :], y_ref[0, 0, :]], dim=0)
            pert_state_0 = torch.cat([x_pert[0, 0, :], y_pert[0, 0, :]], dim=0)
        else:
            ref_state_0 = torch.cat([z_real_ref[0, 0, :], z_imag_ref[0, 0, :]], dim=0)
            pert_state_0 = torch.cat([z_real_pert[0, 0, :], z_imag_pert[0, 0, :]], dim=0)

        initial_distance = torch.norm(pert_state_0 - ref_state_0).item()
        if initial_distance < 1e-10:
            continue

        divergences = []
        log_div_sum = 0.0
        n_valid_steps = 0

        for step in range(1, num_timesteps):
            if dynamics_type == "dho":
                ref_state = torch.cat([x_ref[0, step, :], y_ref[0, step, :]], dim=0)
                pert_state = torch.cat([x_pert[0, step, :], y_pert[0, step, :]], dim=0)
            else:
                ref_state = torch.cat([z_real_ref[0, step, :], z_imag_ref[0, step, :]], dim=0)
                pert_state = torch.cat([z_real_pert[0, step, :], z_imag_pert[0, step, :]], dim=0)

            distance = torch.norm(pert_state - ref_state).item()
            if distance > 0:
                divergences.append(distance)
                log_div_sum += np.log(distance / initial_distance + 1e-10)
                n_valid_steps += 1

        if n_valid_steps > 10 and len(divergences) > 10:
            all_lyap_exponents.append(log_div_sum / n_valid_steps)
            divergence_rates.append(divergences)

    if len(all_lyap_exponents) == 0:
        print("Warning: Could not compute Lyapunov exponents")
        return None, None

    lyapunov_exponents = np.array(all_lyap_exponents)
    largest_lyap = np.mean(lyapunov_exponents)
    std_lyap = np.std(lyapunov_exponents)

    print(f"Largest Lyapunov exponent: {largest_lyap:.6f} ± {std_lyap:.6f}")
    plot_lyapunov_exponents(lyapunov_exponents, divergence_rates, output_dir, epoch=epoch)

    results = {
        'largest_lyapunov_exponent': float(largest_lyap),
        'lyapunov_std': float(std_lyap),
        'n_trajectories': len(all_lyap_exponents),
        'n_steps': n_steps,
        'is_chaotic': bool(largest_lyap > 0.01),
    }
    return lyapunov_exponents, results


def plot_lyapunov_exponents(lyapunov_exponents, divergence_rates, output_dir, epoch=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes

    ax1.hist(lyapunov_exponents, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(np.mean(lyapunov_exponents), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(lyapunov_exponents):.6f}')
    ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Zero (neutral)')
    ax1.set_xlabel('Lyapunov Exponent')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Largest Lyapunov Exponents')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if divergence_rates:
        n_show = min(5, len(divergence_rates))
        for i in range(n_show):
            if len(divergence_rates[i]) > 0:
                times = np.arange(len(divergence_rates[i]))
                ax2.semilogy(times, divergence_rates[i], alpha=0.6, linewidth=1.5,
                           label=f'Trajectory {i+1} ($\\lambda$={lyapunov_exponents[i]:.4f})')
        if len(divergence_rates[0]) > 0:
            times = np.arange(len(divergence_rates[0]))
            mean_lyap = np.mean(lyapunov_exponents)
            exp_fit = divergence_rates[0][0] * np.exp(mean_lyap * times)
            ax2.semilogy(times, exp_fit, 'r--', linewidth=2, alpha=0.8,
                        label=f'Exponential fit ($\\lambda$={mean_lyap:.4f})')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Distance (log scale)')
        ax2.set_title('Trajectory Divergence Over Time')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    filename = f"{output_dir}/lyapunov_exponents_epoch{epoch:02d}.png" if epoch is not None else f"{output_dir}/lyapunov_exponents.png"
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def analyze_manifold_dimension(data_loader, model, dynamics_type, output_dir, epoch=None,
                               batch_size_test=64, max_samples=10000,
                               variance_threshold=0.95, is_imdb=False):
    print("Collecting hidden states...")
    hidden_states, timesteps = collect_hidden_states(
        data_loader, model, dynamics_type, max_samples=max_samples,
        batch_size_test=batch_size_test, is_imdb=is_imdb
    )

    if hidden_states is None:
        print("Warning: No hidden states collected")
        return None

    print(f"Collected {len(hidden_states)} hidden states of dimension {hidden_states.shape[1]}")

    print("Computing PCA dimension...")
    effective_dim, explained_variance, pca = compute_pca_dimension(
        hidden_states, variance_threshold=variance_threshold
    )

    print("Computing correlation dimension...")
    correlation_dim, _, _, log_r, log_c_r = compute_correlation_dimension(
        hidden_states, n_samples_max=min(5000, len(hidden_states))
    )

    print("Generating plots...")
    if explained_variance is not None and effective_dim is not None:
        plot_pca_analysis(explained_variance, effective_dim, output_dir, epoch=epoch)
        plot_state_space(hidden_states, pca, output_dir, epoch=epoch, timesteps=timesteps)

    if log_r is not None and log_c_r is not None:
        plot_correlation_dimension(log_r, log_c_r, correlation_dim, output_dir, epoch=epoch)

    print("Computing Lyapunov exponents...")
    _, lyap_results = compute_lyapunov_exponents(
        data_loader, model, dynamics_type, output_dir,
        epoch=epoch, batch_size_test=batch_size_test,
        n_trajectories=20, n_steps=50, is_imdb=is_imdb
    )

    results = {
        'effective_dim_pca': int(effective_dim) if effective_dim is not None else None,
        'correlation_dim': float(correlation_dim) if correlation_dim is not None else None,
        'n_samples': int(len(hidden_states)),
        'state_dim': int(hidden_states.shape[1]),
        'explained_variance_95': float(explained_variance[effective_dim - 1]) if effective_dim is not None and effective_dim > 0 else None,
    }
    if lyap_results is not None:
        results.update(lyap_results)

    print(f"PCA effective dimension: {effective_dim}")
    print(f"Correlation dimension: {correlation_dim}")
    if lyap_results is not None:
        print(f"Largest Lyapunov exponent: {lyap_results.get('largest_lyapunov_exponent', 'N/A')}")

    return results


def collect_and_save_final_states(data_loader, model, dynamics_type, output_dir,
                                   batch_size_test=64, max_samples=50000, is_imdb=False):
    print(f"\nCollecting final hidden states (max {max_samples} samples)...")
    hidden_states, timesteps = collect_hidden_states(
        data_loader, model, dynamics_type, max_samples=max_samples,
        batch_size_test=batch_size_test, is_imdb=is_imdb
    )

    if hidden_states is None:
        print("Warning: No hidden states collected")
        return None

    print(f"Collected {len(hidden_states)} hidden states of dimension {hidden_states.shape[1]}")

    states_file = f"{output_dir}/final_hidden_states.npz"
    if timesteps is not None:
        np.savez_compressed(states_file, hidden_states=hidden_states, timesteps=timesteps)
    else:
        np.savez_compressed(states_file, hidden_states=hidden_states)
    print(f"Saved hidden states to {states_file}")

    print("Computing PCA for animation...")
    _, _, pca = compute_pca_dimension(hidden_states, variance_threshold=0.95)

    if pca is not None:
        print("Creating state space animation...")
        create_state_space_animation(
            hidden_states, pca, output_dir, timesteps=timesteps,
            max_points=min(20000, len(hidden_states)), frames=150, duration=0.03
        )
    else:
        print("Warning: Could not compute PCA for animation")

    return states_file
