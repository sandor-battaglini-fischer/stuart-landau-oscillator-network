# Analysis utilities for SLON model
import numpy as np
import torch


def extract_model_parameters(model, dynamics_type):
    """Extract all trainable parameters from the model."""
    core = model.slon if hasattr(model, "slon") else model

    params = {
        "i2h_weight": core.i2h.weight.detach().cpu().numpy().tolist(),
        "i2h_bias": core.i2h.bias.detach().cpu().numpy().tolist(),
        "h2h_weight": core.h2h.weight.detach().cpu().numpy().tolist(),
        "h2h_bias": core.h2h.bias.detach().cpu().numpy().tolist(),
        "h2o_weight": core.h2o.weight.detach().cpu().numpy().tolist(),
        "h2o_bias": core.h2o.bias.detach().cpu().numpy().tolist(),
    }

    if dynamics_type in ("sl", "lo"):
        params["lambda_param"] = core.lambda_param.detach().cpu().numpy().tolist() if isinstance(core.lambda_param, torch.Tensor) else [core.lambda_param]
        params["omega_param"] = core.omega_param.detach().cpu().numpy().tolist() if isinstance(core.omega_param, torch.Tensor) else [core.omega_param]
        params["gamma_real"] = core.gamma_real.detach().cpu().numpy().tolist() if isinstance(core.gamma_real, torch.Tensor) else [core.gamma_real]
        params["gamma_imag"] = core.gamma_imag.detach().cpu().numpy().tolist() if isinstance(core.gamma_imag, torch.Tensor) else [core.gamma_imag]
    elif dynamics_type == "dho":
        params["omega"] = float(core.omega)
        params["gamma"] = float(core.gamma)

    params["h"] = float(core.h)
    params["alpha"] = float(core.alpha)

    if hasattr(model, "embedding"):
        params["embedding_weight"] = model.embedding.weight.detach().cpu().numpy().tolist()

    return params


def compute_parameter_statistics(params):
    """Compute statistics for parameter tracking."""
    stats = {}

    for key, value in params.items():
        if key in ["h", "alpha"]:
            stats[key] = value
        elif isinstance(value, list):
            arr = np.array(value)
            if arr.size > 0:
                stats[f"{key}_mean"] = float(np.mean(arr))
                stats[f"{key}_std"] = float(np.std(arr))
                stats[f"{key}_min"] = float(np.min(arr))
                stats[f"{key}_max"] = float(np.max(arr))
                stats[f"{key}_abs_mean"] = float(np.mean(np.abs(arr)))
                if arr.size > 1:
                    if arr.ndim == 1:
                        stats[f"{key}_norm"] = float(np.linalg.norm(arr))
                    else:
                        stats[f"{key}_norm"] = float(np.linalg.norm(arr, ord="fro"))

    return stats
