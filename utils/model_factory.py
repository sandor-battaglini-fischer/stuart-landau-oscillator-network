import math

DYNAMICS_CHOICES = ("sl", "lo", "dho")


def dho_params_matching_linear_sl(omega, gamma, lambda_param=None):
    """Map linear SL eigenvalues λ ± iω onto HORN's under/overdamped form.

    HORN: ẍ + 2γẋ + Ω²x = F has poles -γ ± sqrt(γ² - Ω²).
    Choosing γ = -λ and Ω = sqrt(ω² + γ²) yields poles λ ± iω.
    If lambda_param is None, pass gamma/omega through unchanged.
    """
    if lambda_param is None:
        return float(omega), float(gamma)
    lam = float(lambda_param)
    dho_gamma = -lam
    om = float(omega)
    dho_omega = math.sqrt(om * om + dho_gamma * dho_gamma)
    return dho_omega, dho_gamma


def build_oscillator(
    dynamics,
    num_input,
    num_nodes,
    num_output,
    h,
    alpha,
    omega,
    gamma,
    *,
    lambda_param=None,
    gamma_real=None,
    gamma_imag=None,
    use_tanh=True,
):
    dynamics = dynamics.lower()
    if dynamics not in DYNAMICS_CHOICES:
        raise ValueError(f"dynamics must be one of {DYNAMICS_CHOICES}, got {dynamics!r}")

    if dynamics == "dho":
        from models.horn import HORN

        dho_omega, dho_gamma = dho_params_matching_linear_sl(omega, gamma, lambda_param)
        return HORN(
            num_input,
            num_nodes,
            num_output,
            h,
            alpha,
            dho_omega,
            dho_gamma,
            use_tanh=use_tanh,
        )

    from models.stuart_landau import SLON

    return SLON(
        num_input,
        num_nodes,
        num_output,
        h,
        alpha,
        omega,
        gamma,
        lambda_param=lambda_param,
        gamma_real=gamma_real,
        gamma_imag=gamma_imag,
        use_tanh=use_tanh,
        linear_dynamics=(dynamics == "lo"),
    )


def manifold_dynamics_type(dynamics):
    return "dho" if dynamics == "dho" else "sl"
