DYNAMICS_CHOICES = ("sl", "lo", "dho")


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

        return HORN(
            num_input,
            num_nodes,
            num_output,
            h,
            alpha,
            omega,
            gamma,
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
