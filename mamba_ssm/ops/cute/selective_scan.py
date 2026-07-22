from __future__ import annotations


def selective_scan_cute(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    result = selective_scan_ref(
        u,
        delta,
        A,
        B,
        C,
        D,
        z,
        delta_bias,
        delta_softplus,
        return_last_state,
    )
    if not return_last_state:
        return result
    output, last_state = result
    return output, last_state.detach()


def mamba_inner_cute(
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    out_proj_weight,
    out_proj_bias,
    A,
    B=None,
    C=None,
    D=None,
    delta_bias=None,
    B_proj_bias=None,
    C_proj_bias=None,
    delta_softplus=True,
    checkpoint_lvl=1,
    b_rms_weight=None,
    c_rms_weight=None,
    dt_rms_weight=None,
    b_c_dt_rms_eps=1e-6,
):
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn

    return mamba_inner_fn(
        xz,
        conv1d_weight,
        conv1d_bias,
        x_proj_weight,
        delta_proj_weight,
        out_proj_weight,
        out_proj_bias,
        A,
        B,
        C,
        D,
        delta_bias,
        B_proj_bias,
        C_proj_bias,
        delta_softplus,
        checkpoint_lvl,
        b_rms_weight,
        c_rms_weight,
        dt_rms_weight,
        b_c_dt_rms_eps,
    )


__all__ = ["mamba_inner_cute", "selective_scan_cute"]
