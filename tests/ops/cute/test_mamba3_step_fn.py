import pytest
import torch


def _require_cute():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Hopper or newer CUDA GPU is required")
    pytest.importorskip("cutlass.cute")


@pytest.mark.parametrize(
    "dstate,mimo,has_z,has_outproj,inplace",
    [
        (32, 1, False, True, False),
        (64, 1, True, True, True),
        (128, 4, False, False, False),
        (128, 4, True, True, True),
    ],
)
def test_mamba3_step_fn(dstate, mimo, has_z, has_outproj, inplace):
    _require_cute()
    from mamba_ssm.ops.cute.mamba3.mamba3_step_fn import (
        mamba3_step_fn,
        selective_state_update_fused_ref_v2,
    )

    torch.manual_seed(0)
    batch, nheads, hdim = 2, 4, 64
    device = "cuda"
    state = torch.randn(batch, nheads, hdim, dstate, device=device)
    state_before = state.clone()
    Bstate = torch.randn(batch, mimo, nheads, dstate, device=device, dtype=torch.bfloat16)
    Xstate = torch.randn(batch, nheads, hdim, device=device, dtype=torch.bfloat16)
    A = -torch.rand(batch, nheads, device=device)
    B = torch.randn_like(Bstate)
    C = torch.randn_like(Bstate)
    D = torch.randn(nheads, device=device)
    x = torch.randn(batch, nheads, hdim, device=device, dtype=torch.bfloat16)
    dt = torch.rand(batch, nheads, device=device) * 0.1
    trap = torch.rand(batch, nheads, device=device)
    xproj = torch.randn(mimo, nheads, hdim, device=device, dtype=torch.bfloat16)
    z = torch.randn_like(x) if has_z else None
    zproj = torch.randn_like(xproj) if has_z else None
    outproj = torch.randn_like(xproj) if has_outproj else None
    out_shape = (batch, nheads, hdim) if has_outproj else (batch, mimo, nheads, hdim)
    out = torch.empty(out_shape, device=device, dtype=torch.bfloat16)
    state_out = None if inplace else torch.empty_like(state)

    out_ref, state_ref = selective_state_update_fused_ref_v2(
        state_before,
        A,
        B,
        C,
        xproj,
        x,
        zproj,
        z,
        dt,
        Bstate,
        Xstate,
        trap,
        D,
        outproj,
    )
    mamba3_step_fn(
        state,
        Bstate,
        Xstate,
        A,
        B,
        C,
        D,
        x,
        dt,
        trap,
        xproj,
        outproj,
        state_out,
        out,
        z,
        zproj,
        tile_D=64,
        num_warps=4,
    )

    actual_state = state if inplace else state_out
    torch.testing.assert_close(actual_state, state_ref, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(out, out_ref.to(out.dtype), rtol=2e-2, atol=3e-2)
    if not inplace:
        torch.testing.assert_close(state, state_before, rtol=0, atol=0)
