import pytest
import torch


def _require_cute():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Hopper or newer CUDA GPU is required")
    pytest.importorskip("cutlass.cute")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("has_heads", [False, True])
@pytest.mark.parametrize("dstate", [16, 64])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("use_indices", [False, True])
def test_selective_state_update_cute(dtype, has_heads, dstate, has_z, use_indices):
    _require_cute()
    from mamba_ssm.ops.cute.selective_state_update import selective_state_update_cute
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update_ref

    torch.manual_seed(0)
    batch, dim = 3, 128
    nheads = 4 if has_heads else 1
    head_dim = dim // nheads if has_heads else dim
    state_batches = 8 if use_indices else batch
    state_shape = (state_batches, nheads, head_dim, dstate) if has_heads else (state_batches, dim, dstate)
    x_shape = (batch, nheads, head_dim) if has_heads else (batch, dim)
    param_shape = (nheads, head_dim, dstate) if has_heads else (dim, dstate)
    vector_shape = (nheads, head_dim) if has_heads else (dim,)
    bc_shape = (batch, 2, dstate) if has_heads else (batch, dstate)

    state = torch.randn(state_shape, device="cuda", dtype=dtype)
    state_ref = state.clone()
    x = torch.randn(x_shape, device="cuda", dtype=dtype)
    dt = torch.randn(x_shape, device="cuda", dtype=dtype)
    A = -torch.rand(param_shape, device="cuda", dtype=torch.float32) - 1
    B = torch.randn(bc_shape, device="cuda", dtype=torch.float32)
    C = torch.randn_like(B)
    D = torch.randn(vector_shape, device="cuda", dtype=torch.float32)
    z = torch.randn_like(x) if has_z else None
    dt_bias = torch.rand(vector_shape, device="cuda", dtype=torch.float32) - 4
    indices = torch.tensor([6, -1, 2], device="cuda", dtype=torch.int32) if use_indices else None

    if indices is None:
        ref_state = state_ref
    else:
        valid = indices >= 0
        ref_state = state_ref[indices[valid]].clone()
    out = selective_state_update_cute(
        state, x, dt, A, B, C, D, z, dt_bias, True, indices
    )
    if indices is None:
        out_ref = selective_state_update_ref(ref_state, x, dt, A, B, C, D, z, dt_bias, True)
        updated_state = state
    else:
        out_ref = torch.zeros_like(out)
        out_ref[valid] = selective_state_update_ref(
            ref_state, x[valid], dt[valid], A, B[valid], C[valid], D, z[valid] if z is not None else None, dt_bias, True
        )
        updated_state = state[indices[valid]]

    rtol, atol = ((1e-2, 1e-1) if dtype == torch.bfloat16 else (5e-3, 3e-2))
    if dtype == torch.float32:
        rtol, atol = 3e-4, 1e-3
    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(updated_state, ref_state, rtol=rtol, atol=atol)
