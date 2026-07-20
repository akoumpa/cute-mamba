import pytest
import torch


def _require_cute():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Hopper or newer CUDA GPU is required")
    pytest.importorskip("cutlass.cute")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("has_initial,has_seq_idx", [(False, False), (True, False), (True, True)])
def test_state_passing_fwd_cute(dtype, has_initial, has_seq_idx):
    _require_cute()
    from mamba_ssm.ops.cute.ssd_state_passing import state_passing_fwd_cute
    from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd

    torch.manual_seed(0)
    batch, nchunks, nheads, dim, chunk_size = 2, 5, 3, 130, 16
    states = torch.randn(batch, nchunks, nheads, dim, device="cuda", dtype=dtype)
    decay = -torch.rand(batch, nheads, nchunks, device="cuda")
    initial = torch.randn(batch, nheads, dim, device="cuda") if has_initial else None
    seq_idx = None
    if has_seq_idx:
        seq_idx = torch.zeros(batch, nchunks * chunk_size - 3, device="cuda", dtype=torch.int32)
        seq_idx[:, 2 * chunk_size :] = 1

    actual = state_passing_fwd_cute(states, decay, initial, seq_idx, chunk_size)
    expected = _state_passing_fwd(states, decay, initial, seq_idx, chunk_size)
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("has_final,has_initial,convert_states", [(False, False, False), (True, True, False), (True, True, True)])
def test_state_passing_bwd_cute(has_final, has_initial, convert_states):
    _require_cute()
    from mamba_ssm.ops.cute.ssd_state_passing import state_passing_bwd_cute
    from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_bwd

    torch.manual_seed(1)
    batch, nchunks, nheads, dim = 2, 5, 3, 513
    states = torch.randn(batch, nchunks, nheads, dim, device="cuda", dtype=torch.bfloat16)
    decay = -torch.rand(batch, nheads, nchunks, device="cuda")
    dout = torch.randn_like(states)
    dfinal = torch.randn(batch, nheads, dim, device="cuda") if has_final else None
    states_dtype = torch.float32 if convert_states else None
    actual = state_passing_bwd_cute(
        states,
        decay,
        dout,
        dfinal,
        has_initial_states=has_initial,
        states_dtype=states_dtype,
    )
    expected = _state_passing_bwd(
        states,
        decay,
        dout,
        dfinal,
        has_initial_states=has_initial,
        states_dtype=states_dtype,
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        if actual_tensor is not None:
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=4e-3, atol=5e-3)
