import pytest
import torch


def _require_cute():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Hopper or newer CUDA GPU is required")
    pytest.importorskip("cutlass.cute")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("output_dtype", [None, torch.float32])
def test_bmm_chunk_fwd_cute_matches_triton(dtype, grouped, output_dtype):
    _require_cute()
    from mamba_ssm.ops.cute.ssd_bmm import _bmm_chunk_fwd_cute
    from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd

    torch.manual_seed(0)
    shape = (2, 128, 1, 64) if grouped else (2, 128, 64)
    a = torch.randn(shape, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    out = _bmm_chunk_fwd_cute(a, b, 64, causal=True, output_dtype=output_dtype)
    ref = _bmm_chunk_fwd(a, b, 64, causal=True, output_dtype=output_dtype)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("has_residual", [False, True])
def test_bmm_chunk_bwd_cute_matches_triton(dtype, grouped, has_residual):
    _require_cute()
    from mamba_ssm.ops.cute.ssd_bmm import _bmm_chunk_bwd_cute
    from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_bwd

    torch.manual_seed(1)
    shape = (2, 128, 1, 64) if grouped else (2, 128, 64)
    dout_shape = (2, 2, 1, 64, 64) if grouped else (2, 2, 64, 64)
    a = torch.randn(shape, device="cuda", dtype=dtype)
    dout = torch.randn(dout_shape, device="cuda", dtype=dtype)
    residual = torch.randn_like(a) if has_residual else None
    out_buffer = torch.empty_like(a)
    out = _bmm_chunk_bwd_cute(a, dout, residual=residual, out=out_buffer)
    ref = _bmm_chunk_bwd(a, dout, residual=residual)
    assert out.data_ptr() == out_buffer.data_ptr()
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bmm_chunk_cute_falls_back_for_unaligned_state(dtype):
    _require_cute()
    from mamba_ssm.ops.cute.ssd_bmm import (
        _bmm_chunk_bwd_cute,
        _bmm_chunk_fwd_cute,
    )
    from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_bwd, _bmm_chunk_fwd

    torch.manual_seed(2)
    a = torch.randn(1, 64, 1, 19, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    actual = _bmm_chunk_fwd_cute(a, b, 32)
    expected = _bmm_chunk_fwd(a, b, 32)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    dout = torch.randn(1, 2, 1, 32, 32, device="cuda", dtype=dtype)
    actual_bwd = _bmm_chunk_bwd_cute(a, dout)
    expected_bwd = _bmm_chunk_bwd(a, dout)
    torch.testing.assert_close(actual_bwd, expected_bwd, rtol=2e-2, atol=2e-2)


def test_bmm_chunk_fwd_dispatches_large_aligned_workload(monkeypatch):
    _require_cute()
    import mamba_ssm.ops.cute.ssd_bmm as cute_bmm
    import mamba_ssm.ops.triton.ssd_bmm as triton_bmm

    sentinel = object()
    monkeypatch.setattr(triton_bmm, "is_cute_dsl_available", lambda: True)
    monkeypatch.setattr(cute_bmm, "_bmm_chunk_fwd_cute", lambda *args: sentinel)

    a = torch.empty(8, 2048, 64, device="cuda", dtype=torch.bfloat16)
    assert triton_bmm._bmm_chunk_fwd(a, a, 256) is sentinel
