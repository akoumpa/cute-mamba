import pytest
import torch
import torch.nn.functional as F


def _require_cute():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Hopper or newer CUDA GPU is required")
    pytest.importorskip("cutlass.cute")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(7, 128), (3, 130), (2, 3, 2048), (1, 8192)])
def test_swiglu_forward_backward(dtype, shape):
    _require_cute()
    from mamba_ssm.ops.cute.activations import _swiglu_bwd, _swiglu_fwd

    torch.manual_seed(0)
    xy = torch.randn(shape, device="cuda", dtype=dtype)
    dout = torch.randn((*shape[:-1], shape[-1] // 2), device="cuda", dtype=dtype)
    x, y = xy.float().chunk(2, dim=-1)
    out_ref = (F.silu(x) * y).to(dtype)
    rtol, atol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (2e-3, 2e-3)

    out = _swiglu_fwd(xy)
    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)
    out_buffer = torch.empty(
        xy.numel() // shape[-1], shape[-1] // 2, device="cuda", dtype=dtype
    )
    out_buffer_result = _swiglu_fwd(xy, out=out_buffer)
    assert out_buffer_result.shape == out_ref.shape
    torch.testing.assert_close(out_buffer_result, out_ref, rtol=rtol, atol=atol)

    xy_ref = xy.float().detach().requires_grad_(True)
    (F.silu(xy_ref.chunk(2, dim=-1)[0]) * xy_ref.chunk(2, dim=-1)[1]).backward(
        dout.float()
    )
    dxy = _swiglu_bwd(xy, dout)
    grad_rtol, grad_atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (4e-3, 4e-3)
    torch.testing.assert_close(
        dxy, xy_ref.grad.to(dtype), rtol=grad_rtol, atol=grad_atol
    )
    dxy_buffer = torch.empty(
        xy.numel() // shape[-1], shape[-1], device="cuda", dtype=dtype
    )
    dxy_buffer_result = _swiglu_bwd(xy, dout, dxy=dxy_buffer)
    assert dxy_buffer_result.shape == xy.shape
    torch.testing.assert_close(
        dxy_buffer_result, xy_ref.grad.to(dtype), rtol=grad_rtol, atol=grad_atol
    )

    dxy_recomputed, out_recomputed = _swiglu_bwd(xy, dout, recompute_output=True)
    torch.testing.assert_close(dxy_recomputed, dxy, rtol=0, atol=0)
    torch.testing.assert_close(out_recomputed, out_ref, rtol=rtol, atol=atol)
