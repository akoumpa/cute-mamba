import pytest
import torch


def _require_cute():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Hopper or newer CUDA GPU is required")
    pytest.importorskip("cutlass.cute")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("varlen", [False, True])
def test_angle_dt_fwd_cute(dtype, varlen):
    _require_cute()
    from mamba_ssm.ops.cute.mamba3.angle_dt import angle_dt_fwd_cute
    from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd

    torch.manual_seed(0)
    batch, seqlen, nheads, dim = (1, 37, 4, 18) if varlen else (3, 37, 4, 18)
    angle = torch.randn(batch, seqlen, nheads, dim, device="cuda", dtype=dtype)
    dt = torch.randn(batch, nheads, seqlen, device="cuda", dtype=dtype)
    cu_seqlens = torch.tensor([0, 11, 23, 37], device="cuda", dtype=torch.int32) if varlen else None
    num_sequences = 3 if varlen else batch
    initial = torch.randn(num_sequences, nheads, dim, device="cuda")
    actual = angle_dt_fwd_cute(angle, dt, initial, 16, True, cu_seqlens)
    expected = angle_dt_fwd(angle, dt, initial, 16, True, cu_seqlens)
    torch.testing.assert_close(actual[0], expected[0], rtol=3e-3, atol=5e-3)
    torch.testing.assert_close(actual[1], expected[1], rtol=3e-3, atol=5e-3)
