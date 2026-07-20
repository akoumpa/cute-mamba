import pytest
import torch


def _require_cute():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Hopper or newer CUDA GPU is required")
    pytest.importorskip("cutlass.cute")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_heads,num_groups", [(8, 8), (16, 4), (64, 4)])
def test_reduce_grouped_qk_grads_cute(dtype, num_heads, num_groups):
    _require_cute()
    from mamba_ssm.ops.cute.mamba3.grouped_head_reduction import reduce_grouped_qk_grads_cute
    from mamba_ssm.ops.triton.mamba3.grouped_head_reduction import reduce_grouped_qk_grads

    torch.manual_seed(0)
    x = torch.randn(2, 17, 4, num_heads, 64, device="cuda", dtype=dtype)
    actual = reduce_grouped_qk_grads_cute(x, num_groups)
    expected = reduce_grouped_qk_grads(x, num_groups)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-3)
