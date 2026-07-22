import pytest
import torch
import torch.nn.functional as F


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def _compute_dacs_segsum_padded_ref(da, chunk_size):
    seqlen = da.shape[-1]
    padded = F.pad(da, (0, (-seqlen) % chunk_size))
    chunks = padded.unflatten(-1, (-1, chunk_size))
    prefix = chunks.cumsum(-1)
    da_cs = prefix.clamp_max(0).flatten(-2)[..., :seqlen]
    da_cs_rev = (chunks.sum(-1, keepdim=True) - prefix).clamp_max(0)
    da_cs_rev = da_cs_rev.flatten(-2)[..., :seqlen]
    mask = torch.ones(chunk_size, chunk_size, device=da.device, dtype=torch.bool).tril(
        -1
    )
    segsum = (prefix.unsqueeze(-1) - prefix.unsqueeze(-2)).clamp_max(0)
    return da_cs, da_cs_rev, segsum.masked_fill(~mask, 0)


def _bwd_dadt_eager_ref(dSSdA, SSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev):
    shape = (*ddA_cs.shape[:2], dSSdA.shape[2], dSSdA.shape[-1])
    term_cs = (dA_cs.exp() * ddA_cs).view(shape).flip(-1).cumsum(-1).flip(-1)
    scaled_rev = (dA_cs_rev.exp() * ddA_cs_rev).view(shape)
    term_rev = F.pad(scaled_rev.cumsum(-1)[..., :-1], (1, 0))
    weighted = dSSdA.transpose(-1, -2).float() * SSdA.float().exp()
    mask = torch.ones(
        dSSdA.shape[-1], dSSdA.shape[-1], device=dSSdA.device, dtype=torch.bool
    ).tril(-1)
    term_seg = weighted.flip(-2).cumsum(-2).flip(-2).masked_fill(~mask, 0).sum(-1)
    return (term_cs.float() + term_rev.float() + term_seg).flatten(-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seqlen,chunk_size", [(64, 16), (75, 32), (128, 64)])
def test_compute_dacs_segsum_cute(dtype, seqlen, chunk_size):
    _require_cuda()
    from mamba_ssm.ops.cute.mamba3.dacs_segsum import (
        compute_dacs_segsum_fixed_cute,
    )
    from mamba_ssm.ops.cute.mamba3.mamba3_mimo_utils import compute_dacs_segsum_cute
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import compute_dacs_segsum_triton

    torch.manual_seed(0)
    da = torch.randn(2, 3, seqlen, device="cuda", dtype=dtype) * 0.1
    actual = compute_dacs_segsum_fixed_cute(da, chunk_size)
    expected = (
        compute_dacs_segsum_triton(da, chunk_size)
        if seqlen % chunk_size == 0
        else _compute_dacs_segsum_padded_ref(da, chunk_size)
    )
    rtol, atol = (1e-2, 1.3e-1) if dtype == torch.bfloat16 else (2e-3, 3e-3)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=rtol, atol=atol)
    dispatched = compute_dacs_segsum_cute(da, chunk_size)
    for actual_tensor, dispatched_tensor in zip(actual, dispatched):
        torch.testing.assert_close(
            actual_tensor, dispatched_tensor, rtol=rtol, atol=atol
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("chunk_size", [16, 32, 64])
def test_bwd_dadt_fused_cute(dtype, chunk_size):
    _require_cuda()
    from mamba_ssm.ops.cute.mamba3.mamba3_mimo_utils import bwd_dadt_fused_cute
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import bwd_dadt_fused_triton

    torch.manual_seed(1)
    batch, nheads, seqlen = 2, 3, 128
    nchunks = seqlen // chunk_size
    dA_cs = -torch.rand(batch, nheads, seqlen, device="cuda", dtype=dtype)
    dA_cs_rev = -torch.rand_like(dA_cs)
    ddA_cs = torch.randn_like(dA_cs)
    ddA_cs_rev = torch.randn_like(dA_cs)
    SSdA = (
        torch.randn(
            batch,
            nheads,
            nchunks,
            chunk_size,
            chunk_size,
            device="cuda",
            dtype=dtype,
        )
        * 0.1
    )
    dSSdA = torch.randn_like(SSdA)
    actual = bwd_dadt_fused_cute(
        dSSdA, SSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, chunk_size
    )
    expected = (
        bwd_dadt_fused_triton(
            dSSdA, SSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, chunk_size
        )
        if dtype == torch.float32
        else _bwd_dadt_eager_ref(dSSdA, SSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev)
    )
    if dtype == torch.bfloat16:
        rtol, atol = 2e-2, 1e-1
    elif dtype == torch.float16:
        rtol, atol = 1e-2, 2e-2
    else:
        rtol, atol = 2e-3, 3e-3
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bwd_dtrap_ddt_cute(dtype):
    _require_cuda()
    from mamba_ssm.ops.cute.mamba3.mamba3_mimo_utils import bwd_dtrap_ddt_cute
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import bwd_dtrap_ddt_triton

    torch.manual_seed(2)
    tensors = [torch.randn(2, 3, 64, device="cuda", dtype=dtype) for _ in range(4)]
    actual = bwd_dtrap_ddt_cute(*tensors, 32)
    expected = bwd_dtrap_ddt_triton(*tensors, 32)
    rtol, atol = (2e-2, 1e-2) if dtype == torch.bfloat16 else (3e-3, 5e-3)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=rtol, atol=atol)


def test_mamba3_mimo_utils_varlen_cute():
    _require_cuda()
    from mamba_ssm.ops.cute.mamba3.mamba3_mimo_utils import (
        bwd_dadt_fused_cute_varlen,
        bwd_dtrap_ddt_cute_varlen,
        compute_dacs_segsum_cute_varlen,
    )
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import (
        bwd_dadt_fused_triton_varlen,
        bwd_dtrap_ddt_triton_varlen,
        compute_dacs_segsum_triton_varlen,
    )

    torch.manual_seed(3)
    batch, nheads, seqlen, chunk_size = 1, 2, 75, 16
    cu = torch.tensor([0, 23, 23, 75], device="cuda", dtype=torch.int32)
    da = -torch.rand(batch, nheads, seqlen, device="cuda")
    actual = compute_dacs_segsum_cute_varlen(da, chunk_size, cu)
    expected = compute_dacs_segsum_triton_varlen(da, chunk_size, cu)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=2e-3, atol=3e-3)

    _, _, segsum = expected
    dSSdA = torch.randn_like(segsum)
    SSdA = torch.randn_like(segsum) * 0.1
    ddA_cs = torch.randn_like(da)
    ddA_cs_rev = torch.randn_like(da)
    actual_dadt = bwd_dadt_fused_cute_varlen(
        dSSdA, SSdA, ddA_cs, ddA_cs_rev, expected[0], expected[1], chunk_size, cu
    )
    expected_dadt = bwd_dadt_fused_triton_varlen(
        dSSdA, SSdA, ddA_cs, ddA_cs_rev, expected[0], expected[1], chunk_size, cu
    )
    torch.testing.assert_close(actual_dadt, expected_dadt, rtol=2e-3, atol=3e-3)

    trap, dt, dfactor, dgamma = [torch.randn_like(da) for _ in range(4)]
    actual_trap = bwd_dtrap_ddt_cute_varlen(trap, dt, dfactor, dgamma, chunk_size, cu)
    expected_trap = bwd_dtrap_ddt_triton_varlen(
        trap, dt, dfactor, dgamma, chunk_size, cu
    )
    for actual_tensor, expected_tensor in zip(actual_trap, expected_trap):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=2e-3, atol=3e-3)
