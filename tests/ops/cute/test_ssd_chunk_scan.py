import pytest
import torch


def _require_cute():
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("quack")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("options", ["plain", "d_scalar", "d_vector_z", "seq"])
def test_chunk_scan_fwd_cute(dtype, options):
    _require_cute()
    from mamba_ssm.ops.cute.ssd_chunk_scan import chunk_scan_fwd_cute
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd

    torch.manual_seed(8)
    batch, seqlen, nheads, ngroups = 2, 47, 4, 2
    headdim, dstate, chunk_size = 17, 19, 32
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    x = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=dtype)
    C = torch.randn(batch, seqlen, ngroups, dstate, device="cuda", dtype=dtype)
    cb = torch.randn(
        batch,
        nchunks,
        ngroups,
        chunk_size,
        chunk_size,
        device="cuda",
        dtype=dtype,
    ).tril()
    dt = torch.rand(batch, nheads, nchunks, chunk_size, device="cuda")
    dA = -torch.rand_like(dt).cumsum(-1)
    states = torch.randn(
        batch, nchunks, nheads, headdim, dstate, device="cuda", dtype=dtype
    )
    D = None
    z = None
    seq_idx = None
    if options == "d_scalar":
        D = torch.randn(nheads, device="cuda")
    elif options == "d_vector_z":
        D = torch.randn(nheads, headdim, device="cuda")
        z = torch.randn_like(x)
    elif options == "seq":
        seq_idx = torch.zeros(batch, seqlen, device="cuda", dtype=torch.int32)
        seq_idx[:, 35:] = 1
        cb[:, 1, :, :3, :] = 0
    actual = chunk_scan_fwd_cute(cb, x, dt, dA, C, states, D, z, seq_idx)
    expected = _chunk_scan_fwd(cb, x, dt, dA, C, states, D, z, seq_idx)
    rtol = 1e-2 if dtype == torch.bfloat16 else 6e-3
    atol = 5e-2 if dtype == torch.bfloat16 else 1e-2
    for actual_tensor, expected_tensor in zip(actual, expected):
        if actual_tensor is not None:
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=rtol, atol=atol
            )
