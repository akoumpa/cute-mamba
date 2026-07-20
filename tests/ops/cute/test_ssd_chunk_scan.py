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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "options", ["plain", "no_ddacs", "d_scalar", "d_vector_recompute"]
)
def test_chunk_scan_bwd_dz_cute(dtype, options):
    _require_cute()
    from mamba_ssm.ops.cute.ssd_chunk_scan import chunk_scan_bwd_dz_cute
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dz

    torch.manual_seed(9)
    batch, seqlen, nheads, headdim, chunk_size = 2, 47, 4, 17, 32
    shape = (batch, seqlen, nheads, headdim)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    z = torch.randn_like(x)
    out = torch.randn_like(x)
    dout = torch.randn_like(x)
    D = None
    has_ddacs = options != "no_ddacs"
    recompute = options == "d_vector_recompute"
    if options == "d_scalar":
        D = torch.randn(nheads, device="cuda")
    elif options == "d_vector_recompute":
        D = torch.randn(nheads, headdim, device="cuda")
    dz = torch.empty_like(z)
    actual = chunk_scan_bwd_dz_cute(
        x, z, out, dout, chunk_size, has_ddacs, D, dz, recompute
    )
    expected = _chunk_scan_bwd_dz(
        x, z, out, dout, chunk_size, has_ddacs, D, recompute_output=recompute
    )
    assert actual[0].data_ptr() == dz.data_ptr()
    rtol = 8e-3 if dtype == torch.bfloat16 else 5e-3
    atol = 3e-2 if dtype == torch.bfloat16 else 8e-3
    for output_idx, (actual_tensor, expected_tensor) in enumerate(
        zip(actual, expected)
    ):
        if actual_tensor is not None:
            if has_ddacs and output_idx == 3:
                nchunks = (seqlen + chunk_size - 1) // chunk_size
                valid = (
                    torch.arange(nchunks * chunk_size, device="cuda").reshape(
                        nchunks, chunk_size
                    )
                    < seqlen
                )
                valid = valid[None, None].expand(batch, nheads, -1, -1)
                actual_tensor = actual_tensor[valid]
                expected_tensor = expected_tensor[valid]
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=rtol, atol=atol
            )
