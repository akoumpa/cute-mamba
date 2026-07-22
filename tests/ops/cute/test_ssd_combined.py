import pytest
import torch
from einops import rearrange


def _require_cute():
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("quack")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("d_layout", [None, "head", "head_feature"])
def test_chunk_scan_chunk_state_bwd_dx_cute(dtype, d_layout):
    _require_cute()
    from mamba_ssm.ops.cute.ssd_combined import (
        chunk_scan_chunk_state_bwd_dx_cute,
    )
    from mamba_ssm.ops.triton.ssd_combined import _chunk_scan_chunk_state_bwd_dx

    torch.manual_seed(0)
    batch, seqlen, nheads, ngroups = 2, 47, 4, 2
    headdim, dstate, chunk_size = 17, 19, 32
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    x = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=dtype)
    B = torch.randn(batch, seqlen, ngroups, dstate, device="cuda", dtype=dtype)
    CB = torch.randn(
        batch,
        nchunks,
        ngroups,
        chunk_size,
        chunk_size,
        device="cuda",
        dtype=torch.float32,
    ).tril()
    dt = torch.rand(batch, nheads, nchunks, chunk_size, device="cuda")
    dA_cumsum = -torch.rand_like(dt).cumsum(-1)
    dout = torch.randn_like(x)
    dstates = torch.randn(
        batch, nchunks, nheads, headdim, dstate, device="cuda", dtype=dtype
    )
    D = None
    if d_layout == "head":
        D = torch.randn(nheads, device="cuda")
    elif d_layout == "head_feature":
        D = torch.randn(nheads, headdim, device="cuda")

    dx_buffer = torch.empty_like(x)
    actual = chunk_scan_chunk_state_bwd_dx_cute(
        x, dt, dA_cumsum, B, CB, dout, dstates, D, dx=dx_buffer
    )
    expected = _chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum, B, CB, dout, dstates, D)

    assert actual[0].data_ptr() == dx_buffer.data_ptr()
    torch.testing.assert_close(actual[0], expected[0], rtol=7e-3, atol=1e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-2, atol=2e-2)
    if D is None:
        assert actual[2] is None
    else:
        torch.testing.assert_close(actual[2], expected[2], rtol=7e-3, atol=1e-2)


def test_chunk_scan_chunk_state_bwd_dx_cute_seq_fallback(monkeypatch):
    _require_cute()
    from mamba_ssm.ops.cute.ssd_combined import (
        chunk_scan_chunk_state_bwd_dx_cute,
    )

    batch, seqlen, nheads, ngroups = 1, 16, 2, 1
    headdim, dstate, chunk_size = 8, 8, 16
    x = torch.randn(batch, seqlen, nheads, headdim, device="cuda")
    dt = torch.randn(batch, nheads, 1, chunk_size, device="cuda")
    dA = torch.randn_like(dt)
    B = torch.randn(batch, seqlen, ngroups, dstate, device="cuda")
    CB = torch.randn(batch, 1, ngroups, chunk_size, chunk_size, device="cuda")
    dout = torch.randn_like(x)
    dstates = torch.randn(batch, 1, nheads, headdim, dstate, device="cuda")
    seq_idx = torch.zeros(batch, seqlen, device="cuda", dtype=torch.int32)
    sentinel = (object(), object(), object())

    monkeypatch.setattr(
        "mamba_ssm.ops.triton.ssd_combined._chunk_scan_chunk_state_bwd_dx",
        lambda *args: sentinel,
    )
    assert (
        chunk_scan_chunk_state_bwd_dx_cute(
            x, dt, dA, B, CB, dout, dstates, seq_idx=seq_idx
        )
        == sentinel
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("options", ["plain", "gated", "initial_final"])
def test_mamba_chunk_scan_combined_cute(dtype, options):
    _require_cute()
    from mamba_ssm.ops.cute.ssd_combined import mamba_chunk_scan_combined_cute
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    torch.manual_seed(1)
    batch, seqlen, nheads, ngroups = 1, 47, 4, 2
    headdim, dstate, chunk_size = 16, 24, 32
    values = (
        torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=dtype),
        torch.randn(batch, seqlen, nheads, device="cuda", dtype=dtype),
        -torch.rand(nheads, device="cuda"),
        torch.randn(batch, seqlen, ngroups, dstate, device="cuda", dtype=dtype) / 5,
        torch.randn(batch, seqlen, ngroups, dstate, device="cuda", dtype=dtype) / 5,
    )
    actual_inputs = tuple(value.detach().requires_grad_() for value in values)
    expected_inputs = tuple(value.detach().clone().requires_grad_() for value in values)
    D = torch.randn(nheads, headdim, device="cuda", requires_grad=True)
    D_ref = D.detach().clone().requires_grad_()
    z = torch.randn_like(values[0], requires_grad=True) if options == "gated" else None
    z_ref = z.detach().clone().requires_grad_() if z is not None else None
    initial = None
    initial_ref = None
    return_final = options == "initial_final"
    if return_final:
        initial = torch.randn(
            batch,
            nheads,
            headdim,
            dstate,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )
        initial_ref = initial.detach().clone().requires_grad_()

    kwargs = dict(
        chunk_size=chunk_size,
        D=D,
        z=z,
        dt_bias=torch.randn(nheads, device="cuda"),
        initial_states=initial,
        dt_softplus=True,
        dt_limit=(0.001, 0.2),
        return_final_states=return_final,
    )
    expected_kwargs = {**kwargs, "D": D_ref, "z": z_ref, "initial_states": initial_ref}
    actual = mamba_chunk_scan_combined_cute(*actual_inputs, **kwargs)
    expected = mamba_chunk_scan_combined(*expected_inputs, **expected_kwargs)
    actual_outputs = actual if isinstance(actual, tuple) else (actual,)
    expected_outputs = expected if isinstance(expected, tuple) else (expected,)
    for actual_output, expected_output in zip(actual_outputs, expected_outputs):
        torch.testing.assert_close(actual_output, expected_output, rtol=8e-2, atol=2e-1)

    actual_loss = sum(output.float().square().mean() for output in actual_outputs)
    expected_loss = sum(output.float().square().mean() for output in expected_outputs)
    actual_vars = [*actual_inputs, D, z, initial]
    expected_vars = [*expected_inputs, D_ref, z_ref, initial_ref]
    actual_grads = torch.autograd.grad(
        actual_loss, [value for value in actual_vars if value is not None]
    )
    expected_grads = torch.autograd.grad(
        expected_loss, [value for value in expected_vars if value is not None]
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-1, atol=2e-1)


def test_mamba_chunk_scan_cute_matches_combined():
    _require_cute()
    from mamba_ssm.ops.cute.ssd_combined import (
        mamba_chunk_scan_combined_cute,
        mamba_chunk_scan_cute,
    )

    torch.manual_seed(2)
    batch, seqlen, nheads, ngroups = 1, 33, 2, 1
    headdim, dstate, chunk_size = 16, 24, 32
    args = (
        torch.randn(batch, seqlen, nheads, headdim, device="cuda", requires_grad=True),
        torch.randn(batch, seqlen, nheads, device="cuda", requires_grad=True),
        -torch.rand(nheads, device="cuda", requires_grad=True),
        torch.randn(batch, seqlen, ngroups, dstate, device="cuda", requires_grad=True),
        torch.randn(batch, seqlen, ngroups, dstate, device="cuda", requires_grad=True),
        chunk_size,
    )
    actual = mamba_chunk_scan_cute(*args, dt_softplus=True)
    expected = mamba_chunk_scan_combined_cute(*args, dt_softplus=True)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("combined", [False, True])
def test_mamba_chunk_scan_cute_inference_uses_exact_fallback(monkeypatch, combined):
    from mamba_ssm.ops.cute.ssd_combined import (
        mamba_chunk_scan_combined_cute,
        mamba_chunk_scan_cute,
    )

    sentinel = object()
    symbol = "mamba_chunk_scan_combined" if combined else "mamba_chunk_scan"
    monkeypatch.setattr(
        f"mamba_ssm.ops.triton.ssd_combined.{symbol}",
        lambda *args, **kwargs: sentinel,
    )
    args = (
        torch.randn(1, 8, 2, 4),
        torch.randn(1, 8, 2),
        torch.randn(2),
        torch.randn(1, 8, 1, 4),
        torch.randn(1, 8, 1, 4),
        8,
    )
    function = mamba_chunk_scan_combined_cute if combined else mamba_chunk_scan_cute
    with torch.no_grad():
        assert function(*args) is sentinel


def test_mamba_chunk_scan_combined_cute_varlen_metadata_fallback(monkeypatch):
    from mamba_ssm.ops.cute.ssd_combined import mamba_chunk_scan_combined_cute

    sentinel = object()
    monkeypatch.setattr(
        "mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined",
        lambda *args, **kwargs: sentinel,
    )
    args = (
        torch.randn(1, 8, 2, 4, requires_grad=True),
        torch.randn(1, 8, 2, requires_grad=True),
        torch.randn(2, requires_grad=True),
        torch.randn(1, 8, 1, 4, requires_grad=True),
        torch.randn(1, 8, 1, 4, requires_grad=True),
        8,
    )
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
    assert mamba_chunk_scan_combined_cute(*args, cu_seqlens=cu_seqlens) is sentinel


def test_selective_scan_bwd_cute():
    _require_cute()
    from mamba_ssm.ops.cute.ssd_combined import selective_scan_bwd_cute
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    torch.manual_seed(3)
    batch, seqlen, nheads, ngroups = 1, 31, 2, 1
    headdim, dstate, chunk_size = 8, 16, 32
    x = torch.randn(batch, seqlen, nheads, headdim, device="cuda")
    dt = torch.rand(batch, nheads, 1, chunk_size, device="cuda") * 0.1
    A = -torch.rand(nheads, device="cuda")
    B = torch.randn(batch, seqlen, ngroups, dstate, device="cuda")
    C = torch.randn_like(B)
    D = torch.randn(nheads, headdim, device="cuda")
    z = torch.randn_like(x)
    dout = torch.randn_like(x)

    actual = selective_scan_bwd_cute(dout, x, dt, A, B, C, D, z)
    dt_ref = dt.detach().clone().requires_grad_()
    A_ref = A.detach().clone().requires_grad_()
    dt_tokens = rearrange(dt_ref, "b h c l -> b (c l) h")[:, :seqlen]
    out_ref = mamba_chunk_scan_combined(x, dt_tokens, A_ref, B, C, chunk_size, D=D, z=z)
    expected = torch.autograd.grad(out_ref, (dt_ref, A_ref), dout)
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-1, atol=2e-1)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-1, atol=2e-1)


def test_mamba_split_conv1d_scan_combined_cute():
    _require_cute()
    pytest.importorskip("causal_conv1d")
    from mamba_ssm.ops.cute.ssd_combined import (
        mamba_split_conv1d_scan_combined_cute,
    )
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

    torch.manual_seed(4)
    batch, seqlen, nheads, ngroups = 1, 31, 2, 1
    headdim, dstate, chunk_size, width = 8, 16, 32, 4
    dim = nheads * headdim
    conv_dim = dim + 2 * ngroups * dstate
    input_dim = 2 * dim + 2 * ngroups * dstate + nheads
    values = (
        torch.randn(batch, seqlen, input_dim, device="cuda", requires_grad=True),
        torch.randn(conv_dim, width, device="cuda", requires_grad=True),
        torch.randn(conv_dim, device="cuda", requires_grad=True),
        torch.randn(nheads, device="cuda", requires_grad=True),
        -torch.rand(nheads, device="cuda", requires_grad=True),
        torch.randn(nheads, headdim, device="cuda", requires_grad=True),
    )
    refs = tuple(value.detach().clone().requires_grad_() for value in values)
    kwargs = dict(chunk_size=chunk_size, return_final_states=True)
    actual = mamba_split_conv1d_scan_combined_cute(*values, **kwargs)
    expected = mamba_split_conv1d_scan_combined(*refs, **kwargs)
    for actual_output, expected_output in zip(actual, expected):
        torch.testing.assert_close(actual_output, expected_output, rtol=8e-2, atol=2e-1)
    actual_grads = torch.autograd.grad(
        sum(output.float().square().mean() for output in actual), values
    )
    expected_grads = torch.autograd.grad(
        sum(output.float().square().mean() for output in expected), refs
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-1, atol=2e-1)
