import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.ssd_chunk_state import (
    chunk_state_bwd_db_cute,
    chunk_state_bwd_ddAcs_stable_cute,
    chunk_state_bwd_dx_cute,
    chunk_state_fwd_cute,
    chunk_state_varlen_cute,
)
from mamba_ssm.ops.triton.ssd_chunk_state import (
    _chunk_state_bwd_db,
    _chunk_state_bwd_ddAcs_stable,
    _chunk_state_bwd_dx,
    _chunk_state_fwd,
    chunk_state_varlen,
)


def benchmark_case(dtype, batch, seqlen, nheads, headdim, dstate, chunk_size):
    ngroups = 1
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    x = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=dtype)
    B = torch.randn(batch, seqlen, ngroups, dstate, device="cuda", dtype=dtype)
    dt = torch.rand(batch, nheads, nchunks, chunk_size, device="cuda")
    dA = -torch.rand_like(dt).cumsum(-1)

    def cute_fn():
        return chunk_state_fwd_cute(B, x, dt, dA)

    def triton_fn():
        return _chunk_state_fwd(B, x, dt, dA)

    cute_fn()
    triton_fn()
    cute_ms = do_bench(cute_fn, warmup=20, rep=100)
    triton_ms = do_bench(triton_fn, warmup=20, rep=100)
    print(
        f"op=fwd dtype={dtype} batch={batch} seqlen={seqlen} heads={nheads} "
        f"headdim={headdim} dstate={dstate} chunk={chunk_size} "
        f"speedup={triton_ms / cute_ms:.3f} cute_ms={cute_ms:.4f} "
        f"triton_ms={triton_ms:.4f}",
        flush=True,
    )

    dstates = torch.randn(
        batch, nchunks, nheads, headdim, dstate, device="cuda", dtype=dtype
    )
    operations = (
        (
            "bwd_dx",
            lambda: chunk_state_bwd_dx_cute(B, x, dt, dA, dstates),
            lambda: _chunk_state_bwd_dx(B, x, dt, dA, dstates),
        ),
        (
            "bwd_db",
            lambda: chunk_state_bwd_db_cute(x, dt, dA, dstates, ngroups=ngroups),
            lambda: _chunk_state_bwd_db(x, dt, dA, dstates, ngroups=ngroups),
        ),
        (
            "bwd_ddA",
            lambda: chunk_state_bwd_ddAcs_stable_cute(B, x, dt, dA, dstates),
            lambda: _chunk_state_bwd_ddAcs_stable(B, x, dt, dA, dstates),
        ),
    )
    for operation, cute_fn, triton_fn in operations:
        cute_fn()
        triton_fn()
        cute_ms = do_bench(cute_fn, warmup=20, rep=100)
        triton_ms = do_bench(triton_fn, warmup=20, rep=100)
        print(
            f"op={operation} dtype={dtype} batch={batch} seqlen={seqlen} "
            f"heads={nheads} headdim={headdim} dstate={dstate} "
            f"chunk={chunk_size} speedup={triton_ms / cute_ms:.3f} "
            f"cute_ms={cute_ms:.4f} triton_ms={triton_ms:.4f}",
            flush=True,
        )


def benchmark_varlen_case(
    dtype,
    total_seqlen,
    num_sequences,
    nheads,
    headdim,
    dstate,
    chunk_size,
    states_in_fp32,
):
    ngroups = 1
    nchunks = (total_seqlen + chunk_size - 1) // chunk_size
    sequence_length = total_seqlen // num_sequences
    x = torch.randn(total_seqlen, nheads, headdim, device="cuda", dtype=dtype)
    B = torch.randn(total_seqlen, ngroups, dstate, device="cuda", dtype=dtype)
    dt = torch.rand(nheads, nchunks, chunk_size, device="cuda")
    dA = -torch.rand_like(dt).cumsum(-1)
    cu_seqlens = torch.arange(
        0,
        total_seqlen + 1,
        sequence_length,
        device="cuda",
        dtype=torch.int32,
    )
    state_dtype = torch.float32 if states_in_fp32 else dtype
    chunk_states = torch.randn(
        nchunks,
        nheads,
        headdim,
        dstate,
        device="cuda",
        dtype=state_dtype,
    )

    def cute_fn():
        return chunk_state_varlen_cute(B, x, dt, dA, cu_seqlens, chunk_states)

    def triton_fn():
        return chunk_state_varlen(B, x, dt, dA, cu_seqlens, chunk_states)

    actual = cute_fn()
    expected = triton_fn()
    rtol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    atol = 4e-2 if dtype == torch.bfloat16 else 8e-3
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    cute_ms = do_bench(cute_fn, warmup=20, rep=100)
    triton_ms = do_bench(triton_fn, warmup=20, rep=100)
    print(
        f"op=varlen dtype={dtype} tokens={total_seqlen} seqs={num_sequences} "
        f"heads={nheads} headdim={headdim} dstate={dstate} chunk={chunk_size} "
        f"fp32_states={states_in_fp32} speedup={triton_ms / cute_ms:.3f} "
        f"cute_ms={cute_ms:.4f} triton_ms={triton_ms:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    benchmark_case(torch.bfloat16, 1, 1024, 8, 64, 64, 128)
    benchmark_case(torch.float16, 2, 2048, 8, 64, 128, 256)
    benchmark_varlen_case(torch.bfloat16, 4096, 16, 8, 64, 64, 128, False)
    benchmark_varlen_case(torch.bfloat16, 4096, 64, 8, 64, 64, 128, True)
    benchmark_varlen_case(torch.float16, 8192, 16, 8, 64, 128, 256, False)
