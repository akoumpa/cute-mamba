import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.ssd_chunk_scan import (
    chunk_scan_bwd_dC_cute,
    chunk_scan_bwd_dcb_cute,
    chunk_scan_bwd_ddAcs_prev_cute,
    chunk_scan_bwd_ddAcs_stable_cute,
    chunk_scan_bwd_dstates_cute,
    chunk_scan_bwd_dx_cute,
    chunk_scan_bwd_dz_cute,
    chunk_scan_fwd_cute,
)
from mamba_ssm.ops.triton.ssd_chunk_scan import (
    _chunk_scan_bwd_dC,
    _chunk_scan_bwd_dcb,
    _chunk_scan_bwd_ddAcs_prev,
    _chunk_scan_bwd_ddAcs_stable,
    _chunk_scan_bwd_dstates,
    _chunk_scan_bwd_dx,
    _chunk_scan_bwd_dz,
    _chunk_scan_fwd,
)


CASES = [
    (torch.bfloat16, 1, 1024, 8, 64, 64, 128),
    (torch.bfloat16, 2, 4096, 8, 64, 64, 128),
    (torch.float16, 2, 4096, 8, 64, 64, 128),
]


def benchmark_case(dtype, batch, seqlen, nheads, headdim, dstate, chunk_size):
    torch.manual_seed(0)
    ngroups = 1
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
    D = torch.randn(nheads, headdim, device="cuda")
    z = torch.randn_like(x)

    operations = (
        (
            "fwd",
            lambda: chunk_scan_fwd_cute(cb, x, dt, dA, C, states, D, z),
            lambda: _chunk_scan_fwd(cb, x, dt, dA, C, states, D, z),
        ),
    )
    out = _chunk_scan_fwd(cb, x, dt, dA, C, states, D, z)[0]
    dout = torch.randn_like(out)
    operations += (
        (
            "bwd_dz",
            lambda: chunk_scan_bwd_dz_cute(x, z, out, dout, chunk_size, True, D),
            lambda: _chunk_scan_bwd_dz(x, z, out, dout, chunk_size, True, D),
        ),
        (
            "bwd_dstates",
            lambda: chunk_scan_bwd_dstates_cute(C, dA, dout),
            lambda: _chunk_scan_bwd_dstates(C, dA, dout),
        ),
        (
            "bwd_dC",
            lambda: chunk_scan_bwd_dC_cute(states, dA, dout, C=C, ngroups=ngroups),
            lambda: _chunk_scan_bwd_dC(states, dA, dout, C=C, ngroups=ngroups),
        ),
        (
            "bwd_dx",
            lambda: chunk_scan_bwd_dx_cute(cb, x, dt, dA, dout, D),
            lambda: _chunk_scan_bwd_dx(cb, x, dt, dA, dout, D),
        ),
        (
            "bwd_dcb",
            lambda: chunk_scan_bwd_dcb_cute(x, dt, dA, dout, ngroups=ngroups),
            lambda: _chunk_scan_bwd_dcb(x, dt, dA, dout, ngroups=ngroups),
        ),
        (
            "bwd_ddA_stable",
            lambda: chunk_scan_bwd_ddAcs_stable_cute(x, dt, dA, dout, cb),
            lambda: _chunk_scan_bwd_ddAcs_stable(x, dt, dA, dout, cb),
        ),
        (
            "bwd_ddA_prev",
            lambda: chunk_scan_bwd_ddAcs_prev_cute(states, C, dout, dA),
            lambda: _chunk_scan_bwd_ddAcs_prev(states, C, dout, dA),
        ),
    )

    for operation, cute_fn, triton_fn in operations:
        cute_fn()
        triton_fn()
        cute_ms = do_bench(cute_fn, warmup=25, rep=100)
        triton_ms = do_bench(triton_fn, warmup=25, rep=100)
        print(
            f"op={operation} dtype={dtype} batch={batch} seqlen={seqlen} "
            f"heads={nheads} headdim={headdim} dstate={dstate} chunk={chunk_size} "
            f"speedup={triton_ms / cute_ms:.3f} cute_ms={cute_ms:.4f} "
            f"triton_ms={triton_ms:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    for case in CASES:
        benchmark_case(*case)
