import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.ssd_chunk_scan import (
    chunk_scan_bwd_dz_cute,
    chunk_scan_fwd_cute,
)
from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dz, _chunk_scan_fwd


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
