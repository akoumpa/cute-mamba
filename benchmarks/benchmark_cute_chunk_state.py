import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.ssd_chunk_state import chunk_state_fwd_cute
from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd


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
        f"dtype={dtype} batch={batch} seqlen={seqlen} heads={nheads} "
        f"headdim={headdim} dstate={dstate} chunk={chunk_size} "
        f"speedup={triton_ms / cute_ms:.3f} cute_ms={cute_ms:.4f} "
        f"triton_ms={triton_ms:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    benchmark_case(torch.bfloat16, 1, 1024, 8, 64, 64, 128)
    benchmark_case(torch.float16, 2, 2048, 8, 64, 128, 256)
