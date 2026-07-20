import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.ssd_chunk_state import chunk_cumsum_fwd_cute
from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd


CASES = [
    (torch.float32, 1, 2048, 64, 64, False, False),
    (torch.bfloat16, 8, 4096, 64, 64, True, True),
    (torch.float16, 8, 4096, 128, 128, True, False),
]


def benchmark_case(dtype, batch, seqlen, nheads, chunk_size, has_bias, dt_softplus):
    torch.manual_seed(0)
    dt = torch.randn(batch, seqlen, nheads, device="cuda", dtype=dtype)
    A = -torch.rand(nheads, device="cuda")
    dt_bias = torch.randn(nheads, device="cuda") if has_bias else None
    cute_fn = lambda: chunk_cumsum_fwd_cute(dt, A, chunk_size, dt_bias, dt_softplus)
    triton_fn = lambda: _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias, dt_softplus)
    cute_fn()
    triton_fn()
    cute_ms = do_bench(cute_fn, warmup=50, rep=300)
    triton_ms = do_bench(triton_fn, warmup=50, rep=300)
    print(
        f"dtype={dtype} batch={batch} seqlen={seqlen} heads={nheads} chunk={chunk_size} "
        f"bias={has_bias} softplus={dt_softplus} speedup={triton_ms / cute_ms:.3f} "
        f"cute_ms={cute_ms:.4f} triton_ms={triton_ms:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    for case in CASES:
        benchmark_case(*case)
