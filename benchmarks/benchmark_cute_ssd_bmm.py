import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.ssd_bmm import _bmm_chunk_bwd_cute, _bmm_chunk_fwd_cute
from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_bwd, _bmm_chunk_fwd


CASES = [
    (torch.float16, 1, 2048, 64, 256),
    (torch.bfloat16, 8, 2048, 128, 256),
    (torch.float16, 8, 4096, 128, 256),
]


def benchmark_case(dtype, batch, seqlen, dstate, chunk_size):
    a = torch.randn(batch, seqlen, 1, dstate, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    dout = torch.randn(
        batch,
        seqlen // chunk_size,
        1,
        chunk_size,
        chunk_size,
        device="cuda",
        dtype=dtype,
    )
    for op, cute_fn, triton_fn in (
        (
            "fwd",
            lambda: _bmm_chunk_fwd_cute(a, b, chunk_size, output_dtype=torch.float32),
            lambda: _bmm_chunk_fwd(a, b, chunk_size, output_dtype=torch.float32),
        ),
        (
            "bwd",
            lambda: _bmm_chunk_bwd_cute(a, dout),
            lambda: _bmm_chunk_bwd(a, dout),
        ),
    ):
        cute_fn()
        triton_fn()
        cute_ms = do_bench(cute_fn, warmup=50, rep=300)
        triton_ms = do_bench(triton_fn, warmup=50, rep=300)
        print(
            f"op={op} dtype={dtype} batch={batch} seqlen={seqlen} "
            f"dstate={dstate} chunk={chunk_size} speedup={triton_ms / cute_ms:.3f} "
            f"cute_ms={cute_ms:.4f} triton_ms={triton_ms:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    for case in CASES:
        benchmark_case(*case)
