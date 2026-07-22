import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.mamba3.grouped_head_reduction import (
    reduce_grouped_qk_grads_and_bias_cute,
    reduce_grouped_qk_grads_cute,
)
from mamba_ssm.ops.triton.mamba3.grouped_head_reduction import (
    _reduce_grouped_qk_grads_and_bias_triton,
    reduce_grouped_qk_grads,
)


CASES = [
    (torch.bfloat16, 8, 2048, 4, 64, 4, 128),
    (torch.bfloat16, 2, 4096, 4, 64, 4, 128),
    (torch.float32, 8, 1024, 1, 32, 8, 64),
]


def benchmark_case(dtype, batch, seqlen, rank, heads, groups, dim):
    x = torch.randn(batch, seqlen, rank, heads, dim, device="cuda", dtype=dtype)

    def cute_fn():
        return reduce_grouped_qk_grads_cute(x, groups)

    def torch_fn():
        return reduce_grouped_qk_grads(x, groups)

    cute_fn()
    torch_fn()
    cute_ms = do_bench(cute_fn, warmup=50, rep=300)
    torch_ms = do_bench(torch_fn, warmup=50, rep=300)
    print(
        f"dtype={dtype} batch={batch} seqlen={seqlen} rank={rank} heads={heads} "
        f"groups={groups} dim={dim} speedup={torch_ms / cute_ms:.3f} "
        f"cute_ms={cute_ms:.4f} torch_ms={torch_ms:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    for case in CASES:
        benchmark_case(*case)

    for dtype, batch, seqlen, rank, heads, groups, dim in (
        (torch.bfloat16, 2, 2048, 2, 16, 4, 64),
        (torch.float16, 1, 8192, 1, 64, 4, 32),
        (torch.bfloat16, 1, 1024, 1, 8, 8, 64),
        (torch.bfloat16, 2, 2048, 2, 16, 16, 64),
        (torch.bfloat16, 2, 4096, 2, 16, 16, 64),
        (torch.float16, 1, 8192, 1, 64, 64, 32),
    ):
        dq = torch.randn(batch, seqlen, rank, heads, dim, device="cuda", dtype=dtype)
        dk = torch.randn_like(dq)

        def cute_fn():
            return reduce_grouped_qk_grads_and_bias_cute(dq, dk, groups)

        def triton_fn():
            return _reduce_grouped_qk_grads_and_bias_triton(dq, dk, groups)

        cute_fn()
        triton_fn()
        cute_ms = do_bench(cute_fn, warmup=50, rep=300)
        triton_ms = do_bench(triton_fn, warmup=50, rep=300)
        print(
            f"op=grouped_bias dtype={dtype} batch={batch} seqlen={seqlen} "
            f"rank={rank} heads={heads} groups={groups} dim={dim} "
            f"speedup={triton_ms / cute_ms:.3f} cute_ms={cute_ms:.4f} "
            f"triton_ms={triton_ms:.4f}",
            flush=True,
        )
