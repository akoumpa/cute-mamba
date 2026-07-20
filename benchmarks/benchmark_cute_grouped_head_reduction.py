import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.mamba3.grouped_head_reduction import reduce_grouped_qk_grads_cute
from mamba_ssm.ops.triton.mamba3.grouped_head_reduction import reduce_grouped_qk_grads


CASES = [
    (torch.bfloat16, 8, 2048, 4, 64, 4, 128),
    (torch.bfloat16, 2, 4096, 4, 64, 4, 128),
    (torch.float32, 8, 1024, 1, 32, 8, 64),
]


def benchmark_case(dtype, batch, seqlen, rank, heads, groups, dim):
    x = torch.randn(batch, seqlen, rank, heads, dim, device="cuda", dtype=dtype)
    cute_fn = lambda: reduce_grouped_qk_grads_cute(x, groups)
    torch_fn = lambda: reduce_grouped_qk_grads(x, groups)
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
