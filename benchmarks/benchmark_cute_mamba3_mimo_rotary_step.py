import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.mamba3.mamba3_mimo_rotary_step import (
    apply_rotary_qk_inference_fwd_cute,
)
from mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step import (
    apply_rotary_qk_inference_fwd,
)


def benchmark_case(batch, mimo_dim, nheads, headdim, rotary_dim, rotate_pairwise):
    dtype = torch.bfloat16
    q = torch.randn(batch, mimo_dim, nheads, headdim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    angle = torch.randn(batch, nheads, rotary_dim // 2, device="cuda")
    proj = torch.randn_like(angle)
    dt = torch.randn(batch, nheads, device="cuda")
    bias_q = torch.randn(mimo_dim, nheads, headdim, device="cuda", dtype=dtype)
    bias_k = torch.randn_like(bias_q)

    def cute_fn():
        return apply_rotary_qk_inference_fwd_cute(
            q, k, angle, proj, dt, bias_q, bias_k, rotate_pairwise=rotate_pairwise
        )

    def triton_fn():
        return apply_rotary_qk_inference_fwd(
            q, k, angle, proj, dt, bias_q, bias_k, rotate_pairwise=rotate_pairwise
        )

    cute_fn()
    triton_fn()
    cute_ms = do_bench(cute_fn, warmup=50, rep=300)
    triton_ms = do_bench(triton_fn, warmup=50, rep=300)
    print(
        f"batch={batch} mimo={mimo_dim} heads={nheads} headdim={headdim} "
        f"rotary={rotary_dim} pairwise={rotate_pairwise} "
        f"speedup={triton_ms / cute_ms:.3f} cute_ms={cute_ms:.4f} "
        f"triton_ms={triton_ms:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    for pairwise in (False, True):
        benchmark_case(1, 4, 64, 64, 32, pairwise)
        benchmark_case(8, 4, 64, 64, 32, pairwise)
        benchmark_case(32, 4, 64, 64, 32, pairwise)
