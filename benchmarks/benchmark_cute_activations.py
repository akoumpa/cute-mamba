import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.activations import _swiglu_bwd as cute_bwd
from mamba_ssm.ops.cute.activations import _swiglu_fwd as cute_fwd
from mamba_ssm.ops.triton.k_activations import _swiglu_bwd as triton_bwd
from mamba_ssm.ops.triton.k_activations import _swiglu_fwd as triton_fwd


CASES = [
    (torch.bfloat16, 128, 128),
    (torch.bfloat16, 512, 256),
    (torch.bfloat16, 1024, 256),
    (torch.bfloat16, 2048, 512),
    (torch.float16, 256, 1024),
    (torch.bfloat16, 2048, 2048),
    (torch.bfloat16, 8192, 4096),
]


def benchmark_case(dtype, rows, width):
    xy = torch.randn(rows, width * 2, device="cuda", dtype=dtype)
    dout = torch.randn(rows, width, device="cuda", dtype=dtype)
    for operation, cute_fn, triton_fn in (
        ("fwd", lambda: cute_fwd(xy), lambda: triton_fwd(xy)),
        ("bwd", lambda: cute_bwd(xy, dout), lambda: triton_bwd(xy, dout)),
        (
            "bwd_recompute",
            lambda: cute_bwd(xy, dout, recompute_output=True),
            lambda: triton_bwd(xy, dout, recompute_output=True),
        ),
    ):
        cute_fn()
        triton_fn()
        cute_ms = do_bench(cute_fn, warmup=50, rep=300)
        triton_ms = do_bench(triton_fn, warmup=50, rep=300)
        print(
            f"op={operation} dtype={dtype} rows={rows} width={width} "
            f"speedup={triton_ms / cute_ms:.3f} cute_ms={cute_ms:.4f} "
            f"triton_ms={triton_ms:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    for case in CASES:
        benchmark_case(*case)
