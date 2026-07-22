import torch
import triton

from mamba_ssm.ops.cute.mamba3.mamba3_mimo_utils import bwd_dtrap_ddt_cute
from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import bwd_dtrap_ddt_triton


def bench(fn):
    return triton.testing.do_bench(fn, warmup=100, rep=300)


def benchmark_dtrap_ddt():
    for dtype in (torch.float32, torch.bfloat16):
        for seqlen in (128, 1024, 4096):
            args = [
                torch.randn(2, 16, seqlen, device="cuda", dtype=dtype) for _ in range(4)
            ]
            cute_ms = bench(lambda: bwd_dtrap_ddt_cute(*args, 64))
            triton_ms = bench(lambda: bwd_dtrap_ddt_triton(*args, 64))
            print(
                f"dtrap_ddt dtype={dtype} seqlen={seqlen}: "
                f"cute={cute_ms:.4f} ms triton={triton_ms:.4f} ms "
                f"speedup={triton_ms / cute_ms:.3f}x"
            )


if __name__ == "__main__":
    benchmark_dtrap_ddt()
