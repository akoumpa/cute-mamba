import torch
import torch.nn.functional as F
import triton

from mamba_ssm.ops.cute.mamba3.mamba3_mimo_utils import (
    bwd_dadt_fused_cute,
    bwd_dtrap_ddt_cute,
    bwd_dtrap_ddt_cute_varlen,
    compute_dacs_segsum_cute,
)
from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import (
    bwd_dadt_fused_triton,
    bwd_dtrap_ddt_triton,
    bwd_dtrap_ddt_triton_varlen,
    compute_dacs_segsum_triton,
)


def bench(fn):
    return triton.testing.do_bench(fn, warmup=100, rep=300)


def bwd_dadt_eager(dSSdA, SSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev):
    shape = (*ddA_cs.shape[:2], dSSdA.shape[2], dSSdA.shape[-1])
    term_cs = (dA_cs.exp() * ddA_cs).view(shape).flip(-1).cumsum(-1).flip(-1)
    scaled_rev = (dA_cs_rev.exp() * ddA_cs_rev).view(shape)
    term_rev = F.pad(scaled_rev.cumsum(-1)[..., :-1], (1, 0))
    weighted = dSSdA.transpose(-1, -2).float() * SSdA.float().exp()
    mask = torch.ones(
        dSSdA.shape[-1], dSSdA.shape[-1], device=dSSdA.device, dtype=torch.bool
    ).tril(-1)
    term_seg = weighted.flip(-2).cumsum(-2).flip(-2).masked_fill(~mask, 0).sum(-1)
    return (term_cs.float() + term_rev.float() + term_seg).flatten(-2)


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


def benchmark_dtrap_ddt_varlen():
    for num_sequences in (2, 8, 32):
        lengths = torch.full((num_sequences,), 128, device="cuda", dtype=torch.int32)
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device="cuda", dtype=torch.int32),
                lengths.cumsum(0),
            )
        )
        seqlen = int(cu_seqlens[-1])
        args = [torch.randn(1, 16, seqlen, device="cuda") for _ in range(4)]
        cute_ms = bench(lambda: bwd_dtrap_ddt_cute_varlen(*args, 64, cu_seqlens))
        triton_ms = bench(lambda: bwd_dtrap_ddt_triton_varlen(*args, 64, cu_seqlens))
        print(
            f"dtrap_ddt_varlen sequences={num_sequences}: "
            f"cute={cute_ms:.4f} ms triton={triton_ms:.4f} ms "
            f"speedup={triton_ms / cute_ms:.3f}x"
        )


def benchmark_dacs_segsum():
    for dtype in (torch.float32, torch.bfloat16):
        for chunk_size in (16, 32, 64):
            da = torch.randn(2, 16, 4096, device="cuda", dtype=dtype) * 0.1
            cute_ms = bench(lambda: compute_dacs_segsum_cute(da, chunk_size))
            triton_ms = bench(lambda: compute_dacs_segsum_triton(da, chunk_size))
            print(
                f"dacs_segsum dtype={dtype} chunk={chunk_size}: "
                f"cute={cute_ms:.4f} ms triton={triton_ms:.4f} ms "
                f"speedup={triton_ms / cute_ms:.3f}x"
            )


def benchmark_dadt():
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for chunk_size in (16, 32, 64):
            batch, nheads, seqlen = 2, 16, 4096
            nchunks = seqlen // chunk_size
            matrices = [
                torch.randn(
                    batch,
                    nheads,
                    nchunks,
                    chunk_size,
                    chunk_size,
                    device="cuda",
                    dtype=dtype,
                )
                * 0.1
                for _ in range(2)
            ]
            vectors = [
                torch.randn(batch, nheads, seqlen, device="cuda", dtype=dtype)
                for _ in range(4)
            ]
            args = matrices + vectors
            cute_ms = bench(lambda: bwd_dadt_fused_cute(*args, chunk_size))
            baseline = (
                bwd_dadt_fused_triton if dtype == torch.float32 else bwd_dadt_eager
            )
            baseline_args = (*args, chunk_size) if dtype == torch.float32 else args
            baseline_ms = bench(lambda: baseline(*baseline_args))
            baseline_name = "triton" if dtype == torch.float32 else "eager"
            print(
                f"dadt dtype={dtype} chunk={chunk_size}: "
                f"cute={cute_ms:.4f} ms {baseline_name}={baseline_ms:.4f} ms "
                f"speedup={baseline_ms / cute_ms:.3f}x"
            )


if __name__ == "__main__":
    benchmark_dtrap_ddt()
    benchmark_dtrap_ddt_varlen()
    benchmark_dacs_segsum()
    benchmark_dadt()
