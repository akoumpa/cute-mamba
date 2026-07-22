import torch
import triton

from mamba_ssm.ops.cute.mamba3.mamba3_siso import (
    compute_dzdo_cute,
    mamba3_siso_step_cute,
)
from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import compute_dzdo
from mamba_ssm.ops.triton.mamba3.mamba3_siso_step import mamba3_siso_step


def bench(fn):
    return triton.testing.do_bench(fn, warmup=100, rep=300)


def benchmark_dzdo():
    for seqlen in (128, 1024, 4096):
        shape = (2, seqlen, 16, 64)
        args = [
            torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        ]
        cute_ms = bench(lambda: compute_dzdo_cute(*args))
        triton_ms = bench(lambda: compute_dzdo(*args))
        print(
            f"dzdo seqlen={seqlen}: cute={cute_ms:.4f} ms "
            f"triton={triton_ms:.4f} ms speedup={triton_ms / cute_ms:.3f}x"
        )


def benchmark_step():
    nheads_qk, nheads, qk_dim, v_dim, angle_dim = 2, 16, 128, 64, 32
    for batch in (1, 8, 32):
        q = torch.randn(batch, nheads_qk, qk_dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(batch, nheads, v_dim, device="cuda", dtype=torch.bfloat16)
        adt = -torch.rand(batch, nheads, device="cuda")
        dt = torch.rand(batch, nheads, device="cuda")
        trap = torch.randn(batch, nheads, device="cuda")
        q_bias = torch.randn(nheads, qk_dim, device="cuda", dtype=torch.bfloat16)
        k_bias = torch.randn_like(q_bias)
        angles = torch.randn(batch, nheads, angle_dim, device="cuda")
        states = (
            torch.rand(batch, nheads, angle_dim, device="cuda") * 2 * torch.pi,
            torch.randn(batch, nheads, v_dim, qk_dim, device="cuda"),
            torch.randn(batch, nheads, qk_dim, device="cuda"),
            torch.randn(batch, nheads, v_dim, device="cuda"),
        )
        d = torch.randn(nheads, device="cuda")
        z = torch.randn_like(v)
        args = (q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, states)
        cute_ms = bench(lambda: mamba3_siso_step_cute(*args))
        triton_ms = bench(lambda: mamba3_siso_step(*args))
        print(
            f"step batch={batch}: cute={cute_ms:.4f} ms "
            f"triton={triton_ms:.4f} ms speedup={triton_ms / cute_ms:.3f}x"
        )


if __name__ == "__main__":
    benchmark_dzdo()
    benchmark_step()
