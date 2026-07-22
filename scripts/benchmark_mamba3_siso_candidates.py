import torch
import triton

from mamba_ssm.ops.cute.mamba3.mamba3_siso import (
    apply_dk_state_post_cute,
    compute_ddt_dtrap_dinput_states_cute,
    compute_dzdo_cute,
    mamba3_siso_step_cute,
)
from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import (
    apply_dk_state_post,
    compute_ddt_dtrap_dinput_states,
    compute_dzdo,
)
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


def benchmark_ddt_dtrap():
    for seqlen in (128, 1024, 4096):
        shape = (2, 16, seqlen)
        args = [torch.randn(shape, device="cuda") for _ in range(4)]
        cute_ms = bench(lambda: compute_ddt_dtrap_dinput_states_cute(*args))
        triton_ms = bench(lambda: compute_ddt_dtrap_dinput_states(*args))
        print(
            f"ddt_dtrap seqlen={seqlen}: cute={cute_ms:.4f} ms "
            f"triton={triton_ms:.4f} ms speedup={triton_ms / cute_ms:.3f}x"
        )


def benchmark_dk_state_post():
    seqlen, nheads_qk, nheads, qk_dim, angle_dim = 128, 2, 16, 128, 32
    for batch in (1, 8, 32):
        state = torch.randn(batch, nheads, qk_dim, device="cuda")
        angles = torch.randn(batch, seqlen, nheads, angle_dim, device="cuda")
        k = torch.randn(batch, seqlen, nheads_qk, qk_dim, device="cuda")
        k_bias = torch.randn(nheads, qk_dim, device="cuda")
        outputs = (
            torch.zeros_like(k),
            torch.zeros_like(k_bias),
            torch.zeros_like(angles),
        )
        expected = tuple(tensor.clone() for tensor in outputs)
        cute_ms = bench(
            lambda: apply_dk_state_post_cute(state, angles, k, k_bias, *outputs)
        )
        triton_ms = bench(
            lambda: apply_dk_state_post(state, angles, k, k_bias, *expected)
        )
        print(
            f"dk_state_post batch={batch}: cute={cute_ms:.4f} ms "
            f"triton={triton_ms:.4f} ms speedup={triton_ms / cute_ms:.3f}x"
        )


if __name__ == "__main__":
    benchmark_dzdo()
    benchmark_ddt_dtrap()
    benchmark_dk_state_post()
    benchmark_step()
