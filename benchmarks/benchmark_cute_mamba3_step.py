import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.mamba3.mamba3_step_fn import (
    mamba3_step_fn,
    selective_state_update_fused_ref_v2,
)


CASES = [
    (1, 32),
    (1, 64),
    (4, 128),
]


def benchmark_case(mimo, dstate):
    torch.manual_seed(0)
    batch, nheads, hdim = 16, 64, 64
    state = torch.randn(batch, nheads, hdim, dstate, device="cuda")
    Bstate = torch.randn(batch, mimo, nheads, dstate, device="cuda", dtype=torch.bfloat16)
    Xstate = torch.randn(batch, nheads, hdim, device="cuda", dtype=torch.bfloat16)
    A = -torch.rand(batch, nheads, device="cuda")
    B = torch.randn_like(Bstate)
    C = torch.randn_like(Bstate)
    D = torch.randn(nheads, device="cuda")
    x = torch.randn(batch, nheads, hdim, device="cuda", dtype=torch.bfloat16)
    dt = torch.rand(batch, nheads, device="cuda") * 0.1
    trap = torch.rand(batch, nheads, device="cuda")
    xproj = torch.randn(mimo, nheads, hdim, device="cuda", dtype=torch.bfloat16)
    z = torch.randn_like(x)
    zproj = torch.randn_like(xproj)
    outproj = torch.randn_like(xproj)
    state_out = torch.empty_like(state)
    out = torch.empty_like(x)

    cute_fn = lambda: mamba3_step_fn(
        state,
        Bstate,
        Xstate,
        A,
        B,
        C,
        D,
        x,
        dt,
        trap,
        xproj,
        outproj,
        state_out,
        out,
        z,
        zproj,
        tile_D=64,
        num_warps=4,
    )
    ref_fn = lambda: selective_state_update_fused_ref_v2(
        state, A, B, C, xproj, x, zproj, z, dt, Bstate, Xstate, trap, D, outproj
    )
    cute_fn()
    ref_fn()
    cute_ms = do_bench(cute_fn, warmup=25, rep=100)
    ref_ms = do_bench(ref_fn, warmup=25, rep=100)
    print(
        f"batch={batch} heads={nheads} headdim={hdim} dstate={dstate} mimo={mimo} "
        f"speedup={ref_ms / cute_ms:.2f} cute_ms={cute_ms:.4f} eager_ref_ms={ref_ms:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    for case in CASES:
        benchmark_case(*case)
