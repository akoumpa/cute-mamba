import argparse

import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.selective_state_update import selective_state_update_cute
from mamba_ssm.ops.triton.selective_state_update import selective_state_update


CASES = [
    (torch.float16, 1, 1, 2048, 16, 1, False),
    (torch.float16, 16, 1, 2048, 16, 1, True),
    (torch.bfloat16, 1, 1, 4096, 64, 1, False),
    (torch.bfloat16, 16, 1, 4096, 64, 1, True),
    (torch.float16, 1, 32, 64, 64, 1, False),
    (torch.float16, 16, 32, 64, 64, 4, True),
    (torch.bfloat16, 1, 64, 64, 128, 4, False),
    (torch.bfloat16, 16, 64, 64, 128, 4, True),
]


def benchmark_case(dtype, batch, nheads, headdim, dstate, ngroups, indexed):
    x_shape = (batch, nheads, headdim) if nheads > 1 else (batch, headdim)
    state_batches = batch * 4 if indexed else batch
    state_shape = (
        (state_batches, nheads, headdim, dstate)
        if nheads > 1
        else (state_batches, headdim, dstate)
    )
    a_shape = (nheads, headdim, dstate) if nheads > 1 else (headdim, dstate)
    bc_shape = (batch, ngroups, dstate) if nheads > 1 else (batch, dstate)
    vector_shape = (nheads, headdim) if nheads > 1 else (headdim,)

    state_cute = torch.randn(state_shape, device="cuda", dtype=dtype)
    state_triton = state_cute.clone()
    x = torch.randn(x_shape, device="cuda", dtype=dtype)
    dt = torch.randn_like(x)
    A = -torch.rand(a_shape, device="cuda") - 1
    B = torch.randn(bc_shape, device="cuda")
    C = torch.randn_like(B)
    D = torch.randn(vector_shape, device="cuda")
    z = torch.randn_like(x)
    dt_bias = torch.rand(vector_shape, device="cuda") - 4
    indices = torch.arange(batch, device="cuda", dtype=torch.int32) * 3 if indexed else None

    cute_fn = lambda: selective_state_update_cute(
        state_cute, x, dt, A, B, C, D, z, dt_bias, True, indices
    )
    triton_fn = lambda: selective_state_update(
        state_triton, x, dt, A, B, C, D, z, dt_bias, True, indices
    )
    cute_fn()
    triton_fn()
    cute_ms = do_bench(cute_fn, warmup=50, rep=300)
    triton_ms = do_bench(triton_fn, warmup=50, rep=300)
    print(
        f"dtype={dtype} batch={batch} heads={nheads} headdim={headdim} "
        f"dstate={dstate} groups={ngroups} indexed={indexed} "
        f"speedup={triton_ms / cute_ms:.3f} cute_ms={cute_ms:.4f} triton_ms={triton_ms:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("cute", "triton"))
    parser.add_argument("--case", type=int, default=2)
    args = parser.parse_args()
    torch.manual_seed(0)
    if args.profile is None:
        for case in CASES:
            benchmark_case(*case)
    else:
        dtype, batch, nheads, headdim, dstate, ngroups, indexed = CASES[args.case]
        x_shape = (batch, nheads, headdim) if nheads > 1 else (batch, headdim)
        state_shape = (batch, nheads, headdim, dstate) if nheads > 1 else (batch, headdim, dstate)
        a_shape = (nheads, headdim, dstate) if nheads > 1 else (headdim, dstate)
        bc_shape = (batch, ngroups, dstate) if nheads > 1 else (batch, dstate)
        state = torch.randn(state_shape, device="cuda", dtype=dtype)
        x = torch.randn(x_shape, device="cuda", dtype=dtype)
        dt = torch.randn_like(x)
        A = -torch.rand(a_shape, device="cuda") - 1
        B = torch.randn(bc_shape, device="cuda")
        C = torch.randn_like(B)
        torch.cuda.synchronize()
        fn = selective_state_update_cute if args.profile == "cute" else selective_state_update
        fn(state, x, dt, A, B, C)
        torch.cuda.synchronize()
