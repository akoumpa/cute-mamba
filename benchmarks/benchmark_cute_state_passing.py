import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.ssd_state_passing import state_passing_bwd_cute, state_passing_fwd_cute
from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_bwd, _state_passing_fwd


CASES = [
    (torch.float32, 1, 32, 64, 2048, False),
    (torch.bfloat16, 8, 64, 64, 2048, True),
    (torch.float16, 8, 128, 128, 4096, True),
]


def benchmark_case(dtype, batch, nchunks, nheads, dim, has_initial):
    states = torch.randn(batch, nchunks, nheads, dim, device="cuda", dtype=dtype)
    decay = -torch.rand(batch, nheads, nchunks, device="cuda")
    initial = torch.randn(batch, nheads, dim, device="cuda") if has_initial else None
    cute_fn = lambda: state_passing_fwd_cute(states, decay, initial)
    triton_fn = lambda: _state_passing_fwd(states, decay, initial)
    cute_fn()
    triton_fn()
    cute_ms = do_bench(cute_fn, warmup=50, rep=300)
    triton_ms = do_bench(triton_fn, warmup=50, rep=300)
    print(
        f"dtype={dtype} batch={batch} chunks={nchunks} heads={nheads} dim={dim} "
        f"initial={has_initial} speedup={triton_ms / cute_ms:.3f} "
        f"cute_ms={cute_ms:.4f} triton_ms={triton_ms:.4f}",
        flush=True,
    )
    dout = torch.randn_like(states)
    dfinal = torch.randn(batch, nheads, dim, device="cuda")
    cute_bwd = lambda: state_passing_bwd_cute(
        states, decay, dout, dfinal, has_initial_states=has_initial
    )
    triton_bwd = lambda: _state_passing_bwd(
        states, decay, dout, dfinal, has_initial_states=has_initial
    )
    cute_bwd()
    triton_bwd()
    cute_ms = do_bench(cute_bwd, warmup=50, rep=300)
    triton_ms = do_bench(triton_bwd, warmup=50, rep=300)
    print(
        f"op=bwd dtype={dtype} batch={batch} chunks={nchunks} heads={nheads} dim={dim} "
        f"initial={has_initial} speedup={triton_ms / cute_ms:.3f} "
        f"cute_ms={cute_ms:.4f} triton_ms={triton_ms:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    for case in CASES:
        benchmark_case(*case)
