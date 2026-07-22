import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.mamba3.mamba3_siso_gate import compute_dzdo_cute
from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import compute_dzdo


CASES = [
    (torch.bfloat16, 1, 1024, 8, 64),
    (torch.bfloat16, 2, 4096, 8, 64),
    (torch.bfloat16, 2, 4096, 32, 128),
    (torch.float32, 1, 1024, 8, 64),
]


def benchmark_case(dtype, batch, seqlen, nheads, headdim):
    torch.manual_seed(0)
    shape = (batch, seqlen, nheads, headdim)
    inputs = tuple(torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3))
    implementations = (
        ("cute", lambda: compute_dzdo_cute(*inputs)),
        ("triton", lambda: compute_dzdo(*inputs)),
    )
    times = {}
    for name, function in implementations:
        function()
        times[name] = do_bench(function, warmup=50, rep=300)
    cute_ms, triton_ms = times["cute"], times["triton"]
    print(
        f"dtype={dtype} batch={batch} seqlen={seqlen} heads={nheads} "
        f"headdim={headdim} speedup={triton_ms / cute_ms:.3f} "
        f"cute_ms={cute_ms:.4f} triton_ms={triton_ms:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    for case in CASES:
        benchmark_case(*case)
