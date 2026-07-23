import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.selective_scan import selective_scan_cute
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


CASES = (
    (torch.bfloat16, 8, 256, 1, 16),
    (torch.bfloat16, 8, 256, 4, 16),
    (torch.bfloat16, 8, 256, 16, 16),
    (torch.bfloat16, 16, 512, 1, 64),
)


def benchmark_case(dtype, batch, dim, seqlen, dstate):
    torch.manual_seed(0)
    u = torch.randn(batch, dim, seqlen, device="cuda", dtype=dtype)
    delta = torch.randn_like(u)
    a = -torch.rand(dim, dstate, device="cuda")
    b = torch.randn(batch, dstate, seqlen, device="cuda", dtype=dtype)
    c = torch.randn_like(b)
    d = torch.randn(dim, device="cuda")
    z = torch.randn_like(u)
    delta_bias = torch.randn(dim, device="cuda")
    args = (u, delta, a, b, c, d, z, delta_bias, True, False)

    def cute_fn():
        with torch.no_grad():
            return selective_scan_cute(*args)

    def cuda_fn():
        with torch.no_grad():
            return selective_scan_fn(*args)

    actual = cute_fn()
    expected = cuda_fn()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    cute_ms = do_bench(cute_fn, warmup=50, rep=300)
    cuda_ms = do_bench(cuda_fn, warmup=50, rep=300)
    print(
        f"dtype={dtype} batch={batch} dim={dim} seqlen={seqlen} "
        f"dstate={dstate} speedup={cuda_ms / cute_ms:.3f} "
        f"cute_ms={cute_ms:.4f} cuda_ms={cuda_ms:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    for case in CASES:
        benchmark_case(*case)
