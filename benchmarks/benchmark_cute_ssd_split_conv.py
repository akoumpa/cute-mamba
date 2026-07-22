import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.ssd_combined import mamba_split_conv1d_scan_combined_cute
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


CASES = [
    (torch.bfloat16, 1, 1024, 8, 64, 64, 128),
    (torch.bfloat16, 2, 4096, 8, 64, 64, 128),
]


def benchmark_case(dtype, batch, seqlen, nheads, headdim, dstate, chunk_size):
    torch.manual_seed(0)
    ngroups, conv_width = 1, 4
    dim = nheads * headdim
    conv_dim = dim + 2 * ngroups * dstate
    input_dim = 2 * dim + 2 * ngroups * dstate + nheads
    inputs = (
        torch.randn(
            batch,
            seqlen,
            input_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        ),
        torch.randn(
            conv_dim,
            conv_width,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        ),
        torch.randn(conv_dim, device="cuda", dtype=dtype, requires_grad=True),
        torch.randn(nheads, device="cuda", requires_grad=True),
        -torch.rand(nheads, device="cuda", requires_grad=True),
        torch.randn(nheads, headdim, device="cuda", requires_grad=True),
    )
    kwargs = {"chunk_size": chunk_size}
    implementations = (
        ("cute", lambda: mamba_split_conv1d_scan_combined_cute(*inputs, **kwargs)),
        ("triton", lambda: mamba_split_conv1d_scan_combined(*inputs, **kwargs)),
    )
    grad = torch.randn(batch, seqlen, dim, device="cuda", dtype=dtype)
    times = {}
    for name, function in implementations:

        def inference():
            with torch.no_grad():
                return function()

        inference()
        times[(name, "inference")] = do_bench(inference, warmup=20, rep=50)

        def fwd_bwd():
            return torch.autograd.grad(function(), inputs, grad)

        fwd_bwd()
        times[(name, "fwd_bwd")] = do_bench(fwd_bwd, warmup=20, rep=50)

    for operation in ("inference", "fwd_bwd"):
        cute_ms = times[("cute", operation)]
        triton_ms = times[("triton", operation)]
        print(
            f"op={operation} dtype={dtype} batch={batch} seqlen={seqlen} "
            f"heads={nheads} headdim={headdim} dstate={dstate} chunk={chunk_size} "
            f"speedup={triton_ms / cute_ms:.3f} cute_ms={cute_ms:.4f} "
            f"triton_ms={triton_ms:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    for case in CASES:
        benchmark_case(*case)
