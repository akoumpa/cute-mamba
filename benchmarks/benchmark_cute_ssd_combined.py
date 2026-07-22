import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.ssd_combined import mamba_chunk_scan_combined_cute
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


CASES = [
    (torch.bfloat16, 1, 1024, 8, 64, 64, 128),
    (torch.bfloat16, 2, 4096, 8, 64, 64, 128),
]


def benchmark_case(dtype, batch, seqlen, nheads, headdim, dstate, chunk_size):
    torch.manual_seed(0)
    ngroups = 1
    inputs = (
        torch.randn(
            batch,
            seqlen,
            nheads,
            headdim,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        ),
        torch.randn(
            batch,
            seqlen,
            nheads,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        ),
        -torch.rand(nheads, device="cuda", requires_grad=True),
        torch.randn(
            batch,
            seqlen,
            ngroups,
            dstate,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        ),
        torch.randn(
            batch,
            seqlen,
            ngroups,
            dstate,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        ),
    )
    grad = torch.randn_like(inputs[0])
    kwargs = {"chunk_size": chunk_size, "dt_softplus": True}
    implementations = (
        ("cute", lambda: mamba_chunk_scan_combined_cute(*inputs, **kwargs)),
        ("triton", lambda: mamba_chunk_scan_combined(*inputs, **kwargs)),
    )
    times = {}
    for name, function in implementations:

        def inference():
            with torch.no_grad():
                return function()

        inference()
        times[(name, "inference")] = do_bench(inference, warmup=20, rep=50)

        def fwd_bwd():
            output = function()
            return torch.autograd.grad(output, inputs, grad)

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
