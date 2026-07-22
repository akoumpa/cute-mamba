import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.normalization import (
    layer_norm_fn_cute,
    layernorm_fn_cute,
    rmsnorm_fn_cute,
)
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn
from mamba_ssm.ops.triton.layernorm_gated import layernorm_fn, rmsnorm_fn


CASES = [
    (torch.bfloat16, 128, 256),
    (torch.bfloat16, 512, 256),
    (torch.bfloat16, 1024, 256),
    (torch.bfloat16, 2048, 256),
    (torch.bfloat16, 512, 2048),
    (torch.bfloat16, 1024, 2048),
    (torch.bfloat16, 4096, 2048),
    (torch.bfloat16, 4096, 4096),
    (torch.float16, 4096, 2048),
]


def benchmark_case(dtype, rows, width):
    torch.manual_seed(0)
    x = torch.randn(rows, width, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(width, device="cuda", requires_grad=True)
    bias = torch.randn(width, device="cuda", requires_grad=True)
    grad = torch.randn_like(x)

    implementations = (
        (
            "cute_layer_norm",
            lambda: layer_norm_fn_cute(x, weight, bias, eps=1e-6, is_rms_norm=True),
        ),
        (
            "layer_norm",
            lambda: layer_norm_fn(x, weight, bias, eps=1e-6, is_rms_norm=True),
        ),
        ("cute_gated", lambda: rmsnorm_fn_cute(x, weight, bias, eps=1e-6)),
        ("gated", lambda: rmsnorm_fn(x, weight, bias, eps=1e-6)),
    )
    times = {}
    for name, function in implementations:
        function()
        times[(name, "fwd")] = do_bench(function, warmup=25, rep=100)

        def fwd_bwd():
            output = function()
            return torch.autograd.grad(output, (x, weight, bias), grad)

        fwd_bwd()
        times[(name, "fwd_bwd")] = do_bench(fwd_bwd, warmup=25, rep=100)

    for operation in ("fwd", "fwd_bwd"):
        for candidate, reference in (
            ("cute_layer_norm", "layer_norm"),
            ("cute_gated", "gated"),
        ):
            cute_ms = times[(candidate, operation)]
            reference_ms = times[(reference, operation)]
            print(
                f"op={operation} reference={reference} dtype={dtype} rows={rows} "
                f"width={width} speedup={reference_ms / cute_ms:.3f} "
                f"cute_ms={cute_ms:.4f} triton_ms={reference_ms:.4f}",
                flush=True,
            )


def benchmark_layernorm_case(dtype, rows, width):
    torch.manual_seed(1)
    x = torch.randn(rows, width, device="cuda", dtype=dtype)
    weight = torch.randn(width, device="cuda")
    bias = torch.randn(width, device="cuda")
    implementations = (
        ("cute", lambda: layernorm_fn_cute(x, weight, bias, eps=1e-6)),
        ("gated", lambda: layernorm_fn(x, weight, bias, eps=1e-6)),
        (
            "generic",
            lambda: layer_norm_fn(x, weight, bias, eps=1e-6, is_rms_norm=False),
        ),
    )
    times = {}
    for name, function in implementations:
        function()
        times[name] = do_bench(function, warmup=25, rep=100)
    for reference in ("gated", "generic"):
        print(
            f"op=layernorm_fwd reference={reference} dtype={dtype} rows={rows} "
            f"width={width} speedup={times[reference] / times['cute']:.3f} "
            f"cute_ms={times['cute']:.4f} triton_ms={times[reference]:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    for case in CASES:
        benchmark_case(*case)
    for case in CASES:
        benchmark_layernorm_case(*case)
