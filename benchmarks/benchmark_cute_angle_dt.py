import torch
from triton.testing import do_bench

from mamba_ssm.ops.cute.mamba3.angle_dt import angle_dt_fwd_cute
from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd


def benchmark_case(dtype, batch, seqlen, nheads, dim, varlen):
    angle = torch.randn(batch, seqlen, nheads, dim, device="cuda", dtype=dtype)
    dt = torch.randn(batch, nheads, seqlen, device="cuda", dtype=dtype)
    cu = None
    num_sequences = batch
    if varlen:
        lengths = [seqlen // 4] * 4
        lengths[-1] += seqlen - sum(lengths)
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
        num_sequences = 4
    initial = torch.randn(num_sequences, nheads, dim, device="cuda")
    cute_fn = lambda: angle_dt_fwd_cute(angle, dt, initial, 64, True, cu)
    triton_fn = lambda: angle_dt_fwd(angle, dt, initial, 64, True, cu)
    cute_fn()
    triton_fn()
    cute_ms = do_bench(cute_fn, warmup=50, rep=300)
    triton_ms = do_bench(triton_fn, warmup=50, rep=300)
    print(
        f"dtype={dtype} batch={batch} seqlen={seqlen} heads={nheads} dim={dim} "
        f"varlen={varlen} speedup={triton_ms / cute_ms:.3f} "
        f"cute_ms={cute_ms:.4f} triton_ms={triton_ms:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    benchmark_case(torch.bfloat16, 8, 2048, 64, 32, False)
    benchmark_case(torch.bfloat16, 1, 8192, 64, 32, True)
    benchmark_case(torch.float32, 2, 1024, 32, 16, False)
