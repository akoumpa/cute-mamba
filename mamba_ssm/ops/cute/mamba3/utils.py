"""Fast scalar math helpers for CuTe Mamba-3 kernels."""

from cutlass import Float32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.activation import sigmoid as _sigmoid
from quack.activation import silu as _silu
from quack.activation import tanh as _tanh


def _unary_ptx(op: str, x, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            f"{op}.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cos_approx(x, *, loc=None, ip=None) -> Float32:
    return _unary_ptx("cos", x, loc=loc, ip=ip)


@dsl_user_op
def sin_approx(x, *, loc=None, ip=None) -> Float32:
    return _unary_ptx("sin", x, loc=loc, ip=ip)


@dsl_user_op
def tanh_approx(x, *, loc=None, ip=None) -> Float32:
    return _tanh(x, loc=loc, ip=ip)


@dsl_user_op
def sech2_approx(x, *, loc=None, ip=None) -> Float32:
    tanh_x = _tanh(x, loc=loc, ip=ip)
    return Float32(1.0) - tanh_x * tanh_x


@dsl_user_op
def sigmoid_approx(x, *, loc=None, ip=None) -> Float32:
    return _sigmoid(x, loc=loc, ip=ip)


@dsl_user_op
def silu(x, *, loc=None, ip=None) -> Float32:
    return _silu(x, loc=loc, ip=ip)


__all__ = [
    "cos_approx",
    "sech2_approx",
    "sigmoid_approx",
    "silu",
    "sin_approx",
    "tanh_approx",
]
