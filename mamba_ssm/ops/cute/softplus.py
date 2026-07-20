# Copyright (c) 2024, Tri Dao, Albert Gu.

from cutlass import Float32
from cutlass.cutlass_dsl import dsl_user_op
from quack.activation import softplus as _softplus


@dsl_user_op
def softplus(x: Float32, *, loc=None, ip=None) -> Float32:
    return _softplus(x, loc=loc, ip=ip)


__all__ = ["softplus"]
