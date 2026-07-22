from cutlass import BFloat16, Int32
from cutlass._mlir.dialects import vector
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def i32_to_bf16x2(value: Int32, *, loc=None, ip=None):
    packed = vector.from_elements(
        T.vector(1, T.i32()), (value.ir_value(),), loc=loc, ip=ip
    )
    unpacked = vector.bitcast(T.vector(2, T.bf16()), packed, loc=loc, ip=ip)
    return tuple(
        BFloat16(
            vector.extract(
                unpacked,
                dynamic_position=[],
                static_position=[index],
                loc=loc,
                ip=ip,
            )
        )
        for index in range(2)
    )


@dsl_user_op
def bf16x2_to_i32(first: BFloat16, second: BFloat16, *, loc=None, ip=None):
    unpacked = vector.from_elements(
        T.vector(2, T.bf16()),
        (first.ir_value(), second.ir_value()),
        loc=loc,
        ip=ip,
    )
    packed = vector.bitcast(T.vector(1, T.i32()), unpacked, loc=loc, ip=ip)
    return Int32(
        vector.extract(packed, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    )


__all__ = ["bf16x2_to_i32", "i32_to_bf16x2"]
