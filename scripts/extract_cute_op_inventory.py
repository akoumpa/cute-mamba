#!/usr/bin/env python3
"""Generate the authoritative CuTe migration inventory for mamba_ssm ops."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Route:
    source: str
    symbol: str
    group: str

    @property
    def id(self) -> str:
        return f"{self.source.removeprefix('mamba_ssm/ops/').removesuffix('.py').replace('/', '.')}.{self.symbol}"


def _routes(source: str, group: str, *symbols: str) -> list[Route]:
    return [Route(source, symbol, group) for symbol in symbols]


SURFACE = [
    *_routes(
        "mamba_ssm/ops/selective_scan_interface.py",
        "selective_scan",
        "selective_scan_fn",
        "mamba_inner_fn",
    ),
    *_routes(
        "mamba_ssm/ops/triton/k_activations.py",
        "activation",
        "_swiglu_fwd",
        "_swiglu_bwd",
    ),
    *_routes("mamba_ssm/ops/triton/softplus.py", "shared", "softplus"),
    *_routes(
        "mamba_ssm/ops/triton/layer_norm.py",
        "normalization",
        "layer_norm_fn",
        "rms_norm_fn",
        "layer_norm_linear_fn",
    ),
    *_routes(
        "mamba_ssm/ops/triton/layernorm_gated.py",
        "normalization",
        "layernorm_fn",
        "rmsnorm_fn",
    ),
    *_routes(
        "mamba_ssm/ops/triton/selective_state_update.py",
        "selective_scan",
        "selective_state_update",
    ),
    *_routes(
        "mamba_ssm/ops/triton/ssd_bmm.py", "ssd", "_bmm_chunk_fwd", "_bmm_chunk_bwd"
    ),
    *_routes(
        "mamba_ssm/ops/triton/ssd_chunk_scan.py",
        "ssd",
        "_chunk_scan_fwd",
        "_chunk_scan_bwd_dz",
        "_chunk_scan_bwd_dstates",
        "_chunk_scan_bwd_dC",
        "_chunk_scan_bwd_dcb",
        "_chunk_scan_bwd_dx",
        "_chunk_scan_bwd_ddAcs_stable",
        "_chunk_scan_bwd_ddAcs_prev",
        "chunk_scan",
    ),
    *_routes(
        "mamba_ssm/ops/triton/ssd_chunk_state.py",
        "ssd",
        "_chunk_cumsum_fwd",
        "_chunk_cumsum_bwd",
        "_chunk_state_fwd",
        "_chunk_state_bwd_dx",
        "_chunk_state_bwd_db",
        "_chunk_state_bwd_ddAcs_stable",
        "chunk_state_varlen",
        "chunk_state",
    ),
    *_routes(
        "mamba_ssm/ops/triton/ssd_state_passing.py",
        "ssd",
        "_state_passing_fwd",
        "_state_passing_bwd",
        "state_passing",
    ),
    *_routes(
        "mamba_ssm/ops/triton/ssd_combined.py",
        "ssd",
        "_chunk_scan_chunk_state_bwd_dx",
        "mamba_chunk_scan_combined",
        "mamba_chunk_scan",
        "selective_scan_bwd",
        "mamba_split_conv1d_scan_combined",
    ),
    *_routes(
        "mamba_ssm/ops/triton/mamba3/angle_dt.py",
        "mamba3",
        "angle_dt_fwd",
        "angle_dt_bwd",
    ),
    *_routes(
        "mamba_ssm/ops/triton/mamba3/utils.py",
        "shared",
        "cos_approx",
        "sin_approx",
        "tanh_approx",
        "sech2_approx",
        "sigmoid_approx",
        "silu",
    ),
    *_routes(
        "mamba_ssm/ops/triton/mamba3/grouped_head_reduction.py",
        "mamba3",
        "reduce_grouped_qk_grads",
        "reduce_grouped_qk_grads_and_bias_triton",
    ),
    *_routes(
        "mamba_ssm/ops/triton/mamba3/mamba3_mimo_rotary_step.py",
        "mamba3",
        "apply_rotary_qk_inference_fwd",
    ),
    *_routes(
        "mamba_ssm/ops/triton/mamba3/mamba3_mimo_utils.py",
        "mamba3",
        "bwd_dadt_fused_triton",
        "bwd_dtrap_ddt_triton",
        "compute_dacs_segsum_triton_varlen",
        "compute_dacs_segsum_triton",
        "bwd_dadt_fused_triton_varlen",
        "bwd_dtrap_ddt_triton_varlen",
    ),
    *_routes(
        "mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py", "mamba3", "mamba3_siso_fwd"
    ),
    *_routes(
        "mamba_ssm/ops/triton/mamba3/mamba3_siso_step.py", "mamba3", "mamba3_siso_step"
    ),
    *_routes(
        "mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py",
        "mamba3",
        "compute_dzdo",
        "compute_dqkv",
        "compute_dqktheta",
        "apply_dk_state_post",
        "compute_ddt_dtrap_dinput_states",
    ),
    *_routes(
        "mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py",
        "mamba3",
        "mamba3_siso_combined",
    ),
    *_routes(
        "mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py",
        "mamba3",
        "mamba_mimo_forward",
    ),
    *_routes(
        "mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd_varlen.py",
        "mamba3",
        "mamba_mimo_forward_varlen",
    ),
    *_routes(
        "mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py",
        "mamba3",
        "mamba_mimo_bwd_combined",
    ),
    *_routes(
        "mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd_varlen.py",
        "mamba3",
        "mamba_mimo_bwd_combined_varlen",
    ),
    *_routes("mamba_ssm/ops/tilelang/mamba3/mamba3_mimo.py", "mamba3", "mamba3_mimo"),
    *_routes("mamba_ssm/ops/cute/mamba3/mamba3_step_fn.py", "mamba3", "mamba3_step_fn"),
]


CUTE_ROUTES = {
    "triton.softplus.softplus": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/softplus.py:softplus",
    },
    "triton.k_activations._swiglu_fwd": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/activations.py:_swiglu_fwd",
    },
    "triton.k_activations._swiglu_bwd": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/activations.py:_swiglu_bwd",
    },
    "triton.ssd_bmm._bmm_chunk_fwd": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_bmm.py:_bmm_chunk_fwd_cute",
    },
    "triton.ssd_bmm._bmm_chunk_bwd": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_bmm.py:_bmm_chunk_bwd_cute",
    },
    "triton.layer_norm.rms_norm_fn": {
        "status": "promoted",
        "implementation": "quack.rmsnorm:rmsnorm",
    },
    "triton.selective_state_update.selective_state_update": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/selective_state_update.py:selective_state_update_cute",
    },
    "triton.ssd_chunk_state._chunk_cumsum_fwd": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_chunk_state.py:chunk_cumsum_fwd_cute",
    },
    "triton.ssd_chunk_state._chunk_cumsum_bwd": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_chunk_state.py:chunk_cumsum_bwd_cute",
    },
    "triton.ssd_chunk_state._chunk_state_fwd": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_chunk_state.py:chunk_state_fwd_cute",
    },
    "triton.ssd_chunk_state._chunk_state_bwd_dx": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_chunk_state.py:chunk_state_bwd_dx_cute",
    },
    "triton.ssd_chunk_state._chunk_state_bwd_db": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_chunk_state.py:chunk_state_bwd_db_cute",
    },
    "triton.ssd_chunk_state._chunk_state_bwd_ddAcs_stable": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_chunk_state.py:chunk_state_bwd_ddAcs_stable_cute",
    },
    "triton.ssd_chunk_state.chunk_state": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_chunk_state.py:chunk_state_cute",
    },
    "triton.ssd_chunk_state.chunk_state_varlen": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_chunk_state.py:chunk_state_varlen_cute",
    },
    "triton.ssd_chunk_scan._chunk_scan_fwd": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_chunk_scan.py:chunk_scan_fwd_cute",
    },
    "triton.ssd_chunk_scan._chunk_scan_bwd_dz": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_chunk_scan.py:chunk_scan_bwd_dz_cute",
    },
    "triton.ssd_chunk_scan._chunk_scan_bwd_dstates": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_chunk_scan.py:chunk_scan_bwd_dstates_cute",
    },
    "triton.ssd_chunk_scan._chunk_scan_bwd_dC": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_chunk_scan.py:chunk_scan_bwd_dC_cute",
    },
    "triton.ssd_chunk_scan._chunk_scan_bwd_dx": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_chunk_scan.py:chunk_scan_bwd_dx_cute",
    },
    "triton.ssd_chunk_scan._chunk_scan_bwd_dcb": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_chunk_scan.py:chunk_scan_bwd_dcb_cute",
    },
    "triton.ssd_chunk_scan._chunk_scan_bwd_ddAcs_stable": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_chunk_scan.py:chunk_scan_bwd_ddAcs_stable_cute",
    },
    "triton.ssd_chunk_scan._chunk_scan_bwd_ddAcs_prev": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_chunk_scan.py:chunk_scan_bwd_ddAcs_prev_cute",
    },
    "triton.ssd_chunk_scan.chunk_scan": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_chunk_scan.py:chunk_scan_cute",
    },
    "triton.ssd_state_passing._state_passing_fwd": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_state_passing.py:state_passing_fwd_cute",
    },
    "triton.ssd_state_passing._state_passing_bwd": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_state_passing.py:state_passing_bwd_cute",
    },
    "triton.ssd_state_passing.state_passing": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_state_passing.py:state_passing_cute",
    },
    "triton.ssd_combined._chunk_scan_chunk_state_bwd_dx": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_combined.py:chunk_scan_chunk_state_bwd_dx_cute",
    },
    "triton.ssd_combined.mamba_chunk_scan_combined": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_combined.py:mamba_chunk_scan_combined_cute",
    },
    "triton.ssd_combined.mamba_chunk_scan": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_combined.py:mamba_chunk_scan_cute",
    },
    "triton.ssd_combined.selective_scan_bwd": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_combined.py:selective_scan_bwd_cute",
    },
    "triton.ssd_combined.mamba_split_conv1d_scan_combined": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/ssd_combined.py:mamba_split_conv1d_scan_combined_cute",
    },
    "triton.mamba3.grouped_head_reduction.reduce_grouped_qk_grads": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/mamba3/grouped_head_reduction.py:reduce_grouped_qk_grads_cute",
    },
    "triton.mamba3.grouped_head_reduction.reduce_grouped_qk_grads_and_bias_triton": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/mamba3/grouped_head_reduction.py:reduce_grouped_qk_grads_and_bias_cute",
    },
    "triton.mamba3.angle_dt.angle_dt_fwd": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/mamba3/angle_dt.py:angle_dt_fwd_cute",
    },
    "triton.mamba3.angle_dt.angle_dt_bwd": {
        "status": "present_unverified",
        "implementation": "mamba_ssm/ops/cute/mamba3/angle_dt.py:angle_dt_bwd_cute",
    },
    "triton.mamba3.mamba3_mimo_rotary_step.apply_rotary_qk_inference_fwd": {
        "status": "promoted",
        "implementation": "mamba_ssm/ops/cute/mamba3/mamba3_mimo_rotary_step.py:apply_rotary_qk_inference_fwd_cute",
    },
    **{
        f"triton.mamba3.utils.{symbol}": {
            "status": "present_unverified",
            "implementation": f"mamba_ssm/ops/cute/mamba3/utils.py:{symbol}",
        }
        for symbol in (
            "cos_approx",
            "sin_approx",
            "tanh_approx",
            "sech2_approx",
            "sigmoid_approx",
            "silu",
        )
    },
    "cute.mamba3.mamba3_step_fn.mamba3_step_fn": {
        "status": "promoted",
        "implementation": "mamba_ssm/ops/cute/mamba3/mamba3_step_fn.py:mamba3_step_fn",
    },
}


def _symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def build_inventory(root: Path) -> dict:
    items = []
    for route in SURFACE:
        path = root / route.source
        if not path.is_file():
            raise RuntimeError(f"missing route source: {route.source}")
        if route.symbol not in _symbols(path):
            raise RuntimeError(f"missing route symbol: {route.id}")
        conversion = CUTE_ROUTES.get(
            route.id, {"status": "missing", "implementation": None}
        )
        items.append({**asdict(route), "id": route.id, **conversion})
    counts = {
        status: sum(item["status"] == status for item in items)
        for status in ("promoted", "present_unverified", "missing")
    }
    return {
        "schema_version": 1,
        "total_routes": len(items),
        "counts": counts,
        "routes": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", default="scripts/cute_op_inventory.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    rendered = json.dumps(build_inventory(root), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not output.is_file() or output.read_text() != rendered:
            raise SystemExit(
                f"stale CuTe inventory: regenerate {output.relative_to(root)}"
            )
        return
    output.write_text(rendered)


if __name__ == "__main__":
    main()
