import ast
from pathlib import Path


ROOT = Path(__file__).parents[3]


def _function_contract(path, name):
    tree = ast.parse((ROOT / path).read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    defaults = [None] * (len(positional) - len(arguments.defaults)) + [
        ast.unparse(default) for default in arguments.defaults
    ]
    return list(zip((argument.arg for argument in positional), defaults))


def test_mamba3_mimo_backend_signatures():
    candidate = "mamba_ssm/ops/cute/mamba3/mamba3_mimo_backend.py"
    pairs = [
        (
            "mamba_mimo_forward_cute",
            "mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py",
            "mamba_mimo_forward",
        ),
        (
            "mamba_mimo_forward_varlen_cute",
            "mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd_varlen.py",
            "mamba_mimo_forward_varlen",
        ),
        (
            "mamba_mimo_bwd_combined_cute",
            "mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py",
            "mamba_mimo_bwd_combined",
        ),
        (
            "mamba_mimo_bwd_combined_varlen_cute",
            "mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd_varlen.py",
            "mamba_mimo_bwd_combined_varlen",
        ),
        (
            "mamba3_mimo_cute",
            "mamba_ssm/ops/tilelang/mamba3/mamba3_mimo.py",
            "mamba3_mimo",
        ),
    ]
    for candidate_name, reference_path, reference_name in pairs:
        assert _function_contract(candidate, candidate_name) == _function_contract(
            reference_path, reference_name
        )
