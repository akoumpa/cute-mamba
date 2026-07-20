from unittest.mock import patch

from mamba_ssm.ops.cute.runtime import is_cute_dsl_available


def test_cute_runtime_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MAMBA_CUTE_DSL", "0")
    with patch("mamba_ssm.ops.cute.runtime._host_supports_cute_dsl") as supported:
        assert not is_cute_dsl_available()
        supported.assert_not_called()


def test_backend_dispatch_disable_takes_precedence(monkeypatch):
    monkeypatch.setenv("MAMBA_DISABLE_BACKEND_DISPATCH", "1")
    monkeypatch.setenv("MAMBA_CUTE_DSL", "1")
    with patch("mamba_ssm.ops.cute.runtime._host_supports_cute_dsl") as supported:
        assert not is_cute_dsl_available()
        supported.assert_not_called()


def test_cute_runtime_uses_host_probe_when_enabled(monkeypatch):
    monkeypatch.delenv("MAMBA_DISABLE_BACKEND_DISPATCH", raising=False)
    monkeypatch.delenv("MAMBA_CUTE_DSL", raising=False)
    with patch("mamba_ssm.ops.cute.runtime._host_supports_cute_dsl", return_value=True):
        assert is_cute_dsl_available()
