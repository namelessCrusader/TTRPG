"""Torch LM adapter factory tests (no weight download)."""

from __future__ import annotations

import builtins
import sys

import pytest

from src.sim.lm_adapter import (
    MockLMAdapter,
    TorchLMAdapter,
    get_adapter,
    resolve_torch_model_source,
)


def test_get_adapter_mock_by_default():
    assert isinstance(get_adapter(), MockLMAdapter)


def test_get_adapter_torch(monkeypatch):
    def _init(self, model=TorchLMAdapter.DEFAULT_MODEL, **kwargs):
        self.model_id = model
        self.last_raw_response = None
        self.last_latency_ms = None

    monkeypatch.setattr(TorchLMAdapter, "__init__", _init)
    adapter = get_adapter(use_torch=True, model="hf/test", torch_warmup=False)
    assert isinstance(adapter, TorchLMAdapter)
    assert adapter.model_id == "hf/test"


def test_resolve_torch_path(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    resolved = resolve_torch_model_source(torch_path=str(tmp_path))
    assert resolved == str(tmp_path.resolve())


def test_resolve_torch_path_requires_config(tmp_path):
    with pytest.raises(ValueError, match="config.json"):
        resolve_torch_model_source(torch_path=str(tmp_path))


def test_resolve_torch_model_as_existing_dir(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    resolved = resolve_torch_model_source(model=str(tmp_path))
    assert resolved == str(tmp_path.resolve())


def test_resolve_torch_hub_id():
    assert resolve_torch_model_source(model="org/model-name") == "org/model-name"


def test_get_adapter_torch_path(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}")

    def _init(self, model=TorchLMAdapter.DEFAULT_MODEL, **kwargs):
        self.model_id = model

    monkeypatch.setattr(TorchLMAdapter, "__init__", _init)
    adapter = get_adapter(
        use_torch=True,
        torch_model_path=str(tmp_path),
        torch_warmup=False,
    )
    assert adapter.model_id == str(tmp_path.resolve())


def test_torch_adapter_import_error_when_deps_missing(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ("torch", "transformers"):
            raise ImportError(f"blocked {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    for key in list(sys.modules):
        if key == "torch" or key.startswith("transformers"):
            monkeypatch.delitem(sys.modules, key, raising=False)

    with pytest.raises(ImportError, match="torch and transformers"):
        TorchLMAdapter(model="x", warmup=False)
