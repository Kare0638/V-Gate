# Copyright 2025 the V-Gate authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for inference backend abstraction layer.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from vgate.backends.base import DryRunBackend, InferenceBackend
from vgate.config import ModelConfig


class TestDryRunBackend:
    """Tests for DryRunBackend."""

    def test_implements_protocol(self):
        backend = DryRunBackend()
        assert isinstance(backend, InferenceBackend)

    def test_load_model_noop(self):
        backend = DryRunBackend()
        backend.load_model(ModelConfig())  # Should not raise

    def test_create_sampling_params(self):
        backend = DryRunBackend()
        params = backend.create_sampling_params(0.7, 0.9, 128)
        assert params == {"temperature": 0.7, "top_p": 0.9, "max_tokens": 128}

    def test_generate_returns_correct_format(self):
        backend = DryRunBackend()
        params = backend.create_sampling_params(0.7, 0.9, 128)
        results = backend.generate(["Hello world", "Test prompt"], params)

        assert len(results) == 2
        for result in results:
            assert "text" in result
            assert "token_ids" in result
            assert "num_tokens" in result
            assert "metrics" in result
            assert isinstance(result["text"], str)
            assert result["num_tokens"] == 8
            assert result["text"].startswith("[dry-run] echo:")

    def test_generate_single_prompt(self):
        backend = DryRunBackend()
        params = backend.create_sampling_params(0.5, 0.8, 64)
        results = backend.generate(["Single prompt"], params)
        assert len(results) == 1
        assert "Single prompt" in results[0]["text"]

    def test_shutdown_noop(self):
        backend = DryRunBackend()
        backend.shutdown()  # Should not raise


class TestBackendFactory:
    """Tests for the _create_backend factory function."""

    def test_dry_run_returns_dry_run_backend(self):
        with patch("vgate.engine.DRY_RUN", True):
            from vgate.engine import _create_backend
            backend = _create_backend("vllm")
            assert isinstance(backend, DryRunBackend)

    def test_dry_run_ignores_engine_type(self):
        with patch("vgate.engine.DRY_RUN", True):
            from vgate.engine import _create_backend
            backend = _create_backend("sglang")
            assert isinstance(backend, DryRunBackend)

    def test_unknown_engine_type_raises(self):
        with patch("vgate.engine.DRY_RUN", False):
            from vgate.engine import _create_backend
            with pytest.raises(ValueError, match="Unknown engine_type"):
                _create_backend("unknown_engine")

    def test_vllm_backend_import(self):
        """Test that vllm backend can be instantiated (import path works)."""
        with patch("vgate.engine.DRY_RUN", False):
            from vgate.engine import _create_backend
            # This will try to import vllm which may not be available
            # We mock the import to test the factory logic
            with patch("vgate.backends.vllm_backend.VLLMBackend") as mock_cls:
                mock_cls.return_value = MagicMock()
                backend = _create_backend("vllm")
                mock_cls.assert_called_once()

    def test_sglang_backend_import(self):
        """Test that sglang backend can be instantiated (import path works)."""
        with patch("vgate.engine.DRY_RUN", False):
            from vgate.engine import _create_backend
            with patch("vgate.backends.sglang_backend.SGLangBackend") as mock_cls:
                mock_cls.return_value = MagicMock()
                backend = _create_backend("sglang")
                mock_cls.assert_called_once()


class TestModelConfigEngineType:
    """Tests for engine_type validation in ModelConfig."""

    def test_default_engine_type(self):
        config = ModelConfig()
        assert config.engine_type == "vllm"

    def test_vllm_engine_type(self):
        config = ModelConfig(engine_type="vllm")
        assert config.engine_type == "vllm"

    def test_sglang_engine_type(self):
        config = ModelConfig(engine_type="sglang")
        assert config.engine_type == "sglang"

    def test_invalid_engine_type_raises(self):
        with pytest.raises(ValueError, match="engine_type must be one of"):
            ModelConfig(engine_type="invalid")


class TestVLLMBackendNormalization:
    """Test VLLMBackend output normalization with mocked vLLM internals."""

    def test_generate_normalizes_output(self):
        from vgate.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()

        # Create mock vLLM output
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "Generated text"
        mock_output.outputs[0].token_ids = [1, 2, 3, 4, 5]
        mock_output.metrics = MagicMock()
        mock_output.metrics.first_token_time = 1.1
        mock_output.metrics.arrival_time = 1.0
        mock_output.metrics.finished_time = 1.5

        mock_llm = MagicMock()
        mock_llm.generate.return_value = [mock_output]
        backend.llm = mock_llm

        results = backend.generate(["test prompt"], MagicMock())
        assert len(results) == 1
        assert results[0]["text"] == "Generated text"
        assert results[0]["num_tokens"] == 5
        assert results[0]["token_ids"] == [1, 2, 3, 4, 5]
        assert abs(results[0]["metrics"]["ttft"] - 0.1) < 0.001
        assert abs(results[0]["metrics"]["gen_time"] - 0.4) < 0.001

    def test_generate_without_metrics(self):
        from vgate.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "No metrics text"
        mock_output.outputs[0].token_ids = [1, 2, 3]
        mock_output.metrics = None

        mock_llm = MagicMock()
        mock_llm.generate.return_value = [mock_output]
        backend.llm = mock_llm

        results = backend.generate(["test"], MagicMock())
        assert results[0]["metrics"] == {}


class TestSGLangBackendNormalization:
    """Test SGLangBackend output normalization with mocked SGLang internals."""

    def test_generate_normalizes_output(self):
        from vgate.backends.sglang_backend import SGLangBackend

        backend = SGLangBackend()

        mock_engine = MagicMock()
        mock_engine.generate.return_value = [
            {
                "text": "SGLang generated text",
                "meta_info": {"completion_tokens_ids": [10, 20, 30]},
            }
        ]
        backend.engine = mock_engine

        params = backend.create_sampling_params(0.7, 0.9, 128)
        assert params["max_new_tokens"] == 128

        results = backend.generate(["test prompt"], params)
        assert len(results) == 1
        assert results[0]["text"] == "SGLang generated text"
        assert results[0]["num_tokens"] == 3
        assert results[0]["token_ids"] == [10, 20, 30]
        assert "wall_time" in results[0]["metrics"]

    def test_generate_without_token_ids(self):
        from vgate.backends.sglang_backend import SGLangBackend

        backend = SGLangBackend()

        mock_engine = MagicMock()
        mock_engine.generate.return_value = [
            {"text": "hello world response", "meta_info": {}}
        ]
        backend.engine = mock_engine

        results = backend.generate(["test"], {})
        # Should fallback to word-count estimation
        assert results[0]["num_tokens"] >= 1
        assert results[0]["token_ids"] == []
