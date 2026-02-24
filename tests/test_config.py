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
Tests for V-Gate configuration module.

Tests cover:
- YAML configuration loading
- Environment variable overrides
- Configuration validation
- Priority ordering (env > yaml > defaults)
"""
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from vgate.config import (
    VGateConfig,
    ServerConfig,
    ModelConfig,
    BatchConfig,
    CacheConfig,
    InferenceConfig,
    LoggingConfig,
    MetricsConfig,
    load_yaml_config,
    load_config,
    get_config,
    reset_config,
)


@pytest.fixture(autouse=True)
def clean_config():
    """Reset config before and after each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def clean_env():
    """Clean VGATE_ environment variables for testing."""
    # Store original values
    original_env = {k: v for k, v in os.environ.items() if k.startswith("VGATE_")}

    # Clear VGATE_ variables
    for key in list(os.environ.keys()):
        if key.startswith("VGATE_"):
            del os.environ[key]

    yield

    # Restore original values
    for key in list(os.environ.keys()):
        if key.startswith("VGATE_"):
            del os.environ[key]
    os.environ.update(original_env)


class TestDefaultConfig:
    """Test default configuration values."""

    def test_default_server_config(self, clean_env):
        """Test default server configuration."""
        config = VGateConfig()
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 8000

    def test_default_model_config(self, clean_env):
        """Test default model configuration."""
        config = VGateConfig()
        assert config.model.model_id == "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
        assert config.model.quantization == "awq"
        assert config.model.gpu_memory_utilization == 0.7
        assert config.model.max_model_len == 2048
        assert config.model.trust_remote_code is True
        assert config.model.enforce_eager is True

    def test_default_batch_config(self, clean_env):
        """Test default batch configuration."""
        config = VGateConfig()
        assert config.batch.max_batch_size == 8
        assert config.batch.max_wait_time_ms == 50.0

    def test_default_cache_config(self, clean_env):
        """Test default cache configuration."""
        config = VGateConfig()
        assert config.cache.enabled is True
        assert config.cache.maxsize == 1000

    def test_default_inference_config(self, clean_env):
        """Test default inference configuration."""
        config = VGateConfig()
        assert config.inference.temperature == 0.7
        assert config.inference.top_p == 0.9
        assert config.inference.max_tokens == 256

    def test_default_logging_config(self, clean_env):
        """Test default logging configuration."""
        config = VGateConfig()
        assert config.logging.level == "INFO"
        assert config.logging.json_format is True

    def test_default_metrics_config(self, clean_env):
        """Test default metrics configuration."""
        config = VGateConfig()
        assert config.metrics.enabled is True

    def test_default_version(self, clean_env):
        """Test default version."""
        config = VGateConfig()
        assert config.version == "0.3.2"


class TestYamlLoading:
    """Test YAML configuration loading."""

    def test_load_yaml_config(self, clean_env):
        """Test loading configuration from YAML file."""
        config_data = {
            "version": "1.0.0",
            "server": {"host": "127.0.0.1", "port": 9000},
            "model": {"model_id": "test-model"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            loaded = load_yaml_config(temp_path)
            assert loaded["version"] == "1.0.0"
            assert loaded["server"]["host"] == "127.0.0.1"
            assert loaded["server"]["port"] == 9000
            assert loaded["model"]["model_id"] == "test-model"
        finally:
            os.unlink(temp_path)

    def test_load_yaml_config_file_not_found(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config("/nonexistent/path/config.yaml")

    def test_load_config_from_yaml(self, clean_env):
        """Test load_config with YAML file."""
        config_data = {
            "server": {"port": 9000},
            "batch": {"max_batch_size": 16},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.server.port == 9000
            assert config.batch.max_batch_size == 16
            # Defaults should still apply
            assert config.server.host == "0.0.0.0"
            assert config.model.model_id == "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
        finally:
            os.unlink(temp_path)


class TestEnvironmentOverrides:
    """Test environment variable overrides."""

    def test_env_override_server_port(self, clean_env):
        """Test environment variable overrides server port."""
        os.environ["VGATE_SERVER__PORT"] = "9999"
        config = VGateConfig()
        assert config.server.port == 9999

    def test_env_override_model_id(self, clean_env):
        """Test environment variable overrides model ID."""
        os.environ["VGATE_MODEL__MODEL_ID"] = "custom-model"
        config = VGateConfig()
        assert config.model.model_id == "custom-model"

    def test_env_override_batch_size(self, clean_env):
        """Test environment variable overrides batch size."""
        os.environ["VGATE_BATCH__MAX_BATCH_SIZE"] = "32"
        config = VGateConfig()
        assert config.batch.max_batch_size == 32

    def test_env_override_cache_maxsize(self, clean_env):
        """Test environment variable overrides cache maxsize."""
        os.environ["VGATE_CACHE__MAXSIZE"] = "5000"
        config = VGateConfig()
        assert config.cache.maxsize == 5000

    def test_env_override_logging_level(self, clean_env):
        """Test environment variable overrides logging level."""
        os.environ["VGATE_LOGGING__LEVEL"] = "DEBUG"
        config = VGateConfig()
        assert config.logging.level == "DEBUG"

    def test_env_override_logging_json_format(self, clean_env):
        """Test environment variable overrides logging json_format."""
        os.environ["VGATE_LOGGING__JSON_FORMAT"] = "false"
        config = VGateConfig()
        assert config.logging.json_format is False

    def test_env_override_gpu_memory(self, clean_env):
        """Test environment variable overrides GPU memory utilization."""
        os.environ["VGATE_MODEL__GPU_MEMORY_UTILIZATION"] = "0.9"
        config = VGateConfig()
        assert config.model.gpu_memory_utilization == 0.9


class TestConfigPriority:
    """Test configuration priority ordering."""

    def test_env_overrides_yaml(self, clean_env):
        """Test that environment variables override YAML values."""
        config_data = {
            "server": {"port": 8080},
            "batch": {"max_batch_size": 16},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            # Set env var to override YAML
            os.environ["VGATE_SERVER__PORT"] = "9999"
            os.environ["VGATE_BATCH__MAX_BATCH_SIZE"] = "32"

            config = load_config(temp_path)

            # Env should override YAML
            assert config.server.port == 9999
            assert config.batch.max_batch_size == 32
        finally:
            os.unlink(temp_path)

    def test_yaml_overrides_defaults(self, clean_env):
        """Test that YAML values override defaults."""
        config_data = {
            "server": {"port": 8080},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            # YAML should override default
            assert config.server.port == 8080
            # Default should apply where not specified
            assert config.server.host == "0.0.0.0"
        finally:
            os.unlink(temp_path)


class TestGetConfig:
    """Test get_config singleton behavior."""

    def test_get_config_returns_same_instance(self, clean_env):
        """Test that get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_config_clears_singleton(self, clean_env):
        """Test that reset_config clears the singleton."""
        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2

    def test_get_config_uses_env_path(self, clean_env):
        """Test that get_config uses VGATE_CONFIG_PATH."""
        config_data = {
            "server": {"port": 7777},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            os.environ["VGATE_CONFIG_PATH"] = temp_path
            reset_config()
            config = get_config()
            assert config.server.port == 7777
        finally:
            del os.environ["VGATE_CONFIG_PATH"]
            os.unlink(temp_path)


class TestValidation:
    """Test configuration validation."""

    def test_invalid_port_type(self, clean_env):
        """Test that invalid port type raises error."""
        os.environ["VGATE_SERVER__PORT"] = "not_a_number"
        with pytest.raises(Exception):
            VGateConfig()

    def test_invalid_batch_size_type(self, clean_env):
        """Test that invalid batch size type raises error."""
        os.environ["VGATE_BATCH__MAX_BATCH_SIZE"] = "not_a_number"
        with pytest.raises(Exception):
            VGateConfig()

    def test_invalid_gpu_memory_type(self, clean_env):
        """Test that invalid GPU memory type raises error."""
        os.environ["VGATE_MODEL__GPU_MEMORY_UTILIZATION"] = "not_a_float"
        with pytest.raises(Exception):
            VGateConfig()


class TestSubConfigModels:
    """Test individual config model classes."""

    def test_server_config_model(self):
        """Test ServerConfig model."""
        config = ServerConfig(host="localhost", port=3000)
        assert config.host == "localhost"
        assert config.port == 3000

    def test_model_config_model(self):
        """Test ModelConfig model."""
        config = ModelConfig(
            model_id="my-model",
            quantization="gptq",
            gpu_memory_utilization=0.8,
        )
        assert config.model_id == "my-model"
        assert config.quantization == "gptq"
        assert config.gpu_memory_utilization == 0.8

    def test_batch_config_model(self):
        """Test BatchConfig model."""
        config = BatchConfig(max_batch_size=16, max_wait_time_ms=100.0)
        assert config.max_batch_size == 16
        assert config.max_wait_time_ms == 100.0

    def test_cache_config_model(self):
        """Test CacheConfig model."""
        config = CacheConfig(enabled=False, maxsize=500)
        assert config.enabled is False
        assert config.maxsize == 500

    def test_inference_config_model(self):
        """Test InferenceConfig model."""
        config = InferenceConfig(temperature=0.5, top_p=0.8, max_tokens=512)
        assert config.temperature == 0.5
        assert config.top_p == 0.8
        assert config.max_tokens == 512

    def test_logging_config_model(self):
        """Test LoggingConfig model."""
        config = LoggingConfig(level="DEBUG", json_format=False)
        assert config.level == "DEBUG"
        assert config.json_format is False

    def test_metrics_config_model(self):
        """Test MetricsConfig model."""
        config = MetricsConfig(enabled=False)
        assert config.enabled is False
