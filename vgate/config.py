"""
V-Gate Configuration Module.

Provides Pydantic-based configuration management with support for:
1. YAML configuration files
2. Environment variable overrides (using __ as nested separator)
3. Default values in Pydantic models

Configuration priority (highest to lowest):
1. Environment variables (e.g., VGATE_MODEL__MODEL_ID)
2. YAML configuration file
3. Pydantic model defaults
"""
import os
from pathlib import Path
from typing import Any, Optional, Tuple, Type

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000


class ModelConfig(BaseModel):
    """Model configuration for vLLM engine."""
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
    quantization: str = "awq"
    gpu_memory_utilization: float = 0.7
    max_model_len: int = 2048
    trust_remote_code: bool = True
    enforce_eager: bool = True


class BatchConfig(BaseModel):
    """Request batching configuration."""
    max_batch_size: int = 8
    max_wait_time_ms: float = 50.0


class CacheConfig(BaseModel):
    """Result cache configuration."""
    enabled: bool = True
    maxsize: int = 1000


class InferenceConfig(BaseModel):
    """Default inference parameters."""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    json_format: bool = True


class MetricsConfig(BaseModel):
    """Metrics configuration."""
    enabled: bool = True


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source that loads from a YAML file."""

    def __init__(self, settings_cls: Type[BaseSettings], yaml_file: Optional[Path] = None):
        super().__init__(settings_cls)
        self.yaml_file = yaml_file
        self._yaml_data: Optional[dict] = None

    def _load_yaml(self) -> dict:
        """Load YAML data from file."""
        if self._yaml_data is not None:
            return self._yaml_data

        if self.yaml_file and self.yaml_file.exists():
            with open(self.yaml_file, "r", encoding="utf-8") as f:
                self._yaml_data = yaml.safe_load(f) or {}
        else:
            self._yaml_data = {}

        return self._yaml_data

    def get_field_value(
        self, field: Any, field_name: str
    ) -> Tuple[Any, str, bool]:
        """Get value for a field from YAML data."""
        yaml_data = self._load_yaml()
        field_value = yaml_data.get(field_name)
        return field_value, field_name, field_value is not None

    def __call__(self) -> dict[str, Any]:
        """Return all values from YAML."""
        return self._load_yaml()


# Global YAML file path for config loading
_yaml_config_path: Optional[Path] = None


class VGateConfig(BaseSettings):
    """
    Root configuration for V-Gate.

    Configuration is loaded from:
    1. Environment variables with VGATE_ prefix (e.g., VGATE_SERVER__PORT=8080)
    2. YAML configuration file (if provided)
    3. Default values defined in this model

    Nested values use double underscore (__) as separator in env vars.
    Example: VGATE_MODEL__MODEL_ID, VGATE_BATCH__MAX_BATCH_SIZE
    """
    model_config = SettingsConfigDict(
        env_prefix="VGATE_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    version: str = "0.3.1"
    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """
        Customize settings sources priority.

        Priority (first = highest):
        1. Init settings (passed to constructor)
        2. Environment variables
        3. YAML file
        4. Default values (handled by Pydantic automatically)
        """
        global _yaml_config_path
        yaml_source = YamlConfigSettingsSource(settings_cls, _yaml_config_path)
        return (init_settings, env_settings, yaml_source)


def load_yaml_config(path: str | Path) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration values.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    return config_data or {}


def load_config(path: Optional[str | Path] = None) -> VGateConfig:
    """
    Load and validate configuration from YAML file and environment variables.

    Configuration priority (highest to lowest):
    1. Environment variables (VGATE_* prefix)
    2. YAML configuration file (if provided)
    3. Default values

    Args:
        path: Optional path to YAML configuration file.

    Returns:
        Validated VGateConfig instance.
    """
    global _yaml_config_path

    if path:
        _yaml_config_path = Path(path)
    else:
        _yaml_config_path = None

    return VGateConfig()


# Global configuration singleton
_config: Optional[VGateConfig] = None


def get_config() -> VGateConfig:
    """
    Get the global configuration instance.

    On first call, attempts to load configuration from:
    1. Path specified in VGATE_CONFIG_PATH environment variable
    2. ./config.yaml if it exists
    3. Default values only

    Returns:
        The global VGateConfig instance.
    """
    global _config

    if _config is None:
        config_path = os.getenv("VGATE_CONFIG_PATH")

        if config_path:
            _config = load_config(config_path)
        elif Path("config.yaml").exists():
            _config = load_config("config.yaml")
        else:
            _config = VGateConfig()

    return _config


def reset_config() -> None:
    """
    Reset the global configuration singleton.

    Primarily used for testing to ensure a clean state between tests.
    """
    global _config, _yaml_config_path
    _config = None
    _yaml_config_path = None
