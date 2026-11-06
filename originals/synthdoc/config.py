"""
Configuration management for synthetic document generation.

Supports loading configuration from:
1. YAML/JSON files (~/.synthetic-docs/config.yaml, ./.synthetic-docs.yaml)
2. Environment variables (SYNTHETIC_DOCS_*)
3. Programmatic defaults
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict


@dataclass
class OutputConfig:
    """Output directory and format configuration."""
    base_dir: str = "data"
    images_subdir: str = "images"
    labels_subdir: str = "labels"
    html_subdir: str = "html"
    pdfs_subdir: str = "pdfs"
    formats: List[str] = field(default_factory=lambda: ["json", "ttl", "jsonld"])
    keep_pdfs: bool = True
    create_overlays: bool = True
    create_test_output: bool = False


@dataclass
class GenerationConfig:
    """Document generation configuration."""
    dpi: int = 96  # Standard DPI for 816×1056 px images (8.5"×11" pages)
    image_format: str = "jpg"
    count_per_template: int = 10
    templates: Optional[List[str]] = None  # None = all templates
    entity_types: Optional[List[str]] = None  # None = all entity types
    min_bbox_size: int = 5  # Minimum bbox size in normalized units
    bbox_padding: int = 0  # Padding around bboxes in pixels
    filter_tiny_boxes: bool = True  # Filter out boxes smaller than min_bbox_size
    deterministic_seed: Optional[int] = None  # For reproducible generation
    parallel_workers: int = 1  # Number of parallel workers (1 = sequential)


@dataclass
class ValidationConfig:
    """Validation configuration."""
    strict_mode: bool = False
    validate_before_generation: bool = True
    max_overlap_threshold: float = 0.8  # IoU threshold for overlap warnings


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    log_dir: str = "logs"


@dataclass
class SyntheticDocsConfig:
    """Complete configuration for synthetic document generation."""
    output: OutputConfig = field(default_factory=OutputConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyntheticDocsConfig':
        """Create config from dictionary."""
        return cls(
            output=OutputConfig(**data.get('output', {})),
            generation=GenerationConfig(**data.get('generation', {})),
            validation=ValidationConfig(**data.get('validation', {})),
            logging=LoggingConfig(**data.get('logging', {}))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'output': asdict(self.output),
            'generation': asdict(self.generation),
            'validation': asdict(self.validation),
            'logging': asdict(self.logging)
        }

    def save(self, path: str) -> None:
        """Save configuration to file (YAML or JSON based on extension)."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

        if file_path.suffix in ['.yaml', '.yml']:
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif file_path.suffix == '.json':
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .yaml, .yml, or .json")


def load_config_file(path: str) -> Optional[Dict[str, Any]]:
    """
    Load configuration from YAML or JSON file.

    Args:
        path: Path to config file

    Returns:
        Configuration dictionary or None if file doesn't exist
    """
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return None

    try:
        with open(file_path, 'r') as f:
            if file_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif file_path.suffix == '.json':
                return json.load(f)
            else:
                print(f"Warning: Unsupported config format: {file_path.suffix}")
                return None
    except Exception as e:
        print(f"Warning: Failed to load config from {path}: {e}")
        return None


def load_config_from_env() -> Dict[str, Any]:
    """
    Load configuration from environment variables.

    Environment variables:
        SYNTHETIC_DOCS_OUTPUT_DIR: Output base directory
        SYNTHETIC_DOCS_DPI: Image DPI
        SYNTHETIC_DOCS_FORMATS: Comma-separated output formats (json,ttl,jsonld)
        SYNTHETIC_DOCS_LOG_LEVEL: Logging level
        SYNTHETIC_DOCS_ENTITY_TYPES: Comma-separated entity types to include
        SYNTHETIC_DOCS_MIN_BBOX_SIZE: Minimum bbox size

    Returns:
        Configuration dictionary with values from environment
    """
    config = {}

    # Output config
    if 'SYNTHETIC_DOCS_OUTPUT_DIR' in os.environ:
        config.setdefault('output', {})['base_dir'] = os.environ['SYNTHETIC_DOCS_OUTPUT_DIR']

    if 'SYNTHETIC_DOCS_FORMATS' in os.environ:
        formats = [f.strip() for f in os.environ['SYNTHETIC_DOCS_FORMATS'].split(',')]
        config.setdefault('output', {})['formats'] = formats

    # Generation config
    if 'SYNTHETIC_DOCS_DPI' in os.environ:
        config.setdefault('generation', {})['dpi'] = int(os.environ['SYNTHETIC_DOCS_DPI'])

    if 'SYNTHETIC_DOCS_ENTITY_TYPES' in os.environ:
        entity_types = [e.strip() for e in os.environ['SYNTHETIC_DOCS_ENTITY_TYPES'].split(',')]
        config.setdefault('generation', {})['entity_types'] = entity_types

    if 'SYNTHETIC_DOCS_MIN_BBOX_SIZE' in os.environ:
        config.setdefault('generation', {})['min_bbox_size'] = int(os.environ['SYNTHETIC_DOCS_MIN_BBOX_SIZE'])

    if 'SYNTHETIC_DOCS_SEED' in os.environ:
        config.setdefault('generation', {})['deterministic_seed'] = int(os.environ['SYNTHETIC_DOCS_SEED'])

    # Logging config
    if 'SYNTHETIC_DOCS_LOG_LEVEL' in os.environ:
        config.setdefault('logging', {})['log_level'] = os.environ['SYNTHETIC_DOCS_LOG_LEVEL']

    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Configuration to override base with

    Returns:
        Merged configuration dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def load_config(
    config_path: Optional[str] = None,
    use_defaults: bool = True,
    load_from_env: bool = True
) -> SyntheticDocsConfig:
    """
    Load configuration with priority order:
    1. Explicit config_path parameter (highest priority)
    2. Environment variables
    3. ./.synthetic-docs.yaml (current directory)
    4. ~/.synthetic-docs/config.yaml (user home directory)
    5. Default values (lowest priority)

    Args:
        config_path: Explicit path to config file (highest priority)
        use_defaults: Whether to use default values as base
        load_from_env: Whether to load from environment variables

    Returns:
        SyntheticDocsConfig instance

    Example:
        >>> config = load_config()
        >>> config.generation.dpi
        300
        >>> config.output.formats
        ['json', 'ttl', 'jsonld']
    """
    # Start with defaults
    if use_defaults:
        config_dict = SyntheticDocsConfig().to_dict()
    else:
        config_dict = {}

    # Load from user home config
    home_config = load_config_file("~/.synthetic-docs/config.yaml")
    if home_config:
        config_dict = merge_configs(config_dict, home_config)

    # Load from current directory config
    local_configs = [
        "./.synthetic-docs.yaml",
        "./.synthetic-docs.yml",
        "./.synthetic-docs.json"
    ]
    for local_config_path in local_configs:
        local_config = load_config_file(local_config_path)
        if local_config:
            config_dict = merge_configs(config_dict, local_config)
            break

    # Load from environment variables
    if load_from_env:
        env_config = load_config_from_env()
        if env_config:
            config_dict = merge_configs(config_dict, env_config)

    # Load from explicit path (highest priority)
    if config_path:
        explicit_config = load_config_file(config_path)
        if explicit_config:
            config_dict = merge_configs(config_dict, explicit_config)
        else:
            print(f"Warning: Config file not found: {config_path}")

    return SyntheticDocsConfig.from_dict(config_dict)


# Global config instance (lazy loaded)
_global_config: Optional[SyntheticDocsConfig] = None


def get_config() -> SyntheticDocsConfig:
    """
    Get global configuration instance.

    Loads configuration on first access using default search paths.

    Returns:
        Global SyntheticDocsConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config: SyntheticDocsConfig) -> None:
    """
    Set global configuration instance.

    Args:
        config: Configuration to set as global
    """
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset global configuration to None (will reload on next get_config())."""
    global _global_config
    _global_config = None
