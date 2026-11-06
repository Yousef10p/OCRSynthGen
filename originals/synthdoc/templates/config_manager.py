"""
Configuration management system for document templates.
Handles template variations, batch generation, and styling options.
"""
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from .base import TemplateConfig


class ConfigManager:
    """Manages template configurations and variations"""
    
    def __init__(self, config_dir: str = "template_configs"):
        self.config_dir = Path(config_dir)
    
    def get_config(self, config_name: str) -> TemplateConfig:
        """Load a configuration by name"""
        # Try JSON first
        config_path = self.config_dir / f"{config_name}.json"
        if not config_path.exists():
            # Try YAML
            config_path = self.config_dir / f"{config_name}.yaml"
            if not config_path.exists():
                config_path = self.config_dir / f"{config_name}.yml"
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Filter data to only include valid TemplateConfig fields
            filtered_data = {
                k: v for k, v in config_data.items() 
                if k in TemplateConfig.__dataclass_fields__
            }
            return TemplateConfig(**filtered_data)
        else:
            # Return default config
            return TemplateConfig()
    
    def save_config(self, config: TemplateConfig, config_name: str, format: str = 'json') -> Path:
        """Save a configuration to a file"""
        if format.lower() == 'json':
            config_path = self.config_dir / f"{config_name}.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(config), f, indent=2)
        else:  # YAML
            config_path = self.config_dir / f"{config_name}.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(asdict(config), f, default_flow_style=False)
        
        return config_path
    
    def list_configs(self, category: str = None) -> List[str]:
        """List all available configurations, optionally filtered by category"""
        configs = []
        
        for file_path in self.config_dir.glob("*"):
            if file_path.suffix.lower() in ['.json', '.yaml', '.yml']:
                name = file_path.stem
                if category is None or name.startswith(category):
                    configs.append(name)
        
        return sorted(configs)
    
    def generate_variations(self, base_config_name: str, variation_params: Dict[str, List[Any]]) -> List[str]:
        """
        Generate multiple configuration variations based on parameter options.
        
        Args:
            base_config_name: Name of the base configuration to vary
            variation_params: Dictionary mapping config fields to lists of possible values
        
        Returns:
            List of names of generated variation configurations
        """
        base_config = self.get_config(base_config_name)
        variation_names = []
        
        # Create a list of parameter combinations
        from itertools import product
        
        # Prepare parameter values for product
        param_names = list(variation_params.keys())
        param_values = list(variation_params.values())
        
        # Generate all combinations
        for combo in product(*param_values):
            # Create new config based on base config
            new_config = TemplateConfig(**asdict(base_config))
            
            # Apply parameter combination
            for i, param_name in enumerate(param_names):
                if param_name in TemplateConfig.__dataclass_fields__:
                    setattr(new_config, param_name, combo[i])
            
            # Create config name based on parameters
            param_str = "_".join([f"{k}_{v}" for k, v in zip(param_names, combo)])
            variation_name = f"{base_config_name}_var_{len(variation_names):03d}_{param_str}"
            
            # Save the new configuration
            self.save_config(new_config, variation_name)
            variation_names.append(variation_name)
        
        return variation_names
    
    def create_config_from_params(self, config_name: str, **params) -> TemplateConfig:
        """Create a new configuration with specified parameters"""
        # Start with default config
        config = TemplateConfig()
        
        # Apply parameters that match TemplateConfig fields
        for param_name, param_value in params.items():
            if param_name in TemplateConfig.__dataclass_fields__:
                setattr(config, param_name, param_value)
        
        # Save the new config
        self.save_config(config, config_name)
        return config


# Create a global config manager instance
config_manager = ConfigManager()