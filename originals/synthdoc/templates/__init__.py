"""
Template system for document generation.
This module provides a configurable template system for generating various document types
with different layouts, fonts, and styling options.
"""
from .base import BaseTemplate, Component, TableComponent, HeaderComponent, LineNumbersComponent, TemplateConfig
from .registry import registry, TemplateRegistry
from .config_manager import config_manager, ConfigManager
from .batch_generator import BatchGenerator
from .entity_generator import EntityGenerator
from .discovery import discover, add_template_dir, search_path

__all__ = [
    'BaseTemplate',
    'Component',
    'TableComponent',
    'HeaderComponent',
    'LineNumbersComponent',
    'TemplateConfig',
    'registry',
    'TemplateRegistry',
    'config_manager',
    'ConfigManager',
    'BatchGenerator',
    'EntityGenerator',
    'discover',
    'add_template_dir',
    'search_path',
]

discover()
