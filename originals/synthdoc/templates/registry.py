"""
Template registry for managing different document templates.
"""
from typing import Dict, Type, List
from .base import BaseTemplate
import inspect


class TemplateRegistry:
    """Central registry for all available templates"""
    
    def __init__(self):
        self._templates: Dict[str, Type[BaseTemplate]] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(self, name: str, template_class: Type[BaseTemplate], category: str = "general"):
        """Register a template class with a name and category"""
        self._templates[name] = template_class
        
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)
    
    def get_template(self, name: str) -> Type[BaseTemplate]:
        """Get a template class by name"""
        if name not in self._templates:
            raise ValueError(f"Template '{name}' not found in registry")
        return self._templates[name]
    
    def get_template_instance(self, name: str, *args, **kwargs) -> BaseTemplate:
        """Get an instance of a template by name"""
        template_class = self.get_template(name)
        return template_class(*args, **kwargs)
    
    def list_templates(self, category: str = None) -> List[str]:
        """List all available templates, optionally filtered by category"""
        if category:
            return self._categories.get(category, [])
        return list(self._templates.keys())
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self._categories.keys())
    
    def register_from_module(self, module, category: str = "general"):
        """Register all BaseTemplate subclasses from a module"""
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                issubclass(obj, BaseTemplate) and
                obj != BaseTemplate):
                template_name = name.lower().replace('template', '').replace('doc', '')
                self.register(template_name, obj, category)

    def validate(self, name: str):
        """
        Validate a template before generation.

        Args:
            name: Template name to validate

        Returns:
            ValidationReport from validation module

        Raises:
            ValueError: If template not found in registry

        Example:
            >>> from synthdoc.templates import registry
            >>> report = registry.validate("invoice")
            >>> if report.is_valid:
            ...     print("Template is valid!")
        """
        from ..validation import validate_template

        if name not in self._templates:
            raise ValueError(f"Template '{name}' not found in registry")

        template_class = self._templates[name]
        return validate_template(template_class)

    def validate_all(self):
        """
        Validate all registered templates.

        Returns:
            Dict mapping template names to ValidationReports

        Example:
            >>> from synthdoc.templates import registry
            >>> results = registry.validate_all()
            >>> for name, report in results.items():
            ...     if not report.is_valid:
            ...         print(f"{name}: {len(report.errors)} errors")
        """
        from ..validation import validate_template

        results = {}
        for name, template_class in self._templates.items():
            results[name] = validate_template(template_class)
        return results


# Global registry instance
registry = TemplateRegistry()

# Register built-in categories
registry._categories = {
    "legal": [],
    "healthcare": [],
    "real_estate": [],
    "financial": [],
    "general": []
}