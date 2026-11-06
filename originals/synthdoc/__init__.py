"""
synthdoc: synthetic documents with pixel-accurate entity annotations.

    import synthdoc

    synthdoc.list_templates()                       # every registered template name
    synthdoc.generate(templates=["invoice"], count=5, out_dir="data")

synthdoc bundles a few example templates; the full collection lives in synthdoc-templates
and is added with ``synthdoc.add_template_dir(path)`` or ``SYNTHDOC_TEMPLATE_PATH``.

Templates are HTML + CSS + a Python data generator. Each document is rendered with
Paged.js in a headless browser, so every annotated entity gets a bounding box taken
from the real layout, then exported as page images with JSON, Turtle and JSON-LD labels.
"""
from importlib import import_module

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "generate",
    "list_templates",
    "list_categories",
    "get_template",
    "add_template_dir",
    "registry",
    "BaseTemplate",
    "load_config",
    "SyntheticDocsConfig",
]

# Importing the template registry loads ~260 template modules, so the public names
# resolve on first use rather than on `import synthdoc`.
_LAZY = {
    "generate": ("synthdoc.cli", "generate"),
    "registry": ("synthdoc.templates.registry", "registry"),
    "BaseTemplate": ("synthdoc.templates.base", "BaseTemplate"),
    "load_config": ("synthdoc.config", "load_config"),
    "SyntheticDocsConfig": ("synthdoc.config", "SyntheticDocsConfig"),
}


def __getattr__(name):
    if name in _LAZY:
        module, attr = _LAZY[name]
        if module == "synthdoc.templates.registry":
            import_module("synthdoc.templates")  # registers every template
        value = getattr(import_module(module), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'synthdoc' has no attribute {name!r}")


def list_templates(category=None):
    """Names of the registered templates, optionally only those in ``category``."""
    return sorted(__getattr__("registry").list_templates(category))


def list_categories():
    """Template categories, such as ``financial``, ``medical`` or ``legal``."""
    return sorted(__getattr__("registry").get_categories())


def get_template(name, *args, **kwargs):
    """An instance of the template registered as ``name``."""
    return __getattr__("registry").get_template_instance(name, *args, **kwargs)


def add_template_dir(path):
    """Register the templates in ``path`` (a template folder's parent, or a clone of
    synthdoc-templates) and return their names."""
    __getattr__("registry")
    return import_module("synthdoc.templates.discovery").add_template_dir(path)
