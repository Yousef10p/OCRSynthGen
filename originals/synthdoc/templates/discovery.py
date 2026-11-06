"""
Find template folders on disk and register them.

A template is a folder holding ``template.json``::

    {"name": "invoice", "category": "financial", "class": "InvoiceTemplate"}

next to an ``__init__.py`` that exports that class, plus its ``template.html`` and
``styles.css``. Folders are searched in this order, and a later folder that registers
a name already taken replaces the earlier one:

1. the templates bundled with synthdoc (this package directory);
2. ``library/`` inside this package directory, where the synthdoc-templates
   collection is cloned in a source checkout;
3. every directory in the ``SYNTHDOC_TEMPLATE_PATH`` environment variable
   (separated by ``os.pathsep``);
4. directories passed to :func:`add_template_dir`.

Each search directory may hold template folders directly, or folders of them
(so a cloned collection can sit beside templates of your own).
"""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List

from .registry import registry

PACKAGE_DIR = Path(__file__).parent
LIBRARY_DIR = PACKAGE_DIR / "library"
ENV_VAR = "SYNTHDOC_TEMPLATE_PATH"

_extra_dirs: List[Path] = []


def search_path() -> List[Path]:
    """The directories searched for templates, in registration order."""
    dirs = [PACKAGE_DIR, LIBRARY_DIR]
    env = os.environ.get(ENV_VAR, "")
    dirs += [Path(p).expanduser() for p in env.split(os.pathsep) if p.strip()]
    dirs += _extra_dirs
    return dirs


def _template_folders(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        if (child / "template.json").is_file():
            yield child
        elif child != LIBRARY_DIR:
            # A collection: one more level of template folders
            for grandchild in sorted(child.iterdir()):
                if grandchild.is_dir() and (grandchild / "template.json").is_file():
                    yield grandchild


def _module_name(folder: Path) -> str:
    try:
        rel = folder.resolve().relative_to(PACKAGE_DIR.resolve())
        return "synthdoc.templates." + ".".join(rel.parts)
    except ValueError:
        slug = re.sub(r"\W", "_", str(folder.resolve()))
        return f"synthdoc_external_templates.{slug}"


def _load_class(folder: Path, class_name: str):
    module_name = _module_name(folder)
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            module_name, folder / "__init__.py", submodule_search_locations=[str(folder)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[module_name]
            raise
    return getattr(module, class_name)


def register_folder(folder: Path) -> str:
    """Register the template in ``folder`` and return its name."""
    folder = Path(folder)
    meta = json.loads((folder / "template.json").read_text(encoding="utf-8"))
    template_class = _load_class(folder, meta["class"])
    registry.register(meta["name"], template_class, meta.get("category", "general"))
    return meta["name"]


def discover(dirs: Iterable[Path] = None) -> List[str]:
    """Register every template found in ``dirs`` (default: :func:`search_path`)."""
    names = []
    for root in (search_path() if dirs is None else dirs):
        for folder in _template_folders(Path(root)):
            names.append(register_folder(folder))
    return names


def add_template_dir(path) -> List[str]:
    """Search ``path`` for templates now and on any later :func:`discover`."""
    path = Path(path).expanduser()
    if path not in _extra_dirs:
        _extra_dirs.append(path)
    return discover([path])
