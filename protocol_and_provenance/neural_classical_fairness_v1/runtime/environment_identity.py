"""Identify imported runtime modules; metadata is a separate, lossless inventory."""
from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path
import re

MODULES = {"numpy": "numpy", "pandas": "pandas", "pyarrow": "pyarrow",
           "scikit-learn": "sklearn", "torch": "torch", "psutil": "psutil"}


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def collect(import_module=None, distributions=None):
    """Fail on import/identity errors, never substitute distribution metadata."""
    import_module = import_module or importlib.import_module
    distributions = distributions or importlib.metadata.distributions
    imported = {}
    for package, module_name in MODULES.items():
        module = import_module(module_name)
        version = getattr(module, "__version__", None)
        filename = getattr(module, "__file__", None)
        if not isinstance(version, str) or not version.strip() or not filename:
            raise RuntimeError("RUNTIME_MODULE_IDENTITY_MISSING:" + package)
        imported[package] = dict(module=module_name, version=version,
                                 file=str(Path(filename).resolve()))
    inventory = {}
    for dist in distributions():
        name = dist.metadata.get("Name")
        if name:
            inventory.setdefault(normalize(name), []).append(
                dict(version=str(dist.version), location=str(Path(dist.locate_file("")).resolve())))
    inventory = {k: sorted(v, key=lambda r: (r["location"], r["version"]))
                 for k, v in sorted(inventory.items())}
    return dict(identity_source="imported_module_version_and_file",
                packages={k: r["version"] for k, r in imported.items()},
                imported_modules=imported, distribution_inventory=inventory,
                duplicate_distributions={k: v for k, v in inventory.items() if len(v) > 1})
