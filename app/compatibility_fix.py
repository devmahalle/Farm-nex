#!/usr/bin/env python3
"""
Compatibility fix for Python 3.12+ and 3.13 compatibility issues
This should be imported before any other modules
"""

import sys
import pkgutil

# Fix for Python 3.13 compatibility - pkgutil.ImpImporter
if sys.version_info >= (3, 13):
    # Monkey patch pkgutil to add the missing ImpImporter attribute
    if not hasattr(pkgutil, 'ImpImporter'):
        class ImpImporter:
            def __init__(self, *args, **kwargs):
                pass
        pkgutil.ImpImporter = ImpImporter

# Fix for Python 3.12+ compatibility - distutils.core
if sys.version_info >= (3, 12):
    try:
        # Try to import distutils from setuptools (which provides compatibility)
        import setuptools
        # Ensure distutils is available via setuptools
        if not hasattr(sys.modules.get('distutils', None), 'core'):
            try:
                from setuptools import distutils
                sys.modules['distutils'] = distutils
                sys.modules['distutils.core'] = distutils.core
            except (ImportError, AttributeError):
                # If setuptools doesn't provide distutils, create a minimal shim
                import types
                distutils_module = types.ModuleType('distutils')
                distutils_core = types.ModuleType('distutils.core')
                distutils_module.core = distutils_core
                sys.modules['distutils'] = distutils_module
                sys.modules['distutils.core'] = distutils_core
    except ImportError:
        # If setuptools is not available, create a minimal shim
        import types
        distutils_module = types.ModuleType('distutils')
        distutils_core = types.ModuleType('distutils.core')
        distutils_module.core = distutils_core
        sys.modules['distutils'] = distutils_module
        sys.modules['distutils.core'] = distutils_core

# Apply the fix
fixes_applied = []
if hasattr(pkgutil, 'ImpImporter'):
    fixes_applied.append("pkgutil.ImpImporter")
if 'distutils' in sys.modules:
    fixes_applied.append("distutils.core")

if fixes_applied:
    print(f"✅ Compatibility fixes applied: {', '.join(fixes_applied)}")
else:
    print("⚠️  Compatibility fixes not needed")
