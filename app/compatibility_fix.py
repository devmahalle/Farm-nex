#!/usr/bin/env python3
"""
Compatibility fix for Python 3.13 pkgutil.ImpImporter issue
This should be imported before any other modules
"""

import sys
import pkgutil

# Fix for Python 3.13 compatibility
if sys.version_info >= (3, 13):
    # Monkey patch pkgutil to add the missing ImpImporter attribute
    if not hasattr(pkgutil, 'ImpImporter'):
        class ImpImporter:
            def __init__(self, *args, **kwargs):
                pass
        pkgutil.ImpImporter = ImpImporter

# Apply the fix
if hasattr(pkgutil, 'ImpImporter'):
    print("✅ Compatibility fix applied for Python 3.13")
else:
    print("⚠️  Compatibility fix not needed")
