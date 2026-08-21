"""Runtime hook: install setuptools' distutils shim without importing setuptools.

Replaces the auto-injected ``pyi_rth_setuptools``, whose body is::

    import setuptools
    setuptools_major = int(setuptools.__version__.split('.')[0])
    default_value = "stdlib" if setuptools_major < 60 else "local"
    if os.environ.get("SETUPTOOLS_USE_DISTUTILS", default_value) == "local":
        import _distutils_hack
        _distutils_hack.add_shim()

That ``import setuptools`` costs ~136 ms at every launch and exists only to
read a version number and pick a default. ``_distutils_hack.add_shim()``
itself is free (measured 0.0 ms) and pulls in nothing.

The shim is not optional: distutils left the stdlib in 3.12, so it is what
makes ``import distutils`` resolve at all. It stays installed, just without
the version lookup.

The default is hardcoded to "local", which is what setuptools >= 60 picks.
Anything older would want "stdlib" and no shim at all — long past EOL, and
``tests/test_pyinstaller_rthooks.py`` fails if the pinned setuptools ever
drops below that line. An explicit
``SETUPTOOLS_USE_DISTUTILS`` still wins, as before.
"""

import os

if os.environ.get("SETUPTOOLS_USE_DISTUTILS", "local") == "local":
    try:
        import _distutils_hack

        _distutils_hack.add_shim()
    except Exception:
        # Same posture as the hook this replaces: a missing shim surfaces
        # later as an ImportError on distutils, which is a clearer failure
        # than dying in a runtime hook before any logging exists.
        pass
