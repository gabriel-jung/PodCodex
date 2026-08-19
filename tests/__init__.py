"""Test package init.

Pins the loopback API token via env before any test module imports
``podcodex.api.app`` (whose module-level ``app = create_app()`` would
otherwise read or create the developer's real ``~/.config/podcodex/api_token``
during collection). ``make_client`` sets the same value per test.
"""

import os

os.environ.setdefault("PODCODEX_API_TOKEN", "test-token")
