"""Show-password hashing: the single owner of the stored hash format.

The ``password_hash`` column in the IndexStore (``sha256:<hex>``) is written
from three places: the bot's ``/changepassword`` command, the bot's password
CLI, and the desktop API's bot-access route. They all go through the helpers
here so the scheme has one definition.

Dependency-free on purpose: the desktop API imports this, and the packaged
sidecar ships without the ``bot`` extra (no discord.py), so this must not
live in a module that imports discord.
"""

from __future__ import annotations

import hashlib
import hmac

_PREFIX = "sha256:"


def hash_show_password(password: str) -> str:
    """Hash a show password into the stored ``sha256:<hex>`` format."""
    return f"{_PREFIX}{hashlib.sha256(password.encode()).hexdigest()}"


def verify_show_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash, in constant time."""
    # Hash before the format check so malformed rows take the same time as
    # valid ones.
    actual = hashlib.sha256(password.encode()).hexdigest()
    if not stored_hash.startswith(_PREFIX):
        return False
    return hmac.compare_digest(actual, stored_hash.removeprefix(_PREFIX))
