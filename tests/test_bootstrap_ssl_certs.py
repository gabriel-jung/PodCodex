"""Tests for bootstrap._wire_ssl_certs.

The PyInstaller-frozen bundle ships an OpenSSL whose compiled-in default
cert path points at the build machine, so stdlib ``ssl`` (urllib,
torch.hub downloads) cannot verify any HTTPS peer on user machines.
``_wire_ssl_certs`` points ``SSL_CERT_FILE`` at certifi's CA bundle,
but must never clobber a user-provided override (corporate proxies).
"""

from __future__ import annotations

import os

import certifi
import pytest

from podcodex import bootstrap


def test_sets_ssl_cert_file_to_certifi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    bootstrap._wire_ssl_certs()
    assert os.environ["SSL_CERT_FILE"] == certifi.where()
    assert os.path.isfile(os.environ["SSL_CERT_FILE"])


def test_respects_existing_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSL_CERT_FILE", "/corporate/ca-bundle.pem")
    bootstrap._wire_ssl_certs()
    assert os.environ["SSL_CERT_FILE"] == "/corporate/ca-bundle.pem"
