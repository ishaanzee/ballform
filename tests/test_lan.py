import json
from types import SimpleNamespace

import pytest

import app.lan as lan
from app.lan import choose_transport, local_ip, tailscale_ip
from fastapi.testclient import TestClient

import app.main as main


def test_local_ip_has_four_octets():
    parts = local_ip().split(".")
    assert len(parts) == 4
    assert all(0 <= int(part) <= 255 for part in parts)


def test_pairing_token_protects_api(monkeypatch):
    monkeypatch.setattr(main, "ACCESS_TOKEN", "test-secret")
    client = TestClient(main.app)
    assert client.get("/api/jobs/missing").status_code == 401
    assert client.get("/api/jobs/missing?token=wrong").status_code == 401
    assert client.get("/api/jobs/missing?token=test-secret").status_code == 404
    assert client.get("/").status_code == 200


def test_reads_connected_tailscale_address(monkeypatch):
    status = {"BackendState": "Running", "TailscaleIPs": ["100.91.2.3", "fd7a:115c:a1e0::1"]}
    monkeypatch.setattr(lan, "_tailscale_command", lambda: "/fake/tailscale")
    monkeypatch.setattr(
        lan.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(status), stderr=""),
    )
    assert tailscale_ip() == ("100.91.2.3", "connected")


def test_required_tailscale_gives_actionable_error(monkeypatch):
    monkeypatch.setattr(lan, "tailscale_ip", lambda: (None, "Tailscale is stopped."))
    with pytest.raises(RuntimeError, match="same account"):
        choose_transport("tailscale")


def test_auto_prefers_tailscale(monkeypatch):
    monkeypatch.setattr(lan, "tailscale_ip", lambda: ("100.80.70.60", "connected"))
    transport, warning = choose_transport("auto")
    assert transport.name == "tailscale"
    assert transport.address == "100.80.70.60"
    assert warning is None
