"""Kronos client: offline formatting, disabled sentinel, and graceful failure."""

import io
import json
import urllib.error

import pytest

from tradingagents.dataflows import kronos


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


@pytest.fixture
def payload():
    return {
        "ticker": "AAPL",
        "as_of": "2026-11-05",
        "last_close": 344.17,
        "horizon": 3,
        "predicted_dates": ["2026-11-06", "2026-11-09", "2026-11-10"],
        "predicted_close": [343.35, 339.96, 334.69],
        "implied_move_pct": -2.75,
        "model": "NeoQuasar/Kronos-small",
        "samples": 3,
        "elapsed_ms": 1800,
    }


def test_disabled_when_url_missing(monkeypatch):
    monkeypatch.delenv("KRONOS_API_URL", raising=False)
    out = kronos.get_kronos_forecast("AAPL")
    assert out.startswith("KRONOS_DISABLED")
    assert "do not estimate" in out


def test_unavailable_on_network_error(monkeypatch):
    monkeypatch.setenv("KRONOS_API_URL", "http://127.0.0.1:9")

    def boom(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(kronos.urllib.request, "urlopen", boom)
    out = kronos.get_kronos_forecast("AAPL")
    assert out.startswith("KRONOS_UNAVAILABLE")
    assert "do not invent" in out


def test_formats_forecast_path(monkeypatch, payload):
    monkeypatch.setenv("KRONOS_API_URL", "http://kronos.local")
    monkeypatch.setattr(
        kronos.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse(json.dumps(payload).encode()),
    )
    out = kronos.get_kronos_forecast("aapl", horizon=3)
    assert "AAPL" in out
    assert "2026-11-06: 343.35" in out
    assert "-2.75%" in out
    assert "never as a standalone buy/sell signal" in out
