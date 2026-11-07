"""Client for the Kronos forecast sidecar.

Kronos is a financial K-line foundation model (AAAI 2026) that forecasts the
next N bars from recent OHLCV. It is a *quantitative signal*, not a decision:
forecasts are stochastic samples and carry no backtested P&L here.

The service is optional. When KRONOS_API_URL is unset or unreachable this
module returns a short sentinel string so the analyst pipeline keeps running.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = float(os.environ.get("KRONOS_TIMEOUT", "180"))


def kronos_api_url() -> str | None:
    """Base URL of the sidecar, or None when the feature is switched off."""
    url = os.environ.get("KRONOS_API_URL", "").strip()
    return url.rstrip("/") or None


def fetch_kronos_forecast(
    ticker: str, horizon: int = 10, lookback: int = 400
) -> dict | None:
    """Call the sidecar and return its JSON payload, or None on any failure."""
    base = kronos_api_url()
    if base is None:
        return None

    query = urllib.parse.urlencode(
        {"ticker": ticker.strip().upper(), "horizon": horizon, "lookback": lookback}
    )
    url = f"{base}/forecast?{query}"
    try:
        with urllib.request.urlopen(url, timeout=DEFAULT_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        logger.warning("Kronos sidecar unavailable (%s): %s", url, exc)
        return None


def format_forecast(payload: dict) -> str:
    """Render the sidecar payload as a compact block for an LLM prompt."""
    dates = payload.get("predicted_dates") or []
    closes = payload.get("predicted_close") or []
    path = ", ".join(f"{d}: {c}" for d, c in zip(dates, closes))
    move = payload.get("implied_move_pct")
    direction = "up" if (move or 0) > 0 else "down"
    return (
        f"KRONOS FORECAST — {payload.get('ticker')} (model {payload.get('model')}, "
        f"{payload.get('samples')} samples)\n"
        f"As of {payload.get('as_of')}, last close {payload.get('last_close')}.\n"
        f"Predicted closes for the next {payload.get('horizon')} trading days: {path}\n"
        f"Implied move over the window: {move:+.2f}% ({direction}).\n"
        "This is a stochastic model sample from price history alone — it ignores "
        "news, earnings, and fundamentals. Treat it as one technical input, never "
        "as a standalone buy/sell signal."
    )


def get_kronos_forecast(ticker: str, horizon: int = 10, lookback: int = 400) -> str:
    """Forecast the next `horizon` daily closes with Kronos, as prompt text."""
    payload = fetch_kronos_forecast(ticker, horizon=horizon, lookback=lookback)
    if payload is None:
        if kronos_api_url() is None:
            return (
                "KRONOS_DISABLED: no KRONOS_API_URL configured. Proceed with the "
                "other technical indicators and do not estimate a forecast."
            )
        return (
            "KRONOS_UNAVAILABLE: the forecast service did not respond. Proceed "
            "with the other technical indicators and do not invent forecast values."
        )
    return format_forecast(payload)
