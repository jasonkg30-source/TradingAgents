"""Kronos forecast sidecar: one HTTP hop for TradingAgents' technical analyst.

Kept out of the main TradingAgents image on purpose — torch CPU alone is ~1 GB,
which would triple the Cloud Run image and slow every cold start. This service
loads the model once and answers /forecast?ticker=AAPL&horizon=10.

Pass `as_of=YYYY-MM-DD` to replay a past decision point: the forecast is built
from bars up to that date and, when later bars exist, the response carries the
realized outcome so hit rates can be measured instead of assumed.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

KRONOS_PATH = os.environ.get("KRONOS_PATH", "/opt/Kronos")
if KRONOS_PATH not in sys.path:
    sys.path.insert(0, KRONOS_PATH)

from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

MODEL_ID = os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-small")
TOKENIZER_ID = os.environ.get("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
MAX_CONTEXT = int(os.environ.get("KRONOS_MAX_CONTEXT", "512"))
SAMPLES = int(os.environ.get("KRONOS_SAMPLES", "3"))

YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
UA = {"User-Agent": "Mozilla/5.0 (compatible; tradingagents-kronos/1.0)"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kronos")

app = FastAPI(title="Kronos forecast sidecar", version="1.1.0")

_predictor: KronosPredictor | None = None
_load_lock = threading.Lock()


class Outcome(BaseModel):
    actual_close: list[float]
    actual_move_pct: float
    direction_correct: bool
    mae_pct: float


class Forecast(BaseModel):
    ticker: str
    as_of: str
    last_close: float
    horizon: int
    predicted_dates: list[str]
    predicted_close: list[float]
    implied_move_pct: float
    model: str
    samples: int
    elapsed_ms: int
    outcome: Outcome | None = None


def get_predictor() -> KronosPredictor:
    """Load the tokenizer + model once, on first request."""
    global _predictor
    if _predictor is None:
        with _load_lock:
            if _predictor is None:
                t0 = time.time()
                tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID)
                model = Kronos.from_pretrained(MODEL_ID)
                model.eval()
                _predictor = KronosPredictor(model, tokenizer, max_context=MAX_CONTEXT)
                log.info("loaded %s in %.1fs", MODEL_ID, time.time() - t0)
    return _predictor


def fetch_daily(ticker: str) -> pd.DataFrame:
    """Five years of daily OHLCV from Yahoo's public chart endpoint."""
    try:
        r = requests.get(
            YAHOO.format(ticker=ticker),
            params={"range": "5y", "interval": "1d", "includePrePost": "false"},
            headers=UA,
            timeout=20,
        )
        r.raise_for_status()
        payload = r.json()
    except (requests.RequestException, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"price feed failed: {exc}") from exc

    chart = (payload.get("chart") or {}).get("result") or []
    if not chart:
        raise HTTPException(status_code=404, detail=f"no price data for {ticker}")
    res = chart[0]
    quote = (res.get("indicators") or {}).get("quote") or [{}]
    q = quote[0]

    df = pd.DataFrame(
        {
            "timestamps": pd.to_datetime(res.get("timestamp", []), unit="s").normalize(),
            "open": q.get("open", []),
            "high": q.get("high", []),
            "low": q.get("low", []),
            "close": q.get("close", []),
            "volume": q.get("volume", []),
        }
    ).dropna()
    if len(df) < 60:
        raise HTTPException(status_code=404, detail=f"only {len(df)} usable bars for {ticker}")
    df["amount"] = df["close"] * df["volume"]
    return df.reset_index(drop=True)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_ID, "loaded": _predictor is not None}


@app.get("/forecast", response_model=Forecast)
def forecast(
    ticker: str = Query(..., min_length=1, max_length=12),
    horizon: int = Query(10, ge=1, le=60),
    lookback: int = Query(400, ge=60, le=MAX_CONTEXT),
    samples: int = Query(SAMPLES, ge=1, le=10),
    as_of: str | None = Query(None, description="Replay from this date (YYYY-MM-DD)"),
) -> Forecast:
    ticker = ticker.strip().upper()
    t0 = time.time()

    full = fetch_daily(ticker)
    if as_of:
        try:
            cut = pd.Timestamp(as_of).normalize()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"bad as_of: {as_of}") from exc
        hist = full[full["timestamps"] <= cut]
    else:
        hist = full

    if len(hist) < lookback:
        raise HTTPException(
            status_code=400,
            detail=f"need {lookback} bars before {as_of or 'today'}, have {len(hist)}",
        )

    ctx = hist.iloc[-lookback:].reset_index(drop=True)
    last_row = ctx.iloc[-1]

    # Prefer real subsequent trading days: outcomes then line up bar-for-bar and
    # backtests stay on the exchange calendar instead of business-day guesses.
    later = full[full["timestamps"] > last_row["timestamps"]].head(horizon)
    if len(later) == horizon:
        future = pd.Series(later["timestamps"].values)
    else:
        future = pd.Series(
            pd.bdate_range(
                pd.Timestamp(last_row["timestamps"]) + pd.Timedelta(days=1), periods=horizon
            )
        )

    pred = get_predictor().predict(
        df=ctx[["open", "high", "low", "close", "volume", "amount"]],
        x_timestamp=pd.Series(ctx["timestamps"].values),
        y_timestamp=future,
        pred_len=horizon,
        T=1.0,
        top_p=0.9,
        sample_count=samples,
        verbose=False,
    )

    closes = [round(float(v), 2) for v in pred["close"].tolist()]
    last_close = round(float(last_row["close"]), 2)
    implied = round((closes[-1] / last_close - 1) * 100, 2)

    outcome = None
    if len(later) == horizon:
        actual = [round(float(v), 2) for v in later["close"].tolist()]
        actual_move = round((actual[-1] / last_close - 1) * 100, 2)
        mae = sum(abs(p - a) / a for p, a in zip(closes, actual)) / len(actual) * 100
        outcome = Outcome(
            actual_close=actual,
            actual_move_pct=actual_move,
            direction_correct=(implied > 0) == (actual_move > 0),
            mae_pct=round(mae, 2),
        )

    return Forecast(
        ticker=ticker,
        as_of=pd.Timestamp(last_row["timestamps"]).strftime("%Y-%m-%d"),
        last_close=last_close,
        horizon=horizon,
        predicted_dates=[d.strftime("%Y-%m-%d") for d in future],
        predicted_close=closes,
        implied_move_pct=implied,
        model=MODEL_ID,
        samples=samples,
        elapsed_ms=int((time.time() - t0) * 1000),
        outcome=outcome,
    )
