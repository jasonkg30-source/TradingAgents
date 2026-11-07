# Kronos forecast sidecar

HTTP wrapper around [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos), a
K-line foundation model (AAAI 2026). TradingAgents' market analyst calls it for a
quantitative price path on top of its indicator work.

**Why a separate service instead of a local import:** CPU-only torch is ~1 GB.
Bundling it into the main TradingAgents image would triple the image and slow
every Cloud Run cold start. Here the main image stays untouched and this service
scales to zero on its own.

## API

```
GET /health
GET /forecast?ticker=AAPL&horizon=10&lookback=400&samples=3
GET /forecast?ticker=AAPL&horizon=10&as_of=2026-06-01   # replay + realized outcome
```

```json
{
  "ticker": "AAPL",
  "as_of": "2026-11-05",
  "last_close": 344.17,
  "horizon": 5,
  "predicted_dates": ["2026-11-06", "..."],
  "predicted_close": [335.99, "..."],
  "implied_move_pct": -4.53,
  "model": "NeoQuasar/Kronos-small",
  "samples": 3,
  "elapsed_ms": 4044
}
```

Prices come from Yahoo's public chart endpoint (daily bars). Missing or
insufficient data returns 404/502 rather than a fabricated forecast.

## Run locally

The venv from the Kronos clone already has torch:

```bash
cd projects/TradingAgents/services/kronos
KRONOS_PATH=../../Kronos \
  ../../Kronos/.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8100
```

Then point TradingAgents at it:

```bash
export KRONOS_API_URL=http://127.0.0.1:8100
```

With `KRONOS_API_URL` unset the analyst tool returns a `KRONOS_DISABLED` sentinel
and the run proceeds unchanged — the feature is strictly opt-in.

## Docker / Cloud Run

Build context is this directory (the Dockerfile clones Kronos into `/opt/Kronos`):

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/kronos-forecast:1.0.0 .
gcloud run deploy kronos-forecast \
  --image gcr.io/PROJECT_ID/kronos-forecast:1.0.0 \
  --region us-central1 --memory 4Gi --cpu 2 \
  --timeout 300 --concurrency 4 \
  --min-instances 0 --max-instances 2 --no-allow-unauthenticated
```

Then set `KRONOS_API_URL` on the TradingAgents service to the Cloud Run URL and
grant it `roles/run.invoker` (or use `--allow-unauthenticated` inside a private
VPC and skip the token dance).

Cost: `--min-instances 0` means idle spend is zero; a forecast costs a few
seconds of 2 vCPU / 4 GiB, i.e. fractions of a cent. First request after idle
pays a cold start (image pull + torch import + weights download on a fresh
instance), roughly 40-60 s.

## Measured performance (do not skip)

Zero-shot `Kronos-small`, 400 daily bars, 20 US mega caps, 7 replay dates
(2025-08 → 2026-08), `backtest.py`. Both horizons show **no directional edge**:

| horizon | samples | hit rate (95% CI) | always-up baseline | MAE Kronos | MAE flat |
| --- | --- | --- | --- | --- | --- |
| 5 days | 139 | 49.6% (41.5–57.8) | 64.0% | 6.49% | 3.66% |
| 10 days | 112 | 50.9% (41.8–60.0) | 58.0% | 8.30% | 5.17% |

Prediction correlation with realized moves: 0.04 (5d) and 0.26 (10d). The model
also carries a bearish tilt on this sample — 59–68% of forecasts were negative
while the average realized move was +1.4% to +2.0% — so feeding it to an analyst
would inject a directional bias that the data does not support.

Caveats: small sample, zero-shot only, daily bars, US large caps. The paper's
gains are cross-sectional RankIC on its own pretraining distribution, which is a
different task from "will this ticker be up in 10 days". Fine-tuning on US data
is the untested path.

**Consequence:** keep this as an opt-in experiment. Do not present its output as
evidence, and do not let it override indicators or the verified snapshot.

## Env vars

| var | default | meaning |
| --- | --- | --- |
| `KRONOS_PATH` | `/opt/Kronos` | checkout that provides the `model` package |
| `KRONOS_MODEL` | `NeoQuasar/Kronos-small` | HF id (`Kronos-mini` / `-small` / `-base`) |
| `KRONOS_TOKENIZER` | `NeoQuasar/Kronos-Tokenizer-base` | HF id |
| `KRONOS_MAX_CONTEXT` | `512` | model context cap |
| `KRONOS_SAMPLES` | `3` | samples averaged per forecast |

## Limits

- Forecasts are stochastic: two calls on the same day differ. Treat the path as
  a distribution sample, not a price target.
- Input is price history only — no news, earnings, or fundamentals.
- `Kronos-large` (499M) is not public; `base` (102M) is the biggest usable model.
- No backtested P&L ships with this service. Do not present its output as one.
