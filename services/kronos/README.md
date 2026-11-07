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
