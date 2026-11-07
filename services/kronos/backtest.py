"""Walk-forward check: does the Kronos forecast beat a naive no-change baseline?

Calls the sidecar with as_of dates in the past, compares each forecast with what
actually happened, and prints directional hit rate (with a 95% Wilson interval),
MAE against a flat-price baseline, and the correlation of predicted vs realized
moves. Run the service first.

    python backtest.py --api http://127.0.0.1:8100 --horizon 10 --workers 3
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_TICKERS = (
    "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,AVGO,JPM,XOM,"
    "UNH,V,JNJ,PG,HD,COST,WMT,DIS,BA,CAT"
)
DEFAULT_DATES = "2025-08-01,2025-10-01,2025-12-01,2026-02-02,2026-04-01,2026-06-01,2026-08-03"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / d, (centre + margin) / d)


def fetch(api: str, ticker: str, as_of: str, horizon: int) -> dict:
    query = urllib.parse.urlencode({"ticker": ticker, "as_of": as_of, "horizon": horizon})
    try:
        with urllib.request.urlopen(f"{api}/forecast?{query}", timeout=300) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the sweep
        return {"ticker": ticker, "as_of": as_of, "error": str(exc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8100")
    ap.add_argument("--tickers", default=DEFAULT_TICKERS)
    ap.add_argument("--dates", default=DEFAULT_DATES)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--csv", default="kronos_backtest.csv")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    jobs = [(t, d) for d in dates for t in tickers]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda j: fetch(args.api, j[0], j[1], args.horizon), jobs))

    rows, errors, error_samples = [], 0, []
    for r in results:
        outcome = r.get("outcome")
        if not outcome:
            errors += 1
            if len(error_samples) < 3:
                error_samples.append(f"{r['ticker']} {r.get('as_of')}: {r.get('error', 'no outcome')}")
            continue
        rows.append(
            {
                "ticker": r["ticker"],
                "as_of": r["as_of"],
                "pred_move": r["implied_move_pct"],
                "actual_move": outcome["actual_move_pct"],
                "hit": outcome["direction_correct"],
                "mae_kronos": outcome["mae_pct"],
                # Flat baseline: predicting "no change" errs by the realized move.
                "mae_flat": abs(outcome["actual_move_pct"]),
            }
        )

    n = len(rows)
    if n == 0:
        print(json.dumps({"samples": 0, "errors": errors}, indent=2))
        return

    hits = sum(1 for x in rows if x["hit"])
    lo, hi = wilson(hits, n)
    pred = [x["pred_move"] for x in rows]
    act = [x["actual_move"] for x in rows]
    mp, ma = statistics.mean(pred), statistics.mean(act)
    cov = sum((p - mp) * (a - ma) for p, a in zip(pred, act))
    den = math.sqrt(sum((p - mp) ** 2 for p in pred) * sum((a - ma) ** 2 for a in act))

    print(
        json.dumps(
            {
                "samples": n,
                "errors": errors,
                "error_samples": error_samples,
                "hit_rate_pct": round(hits / n * 100, 1),
                "hit_rate_ci95": [round(lo * 100, 1), round(hi * 100, 1)],
                "always_up_baseline_pct": round(sum(1 for a in act if a > 0) / n * 100, 1),
                "mae_kronos_pct": round(statistics.mean(x["mae_kronos"] for x in rows), 2),
                "mae_flat_baseline_pct": round(statistics.mean(x["mae_flat"] for x in rows), 2),
                "mean_abs_actual_move_pct": round(statistics.mean(abs(a) for a in act), 2),
                "corr_pred_vs_actual": round(cov / den, 3) if den else None,
            },
            indent=2,
        )
    )

    with open(args.csv, "w") as fh:
        fh.write("ticker,as_of,pred_move,actual_move,hit,mae_kronos,mae_flat\n")
        for x in rows:
            fh.write(
                f"{x['ticker']},{x['as_of']},{x['pred_move']},{x['actual_move']},"
                f"{int(x['hit'])},{x['mae_kronos']},{x['mae_flat']:.2f}\n"
            )


if __name__ == "__main__":
    main()
