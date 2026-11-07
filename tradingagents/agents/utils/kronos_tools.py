from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.kronos import get_kronos_forecast as _get_kronos_forecast


@tool
def get_kronos_forecast(
    symbol: Annotated[str, "ticker symbol, e.g. AAPL"],
    horizon: Annotated[int, "trading days to forecast ahead (1-60), default 10"] = 10,
) -> str:
    """Forecast the next daily closes with the Kronos K-line foundation model.

    Returns a numeric price path from the last close plus the implied move over
    the window. Use it as one more technical input alongside the indicators —
    it is a stochastic sample from price history and ignores news, earnings and
    fundamentals, so it must not decide a trade on its own.
    """
    return _get_kronos_forecast(symbol, horizon=horizon)
