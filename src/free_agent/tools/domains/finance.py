"""Finance tools — market data, FX, time value of money, cash-flow and portfolio analytics."""
from __future__ import annotations

import math
import statistics
from datetime import UTC, datetime

from langchain_core.tools import tool

from free_agent.tools.domains._common import http_json, net_error, parse_numbers, truncate

_RANGES = ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")


def _fetch_closes(symbol: str, period: str, interval: str) -> tuple[dict, list[tuple[int, float]]]:
    data = http_json(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.strip().upper()}",
        {"range": period, "interval": interval},
    )
    result = (data.get("chart", {}).get("result") or [None])[0]
    if not result:
        err = data.get("chart", {}).get("error") or {}
        raise ValueError(err.get("description") or f"no data for {symbol!r}")
    closes = result["indicators"]["quote"][0].get("close") or []
    series = [(t, c) for t, c in zip(result.get("timestamp") or [], closes, strict=False) if c is not None]
    return result["meta"], series


@tool
def market_quote(symbol: str, period: str = "1mo", interval: str = "1d") -> str:
    """Latest price and recent performance for a stock, ETF, index, crypto or FX pair (Yahoo Finance).

    Returns last price, change over the period, high/low, annualized volatility and recent closes.
    Data may be delayed — always state the timestamp when quoting it.

    Args:
      symbol: Ticker, e.g. `AAPL`, `^GSPC`, `BTC-USD`, `EURUSD=X`, `ECOPETROL.CL`.
      period: One of 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
      interval: Bar size, e.g. 1d, 1wk, 1mo, 1h, 15m.
    """
    if period not in _RANGES:
        return f"[invalid period {period!r} — use one of {', '.join(_RANGES)}]"
    try:
        meta, series = _fetch_closes(symbol, period, interval)
    except Exception as exc:
        return net_error(exc) if not isinstance(exc, ValueError) else f"[{exc}]"
    if not series:
        return f"[no price data for {symbol}]"
    closes = [c for _, c in series]
    first, last = closes[0], closes[-1]
    rets = [math.log(b / a) for a, b in zip(closes, closes[1:], strict=False) if a > 0 and b > 0]
    per_year = {"1d": 252, "1wk": 52, "1mo": 12}.get(interval, 252)
    vol = statistics.stdev(rets) * math.sqrt(per_year) if len(rets) > 2 else float("nan")
    ts = datetime.fromtimestamp(series[-1][0], tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    tail = ", ".join(
        f"{datetime.fromtimestamp(t, tz=UTC):%m-%d}:{c:.2f}" for t, c in series[-10:]
    )
    return (
        f"{meta.get('symbol')} ({meta.get('longName') or meta.get('shortName') or ''}) "
        f"· {meta.get('exchangeName', '')} · {meta.get('currency', '')}\n"
        f"last={last:.4f} as of {ts} (regularMarketPrice={meta.get('regularMarketPrice')})\n"
        f"{period} change: {last - first:+.4f} ({(last / first - 1) * 100:+.2f}%)\n"
        f"high={max(closes):.4f} low={min(closes):.4f}  annualized vol≈{vol * 100:.1f}%\n"
        f"52w range: {meta.get('fiftyTwoWeekLow')} – {meta.get('fiftyTwoWeekHigh')}\n"
        f"recent closes: {tail}"
    )


@tool
def fx_convert(amount: float, from_currency: str, to_currency: str, date: str = "") -> str:
    """Convert between currencies using European Central Bank reference rates (frankfurter.app).

    Args:
      amount: Amount in the source currency.
      from_currency: ISO code, e.g. `USD`.
      to_currency: ISO code(s), comma separated, e.g. `EUR` or `EUR,GBP,JPY`.
      date: Historical date `YYYY-MM-DD`. Empty = latest.
    """
    endpoint = date.strip() or "latest"
    try:
        data = http_json(
            f"https://api.frankfurter.app/{endpoint}",
            {"amount": amount, "from": from_currency.upper(), "to": to_currency.upper().replace(" ", "")},
        )
    except Exception as exc:
        return net_error(exc)
    rates = data.get("rates", {})
    if not rates:
        return "[no rates returned — check the currency codes]"
    return f"{amount} {data.get('base')} on {data.get('date')} (ECB):\n" + "\n".join(
        f"  = {v:,.4f} {k}" for k, v in rates.items()
    )


@tool
def time_value_of_money(
    present_value: float = 0.0,
    payment: float = 0.0,
    annual_rate_pct: float = 5.0,
    years: float = 10.0,
    periods_per_year: int = 12,
    future_value: float = 0.0,
    solve_for: str = "fv",
) -> str:
    """Solve the time-value-of-money equation for FV, PV or PMT (end-of-period payments).

    Sign convention: money you put in is positive in `present_value`/`payment`.

    Args:
      present_value: Initial lump sum.
      payment: Periodic contribution.
      annual_rate_pct: Nominal annual rate in percent (e.g. 7.5).
      years: Horizon in years.
      periods_per_year: Compounding/payment periods per year (12 = monthly).
      future_value: Target FV (used when solving for pv or pmt).
      solve_for: `fv`, `pv` or `pmt`.
    """
    m = max(1, int(periods_per_year))
    n = years * m
    r = annual_rate_pct / 100 / m
    growth = (1 + r) ** n
    annuity = (growth - 1) / r if r else n
    s = solve_for.lower()
    if s == "fv":
        fv = present_value * growth + payment * annuity
        contributed = present_value + payment * n
        return (f"FV = {fv:,.2f}\ncontributed = {contributed:,.2f}\ngrowth = {fv - contributed:,.2f}\n"
                f"effective annual rate = {((1 + r) ** m - 1) * 100:.4f}%")
    if s == "pv":
        pv = (future_value - payment * annuity) / growth
        return f"PV needed today = {pv:,.2f} (to reach {future_value:,.2f} in {years} years)"
    if s == "pmt":
        pmt = (future_value - present_value * growth) / annuity
        return f"payment per period = {pmt:,.2f} ({m}×/year for {years} years)"
    return "[solve_for must be fv, pv or pmt]"


@tool
def loan_schedule(principal: float, annual_rate_pct: float, years: float, periods_per_year: int = 12) -> str:
    """Amortizing loan: periodic payment, total interest, and a yearly amortization summary.

    Args:
      principal: Amount borrowed.
      annual_rate_pct: Nominal annual interest rate in percent.
      years: Loan term in years.
      periods_per_year: Payments per year (12 = monthly).
    """
    m = max(1, int(periods_per_year))
    n = round(years * m)
    r = annual_rate_pct / 100 / m
    pmt = principal / n if r == 0 else principal * r / (1 - (1 + r) ** -n)
    bal, rows, yi, yp = principal, [], 0.0, 0.0
    for k in range(1, n + 1):
        interest = bal * r
        princ = pmt - interest
        bal -= princ
        yi += interest
        yp += princ
        if k % m == 0 or k == n:
            rows.append(f"  year {math.ceil(k / m):>3}: interest {yi:>12,.2f}  principal {yp:>12,.2f}  balance {max(bal, 0):>14,.2f}")
            yi = yp = 0.0
    total = pmt * n
    return truncate(
        f"payment = {pmt:,.2f} per period ({n} payments)\ntotal paid = {total:,.2f}\n"
        f"total interest = {total - principal:,.2f}\n" + "\n".join(rows)
    )


def _npv(rate: float, flows: list[float]) -> float:
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(flows))


def _irr(flows: list[float]) -> float | None:
    lo, hi = -0.9999, 10.0
    f_lo, f_hi = _npv(lo, flows), _npv(hi, flows)
    if f_lo * f_hi > 0:
        return None
    for _ in range(300):
        mid = (lo + hi) / 2
        f_mid = _npv(mid, flows)
        if abs(f_mid) < 1e-9:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


@tool
def cash_flow_analysis(cash_flows: str, discount_rate_pct: float = 10.0) -> str:
    """Capital-budgeting metrics for a cash-flow series: NPV, IRR, payback, discounted payback, PI.

    Args:
      cash_flows: Period cash flows starting at t=0, e.g. `-1000, 300, 400, 500, 200`.
      discount_rate_pct: Discount rate (WACC / hurdle) in percent per period.
    """
    try:
        flows = parse_numbers(cash_flows)
    except ValueError as exc:
        return f"[{exc}]"
    if len(flows) < 2:
        return "[need at least 2 cash flows]"
    r = discount_rate_pct / 100
    npv = _npv(r, flows)
    irr = _irr(flows)

    def payback(series: list[float]) -> str:
        cum = 0.0
        for t, cf in enumerate(series):
            prev = cum
            cum += cf
            if t > 0 and prev < 0 <= cum:
                return f"{t - 1 + (-prev / cf):.2f} periods"
        return "never"

    disc = [cf / (1 + r) ** t for t, cf in enumerate(flows)]
    invest = -sum(d for d in disc if d < 0)
    pi = sum(d for d in disc if d > 0) / invest if invest else float("nan")
    irr_str = f"{irr * 100:.3f}%" if irr is not None else "undefined (no sign change)"
    decision = "ACCEPT (NPV > 0)" if npv > 0 else "REJECT (NPV ≤ 0)"
    return (
        f"NPV @ {discount_rate_pct}% = {npv:,.2f}\nIRR = {irr_str}\n"
        f"payback = {payback(flows)}\ndiscounted payback = {payback(disc)}\n"
        f"profitability index = {pi:.3f}\ndecision: {decision}"
    )


@tool
def portfolio_risk(prices: str = "", returns_pct: str = "", periods_per_year: int = 252, risk_free_pct: float = 0.0) -> str:
    """Risk/return metrics from a price series or periodic returns.

    Computes CAGR, annualized return & volatility, Sharpe, Sortino, max drawdown,
    and historical 95% VaR / CVaR. Pair with `market_quote` or user data.

    Args:
      prices: Price levels in chronological order (comma/space separated). Takes precedence.
      returns_pct: Alternatively, periodic returns in percent.
      periods_per_year: 252 daily, 52 weekly, 12 monthly.
      risk_free_pct: Annual risk-free rate in percent.
    """
    try:
        if prices.strip():
            px = parse_numbers(prices)
            rets = [b / a - 1 for a, b in zip(px, px[1:], strict=False)]
        else:
            rets = [x / 100 for x in parse_numbers(returns_pct)]
    except ValueError as exc:
        return f"[{exc}]"
    if len(rets) < 3:
        return "[need at least 3 returns]"
    k = max(1, int(periods_per_year))
    mu, sd = statistics.fmean(rets), statistics.stdev(rets)
    rf = risk_free_pct / 100 / k
    downside = [min(0.0, x - rf) for x in rets]
    dd_sd = math.sqrt(sum(d * d for d in downside) / len(downside))
    equity, peak, mdd = 1.0, 1.0, 0.0
    for x in rets:
        equity *= 1 + x
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1)
    cagr = equity ** (k / len(rets)) - 1
    srt = sorted(rets)
    idx = max(0, int(0.05 * len(srt)) - 1)
    var95 = -srt[idx]
    cvar95 = -statistics.fmean(srt[: idx + 1])
    return (
        f"periods={len(rets)}  total return={(equity - 1) * 100:.2f}%  CAGR={cagr * 100:.2f}%\n"
        f"annualized mean={mu * k * 100:.2f}%  annualized vol={sd * math.sqrt(k) * 100:.2f}%\n"
        f"Sharpe={(mu - rf) / sd * math.sqrt(k) if sd else float('nan'):.3f}  "
        f"Sortino={(mu - rf) / dd_sd * math.sqrt(k) if dd_sd else float('nan'):.3f}\n"
        f"max drawdown={mdd * 100:.2f}%\n1-period 95% VaR={var95 * 100:.2f}%  CVaR={cvar95 * 100:.2f}%"
    )


FINANCE_TOOLS = [market_quote, fx_convert, time_value_of_money, loan_schedule, cash_flow_analysis, portfolio_risk]
