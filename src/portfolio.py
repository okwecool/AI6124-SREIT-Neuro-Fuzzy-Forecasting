"""Causal observed-open long/cash wealth ledger for SRT forecasts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import RUN_KEY, Dataset


STRATEGIES = ("signal_long_cash", "momentum_long_cash", "always_long", "always_cash")


def verified_distributions(root: Path) -> pd.DataFrame:
    """Refuse a total-wealth run if the issuer audit diverges from saved events."""
    root = Path(root)
    verified = pd.read_csv(root / "data/reference/srt_distributions_verified.csv",
                           parse_dates=["ex_date", "payment_date"])
    yahoo = pd.read_csv(root / "data/processed/srt_dividends_yahoo.csv", parse_dates=["date"])
    if verified.ex_date.duplicated().any() or verified.payment_date.lt(verified.ex_date).any():
        raise ValueError("Invalid issuer distribution dates")
    left = verified.sort_values("ex_date").reset_index(drop=True)
    right = yahoo.sort_values("date").reset_index(drop=True)
    if len(left) != 9 or not left.ex_date.equals(right.date) or not np.allclose(
        left.amount_sgd_per_unit, right.cash_dividend_sgd, atol=1e-9, rtol=0
    ):
        raise ValueError("Issuer distribution audit does not match the frozen Yahoo events")
    return left


def wealth_ledger(
    data: Dataset, predictions: pd.DataFrame, *, cost_bps: int = 10,
    threshold: float = 0.0, wealth_mode: str = "price_only",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trade at each observed open using only a signal dated before that open.

    At an unobserved open, retain shares and cash. A distribution is earned by
    shares held *before* its ex-date open, and paid cash is never auto-reinvested.
    """
    if cost_bps < 0 or threshold < 0 or wealth_mode not in ("price_only", "gross_distribution"):
        raise ValueError("Invalid cost, threshold, or wealth mode")
    forecast = data.validate_predictions(predictions)
    distributions = verified_distributions(data.root) if wealth_mode == "gross_distribution" else None
    daily = data.daily.sort_values("date").set_index("date")
    cost = cost_bps / 10000
    all_rows, summaries = [], []
    for run_key, group in forecast.groupby(RUN_KEY, sort=True):
        group = group.sort_values("target_date")
        first, last = group.target_date.iloc[0], group.target_date.iloc[-1]
        calendar = daily.loc[first:last, ["srt_open"]].copy()
        if len(calendar) != len(group) or not calendar.index.equals(pd.DatetimeIndex(group.target_date)):
            raise ValueError("Forecast target dates must cover the complete SRT evaluation calendar")
        if pd.isna(calendar.srt_open.iloc[-1]):
            raise ValueError("The declared fold end needs an observed open for planned liquidation")
        signal_by_date = group.set_index("target_date")
        if distributions is not None:
            events = distributions.loc[distributions.ex_date.between(first, last) |
                                       distributions.payment_date.between(first, last)]
            calendar = calendar.reindex(calendar.index.union(events.ex_date).union(events.payment_date).sort_values())
            calendar = calendar.loc[first:last]
            ex_amount = events.groupby("ex_date").amount_sgd_per_unit.sum().to_dict()
        else:
            ex_amount = {}
        last_observed = calendar.srt_open.dropna().index[-1]
        for strategy in STRATEGIES:
            cash, shares, receivable = 1.0, 0.0, 0.0
            pending: dict[pd.Timestamp, float] = {}
            mark_price = np.nan
            turnover_notional, fees_total, trade_count = 0.0, 0.0, 0
            path_rows = []
            for date, day in calendar.iterrows():
                held_before_ex_open = shares
                ex_credit = held_before_ex_open * float(ex_amount.get(date, 0.0))
                if ex_credit:
                    pay_date = distributions.loc[distributions.ex_date.eq(date), "payment_date"].iloc[0]
                    pending[pay_date] = pending.get(pay_date, 0.0) + ex_credit
                    receivable += ex_credit
                payment = pending.pop(date, 0.0)
                cash += payment
                receivable -= payment
                opened = pd.notna(day.srt_open)
                price = float(day.srt_open) if opened else np.nan
                signal_row = signal_by_date.loc[date] if date in signal_by_date.index else None
                signal_time = signal_row.forecast_date if signal_row is not None else pd.NaT
                model_signal = float(signal_row.y_pred) if signal_row is not None else np.nan
                if signal_row is not None and not signal_time < date:
                    raise ValueError("Signal was not known before the open")
                momentum = (float(daily.loc[signal_time, "srt_momentum_5"]) / 5
                            if signal_row is not None else np.nan)
                if strategy == "signal_long_cash":
                    selected_signal = model_signal
                elif strategy == "momentum_long_cash":
                    selected_signal = momentum
                else:
                    selected_signal = np.nan
                requested = int(shares > 0)
                fee, traded_notional, action = 0.0, 0.0, "missing_open_carry"
                if opened:
                    mark_price = price
                    before_trade = cash + shares * price + receivable
                    if date == last_observed:
                        requested = 0  # Known fold boundary; never use a later fold's quote.
                    elif strategy == "always_long":
                        requested = 1
                    elif strategy == "always_cash":
                        requested = 0
                    else:
                        if not np.isfinite(selected_signal):
                            raise ValueError("Missing model or momentum signal at an observed open")
                        requested = int(selected_signal > threshold)
                    if requested and shares == 0:
                        traded_notional = cash / (1 + cost)
                        fee = traded_notional * cost
                        shares = traded_notional / price
                        cash = 0.0
                        action = "buy"
                    elif not requested and shares > 0:
                        traded_notional = shares * price
                        fee = traded_notional * cost
                        cash += traded_notional - fee
                        shares = 0.0
                        action = "planned_liquidation" if date == last_observed else "sell"
                    else:
                        action = "hold_long" if shares > 0 else "hold_cash"
                    after_trade = cash + shares * price + receivable
                    if not np.isclose(before_trade - after_trade, fee, atol=1e-10, rtol=0):
                        raise ArithmeticError("Ledger violates self-financing accounting")
                    turnover_notional += traded_notional
                    fees_total += fee
                    trade_count += int(traded_notional > 0)
                wealth = cash + (shares * mark_price if np.isfinite(mark_price) else 0.0) + receivable
                row = dict(zip(RUN_KEY, run_key))
                row.update({
                    "strategy": strategy, "wealth_mode": wealth_mode, "cost_bps_per_side": cost_bps,
                    "threshold": threshold, "date": date, "observed_open": opened,
                    "open_sgd": price, "signal_time": signal_time,
                    "signal_value": selected_signal, "requested_position": requested,
                    "actual_position": int(shares > 0), "action": action,
                    "traded_notional": traded_notional, "fee": fee,
                    "shares": shares, "cash": cash, "receivable": receivable,
                    "ex_credit": ex_credit, "distribution_payment": payment, "wealth": wealth,
                })
                path_rows.append(row)
            path = pd.DataFrame(path_rows)
            observed = path.loc[path.observed_open]
            if not np.isclose(path.iloc[-1].shares, 0, atol=1e-12):
                raise ArithmeticError("Fold must finish in cash")
            terminal = float(path.iloc[-1].wealth)
            if not np.isclose(path.iloc[-1].receivable, 0, atol=1e-12):
                raise ValueError("Outstanding distribution at fold end; extend payment accounting")
            peak = observed.wealth.cummax().clip(lower=1.0)
            days = (observed.date.iloc[-1] - observed.date.iloc[0]).days
            summary = dict(zip(RUN_KEY, run_key))
            summary.update({
                "strategy": strategy, "wealth_mode": wealth_mode,
                "cost_bps_per_side": cost_bps, "threshold": threshold,
                "calendar_first_open": observed.date.iloc[0],
                "calendar_last_open": observed.date.iloc[-1],
                "observed_opens": len(observed), "missing_open_events": int((~path.observed_open).sum()),
                "terminal_wealth_multiple": terminal, "cumulative_return": terminal - 1,
                "calendar_annualized_return": terminal ** (365.25 / days) - 1 if days else np.nan,
                "max_drawdown": float((1 - observed.wealth / peak).max()),
                "one_way_traded_notional_over_initial_wealth": turnover_notional,
                "fees_sgd_per_initial_sgd": fees_total, "one_way_trade_count": trade_count,
                "distribution_cash_sgd_per_initial_sgd": float(path.distribution_payment.sum()),
            })
            all_rows.extend(path_rows)
            summaries.append(summary)
    return pd.DataFrame(all_rows), pd.DataFrame(summaries)
