# PURE US (Limited Universe) VERSION – Yield-curve fly strategy
#
# Note: This version only changes the systematic signalling logic for the US curve flies (B1)
# (2s5s10s, 5s10s30s). The portfolio construction, cost model, and backtest engine remain
# the same as in the multi-country version. We focus on:
# - computing CRD-based signals for US flies;
# - running a US-only, cost-aware backtest;
# - reporting performance and turnover statistics; and
# - plotting the cumulative net PnL of the strategy.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("default")

# Global constants
ANNUAL_DAYS = 252

# Total DV01 budget (in DV01 dollars)
DV01_BUDGET_TOTAL = 100_000

# Maximum DV01 per fly
DV01_LIMIT_FLY = 0.25 * DV01_BUDGET_TOTAL

# Notional per unit DV01 (used to map DV01 to "capital")
CAPITAL_PER_DV01 = 100.0

# Horizon in months
H_MONTH = 1

# Horizon in years
H = H_MONTH / 12.0

# 6-month lookback window (trading days)
ROLL_WIN_6M = 126

# Smoothing window for the signal
MA_WIN = 5

ENTRY_Z = 1.6
EXIT_Z = 0.2

xlsx_path = Path("Yield curve arb.xlsx")

# 1. Load US gov yields only
raw = pd.read_excel(xlsx_path, sheet_name="Yield Signals")

us_cols = [
    "Date",   "USG2YR",
    "Date.1", "USG5YR",
    "Date.2", "US10GYR",
    "Date.3", "US30GYR",
]

raw = raw[us_cols]

series_list = []
for date_col, yld_col in zip(us_cols[0::2], us_cols[1::2]):
    df_i = raw[[date_col, yld_col]].copy()
    df_i.columns = ["Date", yld_col]
    df_i = df_i.dropna(how="all")
    df_i["Date"] = pd.to_datetime(df_i["Date"], errors="coerce")
    df_i = df_i.dropna(subset=["Date"])
    df_i = df_i.set_index("Date").sort_index()
    df_i = df_i[~df_i.index.duplicated(keep="last")]
    series_list.append(df_i)

yields_us = pd.concat(series_list, axis=1).sort_index()

# Convert yields to decimals and forward-fill gaps
yields_us = yields_us / 100.0
yields_us = yields_us.ffill()

print("US yield panel:")
display(yields_us.head())

# 2. Build US curve (tenors 2, 5, 10, 30)
TENOR_MAP_US = {
    "USG2YR": 2.0,
    "USG5YR": 5.0,
    "US10GYR": 10.0,
    "US30GYR": 30.0,
}

# Build a tenor-indexed US zero-coupon yield curve from raw yield columns
def build_us_curve(yields_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["USG2YR", "USG5YR", "US10GYR", "US30GYR"]
    sub = yields_df[cols].copy()
    tenors = [TENOR_MAP_US[c] for c in cols]
    sub.columns = tenors
    sub = sub[sorted(sub.columns)]
    return sub

curve_us = build_us_curve(yields_us)
print("US curve:")
display(curve_us.head())

# 3. Curve helper functions
# Zero-coupon bond price and DV01 for maturity T and yield y
def zero_coupon(y: float, T: float):
    P = np.exp(-y * T)
    dv01 = P * T * 1e-4
    return P, dv01

# Interpolate a yield at maturity T from a discrete curve row
def interp_yield(curve_row: pd.Series, T: float) -> float:
    xs = curve_row.index.values.astype(float)
    ys = curve_row.values.astype(float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    return float(np.interp(T, xs, ys))

# Zero-coupon price at maturity T from a curve row
def zero_price_from_curve(curve_row: pd.Series, T: float):
    y = interp_yield(curve_row, T)
    P, _ = zero_coupon(y, T)
    return P

# Change in yield from T to T-H along the same curve
def roll_down_dy(curve_row: pd.Series, T: float, H: float) -> float:
    if T - H <= 0:
        return 0.0
    y_T = interp_yield(curve_row, T)
    y_TH = interp_yield(curve_row, T - H)
    return y_TH - y_T

# Carry + roll-down (CRD) return of a single leg over horizon H
def crd_single_leg(curve_row: pd.Series, T: float, H: float, fund_rate: float):
    y = interp_yield(curve_row, T)
    P, dv01 = zero_coupon(y, T)

    dy_roll = roll_down_dy(curve_row, T, H)
    dy_bps = dy_roll * 1e4
    roll_pnl = -dv01 * dy_bps

    carry = (y - fund_rate) * H * P
    crd_ret = (roll_pnl + carry) / P
    return crd_ret, P, dv01

# DV01-neutral fly weights (wL, 1, wR) for (L, B, R) legs
def fly_weight_DV01_neutral(dv01_L, dv01_B, dv01_R):
    """
    Solve for wL, wR given:
        dv01_L * wL + dv01_B * 1 + dv01_R * wR = 0
        wL + wR = -1
    """
    A = np.array([[dv01_L, dv01_R],
                  [1.0,     1.0]])
    b = np.array([-dv01_B, -1.0])
    w_L, w_R = np.linalg.solve(A, b)
    return float(w_L), 1.0, float(w_R)

# 4. Signal generation for US flies (CRD-based z-scores)
# Compute CRD-based signal and z-score for a given fly (T_L, T_B, T_R)
def compute_signal_for_fly(curve_df: pd.DataFrame,
                           tenors: tuple,
                           H: float,
                           roll_win: int = ROLL_WIN_6M,
                           ma_win: int = MA_WIN) -> pd.DataFrame:
    T_L, T_B, T_R = tenors
    rows = []

    for date, curve_row in curve_df.iterrows():
        if curve_row.isna().any():
            rows.append((date, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue

        # Use 2-year yield as funding rate
        fund_rate = curve_row.loc[2.0]

        crd_L, P_L, dv01_L = crd_single_leg(curve_row, T_L, H, fund_rate)
        crd_B, P_B, dv01_B = crd_single_leg(curve_row, T_B, H, fund_rate)
        crd_R, P_R, dv01_R = crd_single_leg(curve_row, T_R, H, fund_rate)

        wL, wB, wR = fly_weight_DV01_neutral(dv01_L, dv01_B, dv01_R)

        fly_dv01 = dv01_L * wL + dv01_B * wB + dv01_R * wR
        fly_dv01_abs = (
            abs(dv01_L * wL) +
            abs(dv01_B * wB) +
            abs(dv01_R * wR)
        )

        fly_crd = wL * crd_L + wB * crd_B + wR * crd_R

        rows.append((date, fly_crd, wL, wB, wR, fly_dv01, fly_dv01_abs, fund_rate))

    sig_df = pd.DataFrame(
        rows,
        columns=[
            "Date",
            "sig_raw",
            "wL", "wB", "wR",
            "fly_dv01",
            "fly_dv01_abs",
            "fund_rate",
        ]
    ).set_index("Date")

    # Smooth signal and build 6-month z-score
    sig_df["sig_smooth"] = sig_df["sig_raw"].rolling(ma_win, min_periods=1).mean()
    roll = sig_df["sig_smooth"].rolling(roll_win, min_periods=20)
    sig_df["z_sig_6m"] = (sig_df["sig_smooth"] - roll.mean()) / roll.std(ddof=1)

    return sig_df

# Define US flies and build signal dictionary
FLIES = [
    (2.0, 5.0, 10.0),
    (5.0, 10.0, 30.0),
]

curve_by_cty = {"US": curve_us}
sig_by_key = {}

for tenors in FLIES:
    fly_name = f"{int(tenors[0])}s{int(tenors[1])}s{int(tenors[2])}s"
    sig_df = compute_signal_for_fly(curve_us, tenors, H)
    key = ("US", fly_name)
    sig_by_key[key] = {
        "tenors": tenors,
        "sig_df": sig_df,
    }

print("US signal keys:", list(sig_by_key.keys()))
display(sig_by_key[("US", "2s5s10s")]["sig_df"].head())

# 5. Cost-aware backtest (US-only)
# Approximate round-trip trading cost in basis points by country
COST_ROUNDTRIP_BP_COUNTRY = {
    "US": 0.12,
}

# Liquidity multiplier by fly (higher means more expensive to trade)
FLY_LIQ_MULT = {
    "2s5s10s": 1.0,
    "5s10s30s": 1.3,
}

# US-only fly portfolio backtest with a simple cost model
def run_portfolio_backtest_with_basic_costs(curve_by_cty: dict,
                                            sig_by_key: dict,
                                            dv01_limit_fly: float,
                                            entry_z: float,
                                            exit_z: float,
                                            max_positions: int = 2,
                                            cost_roundtrip_bp_country: dict = None,
                                            fly_liq_mult: dict = None):
    if cost_roundtrip_bp_country is None:
        cost_roundtrip_bp_country = COST_ROUNDTRIP_BP_COUNTRY
    if fly_liq_mult is None:
        fly_liq_mult = FLY_LIQ_MULT

    CAPITAL_TOTAL = max_positions * dv01_limit_fly * CAPITAL_PER_DV01
    CAPITAL_PER_FLY = dv01_limit_fly * CAPITAL_PER_DV01
    dt = 1.0 / ANNUAL_DAYS

    # Master date index
    common_dates = None
    for cty, cdf in curve_by_cty.items():
        common_dates = cdf.index if common_dates is None else common_dates.intersection(cdf.index)
    common_dates = common_dates.sort_values()

    # Per-(country, fly) position state
    positions = {
        key: {
            "pos_dir": 0,
            "wL": 0.0,
            "wB": 0.0,
            "wR": 0.0,
            "scale": 0.0,
            "hold_days": 0,
            "prev_curve": None,
            "trade_pnl": 0.0,
            "entry_date": None,
        }
        for key in sig_by_key.keys()
    }

    portfolio_rows = []
    trade_log = []

    for date in common_dates:
        daily_pnl_gross = 0.0
        trade_cost_today = 0.0

        # PnL from flies (gross)
        for key, state in positions.items():
            country, fly_name = key
            curve_df = curve_by_cty[country]
            if date not in curve_df.index:
                continue

            curve_today = curve_df.loc[date]
            prev_curve = state["prev_curve"]
            pos_dir = state["pos_dir"]
            scale = state["scale"]
            wL = state["wL"]
            wB = state["wB"]
            wR = state["wR"]

            pnl_m2m = 0.0
            pnl_carry = 0.0

            if prev_curve is not None and pos_dir != 0 and scale > 0:
                T_L, T_B, T_R = sig_by_key[key]["tenors"]

                P_L0 = zero_price_from_curve(prev_curve, T_L)
                P_B0 = zero_price_from_curve(prev_curve, T_B)
                P_R0 = zero_price_from_curve(prev_curve, T_R)
                P_L1 = zero_price_from_curve(curve_today, T_L)
                P_B1 = zero_price_from_curve(curve_today, T_B)
                P_R1 = zero_price_from_curve(curve_today, T_R)

                fly_leg_pnl = (
                    wL * (P_L1 - P_L0) +
                    wB * (P_B1 - P_B0) +
                    wR * (P_R1 - P_R0)
                )

                fund_prev = float(prev_curve.iloc[0])
                y_L_prev = interp_yield(prev_curve, T_L)
                y_B_prev = interp_yield(prev_curve, T_B)
                y_R_prev = interp_yield(prev_curve, T_R)

                carry_L = (y_L_prev - fund_prev) * dt * P_L0
                carry_B = (y_B_prev - fund_prev) * dt * P_B0
                carry_R = (y_R_prev - fund_prev) * dt * P_R0

                fly_carry = wL * carry_L + wB * carry_B + wR * carry_R

                pnl_m2m = fly_leg_pnl * scale * pos_dir
                pnl_carry = fly_carry * scale * pos_dir

            daily_pnl_fly = pnl_m2m + pnl_carry
            daily_pnl_gross += daily_pnl_fly

            if state["pos_dir"] != 0 and state["scale"] > 0:
                state["trade_pnl"] += daily_pnl_fly

            state["prev_curve"] = curve_today

        # Cash PnL from unused capital (US 2y)
        active_positions_start = sum(1 for st in positions.values() if st["pos_dir"] != 0)
        capital_used = active_positions_start * CAPITAL_PER_FLY
        capital_unused = max(CAPITAL_TOTAL - capital_used, 0.0)

        us_curve = curve_by_cty["US"]
        if date in us_curve.index:
            y_us2 = us_curve.loc[date, 2.0]
            cash_pnl = capital_unused * y_us2 * dt
        else:
            cash_pnl = 0.0

        daily_pnl_gross += cash_pnl

        # Exit rules and cost
        for key, state in positions.items():
            country, fly_name = key
            sig_df = sig_by_key[key]["sig_df"]
            if date not in sig_df.index:
                continue

            z_filt = sig_df.loc[date, "z_sig_6m"]
            pos_dir = state["pos_dir"]

            if pos_dir != 0:
                if (not np.isnan(z_filt)) and (abs(z_filt) < exit_z):
                    base_rt_bp = cost_roundtrip_bp_country.get(country, 0.15)
                    mult = fly_liq_mult.get(fly_name, 1.0)
                    roundtrip_bp = base_rt_bp * mult
                    per_side_bp = roundtrip_bp / 2.0
                    cost_per_side = per_side_bp * dv01_limit_fly

                    trade_cost_today += cost_per_side

                    trade_log.append({
                        "date": date,
                        "country": country,
                        "fly": fly_name,
                        "action": "EXIT",
                        "pos_dir": pos_dir,
                        "pnl": state["trade_pnl"],
                    })

                    state["pos_dir"] = 0
                    state["scale"] = 0.0
                    state["hold_days"] = 0
                    state["trade_pnl"] = 0.0
                    state["entry_date"] = None

        # Entry rules and cost
        active_positions_now = sum(1 for st in positions.values() if st["pos_dir"] != 0)
        remaining_capacity = max_positions - active_positions_now

        if remaining_capacity > 0:
            candidates = []
            for key, state in positions.items():
                if state["pos_dir"] != 0:
                    continue

                country, fly_name = key
                sig_df = sig_by_key[key]["sig_df"]
                if date not in sig_df.index:
                    continue

                row = sig_df.loc[date]
                z_filt = row["z_sig_6m"]

                if np.isnan(z_filt) or abs(z_filt) <= entry_z:
                    continue

                candidates.append((abs(z_filt), z_filt, key, row))

            candidates.sort(key=lambda x: x[0], reverse=True)

            for abs_z, z_filt, key, row in candidates[:remaining_capacity]:
                state = positions[key]
                country, fly_name = key

                wL_sig = row["wL"]
                wB_sig = row["wB"]
                wR_sig = row["wR"]
                fly_dv01_abs_sig = row["fly_dv01_abs"]

                pos_dir = 1 if z_filt > 0 else -1
                scale = dv01_limit_fly / max(fly_dv01_abs_sig, 1e-8)

                base_rt_bp = cost_roundtrip_bp_country.get(country, 0.15)
                mult = fly_liq_mult.get(fly_name, 1.0)
                roundtrip_bp = base_rt_bp * mult
                per_side_bp = roundtrip_bp / 2.0
                cost_per_side = per_side_bp * dv01_limit_fly

                trade_cost_today += cost_per_side

                state["pos_dir"] = pos_dir
                state["wL"] = wL_sig
                state["wB"] = wB_sig
                state["wR"] = wR_sig
                state["scale"] = scale
                state["hold_days"] = 0
                state["trade_pnl"] = 0.0
                state["entry_date"] = date

                trade_log.append({
                    "date": date,
                    "country": country,
                    "fly": fly_name,
                    "action": "ENTER",
                    "pos_dir": pos_dir,
                    "z_sig": z_filt,
                })

        # Snapshot and net PnL
        pos_snapshot = {}
        active_positions_end = 0
        for key, state in positions.items():
            tag = f"{key[0]}_{key[1]}"
            pos_snapshot[tag] = state["pos_dir"]
            if state["pos_dir"] != 0:
                state["hold_days"] += 1
                active_positions_end += 1

        daily_pnl_net = daily_pnl_gross - trade_cost_today

        portfolio_rows.append({
            "date": date,
            "portfolio_daily_pnl_gross": daily_pnl_gross,
            "portfolio_daily_pnl_net": daily_pnl_net,
            "cash_pnl": cash_pnl,
            "trade_cost": trade_cost_today,
            "num_active_positions": active_positions_end,
            **pos_snapshot,
        })

    port_bt = pd.DataFrame(portfolio_rows).set_index("date")
    trade_log_df = pd.DataFrame(trade_log)

    port_bt["cum_pnl_gross"] = port_bt["portfolio_daily_pnl_gross"].cumsum()
    port_bt["cum_pnl_net"] = port_bt["portfolio_daily_pnl_net"].cumsum()

    port_bt["ret_capital_gross"] = port_bt["portfolio_daily_pnl_gross"] / CAPITAL_TOTAL
    port_bt["ret_capital_net"] = port_bt["portfolio_daily_pnl_net"] / CAPITAL_TOTAL

    port_bt["cum_ret_capital_gross"] = (1 + port_bt["ret_capital_gross"].fillna(0)).cumprod() - 1
    port_bt["cum_ret_capital_net"] = (1 + port_bt["ret_capital_net"].fillna(0)).cumprod() - 1

    return port_bt, trade_log_df

# Run US-only backtest
portfolio_bt_cost, trade_log_cost = run_portfolio_backtest_with_basic_costs(
    curve_by_cty,
    sig_by_key,
    dv01_limit_fly=DV01_LIMIT_FLY,
    entry_z=ENTRY_Z,
    exit_z=EXIT_Z,
    max_positions=2,
)

display(portfolio_bt_cost.head())
display(trade_log_cost.head())

# 6. Core stats, turnover and PnL attribution
# Annualised net performance stats including drawdown and tail risk
def compute_core_stats(portfolio_bt_cost, annual_days=ANNUAL_DAYS):
    ret_net = portfolio_bt_cost["ret_capital_net"].replace([np.inf, -np.inf], np.nan).dropna()

    def ann_stats(r):
        if len(r) < 2:
            return np.nan, np.nan, np.nan
        mu = r.mean() * annual_days
        vol = r.std(ddof=1) * np.sqrt(annual_days)
        sharpe = mu / vol if vol > 0 else np.nan
        return mu, vol, sharpe

    ann_ret_net, ann_vol_net, sharpe_net = ann_stats(ret_net)

    cum_ret_net = portfolio_bt_cost["cum_ret_capital_net"].fillna(0.0)
    run_max = cum_ret_net.cummax()
    dd = cum_ret_net - run_max
    max_dd_net = dd.min()
    calmar_net = ann_ret_net / abs(max_dd_net) if max_dd_net < 0 else np.nan

    downside = ret_net[ret_net < 0]
    if len(downside) > 0:
        downside_dev = downside.std(ddof=1) * np.sqrt(annual_days)
        sortino_net = ann_ret_net / downside_dev if downside_dev > 0 else np.nan
    else:
        sortino_net = np.nan

    hit_rate = (ret_net > 0).mean() if len(ret_net) > 0 else np.nan
    avg_win = ret_net[ret_net > 0].mean() if (ret_net > 0).any() else np.nan
    avg_loss = ret_net[ret_net < 0].mean() if (ret_net < 0).any() else np.nan

    def var_es(series, alpha=0.99):
        if len(series) < 10:
            return np.nan, np.nan
        sorted_ret = np.sort(series.values)
        idx = int((1 - alpha) * len(sorted_ret))
        var = sorted_ret[idx]
        tail = sorted_ret[:idx + 1]
        es = tail.mean() if len(tail) > 0 else np.nan
        return var, es

    var99, es99 = var_es(ret_net, 0.99)
    var95, es95 = var_es(ret_net, 0.95)

    stats = {
        "Ann. return (net, % per year)": ann_ret_net * 100.0,
        "Ann. volatility (net, % per year)": ann_vol_net * 100.0,
        "Sharpe (net, unitless)": sharpe_net,
        "Max drawdown (capital, %)": max_dd_net * 100.0,
        "Calmar ratio (unitless)": calmar_net,
        "Sortino ratio (unitless)": sortino_net,
        "Hit rate (% of days profitable)": hit_rate * 100.0 if not np.isnan(hit_rate) else np.nan,
        "Avg. daily win (capital %)": avg_win * 100.0,
        "Avg. daily loss (capital %)": avg_loss * 100.0,
        "VaR(99%, daily, % cap)": var99 * 100.0 if not np.isnan(var99) else np.nan,
        "ES(99%, daily, % cap)": es99 * 100.0 if not np.isnan(es99) else np.nan,
        "VaR(95%, daily, % cap)": var95 * 100.0 if not np.isnan(var95) else np.nan,
        "ES(95%, daily, % cap)": es95 * 100.0 if not np.isnan(es95) else np.nan,
    }
    return stats

# Roundtrips per year, holding periods, and average cost in basis points
def compute_turnover_and_costs(portfolio_bt_cost,
                               trade_log_cost,
                               dv01_limit_fly,
                               annual_days=ANNUAL_DAYS):
    tl = trade_log_cost.sort_values("date").copy()
    num_enters = (tl["action"] == "ENTER").sum()
    num_exits = (tl["action"] == "EXIT").sum()
    num_roundtrips = min(num_enters, num_exits)

    years = len(portfolio_bt_cost) / annual_days if len(portfolio_bt_cost) > 0 else np.nan
    turnover_rt_per_year = num_roundtrips / years if years and years > 0 else np.nan

    open_dates = {}
    holding_days = []
    for _, row in tl.iterrows():
        key = (row["country"], row["fly"])
        date = row["date"]
        if row["action"] == "ENTER":
            open_dates[key] = date
        elif row["action"] == "EXIT":
            if key in open_dates:
                hp = (date - open_dates[key]).days
                holding_days.append(hp)
                del open_dates[key]

    avg_holding = float(np.mean(holding_days)) if holding_days else np.nan
    med_holding = float(np.median(holding_days)) if holding_days else np.nan

    total_trade_cost = portfolio_bt_cost["trade_cost"].sum()
    num_sides = len(tl)
    if num_sides > 0 and dv01_limit_fly > 0:
        avg_cost_per_side_bp = total_trade_cost / (dv01_limit_fly * num_sides)
        avg_cost_per_roundtrip_bp = 2.0 * avg_cost_per_side_bp
    else:
        avg_cost_per_side_bp = np.nan
        avg_cost_per_roundtrip_bp = np.nan

    return {
        "num_enters": num_enters,
        "num_exits": num_exits,
        "num_roundtrips": num_roundtrips,
        "roundtrips_per_year": turnover_rt_per_year,
        "avg_holding_days": avg_holding,
        "median_holding_days": med_holding,
        "avg_cost_per_side_bp": avg_cost_per_side_bp,
        "avg_cost_per_roundtrip_bp": avg_cost_per_roundtrip_bp,
    }

# Attribute total PnL between cash collateral and fly trades
def compute_pnl_attribution(portfolio_bt_cost):
    df = portfolio_bt_cost.copy()
    df["fly_pnl_gross"] = df["portfolio_daily_pnl_gross"] - df["cash_pnl"]

    total_portfolio_pnl = df["portfolio_daily_pnl_gross"].sum()
    total_cash_pnl = df["cash_pnl"].sum()
    total_fly_pnl = df["fly_pnl_gross"].sum()

    frac_cash = total_cash_pnl / total_portfolio_pnl if total_portfolio_pnl != 0 else np.nan
    frac_fly = total_fly_pnl / total_portfolio_pnl if total_portfolio_pnl != 0 else np.nan

    return {
        "total_portfolio_pnl_gross": total_portfolio_pnl,
        "total_cash_pnl": total_cash_pnl,
        "total_fly_pnl": total_fly_pnl,
        "frac_cash_of_total": frac_cash,
        "frac_fly_of_total": frac_fly,
    }

core_stats = compute_core_stats(portfolio_bt_cost)
turnover_stats = compute_turnover_and_costs(portfolio_bt_cost, trade_log_cost, DV01_LIMIT_FLY)
pnl_attr = compute_pnl_attribution(portfolio_bt_cost)

print("\n=== Core performance stats (US-only, net) ===")
display(pd.Series(core_stats).round(4))

print("\n=== Turnover / holding / cost stats ===")
display(pd.Series(turnover_stats).round(4))

print("\n=== P&L attribution (gross) ===")
display(pd.Series(pnl_attr).round(4))

# 7. Cumulative net PnL plot (main result)
daily_pnl_net = portfolio_bt_cost["portfolio_daily_pnl_net"].fillna(0.0)
cum_pnl_net = daily_pnl_net.cumsum()

plt.figure(figsize=(12, 4))
cum_pnl_net.plot()
plt.axhline(0, color="black", linewidth=1, linestyle="--")
plt.title("Cumulative Net PnL – US Yield-Curve Fly Strategy")
plt.ylabel("Cumulative net PnL ($)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
