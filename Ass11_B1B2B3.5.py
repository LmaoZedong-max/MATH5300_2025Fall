# MAFN 5300 – Yield-curve fly strategy (implements B1, B2, B3.5 refinement only)

import numpy as np
import pandas as pd
from pathlib import Path

# basic constants
ANNUAL_DAYS = 252
H_MONTH = 1
H = H_MONTH / 12.0

# z-score and lookback
ROLL_WIN_6M = 126
MA_WIN = 5
ENTRY_Z = 1.6
EXIT_Z = 0.2

# DV01 and capital mapping
DV01_BUDGET_TOTAL = 100_000
DV01_LIMIT_FLY = 0.25 * DV01_BUDGET_TOTAL

# cost model (round-trip in yield bp by country; then scaled by fly type)
COST_ROUNDTRIP_BP_COUNTRY = {
    "US": 0.12,
    "DE": 0.12,
    "UK": 0.18,
    "JP": 0.08,
    "AU": 0.35,
    "CA": 0.25,
    "IT": 0.50,
}

FLY_LIQ_MULT = {
    "2s5s10s": 1.0,
    "5s10s30s": 1.3,
}

# file path
xlsx_path = Path("Yield curve arb.xlsx")


# load yield curves from Excel into a panel
def load_yields(xlsx_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name="Yield Signals")

    cols = [
        "Date", "USG2YR",
        "Date.1", "USG5YR",
        "Date.2", "US10GYR",
        "Date.3", "US30GYR",
        "Date.4", "GDBR2",
        "Date.5", "GDBR5",
        "Date.6", "GDBR10",
        "Date.7", "GDBR30",
        "Date.8", "GUKG2",
        "Date.9", "GUKG5",
        "Date.10", "GUKG10",
        "Date.11", "GUKG30",
        "Date.12", "GBTPGR2",
        "Date.13", "GBTPGR5",
        "Date.14", "GBTPGR10",
        "Date.15", "GBTPGR30",
        "Date.16", "JGBS2",
        "Date.17", "JGBS5",
        "Date.18", "JGBS10",
        "Date.19", "JGBS30",
        "Date.20", "GTAUD2Y",
        "Date.21", "GTAUD5Y",
        "Date.22", "GTAUD10Y",
        "Date.23", "GTAUD30Y",
        "Date.24", "GTCAD2Y",
        "Date.25", "GTCAD5Y",
        "Date.26", "GTCAD10Y",
        "Date.27", "GTCAD30Y",
    ]

    raw = raw[cols]

    series_list = []
    for date_col, yld_col in zip(cols[0::2], cols[1::2]):
        series_name = yld_col
        df_i = raw[[date_col, yld_col]].copy()
        df_i.columns = ["Date", series_name]
        df_i = df_i.dropna(how="all")
        df_i["Date"] = pd.to_datetime(df_i["Date"], errors="coerce")
        df_i = df_i.dropna(subset=["Date"])
        df_i = df_i.set_index("Date").sort_index()
        df_i = df_i[~df_i.index.duplicated(keep="last")]
        series_list.append(df_i)

    yields = pd.concat(series_list, axis=1).sort_index()
    yields = (yields / 100.0).ffill()
    return yields


# curve definitions
CURVE_COLS = {
    "US": ["USG2YR", "USG5YR", "US10GYR", "US30GYR"],
    "DE": ["GDBR2", "GDBR5", "GDBR10", "GDBR30"],
    "UK": ["GUKG2", "GUKG5", "GUKG10", "GUKG30"],
    "IT": ["GBTPGR2", "GBTPGR5", "GBTPGR10", "GBTPGR30"],
    "JP": ["JGBS2", "JGBS5", "JGBS10", "JGBS30"],
    "AU": ["GTAUD2Y", "GTAUD5Y", "GTAUD10Y", "GTAUD30Y"],
    "CA": ["GTCAD2Y", "GTCAD5Y", "GTCAD10Y", "GTCAD30Y"],
}

TENOR_MAP = {
    "USG2YR": 2.0, "USG5YR": 5.0, "US10GYR": 10.0, "US30GYR": 30.0,
    "GDBR2": 2.0, "GDBR5": 5.0, "GDBR10": 10.0, "GDBR30": 30.0,
    "GUKG2": 2.0, "GUKG5": 5.0, "GUKG10": 10.0, "GUKG30": 30.0,
    "GBTPGR2": 2.0, "GBTPGR5": 5.0, "GBTPGR10": 10.0, "GBTPGR30": 30.0,
    "JGBS2": 2.0, "JGBS5": 5.0, "JGBS10": 10.0, "JGBS30": 30.0,
    "GTAUD2Y": 2.0, "GTAUD5Y": 5.0, "GTAUD10Y": 10.0, "GTAUD30Y": 30.0,
    "GTCAD2Y": 2.0, "GTCAD5Y": 5.0, "GTCAD10Y": 10.0, "GTCAD30Y": 30.0,
}

# fly structures
FLIES = [
    (2.0, 5.0, 10.0),
    (5.0, 10.0, 30.0),
]


# helper for per-country curve DataFrame
def build_curve_df_for_country(yields: pd.DataFrame, country: str) -> pd.DataFrame:
    cols = CURVE_COLS[country]
    sub = yields[cols].copy()
    new_cols = [TENOR_MAP[c] for c in cols]
    sub.columns = new_cols
    sub = sub[sorted(sub.columns)]
    return sub


# basic curve math
def zero_coupon(y: float, T: float):
    P = np.exp(-y * T)
    dv01 = P * T * 1e-4
    return P, dv01


def interp_yield(curve_row: pd.Series, T: float) -> float:
    xs = curve_row.index.values.astype(float)
    ys = curve_row.values.astype(float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    return float(np.interp(T, xs, ys))


def roll_down_dy(curve_row: pd.Series, T: float, H: float) -> float:
    if T - H <= 0:
        return 0.0
    y_T = interp_yield(curve_row, T)
    y_TH = interp_yield(curve_row, T - H)
    return y_TH - y_T


def crd_single_leg(curve_row: pd.Series, T: float, H: float, fund_rate: float):
    y = interp_yield(curve_row, T)
    P, dv01 = zero_coupon(y, T)
    dy_roll = roll_down_dy(curve_row, T, H)
    dy_bps = dy_roll * 1e4
    roll_pnl = -dv01 * dy_bps
    carry = (y - fund_rate) * H * P
    crd_ret = (roll_pnl + carry) / P
    return crd_ret, P, dv01


def fly_weight_DV01_neutral(dv01_L, dv01_B, dv01_R):
    A = np.array([[dv01_L, dv01_R],
                  [1.0, 1.0]])
    b = np.array([-dv01_B, -1.0])
    w_L, w_R = np.linalg.solve(A, b)
    return float(w_L), 1.0, float(w_R)


def zero_price_from_curve(curve_row: pd.Series, T: float):
    y = interp_yield(curve_row, T)
    P, _ = zero_coupon(y, T)
    return P


# CRD signal / z-score for a single fly
def compute_signal_for_fly(curve_df: pd.DataFrame,
                           tenors: tuple,
                           H: float,
                           roll_win: int = ROLL_WIN_6M,
                           ma_win: int = MA_WIN) -> pd.DataFrame:
    T_L, T_B, T_R = tenors
    rows = []

    for date, curve_row in curve_df.iterrows():
        if curve_row.isna().any():
            rows.append((date, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue

        if 2.0 in curve_row.index:
            fund_rate = curve_row.loc[2.0]
        else:
            fund_rate = curve_row.iloc[0]

        crd_L, P_L, dv01_L = crd_single_leg(curve_row, T_L, H, fund_rate)
        crd_B, P_B, dv01_B = crd_single_leg(curve_row, T_B, H, fund_rate)
        crd_R, P_R, dv01_R = crd_single_leg(curve_row, T_R, H, fund_rate)

        wL, wB, wR = fly_weight_DV01_neutral(dv01_L, dv01_B, dv01_R)
        fly_dv01 = dv01_L * wL + dv01_B * wB + dv01_R * wR

        fly_crd = wL * crd_L + wB * crd_B + wR * crd_R

        rows.append((date, fly_crd, wL, wB, wR, fly_dv01, fund_rate))

    sig_df = pd.DataFrame(
        rows,
        columns=[
            "Date",
            "sig_raw",
            "wL", "wB", "wR",
            "fly_dv01",
            "fund_rate",
        ]
    ).set_index("Date")

    sig_df["sig_smooth"] = sig_df["sig_raw"].rolling(ma_win, min_periods=1).mean()
    roll = sig_df["sig_smooth"].rolling(roll_win, min_periods=20)
    sig_df["z_sig_6m"] = (sig_df["sig_smooth"] - roll.mean()) / roll.std(ddof=1)

    return sig_df


# cost-aware DV01-limited backtest with cash sweep
def run_portfolio_backtest_with_basic_costs(curve_by_cty: dict,
                                            sig_by_key: dict,
                                            dv01_limit_fly: float,
                                            entry_z: float,
                                            exit_z: float,
                                            max_positions: int = 4,
                                            cost_roundtrip_bp_country: dict = None,
                                            fly_liq_mult: dict = None):
    if cost_roundtrip_bp_country is None:
        cost_roundtrip_bp_country = COST_ROUNDTRIP_BP_COUNTRY
    if fly_liq_mult is None:
        fly_liq_mult = FLY_LIQ_MULT

    CAPITAL_TOTAL = max_positions * dv01_limit_fly * 100.0
    CAPITAL_PER_FLY = dv01_limit_fly * 100.0
    dt = 1.0 / ANNUAL_DAYS

    common_dates = None
    for cty, cdf in curve_by_cty.items():
        common_dates = cdf.index if common_dates is None else common_dates.intersection(cdf.index)
    common_dates = common_dates.sort_values()

    positions = {
        key: {
            "pos_dir": 0,
            "wL": 0.0,
            "wB": 0.0,
            "wR": 0.0,
            "scale": 0.0,
            "hold_days": 0,
            "prev_curve": None,
        }
        for key in sig_by_key.keys()
    }

    portfolio_rows = []
    trade_log = []

    for date in common_dates:
        daily_pnl_gross = 0.0
        trade_cost_today = 0.0

        # existing positions (fly PnL)
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
                    wL * (P_L1 - P_L0)
                    + wB * (P_B1 - P_B0)
                    + wR * (P_R1 - P_R0)
                )

                fund_prev = float(prev_curve.iloc[0])
                y_L_prev = interp_yield(prev_curve, T_L)
                y_B_prev = interp_yield(prev_curve, T_B)
                y_R_prev = interp_yield(prev_curve, T_R)

                carry_L = (y_L_prev - fund_prev) * dt * P_L0
                carry_B = (y_B_prev - fund_prev) * dt * P_B0
                carry_R = (y_R_prev - fund_prev) * dt * P_R0

                fly_carry = (
                    wL * carry_L
                    + wB * carry_B
                    + wR * carry_R
                )

                pnl_m2m = fly_leg_pnl * scale * pos_dir
                pnl_carry = fly_carry * scale * pos_dir

            daily_pnl_fly = pnl_m2m + pnl_carry
            daily_pnl_gross += daily_pnl_fly
            state["prev_curve"] = curve_today

        # cash PnL from unused capital (US 2y)
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

        # exit rules + cost on exits
        for key, state in positions.items():
            country, fly_name = key
            sig_df = sig_by_key[key]["sig_df"]
            if date not in sig_df.index:
                continue

            z_sig = sig_df.loc[date, "z_sig_6m"]
            pos_dir = state["pos_dir"]

            if pos_dir != 0:
                if (not np.isnan(z_sig)) and (abs(z_sig) < exit_z):
                    base_roundtrip_bp = cost_roundtrip_bp_country.get(country, 0.15)
                    mult = fly_liq_mult.get(fly_name, 1.0)
                    roundtrip_bp = base_roundtrip_bp * mult
                    per_side_bp = roundtrip_bp / 2.0
                    cost_per_side = per_side_bp * dv01_limit_fly
                    trade_cost_today += cost_per_side

                    trade_log.append({
                        "date": date,
                        "country": country,
                        "fly": fly_name,
                        "action": "EXIT",
                        "pos_dir": pos_dir,
                    })
                    state["pos_dir"] = 0
                    state["scale"] = 0.0
                    state["hold_days"] = 0

        # entry rules + cost on entries
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
                z_sig = row["z_sig_6m"]
                if np.isnan(z_sig) or abs(z_sig) <= entry_z:
                    continue
                candidates.append((abs(z_sig), z_sig, key, row))

            candidates.sort(key=lambda x: x[0], reverse=True)

            for abs_z, z_sig, key, row in candidates[:remaining_capacity]:
                state = positions[key]
                country, fly_name = key

                wL_sig = row["wL"]
                wB_sig = row["wB"]
                wR_sig = row["wR"]
                fly_dv01_sig = row["fly_dv01"]

                pos_dir = 1 if z_sig > 0 else -1
                scale = dv01_limit_fly / max(fly_dv01_sig, 1e-12)

                base_roundtrip_bp = cost_roundtrip_bp_country.get(country, 0.15)
                mult = fly_liq_mult.get(fly_name, 1.0)
                roundtrip_bp = base_roundtrip_bp * mult
                per_side_bp = roundtrip_bp / 2.0
                cost_per_side = per_side_bp * dv01_limit_fly
                trade_cost_today += cost_per_side

                state["pos_dir"] = pos_dir
                state["wL"] = wL_sig
                state["wB"] = wB_sig
                state["wR"] = wR_sig
                state["scale"] = scale
                state["hold_days"] = 0

                trade_log.append({
                    "date": date,
                    "country": country,
                    "fly": fly_name,
                    "action": "ENTER",
                    "pos_dir": pos_dir,
                    "z_sig": z_sig,
                })

        # snapshot & net PnL
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
    port_bt["cum_pnl_gross"] = port_bt["portfolio_daily_pnl_gross"].cumsum()
    port_bt["cum_pnl_net"] = port_bt["portfolio_daily_pnl_net"].cumsum()

    port_bt["ret_capital_gross"] = port_bt["portfolio_daily_pnl_gross"] / CAPITAL_TOTAL
    port_bt["ret_capital_net"] = port_bt["portfolio_daily_pnl_net"] / CAPITAL_TOTAL

    port_bt["cum_ret_capital_gross"] = (1 + port_bt["ret_capital_gross"].fillna(0)).cumprod() - 1
    port_bt["cum_ret_capital_net"] = (1 + port_bt["ret_capital_net"].fillna(0)).cumprod() - 1

    trade_log_df = pd.DataFrame(trade_log).sort_values("date")

    return port_bt, trade_log_df


# key stats on capital-normalised gross/net returns
def compute_core_stats(portfolio_bt_cost, annual_days=ANNUAL_DAYS):
    ret_gross = portfolio_bt_cost["ret_capital_gross"].dropna()
    ret_net = portfolio_bt_cost["ret_capital_net"].dropna()

    def ann_stats(r):
        if len(r) < 2:
            return np.nan, np.nan, np.nan
        mu = r.mean() * annual_days
        vol = r.std(ddof=1) * np.sqrt(annual_days)
        sharpe = mu / vol if vol > 0 else np.nan
        return mu, vol, sharpe

    ann_ret_gross, ann_vol_gross, sharpe_gross = ann_stats(ret_gross)
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

    var_es = {}
    for alpha in [0.95, 0.99]:
        if len(ret_net) == 0:
            var_es[alpha] = {"VaR": np.nan, "ES": np.nan}
            continue
        q = np.quantile(ret_net, 1 - alpha)
        var = -q
        tail = ret_net[ret_net <= q]
        es = -tail.mean() if len(tail) > 0 else np.nan
        var_es[alpha] = {"VaR": var, "ES": es}

    core_stats = {
        "ann_ret_capital_gross": ann_ret_gross,
        "ann_vol_capital_gross": ann_vol_gross,
        "sharpe_gross": sharpe_gross,
        "ann_ret_capital_net": ann_ret_net,
        "ann_vol_capital_net": ann_vol_net,
        "sharpe_net": sharpe_net,
        "max_drawdown_net_pct": max_dd_net,
        "calmar_net": calmar_net,
        "sortino_net": sortino_net,
        "hit_rate_net": hit_rate,
        "avg_win_net": avg_win,
        "avg_loss_net": avg_loss,
        "VaR_95_net": var_es[0.95]["VaR"],
        "ES_95_net": var_es[0.95]["ES"],
        "VaR_99_net": var_es[0.99]["VaR"],
        "ES_99_net": var_es[0.99]["ES"],
    }
    return core_stats


# turnover, holding periods and average cost in bp
def compute_turnover_and_costs(portfolio_bt_cost,
                               trade_log_cost,
                               dv01_limit_fly,
                               annual_days=ANNUAL_DAYS):
    tl = trade_log_cost.sort_values("date").copy()
    num_enters = (tl["action"] == "ENTER").sum()
    num_exits = (tl["action"] == "EXIT").sum()
    num_roundtrips = min(num_enters, num_exits)

    years = len(portfolio_bt_cost) / annual_days if len(portfolio_bt_cost) > 0 else np.nan
    turnover_trades_per_year = num_roundtrips / years if years and years > 0 else np.nan

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

    if len(holding_days) > 0:
        avg_holding = float(np.mean(holding_days))
        med_holding = float(np.median(holding_days))
    else:
        avg_holding = np.nan
        med_holding = np.nan

    total_trade_cost = portfolio_bt_cost["trade_cost"].sum()
    num_sides = len(tl)
    if num_sides > 0 and dv01_limit_fly > 0:
        avg_cost_per_side_bp = total_trade_cost / (dv01_limit_fly * num_sides)
        avg_cost_per_roundtrip_bp = 2.0 * avg_cost_per_side_bp
    else:
        avg_cost_per_side_bp = np.nan
        avg_cost_per_roundtrip_bp = np.nan

    turnover_stats = {
        "num_enters": num_enters,
        "num_exits": num_exits,
        "num_roundtrips": num_roundtrips,
        "turnover_roundtrips_per_year": turnover_trades_per_year,
        "avg_holding_days": avg_holding,
        "median_holding_days": med_holding,
        "avg_cost_per_side_bp": avg_cost_per_side_bp,
        "avg_cost_per_roundtrip_bp": avg_cost_per_roundtrip_bp,
    }
    return turnover_stats


# simple cash vs fly PnL attribution
def compute_pnl_attribution(portfolio_bt_cost):
    df = portfolio_bt_cost.copy()
    df["fly_pnl_gross"] = df["portfolio_daily_pnl_gross"] - df["cash_pnl"]

    total_portfolio_pnl = df["portfolio_daily_pnl_gross"].sum()
    total_cash_pnl = df["cash_pnl"].sum()
    total_fly_pnl = df["fly_pnl_gross"].sum()

    frac_cash = total_cash_pnl / total_portfolio_pnl if total_portfolio_pnl != 0 else np.nan
    frac_fly = total_fly_pnl / total_portfolio_pnl if total_portfolio_pnl != 0 else np.nan

    attribution = {
        "total_portfolio_pnl_gross": total_portfolio_pnl,
        "total_cash_pnl": total_cash_pnl,
        "total_fly_pnl": total_fly_pnl,
        "frac_cash_of_total": frac_cash,
        "frac_fly_of_total": frac_fly,
    }
    return attribution


# Main
if __name__ == "__main__":
    yields = load_yields(xlsx_path)

    curve_by_cty = {}
    for country in CURVE_COLS.keys():
        curve_by_cty[country] = build_curve_df_for_country(yields, country)

    sig_by_key = {}
    for country, curve_df in curve_by_cty.items():
        for tenors in FLIES:
            if not all(t in curve_df.columns for t in tenors):
                continue
            fly_name = f"{int(tenors[0])}s{int(tenors[1])}s{int(tenors[2])}s"
            sig_df = compute_signal_for_fly(curve_df, tenors, H)
            key = (country, fly_name)
            sig_by_key[key] = {
                "tenors": tenors,
                "sig_df": sig_df,
            }

    portfolio_bt_cost, trade_log_cost = run_portfolio_backtest_with_basic_costs(
        curve_by_cty,
        sig_by_key,
        dv01_limit_fly=DV01_LIMIT_FLY,
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
        max_positions=4,
    )

    core_stats = compute_core_stats(portfolio_bt_cost)
    turnover_stats = compute_turnover_and_costs(
        portfolio_bt_cost,
        trade_log_cost,
        dv01_limit_fly=DV01_LIMIT_FLY,
    )
    pnl_attr = compute_pnl_attribution(portfolio_bt_cost)

    print("\n=== Core performance stats (net, after costs) ===")
    print(pd.Series(core_stats).round(4))

    print("\n=== Turnover / holding / cost stats ===")
    print(pd.Series(turnover_stats).round(4))

    print("\n=== P&L attribution (gross) ===")
    print(pd.Series(pnl_attr).round(4))

