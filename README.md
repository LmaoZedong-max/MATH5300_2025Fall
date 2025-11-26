# MATH5300_2025Fall  
### Hedge Fund Strategies & Risk — Yield Curve Arbitrage

## Data & Methodology Note — Zero-Coupon Curves, Futures, and Bloomberg Limits

For now, our full curve-fly construction — CRD signals, z-score entries/exits, and DV01-neutral portfolio weights — is built off **zero-coupon curves**. This is the standard modelling framework in fixed-income, where theo-P&L, curve shape, carry/roll-down and DV01 are all defined relative to the **zero-coupon term structure**, not directly from raw bond prices.

Previous versions of similar projects often used **benchmark bond prices directly**, but given our Bloomberg restrictions (see below), zero curves are the most reasonable and academically/industry-consistent proxy until the full dataset becomes available at the start of next month.

### Why zero-coupon curves? (Industry-standard foundations)

Across the core interest-rate literature (Brigo & Mercurio; Andersen & Piterbarg; Hull), the treatment is consistent:

- Pricing, hedging and risk attribution are carried out with respect to the **discount curve / zero curve** rather than coupon bonds.  
- Zero curves give an **arbitrage-free, maturity-consistent** representation of the term structure — essential for DV01 buckets, carry, roll-down, and curvature.  
- Curve trades (including flies) are economically trades in **relative zero-rate movements**, even if executed using coupon bonds or futures.

These sources form the baseline for modern fixed-income desk methodology and justify our use of zeros for theoretical curve-P&L.

---

## Realised PnL Using Actual Bond Prices (to be added once data is pulled)

At the **beginning of next month**, once Bloomberg download quotas reset, we will pull the full benchmark government bond history (2y/5y/10y/30y) for all seven sovereigns.  
That will allow us to compute **bond-level realised P&L**, matching what a live desk would see.

The realised P&L engine (already built) works as follows:

- **Fix trade notionals \(N_i\)** at entry using DV01-neutral weights:

      sum_i [ N_i * DV01_i ] = 0

- **Daily realised / MTM PnL**, marked to *dirty close prices*:

      PnL_t = sum_i [ N_i * ( P_{i,t} - P_{i,t-1} ) ]
              + sum_i [ N_i * Coupon_{i,t} ]
              - TC_t

  - Dirty closes incorporate accrued interest naturally.  
  - Explicit coupon cashflows `Coupon_{i,t}` are added on payment dates.  
  - `TC_t` includes bid–ask and execution costs.  

- **FX conversion** for non-USD sovereigns:

      PnL_USD_t = PnL_local_t * FX_t

Once the bond dataset is retrieved, this module will be activated to report **full realised P&L** alongside the theoretical curve P&L from zero curves.

Until then, the zero-curve implementation is an appropriate and academically accepted proxy — and aligns directly with the theoretical foundation used by actual rates desks.

---

## Why our z-score entry/exit thresholds make sense (empirical mean-reversion practice)

Our entry and exit rules  
- `enter: |z| > 1.6`  
- `exit: |z| < 0.2`  

come directly from the **empirical research tradition in mean-reversion strategies**, where signals are normalised (usually via z-scores) and trades are triggered when deviations reach statistically meaningful levels.

Across academic
