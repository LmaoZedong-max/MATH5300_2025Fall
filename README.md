# MATH5300_2025Fall  
### Hedge Fund Strategies & Risk — Yield Curve Arbitrage

## Data & Methodology Note — Zero-Coupon Curves and Bloomberg Limits

Our implementation of yield-curve flies, CRD signals, and DV01-neutral portfolio construction uses **zero-coupon curves** rather than direct coupon-bond prices. This approach follows established practice in fixed-income modelling, where theo-PnL, rate risk, DV01 and curve-shape analysis are defined relative to the **zero-coupon term structure**.

### Why zero-coupon curves? (Industry-standard foundations)

The foundational literature in interest-rate modelling consistently emphasises that:

- Pricing, hedging, and sensitivity computations for fixed-income instruments are performed with respect to the **zero-coupon discount curve**, not directly from coupon-bond prices.
- Zero curves provide the **arbitrage-free, maturity-consistent** representation of the term structure used for DV01, bucket risk, carry, and roll-down.
- Curve trades (including flies) are economically defined via **changes in zero rates** rather than raw bond price moves.

This treatment is described in:
- Brigo & Mercurio — *Interest Rate Models: Theory and Practice* (Ch. 1–2)  
- Andersen & Piterbarg — *Interest Rate Modelling*, Vol. 1 (Ch. 3)  
- Hull — *Options, Futures and Other Derivatives* (Ch. 7)

These texts form the basis for modern desk methodology and justify our use of zero-coupon curves for theoretical PnL.

---

## Realised PnL Using Actual Bond Prices (for future data refresh)

When full benchmark government bond data (2y/5y/10y/30y) is available, realised PnL will be computed exactly as done on fixed-income desks:

- **Fix trade notionals (N_i) at entry using DV01-neutral weights:**

      sum_i [ N_i * DV01_i ] = 0

- **Daily realised / MTM PnL using *dirty close prices* (which include accrued interest):**

      PnL_t = sum_i [ N_i * ( P_{i,t} - P_{i,t-1} ) ]
              + sum_i [ N_i * Coupon_{i,t} ]
              - TC_t

  - Dirty close prices ensure that coupon accrual is captured naturally in daily PnL.
  - Explicit coupon cashflows (Coupon_{i,t}) are added on coupon payment dates.

- **FX conversion for non-USD sovereigns:**

      PnL_USD_t = PnL_local_t * FX_t

This produces executable, bond-level realised PnL suitable for comparison with the zero-curve theoretical PnL.


---

## Bloomberg Data Constraints

We successfully downloaded futures data for liquid markets (US TU/FV/TY/US; Germany Schatz/Bobl/Bund/Buxl; UK gilt futures).  
However, extracting full benchmark bond histories for all seven sovereign markets was not feasible due to:

- Bloomberg **daily and monthly download caps** at Columbia Uris Library  
- Limited access to historical benchmark bonds and older swap tenors  
- Insufficient quota for full multi-country extraction within the project window

Because of this, **zero-coupon curves serve as the primary modelling object**, which aligns with the references listed above and with standard theo-PnL practice on professional desks.

---

## Future Work

Once Bloomberg quotas reset, we will extract:

- Benchmark 2y/5y/10y/30y bond prices  
- CTD-adjusted DV01s for all futures  
- Complete swap-curve histories  
- Country-specific transaction cost surfaces  

This will enable full **realised** (executable) PnL alongside the theoretical curve-PnL.

---

## References

- Brigo, D. & Mercurio, F. (2006). *Interest Rate Models: Theory and Practice*. Springer.  
- Andersen, L. & Piterbarg, V. (2010). *Interest Rate Modelling*, Volume I. Atlantic Financial Press.  
- Hull, J. (2018). *Options, Futures and Other Derivatives* (10th ed.). Pearson.
