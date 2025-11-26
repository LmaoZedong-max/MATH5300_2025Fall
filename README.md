# MATH5300_2025Fall  
### Hedge Fund Strategies & Risk:  Yield Curve Arbitrage

## Data & Methodology Note: Zero-Coupon Curves and Bloomberg Limits

In this project we build our flies, CRD signals and DV01-neutral portfolios off **zero-coupon curves**, not raw coupon-bond prices. That is broadly how fixed-income desks think about curve risk: theo PnL, DV01 and curve shape are all defined relative to the **zero-coupon term structure**, with the actual bonds or futures just used for execution.

### Why zero-coupon curves?

The standard interest-rate texts all lean on zero curves as the basic object:

- Pricing, hedging and sensitivities are done off the **discount / zero curve**, rather than straight from bond prices.
- Zero curves give an **arbitrage-free, maturity-consistent** view of the term structure, which is what you want for DV01 buckets, carry and roll-down.
- Curve trades (including flies) are really trades in **changes in zero rates** across maturities, even if you put them on in bonds or futures.

This is the treatment in:
- Brigo & Mercurio — *Interest Rate Models: Theory and Practice* (Ch. 1–2)  
- Andersen & Piterbarg — *Interest Rate Modelling*, Vol. 1 (Ch. 3)  
- Hull — *Options, Futures and Other Derivatives* (Ch. 7)  

So using zero curves for our theoretical curve PnL is very much in line with how people do this in practice.

---

## Realised PnL Using Actual Bond Prices (for future data refresh)

Once we have full benchmark government bond data (2y/5y/10y/30y) for each country, we’ll back out **realised PnL** in the usual desk way:

- **Fix trade notionals \(N_i\) at entry using DV01-neutral weights**

      sum_i [ N_i * DV01_i ] = 0

- **Mark to dirty close each day**

      PnL_t = sum_i [ N_i * ( P_{i,t} - P_{i,t-1} ) ]
              + sum_i [ N_i * Coupon_{i,t} ]
              - TC_t

  - We work with **dirty close prices**, so coupon accrual is naturally baked into the daily price move.
  - Actual coupon cashflows `Coupon_{i,t}` are added on payment dates.
  - `TC_t` collects bid–ask, fees and any other transaction costs.

- **Convert non-USD legs back to USD**

      PnL_USD_t = PnL_local_t * FX_t

That gives us an executable, bond-level realised PnL that we can line up against the zero-curve theo PnL.

---

## Bloomberg Data Constraints

We’ve already pulled the futures history for the liquid contracts  
(US TU/FV/TY/US; Germany Schatz/Bobl/Bund/Buxl; UK gilt futures).

Pulling **full benchmark bond and swap histories** for all seven sovereigns, however, hit the limits of the Columbia Uris Library Bloomberg terminal:

- Hard **daily and monthly download caps**,
- Patchy access to older benchmark bonds and long-tenor swaps,
- Not enough quota to grab every series we’d like across all countries.

Because of that, the current version leans on **zero-coupon curves as the modelling object**, which still matches the way desks run theo curve PnL and risk.

---

## Future Work

When the Bloomberg quota resets, the plan is to:

- Download benchmark 2y/5y/10y/30y bond prices for each sovereign,
- Map futures via CTD-adjusted DV01s,
- Fill out the swap-curve history,
- Layer in more realistic, country-specific transaction-cost assumptions.

At that point we can report both the **theoretical curve PnL** and the **fully realised, executable PnL** for the same strategy.

---

## References

- Brigo, D. & Mercurio, F. (2006). *Interest Rate Models: Theory and Practice*. Springer.  
- Andersen, L. & Piterbarg, V. (2010). *Interest Rate Modelling*, Volume I. Atlantic Financial Press.  
- Hull, J. (2018). *Options, Futures and Other Derivatives* (10th ed.). Pearson.
