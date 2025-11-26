# MATH5300_2025Fall  
### Hedge Fund Strategies & Risk — Yield Curve Arbitrage

## Data & Methodology Note — Zero-Coupon Curves and Bloomberg Limits

For the current stage of the project, all curve flies, CRD signals, and DV01-neutral weights are built off **zero-coupon curves** rather than raw coupon-bond prices. This is basically how fixed income desks think about curve shape and risk: theo PnL, DV01 and carry/roll-down are all defined relative to the **zero curve**, while the actual bonds or futures are simply the instruments used to express the trade.

Previous implementations of similar strategies often used bond prices directly, but because of the Bloomberg limits we ran into, using the zero curve is the most practical and technically correct proxy for now. It lines up with how the standard modelling texts treat the term structure (Brigo & Mercurio; Andersen & Piterbarg; Hull).

---

## Realised PnL Using Actual Bond Prices (to be added once data is pulled)

When the Bloomberg quota at the Uris Library resets at the start of next month, we will pull full benchmark bond histories for each sovereign (2y, 5y, 10y, 30y). Once that data is in, we can switch on the realised PnL module using the usual desk-style marking:

- **Fix notionals at entry using DV01-neutral weights**

      sum_i [ N_i * DV01_i ] = 0

- **Mark to dirty close each day**

      PnL_t = sum_i [ N_i * ( P_{i,t} - P_{i,t-1} ) ]
              + sum_i [ N_i * Coupon_{i,t} ]
              - TC_t

Dirty closes naturally include accrual, so day-to-day PnL reflects both price moves and interest build-up. Coupon cashflows are added on payment dates. Transaction costs are netted out on entry and exit.

- **Convert non-USD legs into USD**

      PnL_USD_t = PnL_local_t * FX_t

Once we load the bond data, this will give us proper realised PnL that we can compare directly to the zero-curve theo PnL. Until then, the zero-curve version is a reasonable and defensible stand-in.

---

## Why the z-score entry and exit levels look like this

Our entry and exit rules  
- enter when |z| > 1.6  
- exit when |z| < 0.2  

are not arbitrary. They come from the usual empirical work on mean-reversion strategies, where signals are normalised and trades only occur when deviations are far enough from the mean to be meaningful. In the literature on pairs trading, spread trades and OU-type mean-reversion, entry levels around **1.5 to 2.0** standard deviations are common (Gatev, Goetzmann and Rouwenhorst 2006; Elliott, Van der Hoek and Malcolm 2005).

We tested a range of thresholds and 1.6 worked well for our turnover, cost assumptions and multi-country setup. The narrow exit band at 0.2 helps cut positions cleanly once the signal reverts.

---

## Bloomberg data constraints

We were able to pull the futures data we needed (US TU, FV, TY, US; Germany Schatz, Bobl, Bund, Buxl; UK gilt futures). The issue was with pulling a complete set of benchmark bond histories and longer swap tenors across seven sovereign markets. The Uris Library terminal has strict daily and monthly download limits, and we hit those limits before we could extract everything.

This is why the current version uses zero curves for modelling. It is still fully aligned with standard practice for theo curve PnL.

---

## Future work

As soon as the Bloomberg quota resets, we will pull:

- Full benchmark bond prices for 2y, 5y, 10y and 30y  
- CTD-adjusted DV01s for the main futures  
- Complete swap curve histories  
- Better country-specific cost estimates  

That will let us generate both the zero-curve theoretical PnL and the full realised, executable PnL.

---

## References

- Brigo, D and Mercurio, F (2006). *Interest Rate Models: Theory and Practice*. Springer.  
- Andersen, L and Piterbarg, V (2010). *Interest Rate Modelling*, Volume I. Atlantic Financial Press.  
- Hull, J (2018). *Options, Futures and Other Derivatives*, 10th ed. Pearson.  
- Gatev, E; Goetzmann, W; Rouwenhorst, K (2006). *Pairs Trading: Performance of a Relative-Value Arbitrage Rule*.  
- Elliott, R; Van der Hoek, J; Malcolm, W (2005). *Pairs Trading*.  
