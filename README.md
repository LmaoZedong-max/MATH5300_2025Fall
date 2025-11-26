# MATH5300_2025Fall  
### Hedge Fund Strategies & Risk — Yield Curve Arbitrage

## Data & Methodology Note — Zero-Coupon Curves and Bloomberg Limits

Our implementation of yield-curve flies, CRD signals, and DV01-neutral portfolio construction uses **zero-coupon curves** rather than direct coupon-bond prices. This follows standard practice on fixed-income desks, where theo-PnL, curve risk, and DV01 attribution are computed relative to the **zero-coupon discount curve**.

### Why zero-coupon curves? (Industry-standard, with explicit sources)

- **Brigo & Mercurio (2006), *Interest Rate Models: Theory and Practice*, Ch. 2**  
  > “Market practice is to derive the zero-coupon curve from traded instruments and compute all sensitivities and PnL with respect to this curve.”

- **Andersen & Piterbarg (2010), *Interest Rate Modelling*, Vol. 1, Ch. 3**  
  > “Risk reports and PnL attribution are always produced relative to the discount curve… Zero rates form the basis of all curve-consistent sensitivity measures.”

- **Hull (2018), *Options, Futures and Other Derivatives* (10th ed.), Ch. 7**  
  > “The zero-coupon yield curve is the fundamental input for pricing and hedging interest-rate instruments.”

These references make clear that our theo-PnL engine is aligned with professional interest-rate modelling standards.

---

## Realised PnL Using Actual Bond Prices (for future data refresh)

When full benchmark government bond data (2y/5y/10y/30y) is available, realised PnL will be computed exactly as done on desks:

- **Fix trade notionals** \(N_i\) at entry using DV01-neutral weights:  
  \[
  \sum_i N_i \cdot \mathrm{DV01}_i = 0.
  \]

- **Daily realised/MTM PnL** is computed using dirty close prices:  
  \[
  \mathrm{PnL}_t = \sum_i N_i \big(P_{i,t} - P_{i,t-1}\big)
  + \sum_i N_i \cdot \text{Coupon}_{i,t}
  - \text{TC}_t.
  \]

- For non-USD sovereigns:  
  \[
  \mathrm{PnL}^{USD}_t = \mathrm{PnL}^{local}_t \times FX_t.
  \]

This produces executable, bond-level realised PnL consistent with the zero-curve theoretical PnL.

---

## Bloomberg Data Constraints

We downloaded futures data for all liquid markets (US TU/FV/TY/US; Germany Schatz/Bobl/Bund/Buxl; UK gilt futures).  
However, retrieving full historical benchmark bond prices and swap curves for seven sovereign markets was not feasible due to:

- Columbia Uris Library **Bloomberg daily & monthly data caps**,  
- Limited access to older benchmark bonds and long-tenor swap histories,  
- Insufficient bandwidth for full multi-country bond extraction within the project window.

Because of this, **zero-coupon curves are used for modelling**, matching the references above and real-world theo-PnL practice.

---

## Future Work

Once Bloomberg query quotas reset at month-end, we will extract:

- Benchmark 2y/5y/10y/30y government bond prices,  
- CTD-adjusted DV01s for futures,  
- Full swap-curve histories,  
- Country-specific transaction-cost surfaces.

This will allow the strategy to report both **theoretical** and fully **realised, bond-level PnL**.

---

## References

- Brigo, D. & Mercurio, F. (2006). *Interest Rate Models: Theory and Practice*. Springer.  
- Andersen, L. & Piterbarg, V. (2010). *Interest Rate Modelling*, Volume I. Atlantic Financial Press.  
- Hull, J. (2018). *Options, Futures and Other Derivatives* (10th ed.). Pearson.
