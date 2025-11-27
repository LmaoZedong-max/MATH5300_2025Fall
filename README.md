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

### Evidence on z-score thresholds in the literature

Below is a short list of papers and practitioner pieces we are implicitly sitting on top of:

- **Gatev, Goetzmann & Rouwenhorst (2006)** – [“Pairs Trading: Performance of a Relative-Value Arbitrage Rule”](https://ssrn.com/abstract=141615)  
  Use a distance-based spread and open trades when the spread deviates by k = 2 standard deviations from its historical mean, closing as it reverts back towards the mean (entry |z| ≈ 2, exit |z| ≈ 0). :contentReference[oaicite:3]{index=3}  

- **Stübinger (2017) High-Frequency Pairs Trading** – [“Statistical Arbitrage Pairs Trading with High-frequency Data”](https://www.econjournals.com/index.php/ijefi/article/download/5127/pdf/13716)  
  Implements a Gatev-style framework and again sets upper and lower entry bands at k = 2 standard deviations around the mean spread (entry |z| ≈ 2, exit on reversion). :contentReference[oaicite:4]{index=4}  

- **Avellaneda & Lee (2010)** – [“Statistical Arbitrage in the U.S. Equities Market”](https://traders.berkeley.edu/papers/Statistical%20arbitrage%20in%20the%20US%20equities%20market.pdf)  
  Model idiosyncratic residuals as a mean reverting process and enter when the residual z-score exceeds 1.25 in absolute value, exiting when it falls below 0.5 (entry |z| ≥ 1.25, exit |z| < 0.5). :contentReference[oaicite:5]{index=5}  

- **Tokat et al. (2022) – ETF Pairs** – [“Pairs trading: is it applicable to exchange-traded funds?”](https://www.sciencedirect.com/science/article/pii/S2214845021000880)  
  Systematically sweep threshold levels between 0.1 and 3.1 standard deviations in 0.2 increments, so our 1.6 entry sits comfortably inside the empirically tested sigma grid. :contentReference[oaicite:6]{index=6}  

- **Vidyamurthy (2004) plus later summaries** – [*Pairs Trading: Quantitative Methods and Analysis*](https://www.researchgate.net/publication/47801548_Pairs_Trading_Quantitative_Methods_and_Analysis_G_Vidyamurthy)  
  Lays out the standard cointegration-based framework and, as later survey work notes, uses two-standard-deviation style bands as a benchmark before moving to more dynamic signal rules (canonical entry |z| around 2, exit near 0). :contentReference[oaicite:7]{index=7}  

- **QuantInsti (2022) Practitioner Guide** – [“Pairs Trading for Beginners: Correlation, Cointegration, Z-score”](https://blog.quantinsti.com/pairs-trading-basics/)  
  Explicitly recommends z-score entry thresholds in the 1.5–2.0 sigma range and stop-loss levels around 3 sigma, with exits as the spread reverts towards the mean (entry |z| ≈ 1.5–2, exit |z| ≈ 0). :contentReference[oaicite:8]{index=8}  

- **Investopedia Mean-Reversion Overview** – [“What Is Mean Reversion, and How Do Investors Use It?”](https://www.investopedia.com/terms/m/meanreversion.asp)  
  States that a z-score above about 1.5 or 2 (or below −1.5 or −2) is commonly treated as a trading signal in mean reversion setups, which supports using thresholds in this band. :contentReference[oaicite:9]{index=9}  

- **Palomar (2024) Pairs Trading Chapter** – [“Trading the Spread” in *Portfolio Optimization*](https://bookdown.org/palomar/portfoliooptimizationbook/15.5-trading-spread.html)  
  Formalises the choice of a symmetric threshold s₀ for a z-score assumed to be N(0,1) and shows how to optimise s₀ empirically, which again leaves 1–2 sigma as the natural region to search over. :contentReference[oaicite:10]{index=10}  

Overall, our choice of 1.6 for entry and 0.2 for exit is basically a slightly more selective version of the “1.5 to 2 in, near-zero out” pattern that keeps showing up in both the academic studies and the practitioner playbooks.

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

- Andersen, L and Piterbarg, V (2010). *Interest Rate Modelling*, Volume I. Atlantic Financial Press.  
- Avellaneda, M and Lee, J-H (2010). ‘Statistical Arbitrage in the U.S. Equities Market’, *Quantitative Finance*, 10(7), 761–782. :contentReference[oaicite:11]{index=11}  
- Brigo, D and Mercurio, F (2006). *Interest Rate Models: Theory and Practice*. Springer.  
- Elliott, R; Van der Hoek, J; Malcolm, W (2005). ‘Pairs Trading’, working paper. :contentReference[oaicite:12]{index=12}  
- Gatev, E; Goetzmann, W; Rouwenhorst, K (2006). ‘Pairs Trading: Performance of a Relative-Value Arbitrage Rule’, *Review of Financial Studies*, 19(3), 797–827. :contentReference[oaicite:13]{index=13}  
- Hull, J (2018). *Options, Futures and Other Derivatives*, 10th ed. Pearson.  
- Palomar, D P (2024). ‘Trading the Spread’, in *Portfolio Optimization* (online bookdown). :contentReference[oaicite:14]{index=14}  
- Tokat, E et al. (2022). ‘Pairs trading: is it applicable to exchange-traded funds?’, *Journal of Commodity Markets*. :contentReference[oaicite:15]{index=15}  
- Vidyamurthy, G (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley. :contentReference[oaicite:16]{index=16}  
- QuantInsti (2022). ‘Pairs Trading for Beginners: Correlation, Cointegration, Z-score’. :contentReference[oaicite:17]{index=17}  
- Investopedia (n.d.). ‘What Is Mean Reversion, and How Do Investors Use It?’. :contentReference[oaicite:18]{index=18}  
- Stübinger, J (2017). ‘Statistical Arbitrage Pairs Trading with High-frequency Data’, *International Journal of Economics and Financial Issues*. :contentReference[oaicite:19]{index=19}
