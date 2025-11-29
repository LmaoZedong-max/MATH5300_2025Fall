# MATH5300_2025Fall  
### Hedge Fund Strategies & Risk — Yield Curve Arbitrage



## Why the z-score entry and exit levels look like this

Our entry and exit rules  
- enter when |z| > 1.6  
- exit when |z| < 0.2  

are not arbitrary. They come from the usual empirical work on mean-reversion strategies, where signals are normalised and trades only occur when deviations are far enough from the mean to be meaningful. In the literature on pairs trading, spread trades and OU-type mean-reversion, entry levels around **1.5 to 2.0** standard deviations are common (Gatev, Goetzmann and Rouwenhorst 2006; Elliott, Van der Hoek and Malcolm 2005).

We tested a range of thresholds and 1.6 worked well for our turnover, cost assumptions and multi-country setup. The narrow exit band at 0.2 helps cut positions cleanly once the signal reverts.

### Evidence on z-score thresholds in the literature

Below is a short list of papers and practitioner pieces we are implicitly sitting on top of:

- **Gatev, Goetzmann & Rouwenhorst (2006)** – [“Pairs Trading: Performance of a Relative-Value Arbitrage Rule”](https://ssrn.com/abstract=141615)  
  Use a distance-based spread and open trades when the spread deviates by k = 2 standard deviations from its historical mean, closing as it reverts back towards the mean (entry |z| ≈ 2, exit |z| ≈ 0). 

- **Stübinger (2017) High-Frequency Pairs Trading** – [“Statistical Arbitrage Pairs Trading with High-frequency Data”](https://www.econjournals.com/index.php/ijefi/article/download/5127/pdf/13716)  
  Implements a Gatev-style framework and again sets upper and lower entry bands at k = 2 standard deviations around the mean spread (entry |z| ≈ 2, exit on reversion). 

- **Avellaneda & Lee (2010)** – [“Statistical Arbitrage in the U.S. Equities Market”](https://traders.berkeley.edu/papers/Statistical%20arbitrage%20in%20the%20US%20equities%20market.pdf)  
  Model idiosyncratic residuals as a mean reverting process and enter when the residual z-score exceeds 1.25 in absolute value, exiting when it falls below 0.5 (entry |z| ≥ 1.25, exit |z| < 0.5). 

- **Tokat et al. (2022) – ETF Pairs** – [“Pairs trading: is it applicable to exchange-traded funds?”](https://www.sciencedirect.com/science/article/pii/S2214845021000880)  
  Systematically sweep threshold levels between 0.1 and 3.1 standard deviations in 0.2 increments, so our 1.6 entry sits comfortably inside the empirically tested sigma grid. 
- **Vidyamurthy (2004) plus later summaries** – [*Pairs Trading: Quantitative Methods and Analysis*](https://www.researchgate.net/publication/47801548_Pairs_Trading_Quantitative_Methods_and_Analysis_G_Vidyamurthy)  
  Lays out the standard cointegration-based framework and, as later survey work notes, uses two-standard-deviation style bands as a benchmark before moving to more dynamic signal rules (canonical entry |z| around 2, exit near 0). 

- **QuantInsti (2022) Practitioner Guide** – [“Pairs Trading for Beginners: Correlation, Cointegration, Z-score”](https://blog.quantinsti.com/pairs-trading-basics/)  
  Explicitly recommends z-score entry thresholds in the 1.5–2.0 sigma range and stop-loss levels around 3 sigma, with exits as the spread reverts towards the mean (entry |z| ≈ 1.5–2, exit |z| ≈ 0). 

- **Investopedia Mean-Reversion Overview** – [“What Is Mean Reversion, and How Do Investors Use It?”](https://www.investopedia.com/terms/m/meanreversion.asp)  
  States that a z-score above about 1.5 or 2 (or below −1.5 or −2) is commonly treated as a trading signal in mean reversion setups, which supports using thresholds in this band. 

- **Palomar (2024) Pairs Trading Chapter** – [“Trading the Spread” in *Portfolio Optimization*](https://bookdown.org/palomar/portfoliooptimizationbook/15.5-trading-spread.html)  
  Formalises the choice of a symmetric threshold s₀ for a z-score assumed to be N(0,1) and shows how to optimise s₀ empirically, which again leaves 1–2 sigma as the natural region to search over. 

Overall, our choice of 1.6 for entry and 0.2 for exit is basically a slightly more selective version of the “1.5 to 2 in, near-zero out” pattern that keeps showing up in both the academic studies and the practitioner playbooks.

---
