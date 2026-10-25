# phacker

Measure how likely it is that your best backtest is overfit.

If you try many strategy configurations and keep the one with the best in-sample
Sharpe ratio, part of that Sharpe is selection luck. `phacker` estimates the
**Probability of Backtest Overfitting (PBO)**: the probability that the configuration
you picked as best in-sample ranks below the median out-of-sample. It implements
combinatorially symmetric cross-validation (CSCV) from

> Bailey, D., Borwein, J., López de Prado, M., & Zhu, Q. J. (2016).
> *The Probability of Backtest Overfitting.* Journal of Computational Finance, 20(4), 39–69.
> [doi:10.21314/JCF.2016.322](https://doi.org/10.21314/JCF.2016.322)

CSCV is deterministic: there is no Monte Carlo, and the same input always gives the same
answer. The core is a C++ kernel (OpenMP, SIMD) with a pure-NumPy fallback.

## Install

```bash
pip install phacker
```

Wheels are built for CPython 3.11–3.13 on Linux (x86_64, aarch64), macOS (Intel,
Apple Silicon) and Windows (x64). On other platforms pip builds from source, and
if no C++ compiler is available you get the NumPy implementation alone. Both give
the same results; only the speed differs.

```python
import phacker
phacker.backend()   # "cpp" if the compiled kernel is available, else "numpy"
```

## Quick start

You need a T × N matrix of P&L: one column per configuration you tried, one row per
period, with all columns on the same time index.

```python
import numpy as np
from phacker.cscv import cscv

pnl = np.random.default_rng(0).standard_normal((2000, 500))   # 500 pure-noise strategies
pbo, logits = cscv(pnl, S=16)

pbo            # 0.317; noise averages 0.5, but one family can land well away from it
logits.shape   # (12870,) = C(16, 8)
```

**Include every configuration you tried**, including the ones that did badly.
Leaving out failed trials biases PBO downward.

### Reading the result

- **PBO = 0.5** means selection carried no information. The in-sample winner is
  equally likely to land above or below the out-of-sample median. This is what pure
  noise gives, not 0.
- **PBO near 0** means the in-sample winner reliably stays above the median
  out-of-sample.
- **PBO above 0.5** means the selection is worse than random. The in-sample winners
  tend to underperform out-of-sample (mean reversion between halves, for example).

`logits` holds one value per train/test split, λ = ln(ω / (1 − ω)), where ω is the
in-sample winner's relative out-of-sample rank. PBO is the fraction with λ ≤ 0.
With no skill, ω is uniform and λ follows the standard logistic distribution exactly.

## Building a trial matrix from prices

`phacker` includes a grid backtester for threshold rules on technical indicators.

```python
import pandas as pd
from phacker.indicators import indicators
from phacker.grid import Grid, backtest, linspace
from phacker.cscv import cscv

# a DataFrame with Adj_Close, High, Low, Volume columns on a DatetimeIndex
prices = pd.read_csv("NQ.csv", index_col=0, parse_dates=True)
prices = indicators(prices, ["RSI14", "Z20", "BOLL_B20"])

grid = Grid(
    indicators=["RSI14", "Z20", "BOLL_B20"],
    entry=linspace(-2, 2, 17),     # z-score levels
    band=[0.25, 0.5, 1.0, 2.0],    # half-width around each level
    hold=[1, 5, 10, 20],           # bars to hold after a trigger
    side="both",                   # "long", "short" or "both"
    cost_bps=0.5,                  # per unit of position change
)
M = backtest(prices, grid, price_col="Adj_Close", min_periods=60)   # TrialMatrix, 1,632 columns

pbo, logits = cscv(M, S=16)
M.names[0]     # 'RSI14:entry=-2:band=0.25:hold=1:long'
```

For each configuration (indicator, entry, band, hold, side) the rule goes long or short
whenever the indicator's z-score is within `band` of `entry`, and holds for `hold` bars.
Positions are lagged one bar, so there is no lookahead. Indicators are standardized
with an **expanding-window** z-score, so θ means the same thing for every indicator
and no future data leaks into the past. `min_periods` is the warm-up before the
z-score is defined.

Available indicators: `SMA5`, `SMA20`, `EMA5`, `EMA20`, `RSI14`, `ATR14`, `OBV`,
`BOLL_B20`, `ADX14`, `VOL20`, `Z20`.

To bring your own P&L with labels, wrap it in `TrialMatrix(pnl, names=[...])`.

## Diagnostics

```python
from phacker.stats import n_eff, logit_fit, sharpe_se

n_eff(M)            # effective number of independent trials
logit_fit(logits)   # (KS distance from the logistic null, mean logit)
sharpe_se(returns)  # delta-method standard error of each column's Sharpe ratio
```

- **`n_eff`**: the participation ratio of the centred trial matrix's singular values.
  A grid of 1,632 configurations that mostly trade the same signal can have an
  `n_eff` near 1. High PBO on a family with a small `n_eff` may say more about
  redundancy than about overfitting.
- **`logit_fit`** compares the logits to the exact logistic null. It's descriptive
  only: the C(S, S/2) splits share blocks, so the logits are dependent and a
  classical KS p-value would be wrong. None is reported.
- **`sharpe_se`** is √((1 − γ₃·SR + (γ₄ − 1)/4·SR²) / (T − 1)), with skewness γ₃
  and non-excess kurtosis γ₄. Use it to check that Sharpe estimates on
  T/2 rows are reliable enough for your choice of S.

## Choosing S

`S` is the number of blocks the rows are split into. It must be even and at most 32.
CSCV evaluates C(S, S/2) splits, each with T/2 rows on each side:

| S | splits |
|---|---|
| 8 | 70 |
| 12 | 924 |
| 16 | 12,870 |
| 20 | 184,756 |

The paper uses S = 16. A larger S gives more splits but shorter blocks. If T isn't a
multiple of S, the last `T % S` rows are dropped.

## Custom metrics

`cscv` accepts any performance function that maps a `(rows, N)` array to N scores.
Rows arrive in their original time order, so path-dependent metrics such as drawdown
work too:

```python
pbo, logits = cscv(M, S=16, metric=lambda x: x.mean(axis=0))
```

The C++ kernel computes the Sharpe ratio only. Any other metric runs the NumPy
implementation (also available directly as `phacker.cscv.cscv_reference`), which is
much slower at large N.

## Performance

With the Sharpe metric, the kernel sums each block once and then reuses those sums.
Neighbouring splits are visited in revolving-door (Gray code) order, so each split
needs one block added and one removed, and the out-of-sample sums are the totals minus
the in-sample ones. Splits are processed in parallel with OpenMP.

N = 1,000 strategies, T = 2,000 rows, S = 16, Intel i7-1365U (10 cores), Windows:

| backend | time |
|---|---|
| NumPy reference | ~250 s |
| C++ kernel, 1 thread | ~0.3 s |
| C++ kernel, OpenMP | ~0.05–0.07 s |

macOS wheels are built without OpenMP and run single-threaded.

## Behaviour details

- **Flat columns.** A configuration with no nonzero returns in a half (it never traded
  there) gets a Sharpe of 0 in that half rather than NaN.
- **Ties.** The out-of-sample rank is 1 + the number of strategies strictly worse,
  so ties rank low.
- **First winner.** If several strategies share the best in-sample score, the first
  column wins.
- Both backends follow these rules and agree to within 1e-12 on the logits.

## Limitations

1. **PBO measures the selection process, not a strategy.** A family where every
   configuration is genuinely good and roughly equal also gives a high PBO (paper
   §5.2). `n_eff` helps tell these cases apart but doesn't settle it.
2. **The answer is only as honest as the trial list.** Omitting configurations you
   tried biases PBO down, and so does padding the family with configurations
   designed to fail.
3. **Never optimize for low PBO.** Using CSCV as an objective function overfits PBO
   itself.
4. **The splits are dependent**, so there's no calibrated goodness-of-fit p-value on
   the logits.
5. **Regimes outside the sample are invisible.** PBO can't account for structural
   breaks it never saw.
6. **Order-dependent metrics** (max drawdown, Calmar) have no additive block
   statistics, so they run on the slow NumPy path only.

## Development

```bash
git clone https://github.com/ryousuf569/phacker.git
cd phacker
pip install -e ".[test]"   # builds the C++ kernel; needs a C++20 compiler
pytest
```

`pytest` without installing runs against the source tree on the NumPy backend, and
the C++ parity tests are skipped. On Windows with MinGW on `PATH`, set
`CMAKE_GENERATOR="Visual Studio 17 2022"` so the build uses MSVC.

## License

MIT. See [LICENSE](LICENSE).
