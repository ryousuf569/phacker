from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

Side = Literal["long", "short", "both"]
FloatArray = npt.NDArray[np.float64]

_SIDES: dict[str, tuple[int, ...]] = {"long": (1,), "short": (-1,), "both": (1, -1)}


def linspace(start, stop, num):
    # rounded so grid points like 0.25 print and compare cleanly
    return np.round(np.linspace(start, stop, num), 12)


class Config(NamedTuple):
    indicator: str
    entry: float
    band: float
    hold: int
    side: int

    @property
    def name(self) -> str:
        side = "long" if self.side > 0 else "short"
        return f"{self.indicator}:entry={self.entry:g}:band={self.band:g}:hold={self.hold}:{side}"


@dataclass(frozen=True)
class Grid:
    indicators: Sequence[str]
    entry: Sequence[float] | FloatArray
    band: Sequence[float] | FloatArray
    hold: Sequence[int]
    side: Side = "both"
    cost_bps: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "indicators", tuple(str(i) for i in self.indicators))
        object.__setattr__(self, "entry", tuple(float(e) for e in self.entry))
        object.__setattr__(self, "band", tuple(float(b) for b in self.band))
        object.__setattr__(self, "hold", tuple(int(h) for h in self.hold))
        if not (self.indicators and self.entry and self.band and self.hold):
            raise ValueError("every grid axis needs at least one value")
        if len(set(self.indicators)) != len(self.indicators):
            raise ValueError("indicators must be unique")
        if any(b <= 0 for b in self.band):
            raise ValueError("band widths must be positive")
        if any(h < 1 for h in self.hold):
            raise ValueError("hold periods must be at least 1 bar")
        if self.side not in _SIDES:
            raise ValueError(f"side must be one of {sorted(_SIDES)}, got {self.side!r}")
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")

    @property
    def sides(self):
        return _SIDES[self.side]

    @property
    def shape(self):
        return (len(self.indicators), len(self.entry), len(self.band), len(self.hold), len(self.sides))

    def __len__(self):
        return int(np.prod(self.shape))

    def __iter__(self):
        # order matches the column order produced by backtest
        for ind, e, b, h, s in product(self.indicators, self.entry, self.band, self.hold, self.sides):
            yield Config(ind, e, b, h, s)

    def configs(self):
        return list(self)


@dataclass
class TrialMatrix:
    pnl: FloatArray
    names: list[str] = field(default_factory=list)
    index: pd.Index | None = None

    def __post_init__(self):
        self.pnl = np.ascontiguousarray(self.pnl, dtype=np.float64)
        if self.pnl.ndim != 2:
            raise ValueError(f"pnl must be 2-D (T x N), got shape {self.pnl.shape}")
        if not self.names:
            self.names = [f"trial_{n}" for n in range(self.pnl.shape[1])]
        if len(self.names) != self.pnl.shape[1]:
            raise ValueError(f"got {len(self.names)} names for {self.pnl.shape[1]} columns")
        if self.index is not None and len(self.index) != self.pnl.shape[0]:
            raise ValueError(f"index length {len(self.index)} does not match T={self.pnl.shape[0]}")
        if not np.all(np.isfinite(self.pnl)):
            raise ValueError("pnl contains NaN or inf")

    @property
    def T(self):
        return int(self.pnl.shape[0])

    @property
    def N(self):
        return int(self.pnl.shape[1])

    @property
    def shape(self):
        return (self.T, self.N)

    def __array__(self, dtype: npt.DTypeLike = None, copy: bool | None = None):
        return self.pnl if dtype is None else self.pnl.astype(dtype)

    def to_frame(self):
        return pd.DataFrame(self.pnl, index=self.index, columns=self.names)


def expanding_zscore(x, min_periods):
    # stats at t use only rows 0..t, so there is no lookahead
    if min_periods < 2:
        raise ValueError("min_periods must be at least 2")
    ex = x.astype(np.float64).expanding(min_periods=min_periods)
    sd = ex.std(ddof=1)
    return (x - ex.mean()) / sd.where(sd > 0)


def _held(signal, hold):
    # position is on if any trigger fired in the last `hold` bars, via a windowed cumsum
    T = signal.shape[0]
    cs = np.zeros((T + hold,) + signal.shape[1:], dtype=np.int32)
    np.cumsum(signal, axis=0, out=cs[hold:])
    return (cs[hold:] - cs[:T]) > 0


def backtest(prices, grid, price_col, min_periods):

    missing = [c for c in (price_col, *grid.indicators) if c not in prices.columns]
    if missing:
        raise KeyError(f"prices is missing columns: {missing}")

    px = prices[price_col].to_numpy(dtype=np.float64)
    if px.shape[0] < 3:
        raise ValueError("need at least 3 price rows")
    if not np.all(np.isfinite(px)) or np.any(px <= 0):
        raise ValueError(f"{price_col} must be finite and positive")

    T = px.shape[0] - 1
    r = np.diff(np.log(px))
    cost = grid.cost_bps * 1e-4
    entry = np.asarray(grid.entry)
    band = np.asarray(grid.band)
    I, K, B, H, S = grid.shape

    out = np.empty((T, I, K, B, H, S), dtype=np.float64)

    for i, ind in enumerate(grid.indicators):
        z = expanding_zscore(prices[ind], min_periods).to_numpy(dtype=np.float64)
        # NaN z (warmup or zero spread) compares False, so those rows stay flat
        with np.errstate(invalid="ignore"):
            signal = np.abs(z[:, None, None] - entry[None, :, None]) <= band[None, None, :]

        for j, h in enumerate(grid.hold):
            pos = _held(signal, h).astype(np.float64)
            # pnl_t = pos_{t-1} * r_t, trading cost charged on the bar the position changes
            gross = pos[:-1] * r[:, None, None]
            turn = cost * np.abs(np.diff(pos, axis=0))
            for k, s in enumerate(grid.sides):
                out[:, i, :, :, j, k] = s * gross - turn

    index = prices.index[1:] if not isinstance(prices.index, pd.RangeIndex) else None
    return TrialMatrix(out.reshape(T, -1), names=[c.name for c in grid], index=index)
