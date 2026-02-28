"""
SHIELD — rainfall.py
Data-driven Seasonal Climatological Rainfall Forecaster.

Replaces the random np.random.normal() calls in all previous scripts.
Instead of "pick a random number based on whether it's monsoon",
this fits a per-month distribution from the actual historical data
and samples from that — making predictions reproducible and data-grounded.

Usage
-----
    from shield.rainfall import SeasonalRainfallModel
    model = SeasonalRainfallModel()
    model.fit(df["date"], df["rainfall_mm"])      # call once during training
    model.save("saved_models/monthly_rain.pkl")   # persist alongside ML models

    # During prediction
    model = SeasonalRainfallModel.load("saved_models/monthly_rain.pkl")
    rain_day5 = model.predict(month=7, seed=42)   # deterministic with seed
"""

import joblib
import logging
import numpy as np
import pandas as pd
from typing import Optional

log = logging.getLogger(__name__)

# Absolute upper cap on sampled rainfall (physical maximum for India)
MAX_RAIN_MM = 200.0


class SeasonalRainfallModel:
    """
    Per-month Gamma-distribution rainfall model.

    Gamma is the standard meteorological choice for daily rainfall
    because it is bounded below by 0 and right-skewed (rare heavy events).

    If a month has too few samples (< 5 days), it falls back to a
    simple mean ± std Gaussian (clipped to 0).
    """

    def __init__(self):
        # Dict: month_int -> {"mean": float, "std": float, "max": float,
        #                      "alpha": float, "beta": float, "n": int}
        self._stats: dict = {}
        self._global_mean: float = 5.0
        self._global_std:  float = 10.0
        self._global_max:  float = MAX_RAIN_MM
        self._fitted: bool = False

    # ─────────────────────────────────────────────
    # Fitting
    # ─────────────────────────────────────────────

    def fit(self, dates: pd.Series, rainfall: pd.Series) -> "SeasonalRainfallModel":
        """
        Fit the per-month model from historical data.

        Parameters
        ----------
        dates    : Series of datetime values
        rainfall : Series of daily rainfall (mm), aligned with dates
        """
        df = pd.DataFrame({"month": pd.DatetimeIndex(dates).month, "rain": rainfall.values})
        df["rain"] = df["rain"].clip(lower=0.0)

        self._global_mean = float(df["rain"].mean()) if len(df) > 0 else 5.0
        self._global_std  = float(df["rain"].std())  if len(df) > 0 else 10.0
        self._global_max  = min(float(df["rain"].max()) * 1.2, MAX_RAIN_MM)

        for month in range(1, 13):
            subset = df.loc[df["month"] == month, "rain"].dropna()
            n = len(subset)

            if n >= 10:
                mean_r = float(subset.mean())
                std_r  = float(subset.std())
                max_r  = min(float(subset.max()) * 1.3, MAX_RAIN_MM)

                # Gamma parameters: shape alpha = (mean/std)^2, scale beta = std^2/mean
                if mean_r > 0.01 and std_r > 0.01:
                    alpha = (mean_r / std_r) ** 2
                    beta  = std_r ** 2 / mean_r
                else:
                    alpha, beta = None, None

                self._stats[month] = {
                    "mean": mean_r, "std": std_r, "max": max_r,
                    "alpha": alpha, "beta": beta, "n": n,
                }
            elif n >= 3:
                # Insufficient for Gamma — fall back to Gaussian
                self._stats[month] = {
                    "mean": float(subset.mean()),
                    "std":  float(subset.std()) if n > 1 else 5.0,
                    "max":  min(float(subset.max()) * 1.3, MAX_RAIN_MM),
                    "alpha": None, "beta": None, "n": n,
                }
            else:
                # No data for this month — will use global stats
                self._stats[month] = None

        self._fitted = True
        month_coverage = sum(1 for v in self._stats.values() if v is not None)
        log.info(
            f"SeasonalRainfallModel fitted: {month_coverage}/12 months have data, "
            f"global mean={self._global_mean:.1f}mm, max={self._global_max:.1f}mm"
        )
        return self

    # ─────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────

    def predict(
        self,
        month: int,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        Sample a single day's rainfall for the given month.

        Parameters
        ----------
        month : Calendar month (1–12)
        seed  : Optional int seed for reproducibility. Ignored if rng is provided.
        rng   : Optional pre-seeded numpy random Generator (preferred for sequences).

        Returns
        -------
        float: Predicted rainfall in mm (>= 0)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        if rng is None:
            rng = np.random.default_rng(seed)

        stats = self._stats.get(month)

        if stats is None:
            # No monthly data — use global Gaussian fallback
            val = rng.normal(self._global_mean, self._global_std)
        elif stats["alpha"] is not None:
            # Sample from Gamma distribution
            val = rng.gamma(shape=stats["alpha"], scale=stats["beta"])
        else:
            # Gaussian fallback for this month
            val = rng.normal(stats["mean"], stats["std"])

        max_r = (stats["max"] if stats else self._global_max)
        return float(np.clip(val, 0.0, max_r))

    def predict_sequence(
        self,
        months: list,
        seed: int = 42,
    ) -> list:
        """
        Generate a reproducible sequence of rainfall values (one per day).
        Uses a single seeded RNG so the entire sequence is deterministic.

        Parameters
        ----------
        months : List of calendar month ints (one per future day)
        seed   : Random seed — same seed → same sequence every run

        Returns
        -------
        List of float rainfall values (mm), one per element in months
        """
        rng = np.random.default_rng(seed)
        return [self.predict(m, rng=rng) for m in months]

    # ─────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        log.info(f"SeasonalRainfallModel saved to {path}")

    @staticmethod
    def load(path: str) -> "SeasonalRainfallModel":
        model = joblib.load(path)
        log.info(f"SeasonalRainfallModel loaded from {path}")
        return model

    # ─────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of per-month statistics."""
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        lines = ["Month       Mean(mm)   Std(mm)   Max(mm)   N     Dist"]
        lines.append("─" * 60)
        for i, name in enumerate(month_names, start=1):
            s = self._stats.get(i)
            if s:
                dist = "Gamma" if s["alpha"] else "Gauss"
                lines.append(
                    f"{name:<10}  {s['mean']:>6.1f}    {s['std']:>5.1f}    "
                    f"{s['max']:>6.1f}    {s['n']:>4}   {dist}"
                )
            else:
                lines.append(f"{name:<10}  (no data — using global fallback)")
        return "\n".join(lines)
