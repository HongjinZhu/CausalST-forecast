from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import warnings

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller
    _HAS_SM = True
except Exception:
    _HAS_SM = False
    grangercausalitytests = None
    adfuller = None


# basics

def _ensure_time_index(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    if time_col not in df.columns:
        raise KeyError(f"Expected '{time_col}' in dataframe.")
    g = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(g[time_col]):
        g[time_col] = pd.to_datetime(g[time_col], utc=False)
    return g.set_index(time_col).sort_index()


def corr_matrix(
    df: pd.DataFrame,
    cols: Sequence[str],
    time_col: str = "time",
    method: str = "pearson",
    min_periods: int = 3
) -> pd.DataFrame:
    """
    Pairwise correlation matrix over given columns.
    """
    X = _ensure_time_index(df, time_col=time_col)[list(cols)]
    return X.corr(method=method, min_periods=min_periods)


def rolling_corr_series(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    time_col: str = "time",
    window: str = "168H"  # 1 week at hourly cadence
) -> pd.Series:
    """
    Rolling correlation between two series (time-based window, e.g., '168H').
    """
    X = _ensure_time_index(df, time_col=time_col)[[x_col, y_col]].dropna()
    return X[x_col].rolling(window).corr(X[y_col])


# ADF + differencing utilities

@dataclass
class StationarityCheck:
    is_stationary: bool
    pvalue: float
    usedlag: Optional[int]


def adf_check(series: pd.Series, alpha: float = 0.05, autolag: str = "AIC") -> StationarityCheck:
    if not _HAS_SM:
        # If statsmodels not available, just say "unknown" -> treat as non-stationary
        return StationarityCheck(is_stationary=False, pvalue=np.nan, usedlag=None)
    s = series.dropna()
    if len(s) < 8:
        # too short for meaningful ADF
        return StationarityCheck(is_stationary=False, pvalue=np.nan, usedlag=None)
    try:
        res = adfuller(s.values, autolag=autolag)
        pval = float(res[1])
        usedlag = int(res[2])
        return StationarityCheck(is_stationary=(pval < alpha), pvalue=pval, usedlag=usedlag)
    except Exception:
        return StationarityCheck(is_stationary=False, pvalue=np.nan, usedlag=None)


def maybe_difference(series: pd.Series, do_diff: bool) -> pd.Series:
    return series.diff() if do_diff else series


# Granger wrapper

@dataclass
class GrangerResult:
    pair: Tuple[str, str]
    maxlag: int
    pvalues_by_lag: Dict[int, float]   # {lag: pvalue (SSRF-test)}
    significant_lags: List[int]        # lags where p < alpha
    best_lag: Optional[int]            # lag with min pvalue (if any)
    adf_x: StationarityCheck
    adf_y: StationarityCheck
    differenced: bool                  # whether we applied 1st diff to both series


def granger_pair(
    df: pd.DataFrame,
    cause_col: str,
    effect_col: str,
    time_col: str = "time",
    maxlag: int = 6,
    alpha: float = 0.05,
    force_diff: Optional[bool] = None,
    dropna: str = "any",
    verbose: bool = False,
) -> GrangerResult:
    """
    Test whether `cause_col` Granger-causes `effect_col` up to `maxlag`.
    Returns p-values by lag from the SSR F-test (a common choice).
    """
    if not _HAS_SM:
        raise ImportError("statsmodels is required for Granger causality. Try: pip install statsmodels")

    X = _ensure_time_index(df, time_col=time_col)[[cause_col, effect_col]].dropna(how=dropna)
    X = X.rename(columns={cause_col: "cause", effect_col: "effect"})

    # Stationarity checks
    adf_c = adf_check(X["cause"])
    adf_e = adf_check(X["effect"])

    # Decide differencing
    if force_diff is None:
        do_diff = not (adf_c.is_stationary and adf_e.is_stationary)
    else:
        do_diff = bool(force_diff)

    Y = pd.DataFrame({
        "effect": maybe_difference(X["effect"], do_diff),
        "cause":  maybe_difference(X["cause"], do_diff),
    }).dropna()

    if len(Y) < (maxlag + 5):
        # Too short for requested maxlag; reduce
        maxlag = max(1, min(maxlag, max(1, len(Y) // 3)))
        if verbose:
            warnings.warn(f"Series too short; reducing maxlag to {maxlag}")

    # statsmodels expects a 2-col array with order: [effect, cause]
    try:
        res = grangercausalitytests(Y[["effect", "cause"]].values, maxlag=maxlag, verbose=False)
    except Exception as e:
        # If failure, try one more time with smaller maxlag
        if maxlag > 2:
            maxlag2 = 2
            if verbose:
                warnings.warn(f"Granger test failed for maxlag={maxlag}. Retrying with maxlag={maxlag2}. Error: {e}")
            res = grangercausalitytests(Y[["effect", "cause"]].values, maxlag=maxlag2, verbose=False)
            maxlag = maxlag2
        else:
            raise

    pvals = {}
    for lag, out in res.items():
        # 'ssr_ftest' is common: (F-stat, pvalue, df_denom, df_num)
        ssr_ftest = out[0].get("ssr_ftest", None)
        if ssr_ftest is None:
            continue
        pvals[lag] = float(ssr_ftest[1])

    sig = [lag for lag, p in pvals.items() if p < alpha]
    best = min(pvals, key=pvals.get) if pvals else None

    return GrangerResult(
        pair=(cause_col, effect_col),
        maxlag=maxlag,
        pvalues_by_lag=pvals,
        significant_lags=sorted(sig),
        best_lag=best,
        adf_x=adf_c,
        adf_y=adf_e,
        differenced=do_diff
    )


def granger_batch(
    df: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]],
    time_col: str = "time",
    maxlag: int = 6,
    alpha: float = 0.05,
    force_diff: Optional[bool] = None,
    dropna: str = "any",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run Granger tests over many (cause, effect) pairs.
    Returns a tidy DataFrame with one row per (pair, lag).
    """
    rows = []
    for cause, effect in pairs:
        try:
            g = granger_pair(
                df, cause_col=cause, effect_col=effect, time_col=time_col,
                maxlag=maxlag, alpha=alpha, force_diff=force_diff, dropna=dropna, verbose=verbose
            )
            for lag, p in g.pvalues_by_lag.items():
                rows.append({
                    "cause": cause,
                    "effect": effect,
                    "lag": lag,
                    "pvalue": p,
                    "significant": (p < alpha),
                    "best_lag_for_pair": (lag == g.best_lag),
                    "differenced": g.differenced,
                    "adf_cause_p": g.adf_x.pvalue,
                    "adf_effect_p": g.adf_y.pvalue,
                    "maxlag_used": g.maxlag,
                })
        except Exception as e:
            rows.append({
                "cause": cause, "effect": effect, "lag": np.nan, "pvalue": np.nan,
                "significant": False, "best_lag_for_pair": False, "differenced": np.nan,
                "adf_cause_p": np.nan, "adf_effect_p": np.nan, "maxlag_used": np.nan,
                "error": repr(e)
            })
    return pd.DataFrame(rows)


# small viz helpers

import matplotlib.pyplot as plt

def plot_granger_matrix(
    results_df: pd.DataFrame,
    cause: str,
    effect: str,
    title: Optional[str] = None
):
    """
    Plot p-value by lag as a bar chart (lower is stronger evidence).
    """
    D = results_df[(results_df["cause"] == cause) & (results_df["effect"] == effect)].dropna(subset=["lag", "pvalue"])
    if D.empty:
        warnings.warn("No results to plot for this pair.")
        return
    D = D.sort_values("lag")
    plt.figure(figsize=(6,3))
    plt.bar(D["lag"].astype(int), D["pvalue"])
    plt.axhline(0.05, linestyle="--")  # alpha
    plt.xlabel("Lag (hours)")
    plt.ylabel("p-value (SSR F-test)")
    plt.title(title or f"Granger p-values: {cause} → {effect}")
    plt.tight_layout()


def plot_rolling_corr(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    time_col: str = "time",
    window: str = "168H",
    title: Optional[str] = None
):
    rc = rolling_corr_series(df, x_col, y_col, time_col=time_col, window=window)
    plt.figure(figsize=(12,3))
    plt.plot(rc.index, rc.values)
    plt.axhline(0, linestyle="--")
    plt.title(title or f"Rolling corr ({window}): {x_col} vs {y_col}")
    plt.ylabel("corr"); plt.xlabel("time")
    plt.tight_layout()