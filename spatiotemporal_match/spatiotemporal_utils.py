# === spatiotemporal_utils.py ===============================================
# Tools to (1) aggregate pixels over a plant polygon with robust stats,
# (2) analyze dataset cadence vs forecast horizon,
# (3) derive latency & gap-aware provenance features,
# (4) enforce per-dataset matching/interp policy.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Literal, Tuple
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import shape, Polygon, MultiPolygon
    import rioxarray as rxr
    _HAS_GEO = True
except Exception:
    _HAS_GEO = False


# 1) Polygon aggregation helpers

def _iqr(a: np.ndarray) -> float:
    q75, q25 = np.nanpercentile(a, [75, 25])
    return float(q75 - q25)

def robust_stats(values: np.ndarray) -> Dict[str, float]:
    """Median + IQR + valid fraction; ignores NaNs."""
    arr = np.asarray(values).astype(float)
    m = float(np.nanmedian(arr)) if np.isfinite(arr).any() else np.nan
    spread = _iqr(arr) if np.isfinite(arr).any() else np.nan
    valid_frac = float(np.isfinite(arr).mean()) if arr.size else 0.0
    return {"median": m, "iqr": spread, "valid_frac": valid_frac}

def aggregate_raster_over_polygon(
    raster_path: str,
    polygon_gdf: "gpd.GeoDataFrame",
    band: int = 1
) -> Dict[str, float]:
    """
    Robustly summarize one raster band over the plant polygon.
    Returns dict with median/iqr/valid_frac.
    """
    if not _HAS_GEO:
        raise ImportError("Install geopandas/shapely/rioxarray for local raster aggregation.")
    # Load raster as DataArray with CRS
    da = rxr.open_rasterio(raster_path).sel(band=band)
    da = da.squeeze(drop=True)
    # Reproject polygon to raster CRS
    pg = polygon_gdf.to_crs(da.rio.crs)
    clipped = da.rio.clip(pg.geometry, all_touched=True, drop=False)
    vals = clipped.values.ravel()
    return robust_stats(vals)


# 2) Cadence analysis vs forecast horizon

@dataclass
class CadenceReport:
    revisit_hours_p50: float
    revisit_hours_p90: float
    long_gap_hours_p95: float
    n_obs: int
    recommendation: Literal["continuous","anchor_only","exclude"]

def analyze_cadence(
    df_obs: pd.DataFrame,
    time_col: str = "time",
    forecast_horizon_hours: int = 24
) -> CadenceReport:
    """Quantify typical revisit interval & suggest how to use the dataset."""
    t = pd.to_datetime(df_obs[time_col]).sort_values().unique()
    if len(t) < 2:
        return CadenceReport(np.inf, np.inf, np.inf, len(t), "exclude")
    gaps = np.diff(t) / np.timedelta64(1, "h")
    p50, p90, p95 = np.percentile(gaps, [50, 90, 95])
    n_obs = len(t)

    # Heuristic: for day-ahead forecasting, we like ≤ 24h cadence.
    if p90 <= forecast_horizon_hours:
        rec = "continuous"
    elif p90 <= 96:  # sparse but still useful as anchors within a few days
        rec = "anchor_only"
    else:
        rec = "exclude"
    return CadenceReport(p50, p90, p95, n_obs, rec)


# 3) Latency & provenance (gap-aware) features

def add_provenance_features(
    df_grid: pd.DataFrame,            # hourly reference (ERA5 times)
    df_obs: pd.DataFrame,             # sparse observations with 'time'
    time_col: str = "time",
    obs_cols: Sequence[str] = (),
    valid_frac_col: Optional[str] = None,
    max_hold_hours: int = 24,
) -> pd.DataFrame:
    """
    On the hourly grid, add:
      - is_anchor (1 if an actual observation hits that hour)
      - time_since_last_obs (hours)
      - gap_to_next_obs (hours)
      - obs_age_bucket (categorical)
      - valid_frac (if provided)
      - last_value_hold for <= max_hold_hours (per obs_col)
    """
    g = df_grid.copy()
    g[time_col] = pd.to_datetime(g[time_col])
    g = g.sort_values(time_col).reset_index(drop=True)

    o = df_obs.copy()
    o[time_col] = pd.to_datetime(o[time_col])
    o = o.sort_values(time_col).drop_duplicates(subset=[time_col])

    # Anchor flag on exact hours (assumes obs already matched/rounded if needed)
    anchors = pd.Series(0, index=g.index)
    anchor_idx = g[time_col].isin(o[time_col]).values
    anchors[anchor_idx] = 1
    g["is_anchor"] = anchors.values

    # Time since last & to next obs
    # (merge_asof gives nearest left/right matches)
    left = pd.merge_asof(
        g[[time_col]], o[[time_col]], on=time_col, direction="backward"
    ).rename(columns={time_col: "last_obs_time"})
    right = pd.merge_asof(
        g[[time_col]], o[[time_col]], on=time_col, direction="forward"
    ).rename(columns={time_col: "next_obs_time"})

    g["time_since_last_obs"] = (g[time_col] - left["last_obs_time"]).dt.total_seconds() / 3600.0
    g["gap_to_next_obs"] = (right["next_obs_time"] - g[time_col]).dt.total_seconds() / 3600.0

    g.loc[left["last_obs_time"].isna(), "time_since_last_obs"] = np.nan
    g.loc[right["next_obs_time"].isna(), "gap_to_next_obs"] = np.nan

    # Age buckets (for the model to down-weight stale values)
    bins = [-np.inf, 3, 12, 48, np.inf]
    labels = ["0-3h", "3-12h", "12-48h", "48h+"]
    g["obs_age_bucket"] = pd.cut(g["time_since_last_obs"], bins=bins, labels=labels)

    # Last-value hold for a limited time window (≤ max_hold_hours)
    if obs_cols:
        # Left-join the last observed values
        o_indexed = o.set_index(time_col).sort_index()
        g_indexed = g.set_index(time_col).sort_index()
        # Build forward fill from obs into the grid
        hold = g_indexed[[]]
        hold = hold.join(o_indexed[list(obs_cols)], how="left")
        hold = hold.ffill()
        # Mask out holds older than max_hold_hours
        too_old = g_indexed["time_since_last_obs"] > max_hold_hours
        for c in obs_cols:
            g_indexed[f"{c}__hold"] = hold[c]
            g_indexed.loc[too_old, f"{c}__hold"] = np.nan
        g = g_indexed.reset_index()

    # Carry valid fraction if provided (nearest backward)
    if valid_frac_col and valid_frac_col in o.columns:
        vf = pd.merge_asof(
            g[[time_col]], o[[time_col, valid_frac_col]], on=time_col, direction="backward"
        )[valid_frac_col]
        g["valid_frac"] = vf.values

    return g


# 4) Gap-aware matching policy for continuous vs. anchors

@dataclass
class MatchingPolicy:
    # Max allowable gap (hours) for methods before we leave NaN
    max_gap_linear: int = 96
    max_gap_spline: int = 48
    max_hold_nearest: int = 24
    allow_edge_fill_hours: int = 12  # ffill/bfill cap

def enforce_gap_caps(
    hourly_df: pd.DataFrame,
    value_cols: Sequence[str],
    policy: MatchingPolicy,
    is_interpolated: bool = True
) -> pd.DataFrame:
    """
    Apply caps so interpolations don’t silently bridge long gaps.
    Requires `time_since_last_obs` and `gap_to_next_obs` columns (from add_provenance_features).
    """
    out = hourly_df.copy()
    if "time_since_last_obs" not in out or "gap_to_next_obs" not in out:
        return out  # nothing to enforce

    # Gaps larger than allowed → set to NaN
    if is_interpolated:
        # Decide per timestamp whether the local gap is too large for spline/linear
        local_gap = out[["time_since_last_obs", "gap_to_next_obs"]].min(axis=1)
        # If you know which method was used, choose threshold; here we use the stricter (spline) threshold for safety
        too_large = local_gap > policy.max_gap_linear
        out.loc[too_large, list(value_cols)] = np.nan

        # Edge fill cap
        too_far_edge = out["time_since_last_obs"] > policy.allow_edge_fill_hours
        out.loc[too_far_edge, list(value_cols)] = np.nan

    # For nearest/holds, already masked via __hold in add_provenance_features
    return out