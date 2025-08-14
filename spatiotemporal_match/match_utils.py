from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence

import numpy as np
import pandas as pd

InterpolationMethod = Literal["nearest", "linear", "spline"]


@dataclass
class MatchSpec:
    """
    Configuration describing how to bring a dataset onto an hourly grid.

    Attributes
    ----------
    name : str
        Key used to look up the dataset in a dict (e.g., "S2_CSPlus").
    time_col : str
        Name of the timestamp column in the source dataframe.
    value_cols : Sequence[str]
        Columns to transfer/interpolate onto the hourly grid.
    prefix : Optional[str]
        Prefix for output columns; defaults to `name` if None.
    how : InterpolationMethod
        "nearest" (use match_nearest), "linear", or "spline" (use match_interpolate).
    spline_order : int
        Order for spline interpolation (only if how == "spline", default=3).
    limit_direction : Literal["forward", "backward", "both"]
        Direction for filling gaps during interpolation; passed to pandas interpolate.
    edge_fill : bool
        If True, forward/backward-fill edges after interpolation to avoid NaNs at ends.
    """
    name: str
    time_col: str
    value_cols: Sequence[str]
    prefix: Optional[str] = None
    how: InterpolationMethod = "linear"
    spline_order: int = 3
    limit_direction: Literal["forward", "backward", "both"] = "both"
    edge_fill: bool = True


def _ensure_time_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Ensure datetime index sorted ascending on `time_col`."""
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], utc=False)  # keep naive to match ERA5 if naive
    return df.set_index(time_col).sort_index()


def find_nearest_hour(ts: pd.Timestamp, hourly_index: pd.DatetimeIndex) -> pd.Timestamp:
    """
    Return the closest timestamp in `hourly_index` to `ts` (tie breaks to the earlier).
    Assumes `hourly_index` is sorted and represents an hourly grid.
    """
    pos = np.searchsorted(hourly_index.values, ts.to_datetime64())
    if pos == 0:
        return hourly_index[0]
    if pos == len(hourly_index):
        return hourly_index[-1]
    before = hourly_index[pos - 1]
    after = hourly_index[pos]
    return before if abs(ts - before) <= abs(after - ts) else after


def match_nearest(
    df_grid: pd.DataFrame,
    df_obs: pd.DataFrame,
    spec: MatchSpec,
    grid_time_col: str = "time",
) -> pd.DataFrame:
    """
    Nearest-hour join of `df_obs[spec.value_cols]` onto `df_grid`.

    Parameters
    ----------
    df_grid : DataFrame
        Hourly grid (e.g., ERA5) with a time column (default "time").
    df_obs : DataFrame
        Sparse observations with `spec.time_col` and `spec.value_cols`.
    spec : MatchSpec
        Dataset configuration; `how` is ignored here.
    grid_time_col : str
        Name of the time column in df_grid.

    Returns
    -------
    DataFrame
        `df_grid` with new columns prefixed by `spec.prefix or spec.name`.
    """
    g = _ensure_time_index(df_grid, grid_time_col)
    o = _ensure_time_index(df_obs, spec.time_col)

    # Map each observation timestamp to its nearest grid timestamp
    nearest = o.index.to_series().apply(lambda t: find_nearest_hour(t, g.index))
    tmp = pd.DataFrame({"matched_time": nearest.values}, index=o.index)

    cols = list(spec.value_cols)
    tmp = tmp.join(o[cols], how="left")
    # If multiple obs map to the same hour, average them
    tmp = tmp.groupby("matched_time")[cols].mean()

    out = g.join(tmp, how="left")
    prefix = spec.prefix or spec.name
    out = out.rename(columns={c: f"{prefix}__{c}" for c in cols}).reset_index()
    return out


def _interpolate_on_union(
    hourly_idx: pd.DatetimeIndex,
    obs_df_idxed: pd.DataFrame,
    cols: Sequence[str],
    method: InterpolationMethod,
    spline_order: int,
    limit_direction: Literal["forward", "backward", "both"],
    edge_fill: bool,
) -> pd.DataFrame:
    """
    Reindex to union(hourly_idx, obs_idx), interpolate on the union,
    then slice back to hourly_idx. Prevents distortion from interpolating
    only on the hourly grid without anchoring at actual obs times.
    """
    union_idx = hourly_idx.union(obs_df_idxed.index)
    arr = obs_df_idxed[cols].reindex(union_idx)

    if method == "linear":
        arr = arr.interpolate(method="time", limit_direction=limit_direction)
    elif method == "spline":
      # Ensure we have enough points: need at least order+1
      n_valid = arr.dropna(how="all").shape[0]
      order_to_use = min(spline_order, max(1, n_valid - 1))
      if order_to_use < 2:
          # too few points for any spline, fallback to linear
          arr = arr.interpolate(method="time", limit_direction=limit_direction)
      else:
          try:
              arr = arr.interpolate(method="spline", order=order_to_use,
                                    limit_direction=limit_direction)
          except Exception:
              # fallback if scipy or method fails
              arr = arr.interpolate(method="time", limit_direction=limit_direction)
    else:
        raise ValueError("Use match_nearest() for 'nearest'.")

    if edge_fill:
        arr = arr.ffill().bfill()

    return arr.reindex(hourly_idx)


def match_interpolate(
    df_grid: pd.DataFrame,
    df_obs: pd.DataFrame,
    spec: MatchSpec,
    grid_time_col: str = "time",
) -> pd.DataFrame:
    """
    Interpolate `df_obs[spec.value_cols]` onto hourly stamps in `df_grid`.

    Supports `how="linear"` (time-based) and `how="spline"` (cubic by default).

    Notes
    -----
    - Rows in `df_obs` that are all-NaN across `value_cols` are dropped before interpolation.
    - After interpolation, leading/trailing NaNs are filled if `edge_fill=True`.
    """
    if spec.how not in ("linear", "spline"):
        raise ValueError("match_interpolate supports only 'linear' or 'spline'. Use match_nearest for 'nearest'.")

    g = _ensure_time_index(df_grid, grid_time_col)
    o = _ensure_time_index(df_obs, spec.time_col)

    cols = list(spec.value_cols)
    o_valid = o[cols].dropna(how="all")

    interp = _interpolate_on_union(
        hourly_idx=g.index,
        obs_df_idxed=o_valid,
        cols=cols,
        method=spec.how,
        spline_order=spec.spline_order,
        limit_direction=spec.limit_direction,
        edge_fill=spec.edge_fill,
    )

    out = g.join(interp, how="left")
    prefix = spec.prefix or spec.name
    out = out.rename(columns={c: f"{prefix}__{c}" for c in cols}).reset_index()
    return out


def align_many(
    df_grid: pd.DataFrame,
    datasets: Dict[str, pd.DataFrame],
    specs: Sequence[MatchSpec],
    grid_time_col: str = "time",
) -> Dict[str, pd.DataFrame]:
    """
    Convenience wrapper to produce (nearest, interpolated) outputs for each dataset.

    Parameters
    ----------
    df_grid : DataFrame
        Hourly grid with a `grid_time_col`.
    datasets : Dict[str, DataFrame]
        Map from `MatchSpec.name` to its source DataFrame.
    specs : Sequence[MatchSpec]
        One spec per dataset.
    grid_time_col : str
        Name of the time column in df_grid.

    Returns
    -------
    Dict[str, DataFrame]
        For each spec.name, returns two keys:
            "{name}__nearest"  -> nearest-hour match (always produced)
            "{name}__interp"   -> interpolated result (only for how in {"linear","spline"})
    """
    out: Dict[str, pd.DataFrame] = {}
    for spec in specs:
        if spec.name not in datasets:
            raise KeyError(f"Dataset '{spec.name}' not found in `datasets` dict.")
        df_obs = datasets[spec.name]

        # Always provide a nearest-hour companion for inspection
        near = match_nearest(df_grid, df_obs, spec, grid_time_col=grid_time_col)
        out[f"{spec.name}__nearest"] = near

        if spec.how in ("linear", "spline"):
            interp = match_interpolate(df_grid, df_obs, spec, grid_time_col=grid_time_col)
            out[f"{spec.name}__interp"] = interp

    return out