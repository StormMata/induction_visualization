"""Ørsted mesoscale-profile loader with India-style output schema.

This module mirrors the user-facing pattern of ``matlab2python.py`` and
``orsted2python.py``:

    import induction_visualization.orstedmeso2python as om2p
    data = om2p.load_orsted_meso_data(root, filters=mask)

It implements the mesoscale protocol from ``orsted_mesoscale_workflow_v3.ipynb``
internally. It reads raw SCADA parquet files, raw ZX lidar CSV files, and the
mesoscale CSV file; builds the hybrid lidar-scaled mesoscale profiles; returns
an India-style dictionary suitable for the same profile/engineering plotting
codes.

Default protocol
----------------
- selected lidar: ZX_zxtm5052;
- SCADA and lidar are resampled to filters['avg_window'] minutes;
- avg_window must be no smaller than the coarsest native timestep of SCADA and
  selected lidar. Mesoscale fields are time-interpolated to that grid, following
  the notebook protocol;
- speed profile returned in ``speed_profiles`` is the hybrid profile:

      U(z,t) = U_lidar,hub(t) * U_meso(z,t) / U_meso,hub(t)

- direction profile returned in ``dir_profiles`` is the hybrid relative
  direction profile: mesoscale direction anomaly about hub height anchored to
  lidar hub-height relative direction;
- by default, hybrid speed, direction, and density profiles are linearly
  extrapolated down to the rotor bottom ``Hub - R`` using the lowest available
  mesoscale levels, so returned profiles span the lower rotor tip;
- power is kW from ActPower_Value_mean;
- turbine rotor speed is rpm from GenRpm_Value_mean;
- turbine TI is read from TurbEst_TurbEst_mean and converted from fraction to
  percent to match the India dataset convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import re

import numpy as np
import pandas as pd

try:
    from scipy.io import savemat
except Exception:  # pragma: no cover
    savemat = None


# -----------------------------------------------------------------------------
# Defaults and hard-coded Ørsted columns
# -----------------------------------------------------------------------------

DEFAULT_LIDAR_KEY = "ZX_zxtm5052"
DEFAULT_TURBINE_ID = "ROW01A04"

DEFAULT_DIAMETER = 154.0
DEFAULT_R = DEFAULT_DIAMETER / 2.0
DEFAULT_HUB = 103.3
DEFAULT_RHO = 1.225
DEFAULT_HUBR = np.nan
DEFAULT_B = 3

POWER_COL = "ActPower_Value_mean"              # kW
NACELLE_HEADING_COL = "NacelPos_Value_mean"    # deg, sparse heading values are averaged circularly
SCADA_WIND_DIR_COL = "AcWindDr_Value_mean"     # deg absolute
TURBINE_HUB_SPEED_COL = "AcWindSp_AcWindSp_mean"  # m/s
GENERATOR_RPM_COL = "GenRpm_Value_mean"        # rpm; direct-drive rotor speed
PITCH_COL = "PitcPosA_Value_mean"              # deg
TURBINE_TI_COL = "TurbEst_TurbEst_mean"        # fractional raw TI; converted to percent
AMBIENT_TEMP_COL = "AmbieTmp_Value_mean"       # C

DEFAULT_NBL_FILES = {
    "ZX_zxtm5005": {
        "filename": "RaceBank_A04_ZXNBL_zxtm5005.csv",
        "instrument": "ZX",
        "subdir_kind": "zx",
    },
    "ZX_zxtm5052": {
        "filename": "RaceBank_A04_ZXNBL_zxtm5052.csv",
        "instrument": "ZX",
        "subdir_kind": "zx",
    },
}

DEFAULT_LIDAR_DIRECTION_IS_ALREADY_RELATIVE = {
    "ZX_zxtm5005": True,
    "ZX_zxtm5052": True,
}


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def wrap_180(angle_deg):
    a = np.asarray(angle_deg, dtype=float)
    return (a + 180.0) % 360.0 - 180.0


def _to_tz_naive_datetime_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def _standardize_time_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = _to_tz_naive_datetime_index(pd.DatetimeIndex(out.index))
    out = out.sort_index()
    out.index.name = "Time"
    return out


def _infer_base_dt(index: pd.DatetimeIndex) -> pd.Timedelta:
    idx = pd.DatetimeIndex(index).dropna().sort_values()
    if len(idx) < 2:
        raise ValueError("Need at least two timestamps to infer native timestep.")
    dt = pd.Series(idx).diff().dropna()
    dt = dt[dt > pd.Timedelta(0)]
    if len(dt) == 0:
        raise ValueError("No positive timestamp differences found.")
    return dt.median()


def _circular_mean_deg_1d(x) -> float:
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan
    rad = np.deg2rad(x)
    mean_angle = np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
    return float(np.rad2deg(mean_angle) % 360.0)


def _circular_mean_signed_deg_1d(x) -> float:
    return float(wrap_180(_circular_mean_deg_1d(x)))


def _numeric_time(index: pd.Index) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        return index.view("int64").astype(float) / 1e9
    return np.arange(len(index), dtype=float)


def _read_csv_flexible(path: Path, nrows: int | None = None) -> pd.DataFrame:
    attempts = [dict(), dict(sep=";"), dict(encoding="latin1"), dict(sep=";", encoding="latin1")]
    last_error = None
    for kwargs in attempts:
        try:
            return pd.read_csv(path, nrows=nrows, **kwargs)
        except Exception as exc:
            last_error = exc
    raise last_error


def _require_columns(df: pd.DataFrame, columns: Sequence[str], source_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required {source_name} columns: {missing}. "
            "The mesoscale loader uses explicit hard-coded Ørsted column names."
        )


def _robust_angle_center_deg(angle_deg) -> float:
    x = pd.Series(wrap_180(angle_deg)).dropna()
    return float(np.nanmedian(x)) if len(x) else np.nan


# -----------------------------------------------------------------------------
# Raw file discovery/loading
# -----------------------------------------------------------------------------

def _resolve_paths(
    root: str | Path,
    *,
    meso_file: str | Path | None = None,
    scada_dir: str | Path | None = None,
    zx_nbl_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    root = Path(root)

    if meso_file is None:
        candidates = [
            root / "Mesoscale data" / "ROW01_mesoscale_data.csv",
            root / "ROW" / "Mesoscale data" / "ROW01_mesoscale_data.csv",
            root / "ROW01_mesoscale_data.csv",
        ]
        meso_file = next((p for p in candidates if p.exists()), None)
    else:
        meso_file = Path(meso_file)

    if scada_dir is None:
        candidates = [root, root / "hfdata ROW01", root / "ROW" / "hfdata ROW01"]
        scada_dir = next((p for p in candidates if p.exists() and any(p.glob("*.parquet"))), None)
    else:
        scada_dir = Path(scada_dir)

    if zx_nbl_dir is None:
        candidates = [
            root,
            root / "LiDAR data" / "ZX LiDAR on A04",
            root / "ROW" / "LiDAR data" / "ZX LiDAR on A04",
        ]
        zx_nbl_dir = next((p for p in candidates if p.exists() and (p / DEFAULT_NBL_FILES[DEFAULT_LIDAR_KEY]["filename"]).exists()), None)
    else:
        zx_nbl_dir = Path(zx_nbl_dir)

    missing = []
    if meso_file is None or not Path(meso_file).exists():
        missing.append("meso_file")
    if scada_dir is None or not Path(scada_dir).exists():
        missing.append("scada_dir")
    if zx_nbl_dir is None or not Path(zx_nbl_dir).exists():
        missing.append("zx_nbl_dir")
    if missing:
        raise FileNotFoundError(
            "Could not resolve raw Ørsted mesoscale input paths: "
            + ", ".join(missing)
            + ". Pass meso_file=..., scada_dir=..., and zx_nbl_dir=... explicitly."
        )

    return Path(meso_file), Path(scada_dir), Path(zx_nbl_dir)


def _list_scada_files(data_path: Path, turbine_id: str = DEFAULT_TURBINE_ID) -> list[Path]:
    files = sorted(Path(data_path).glob("*.parquet"))
    files = [p for p in files if turbine_id in p.name]
    if not files:
        raise FileNotFoundError(f"No parquet files containing {turbine_id!r} found in {data_path}")
    parsed = []
    for p in files:
        m = re.search(r"_(\d{4})_(\d{1,2})\.parquet$", p.name)
        year = int(m.group(1)) if m else 9999
        month = int(m.group(2)) if m else 99
        parsed.append((year, month, p))
    return [p for _, _, p in sorted(parsed)]


def _load_single_scada_parquet(file_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    time_cols = [c for c in df.columns if "Time" in str(c)]
    if "Time" in df.columns:
        time_col = "Time"
    elif len(time_cols) == 1:
        time_col = time_cols[0]
    else:
        raise ValueError(f"Could not identify unique Time column in {file_path.name}: {time_cols}")
    row_times = pd.to_datetime(df[time_col], errors="coerce")
    out = df.drop(columns=time_cols).loc[row_times.notna()].copy()
    out.index = pd.DatetimeIndex(row_times.loc[row_times.notna()])
    out.index.name = "Time"
    return out


def _load_scada_parquet_dir(data_path: Path, turbine_id: str = DEFAULT_TURBINE_ID) -> pd.DataFrame:
    chunks = [_load_single_scada_parquet(p) for p in _list_scada_files(data_path, turbine_id=turbine_id)]
    return _standardize_time_index(pd.concat(chunks, axis=0).sort_index())


def _load_meso_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = "Time" if "Time" in df.columns else None
    if time_col is None:
        candidates = [c for c in df.columns if "time" in str(c).lower()]
        time_col = candidates[0] if candidates else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()
    df.index.name = "Time"
    return _standardize_time_index(df)


def _extract_meso_variable(df: pd.DataFrame, var: str) -> pd.DataFrame:
    cols, heights = [], []
    for c in df.columns:
        s = str(c)
        if "_" not in s:
            continue
        pre, suf = s.split("_", 1)
        if pre != var:
            continue
        m = re.search(r"[-+]?\d*\.?\d+", suf)
        if m:
            cols.append(c)
            heights.append(float(m.group()))
    if not cols:
        raise ValueError(f"No columns found for mesoscale variable {var!r}")
    order = np.argsort(heights)
    out = df[[cols[i] for i in order]].copy()
    out.columns = np.asarray(heights)[order]
    return out


def _load_single_nbl_csv(path: Path) -> pd.DataFrame:
    df = _read_csv_flexible(path)
    time_col = "Time" if "Time" in df.columns else [c for c in df.columns if "Time" in str(c)][0]
    row_times = pd.to_datetime(df[time_col], errors="coerce")
    out = df.drop(columns=[time_col]).loc[row_times.notna()].copy()
    out.index = pd.DatetimeIndex(row_times.loc[row_times.notna()])
    out.index.name = "Time"
    return _standardize_time_index(out)


def _extract_r_distance_from_name(col: str) -> float | None:
    m = re.search(r"r(\d+(?:\.\d+)?)", str(col), flags=re.I)
    return None if m is None else float(m.group(1))


def _extract_h_height_from_name(col: str) -> float | None:
    m = re.search(r"h(\d+(?:\.\d+)?)", str(col), flags=re.I)
    return None if m is None else float(m.group(1))


def _classify_nbl_columns(columns, instrument="ZX") -> pd.DataFrame:
    rows = []
    for c in columns:
        s = str(c)
        sl = s.lower()
        rows.append({
            "column": s,
            "instrument": instrument,
            "r_distance_from_lidar_m": _extract_r_distance_from_name(s),
            "h_height_msl_m": _extract_h_height_from_name(s),
            "is_wind_speed": ("windspeed" in sl or "wind_speed" in sl),
            "is_wind_direction": ("relativedirection" in sl or "relative_direction" in sl or "winddirection" in sl or "wind_direction" in sl),
            "is_relative_direction": ("relativedirection" in sl or "relative_direction" in sl),
            "is_ti": (sl.startswith("ti_") or "ti_h" in sl or "_ti_" in sl),
        })
    return pd.DataFrame(rows)


def _load_zx_lidar(zx_nbl_dir: Path, lidar_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if lidar_key not in DEFAULT_NBL_FILES:
        raise KeyError(f"Unknown lidar_key {lidar_key!r}. Available: {list(DEFAULT_NBL_FILES)}")
    path = Path(zx_nbl_dir) / DEFAULT_NBL_FILES[lidar_key]["filename"]
    if not path.exists():
        raise FileNotFoundError(path)
    raw = _load_single_nbl_csv(path)
    inv = _classify_nbl_columns(raw.columns, instrument="ZX")
    inv.insert(0, "nbl_key", lidar_key)
    return raw, inv


# -----------------------------------------------------------------------------
# Resampling and profile math
# -----------------------------------------------------------------------------

def _make_common_grid(
    scada: pd.DataFrame,
    lidar: pd.DataFrame,
    meso: pd.DataFrame,
    *,
    avg_window_minutes: float | None,
) -> tuple[pd.DatetimeIndex, pd.Timedelta, dict]:
    scada_dt = _infer_base_dt(scada.index)
    lidar_dt = _infer_base_dt(lidar.index)
    meso_dt = _infer_base_dt(meso.index)
    native_required_max = max(scada_dt, lidar_dt)

    if avg_window_minutes is None:
        target_dt = native_required_max
    else:
        target_dt = pd.Timedelta(minutes=float(avg_window_minutes))
        if target_dt < native_required_max:
            raise ValueError(
                "For Ørsted mesoscale, filters['avg_window'] is a time window in minutes and must be "
                "at least as large as the coarsest native timestep among SCADA and selected lidar. "
                "Mesoscale fields are interpolated to this grid following the workflow protocol.\n"
                f"SCADA native timestep: {scada_dt}\n"
                f"Lidar native timestep: {lidar_dt}\n"
                f"Mesoscale native timestep: {meso_dt}\n"
                f"Requested avg_window: {target_dt}\n"
                f"Use avg_window >= {native_required_max / pd.Timedelta(minutes=1):.3g} minutes."
            )

    starts = [scada.index.min(), lidar.index.min(), meso.index.min()]
    ends = [scada.index.max(), lidar.index.max(), meso.index.max()]
    common_start = max(starts).ceil(target_dt)
    common_end = min(ends).floor(target_dt)
    if common_end <= common_start:
        raise ValueError(
            "No overlapping SCADA/lidar/mesoscale time window after snapping to target grid.\n"
            f"SCADA: {scada.index.min()} to {scada.index.max()}\n"
            f"Lidar: {lidar.index.min()} to {lidar.index.max()}\n"
            f"Meso:  {meso.index.min()} to {meso.index.max()}\n"
            f"Target dt: {target_dt}"
        )
    grid = pd.date_range(common_start, common_end, freq=target_dt, inclusive="left")
    diagnostics = {
        "scada_native_dt": scada_dt,
        "lidar_native_dt": lidar_dt,
        "meso_native_dt": meso_dt,
        "target_dt": target_dt,
        "common_start": common_start,
        "common_end": common_end,
        "n_windows": len(grid),
    }
    return grid, target_dt, diagnostics


def _resample_numeric_to_grid(df: pd.DataFrame, grid: pd.DatetimeIndex, freq: pd.Timedelta) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).resample(freq, origin=grid[0], label="left", closed="left").mean().reindex(grid)


def _resample_direction_cols_to_grid(df: pd.DataFrame, cols: Sequence[str], grid: pd.DatetimeIndex, freq: pd.Timedelta, relative=False) -> pd.DataFrame:
    out = pd.DataFrame(index=grid)
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"Missing direction column {c!r}")
        grouped = pd.to_numeric(df[c], errors="coerce").resample(freq, origin=grid[0], label="left", closed="left")
        out[c] = grouped.apply(_circular_mean_signed_deg_1d if relative else _circular_mean_deg_1d).reindex(grid)
    return out


def _interpolate_meso_speed_direction_to_grid(speed_df: pd.DataFrame, dir_df: pd.DataFrame, grid: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    heights = np.asarray(sorted(set(speed_df.columns).intersection(set(dir_df.columns))), dtype=float)
    speed = speed_df[heights].astype(float)
    direction = dir_df[heights].astype(float)
    theta = np.deg2rad(direction)
    uu = speed * np.cos(theta)
    vv = speed * np.sin(theta)
    uu_g = uu.reindex(uu.index.union(grid)).sort_index().interpolate(method="time").loc[grid]
    vv_g = vv.reindex(vv.index.union(grid)).sort_index().interpolate(method="time").loc[grid]
    speed_g = np.sqrt(uu_g ** 2 + vv_g ** 2)
    dir_g = np.rad2deg(np.arctan2(vv_g, uu_g)) % 360.0
    speed_g.columns = heights
    dir_g.columns = heights
    return speed_g, dir_g


def _interpolate_numeric_profile_to_grid(df: pd.DataFrame | None, grid: pd.DatetimeIndex) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.astype(float).reindex(df.index.union(grid)).sort_index().interpolate(method="time").loc[grid]
    out.columns = np.asarray(out.columns, dtype=float)
    return out


def _interpolate_profile_at_height(profile_df: pd.DataFrame, target_height: float) -> pd.Series:
    heights = np.asarray(profile_df.columns, dtype=float)
    values = profile_df.to_numpy(dtype=float)
    out = np.full(values.shape[0], np.nan)
    for i, row in enumerate(values):
        ok = np.isfinite(row) & np.isfinite(heights)
        if ok.sum() >= 2:
            order = np.argsort(heights[ok])
            out[i] = np.interp(target_height, heights[ok][order], row[ok][order])
    return pd.Series(out, index=profile_df.index)



def _linear_extrapolate_profile_to_lower_bound(
    profile_df: pd.DataFrame,
    lower_height: float,
    *,
    n_fit_levels: int = 3,
    angle_degrees: bool = False,
) -> pd.DataFrame:
    """Add a profile value at ``lower_height`` using local linear extrapolation.

    The existing profile values are unchanged.  If ``lower_height`` is already
    within the available vertical grid, the input profile is returned with sorted
    numeric columns.  If ``lower_height`` is below the lowest available level,
    the value at ``lower_height`` is computed independently for each timestamp
    from a straight-line fit to the lowest ``n_fit_levels`` finite values.

    For angular profiles, the fit is done to wrapped angular differences from
    the lowest finite level, which avoids spurious jumps across the -180/180
    boundary for relative-direction profiles.
    """
    if profile_df is None:
        return None

    out = profile_df.copy()
    out.columns = np.asarray(out.columns, dtype=float)
    out = out.reindex(sorted(out.columns), axis=1)

    z_all = np.asarray(out.columns, dtype=float)
    if len(z_all) == 0:
        return out

    lower_height = float(lower_height)

    # If the requested lower bound is already covered by the grid, do not
    # invent a new level.
    if lower_height >= np.nanmin(z_all) or np.any(np.isclose(z_all, lower_height, rtol=0.0, atol=1e-9)):
        return out

    values = out.to_numpy(dtype=float)
    y_lower = np.full(values.shape[0], np.nan)
    n_fit_levels = max(int(n_fit_levels), 2)

    for i, row in enumerate(values):
        ok = np.isfinite(row) & np.isfinite(z_all)
        if np.count_nonzero(ok) < 2:
            continue

        z = z_all[ok]
        y = row[ok]
        order = np.argsort(z)
        z = z[order]
        y = y[order]

        k = min(n_fit_levels, len(z))
        z_fit = z[:k]
        y_fit = y[:k]

        if angle_degrees:
            # Anchor at the lowest finite angle and fit local wrapped angle
            # anomalies relative to that anchor.
            z0 = z_fit[0]
            y0 = y_fit[0]
            dy = wrap_180(y_fit - y0)
            if k >= 2:
                slope = np.polyfit(z_fit - z0, dy, deg=1)[0]
                y_lower[i] = wrap_180(y0 + slope * (lower_height - z0))
        else:
            if k >= 2:
                slope, intercept = np.polyfit(z_fit, y_fit, deg=1)
                y_lower[i] = slope * lower_height + intercept

    out[lower_height] = y_lower
    out = out.reindex(sorted(out.columns), axis=1)
    return out

def _compute_power_law_shear_alpha(speed_profile: pd.DataFrame, hub_height: float, diameter: float) -> pd.Series:
    heights = np.asarray(speed_profile.columns, dtype=float)
    in_rotor = (heights >= hub_height - diameter / 2.0) & (heights <= hub_height + diameter / 2.0)
    z = heights[in_rotor]
    if len(z) < 2:
        raise ValueError("Need at least two heights inside rotor to fit shear.")
    x = np.log(z / hub_height)
    vals = speed_profile.iloc[:, in_rotor].to_numpy(dtype=float)
    alpha = np.full(vals.shape[0], np.nan)
    for i, u in enumerate(vals):
        ok = np.isfinite(u) & (u > 0) & np.isfinite(x)
        if ok.sum() >= 2:
            alpha[i] = np.polyfit(x[ok], np.log(u[ok]), 1)[0]
    return pd.Series(alpha, index=speed_profile.index, name="alpha")


def _compute_top_bottom_veer_rate(rel_dir_profile_deg: pd.DataFrame, hub_height: float, diameter: float) -> pd.Series:
    theta_bottom = _interpolate_profile_at_height(rel_dir_profile_deg, hub_height - diameter / 2.0)
    theta_top = _interpolate_profile_at_height(rel_dir_profile_deg, hub_height + diameter / 2.0)
    return pd.Series(wrap_180(theta_top - theta_bottom) / diameter, index=rel_dir_profile_deg.index)


def _compute_mean_interval_veer_rate(rel_dir_profile_deg: pd.DataFrame, hub_height: float, diameter: float) -> pd.Series:
    heights = np.asarray(rel_dir_profile_deg.columns, dtype=float)
    in_rotor = (heights >= hub_height - diameter / 2.0) & (heights <= hub_height + diameter / 2.0)
    z = heights[in_rotor]
    if len(z) < 2:
        raise ValueError("Need at least two heights inside rotor to fit veer.")
    order = np.argsort(z)
    z = z[order]
    vals = rel_dir_profile_deg.iloc[:, in_rotor].iloc[:, order].to_numpy(dtype=float)
    slopes = wrap_180(np.diff(vals, axis=1)) / np.diff(z)[None, :]
    return pd.Series(np.nanmean(slopes, axis=1), index=rel_dir_profile_deg.index)


def _compute_density_moist_air(P: pd.DataFrame, T: pd.DataFrame, RH: pd.DataFrame) -> pd.DataFrame:
    P = P.astype(float)
    T = T.astype(float)
    RH = RH.astype(float)
    P_pa = P.where(P > 2000, P * 100.0)
    T_k = T.where(T > 100, T + 273.15)
    rh = RH.where(RH <= 1.5, RH / 100.0).clip(0, 1)
    T_c = T_k - 273.15
    e_s = 611.21 * np.exp((18.678 - T_c / 234.5) * (T_c / (257.14 + T_c)))
    e = rh * e_s
    rho = (P_pa - e) / (287.05 * T_k) + e / (461.495 * T_k)
    rho.columns = P.columns.astype(float)
    return rho


def _is_mean_signal(col: str) -> bool:
    s = str(col).lower()
    return not any(t in s for t in ["std", "_min", "_max", "minimum", "maximum"])


def _select_zx_profile_columns(inv: pd.DataFrame, key: str, quantity: str, preferred_range_m: float | None = None, strict=True) -> pd.DataFrame:
    sub = inv[(inv.nbl_key == key) & (inv.instrument.str.lower() == "zx") & (inv.h_height_msl_m.notna())].copy()
    if quantity == "speed":
        sub = sub[sub.is_wind_speed]
    elif quantity == "direction":
        sub = sub[sub.is_wind_direction]
        rel = sub[sub.is_relative_direction]
        if not rel.empty:
            sub = rel
    elif quantity == "ti":
        sub = sub[sub.is_ti]
    else:
        raise ValueError(quantity)
    sub = sub[sub.column.map(_is_mean_signal)]
    if sub.empty:
        if strict:
            raise ValueError(f"No ZX {quantity} profile columns found for {key}.")
        return pd.DataFrame(columns=["column", "h_height_msl_m", "r_distance_from_lidar_m"])
    if preferred_range_m is not None and sub.r_distance_from_lidar_m.notna().any():
        ranges = np.sort(sub.r_distance_from_lidar_m.dropna().unique())
        chosen = float(ranges[np.argmin(np.abs(ranges - preferred_range_m))])
        sub = sub[sub.r_distance_from_lidar_m == chosen].copy()
    return sub[["column", "h_height_msl_m", "r_distance_from_lidar_m"]].sort_values(["h_height_msl_m", "column"])


def _build_profile_from_columns(df: pd.DataFrame, cols: pd.DataFrame, direction=False) -> pd.DataFrame:
    pieces = []
    for h, group in cols.groupby("h_height_msl_m"):
        use = [c for c in group.column if c in df.columns]
        if not use:
            continue
        tmp = df[use].apply(pd.to_numeric, errors="coerce")
        if len(use) == 1:
            s = tmp[use[0]]
        else:
            s = tmp.apply(_circular_mean_signed_deg_1d, axis=1) if direction else tmp.mean(axis=1)
        s.name = float(h)
        pieces.append(s)
    if not pieces:
        raise ValueError("No selected columns present in lidar raw dataframe.")
    return pd.concat(pieces, axis=1).reindex(sorted([p.name for p in pieces]), axis=1)


def _resample_profile_to_grid(profile: pd.DataFrame, grid: pd.DatetimeIndex, freq: pd.Timedelta, direction=False) -> pd.DataFrame:
    out = pd.DataFrame(index=grid)
    for c in profile.columns:
        grouped = pd.to_numeric(profile[c], errors="coerce").resample(freq, origin=grid[0], label="left", closed="left")
        out[float(c)] = grouped.apply(_circular_mean_signed_deg_1d if direction else np.nanmean).reindex(grid)
    return out


def _compute_rews(speed_profile: pd.DataFrame, rel_dir_profile: pd.DataFrame, density_profile: pd.DataFrame, hub_height: float, radius: float, rho_std: float) -> pd.Series:
    h = np.asarray(sorted(set(speed_profile.columns).intersection(rel_dir_profile.columns).intersection(density_profile.columns)), dtype=float)
    z = h
    edges = np.empty(len(z) + 1)
    edges[1:-1] = 0.5 * (z[:-1] + z[1:])
    edges[0] = hub_height - radius
    edges[-1] = hub_height + radius
    edges = np.clip(edges, hub_height - radius, hub_height + radius)

    def F(y):
        y = np.clip(y, -radius, radius)
        return y * np.sqrt(np.maximum(radius ** 2 - y ** 2, 0)) + radius ** 2 * np.arcsin(y / radius)

    areas = np.array([F(b - hub_height) - F(a - hub_height) for a, b in zip(edges[:-1], edges[1:])])
    weights = areas / areas.sum()
    integrand = (density_profile[h] / rho_std) * speed_profile[h] ** 3 * np.cos(np.deg2rad(rel_dir_profile[h])) ** 3
    summed = integrand.mul(weights, axis=1).sum(axis=1, skipna=False)
    return np.sign(summed) * np.abs(summed) ** (1.0 / 3.0)


# -----------------------------------------------------------------------------
# Filtering
# -----------------------------------------------------------------------------

def _case_mask(data: Mapping, filters: Mapping | None) -> np.ndarray:
    n = int(data["nCases"])
    mask = np.ones(n, dtype=bool)
    if not filters:
        return mask
    skip = {"avg_window", "window_min"}
    for key, val in filters.items():
        if key in skip or val is None:
            continue
        if key.endswith("_min"):
            field = key[:-4]
            if field not in data:
                raise KeyError(f"Filter refers to missing field: {field!r}")
            arr = np.asarray(data[field], dtype=float)
            mask &= np.isfinite(arr) & (arr >= float(val))
        elif key.endswith("_max"):
            field = key[:-4]
            if field not in data:
                raise KeyError(f"Filter refers to missing field: {field!r}")
            arr = np.asarray(data[field], dtype=float)
            mask &= np.isfinite(arr) & (arr <= float(val))
    return mask


def _apply_case_mask(data: dict, mask: np.ndarray) -> dict:
    n = int(data["nCases"])
    out = {}
    for key, val in data.items():
        arr = np.asarray(val)
        if key == "nCases":
            continue
        if arr.ndim == 2 and arr.shape[1] == n:
            out[key] = arr[:, mask]
        elif arr.ndim == 1 and arr.size == n:
            out[key] = arr[mask]
        else:
            out[key] = val
    out["nCases"] = int(np.count_nonzero(mask))
    out["nH"] = int(data["nH"])
    return out


# -----------------------------------------------------------------------------
# Public loader
# -----------------------------------------------------------------------------

def load_orsted_meso_data(
    root: str | Path,
    filters: Mapping | None = None,
    *,
    lidar_key: str = DEFAULT_LIDAR_KEY,
    yaw_mode: str = "lidar",
    meso_file: str | Path | None = None,
    scada_dir: str | Path | None = None,
    zx_nbl_dir: str | Path | None = None,
    turbine_id: str = DEFAULT_TURBINE_ID,
    R: float = DEFAULT_R,
    Hub: float = DEFAULT_HUB,
    HubR: float = DEFAULT_HUBR,
    B: int = DEFAULT_B,
    rho_std: float = DEFAULT_RHO,
    turbine_name: str = "Racebank_A04_meso",
    pcurve_u: Sequence[float] | None = None,
    pcurve_p: Sequence[float] | None = None,
    preferred_lidar_range_m: float | None = None,
    lidar_direction_is_already_relative: bool | None = None,
    center_yaw: bool = True,
    extrapolate_to_rotor_bottom: bool = True,
    extrapolate_n_levels: int = 3,
    return_intermediate: bool = False,
) -> dict:
    """Load Ørsted SCADA/lidar/mesoscale data and return India-style dict.

    Parameters mirror ``orsted2python.load_orsted_data``. The most important
    difference is that ``speed_profiles`` and ``dir_profiles`` are the hybrid
    lidar-scaled mesoscale profiles used by the mesoscale protocol.
    """
    root = Path(root)
    filters = dict(filters or {})
    meso_file, scada_dir, zx_nbl_dir = _resolve_paths(root, meso_file=meso_file, scada_dir=scada_dir, zx_nbl_dir=zx_nbl_dir)

    diameter = 2.0 * float(R)
    preferred_lidar_range_m = 2.5 * diameter if preferred_lidar_range_m is None else preferred_lidar_range_m
    if lidar_direction_is_already_relative is None:
        lidar_direction_is_already_relative = DEFAULT_LIDAR_DIRECTION_IS_ALREADY_RELATIVE.get(lidar_key, True)

    scada = _load_scada_parquet_dir(scada_dir, turbine_id=turbine_id)
    meso = _load_meso_csv(meso_file)
    lidar_raw, inventory = _load_zx_lidar(zx_nbl_dir, lidar_key)

    avg_window = filters.get("avg_window", None)
    grid, target_dt, time_diag = _make_common_grid(scada, lidar_raw, meso, avg_window_minutes=avg_window)

    # SCADA reference fields.
    scada_num = _resample_numeric_to_grid(scada, grid, target_dt)
    scada_dir = _resample_direction_cols_to_grid(scada, [NACELLE_HEADING_COL, SCADA_WIND_DIR_COL], grid, target_dt)
    _require_columns(
        scada_num,
        [POWER_COL, TURBINE_HUB_SPEED_COL, GENERATOR_RPM_COL, PITCH_COL, TURBINE_TI_COL, AMBIENT_TEMP_COL],
        "SCADA numeric",
    )
    _require_columns(scada_dir, [NACELLE_HEADING_COL, SCADA_WIND_DIR_COL], "SCADA direction")
    scada_ref = pd.DataFrame(index=grid)
    scada_ref["P_obs"] = scada_num[POWER_COL]
    scada_ref["scada_power"] = scada_num[POWER_COL]
    scada_ref["turbine_hub_speed"] = scada_num[TURBINE_HUB_SPEED_COL]
    scada_ref["turbine_hub_direction_deg"] = scada_dir_df[SCADA_WIND_DIR_COL]
    scada_ref["generator_rpm"] = scada_num[GENERATOR_RPM_COL]
    scada_ref["pitch_deg"] = scada_num[PITCH_COL]
    scada_ref["turbine_TI_percent"] = 100.0 * scada_num[TURBINE_TI_COL]
    scada_ref["ambient_temp_c"] = scada_num[AMBIENT_TEMP_COL]
    scada_ref["nacelle_heading_deg"] = scada_dir[NACELLE_HEADING_COL]
    scada_ref["scada_wind_dir_deg"] = scada_dir[SCADA_WIND_DIR_COL]
    scada_ref["local_yaw_deg"] = wrap_180(scada_ref["scada_wind_dir_deg"] - scada_ref["nacelle_heading_deg"])

    # Mesoscale profiles to grid.
    wind_speed_meso_raw = _extract_meso_variable(meso, "v")
    wind_dir_meso_raw = _extract_meso_variable(meso, "dir")
    pressure_raw = _extract_meso_variable(meso, "P") if any(str(c).startswith("P_") for c in meso.columns) else None
    temp_raw = _extract_meso_variable(meso, "T") if any(str(c).startswith("T_") for c in meso.columns) else None
    rh_raw = _extract_meso_variable(meso, "RH") if any(str(c).startswith("RH_") for c in meso.columns) else None

    meso_speed, meso_abs_dir = _interpolate_meso_speed_direction_to_grid(wind_speed_meso_raw, wind_dir_meso_raw, grid)
    pressure = _interpolate_numeric_profile_to_grid(pressure_raw, grid)
    temp = _interpolate_numeric_profile_to_grid(temp_raw, grid)
    rh = _interpolate_numeric_profile_to_grid(rh_raw, grid)

    meso_U_HH = _interpolate_profile_at_height(meso_speed, Hub)
    meso_shear_ratio = meso_speed.div(meso_U_HH, axis=0)
    meso_dir_HH = _interpolate_profile_at_height(meso_abs_dir, Hub)
    meso_dir_anom = pd.DataFrame(wrap_180(meso_abs_dir.sub(meso_dir_HH, axis=0).to_numpy(dtype=float)), index=grid, columns=meso_abs_dir.columns)
    meso_alpha_shape = _compute_power_law_shear_alpha(meso_shear_ratio, Hub, diameter)
    meso_veer_interval = _compute_mean_interval_veer_rate(meso_dir_anom, Hub, diameter)
    meso_veer_top_bottom = _compute_top_bottom_veer_rate(meso_dir_anom, Hub, diameter)

    if pressure is not None and temp is not None and rh is not None:
        meso_density = _compute_density_moist_air(pressure, temp, rh)
    else:
        meso_density = pd.DataFrame(float(rho_std), index=grid, columns=meso_speed.columns)
    meso_rho_hub = _interpolate_profile_at_height(meso_density, Hub)

    # ZX lidar hub values at selected range.
    speed_cols = _select_zx_profile_columns(inventory, lidar_key, "speed", preferred_range_m=preferred_lidar_range_m, strict=True)
    dir_cols = _select_zx_profile_columns(inventory, lidar_key, "direction", preferred_range_m=preferred_lidar_range_m, strict=True)
    ti_cols = _select_zx_profile_columns(inventory, lidar_key, "ti", preferred_range_m=preferred_lidar_range_m, strict=False)

    speed_raw = _build_profile_from_columns(lidar_raw, speed_cols, direction=False)
    rel_dir_raw = _build_profile_from_columns(lidar_raw, dir_cols, direction=True)
    ti_raw = None if ti_cols.empty else _build_profile_from_columns(lidar_raw, ti_cols, direction=False)

    lidar_speed = _resample_profile_to_grid(speed_raw, grid, target_dt, direction=False)
    lidar_rel_dir = pd.DataFrame(wrap_180(_resample_profile_to_grid(rel_dir_raw, grid, target_dt, direction=True).to_numpy(dtype=float)), index=grid, columns=rel_dir_raw.columns.astype(float))
    lidar_ti = _resample_profile_to_grid(ti_raw, grid, target_dt, direction=False) if ti_raw is not None else None

    U_lidar_HH_raw = _interpolate_profile_at_height(lidar_speed, Hub)
    lidar_rel_dir_hub = wrap_180(_interpolate_profile_at_height(lidar_rel_dir, Hub))

    # Hybrid profiles.
    hybrid_speed = meso_shear_ratio.mul(U_lidar_HH_raw, axis=0)
    hybrid_rel_dir = pd.DataFrame(wrap_180(meso_dir_anom.add(lidar_rel_dir_hub, axis=0).to_numpy(dtype=float)), index=grid, columns=meso_dir_anom.columns)

    # The mesoscale vertical grid may start above the lower rotor tip.  For
    # REWS/REP integration over the full rotor, add a rotor-bottom level using
    # local linear extrapolation from the lowest resolved mesoscale levels.
    # Existing mesoscale levels are left unchanged.
    rotor_bottom = float(Hub) - float(R)
    if extrapolate_to_rotor_bottom:
        hybrid_speed = _linear_extrapolate_profile_to_lower_bound(
            hybrid_speed,
            rotor_bottom,
            n_fit_levels=extrapolate_n_levels,
            angle_degrees=False,
        )
        hybrid_rel_dir = _linear_extrapolate_profile_to_lower_bound(
            hybrid_rel_dir,
            rotor_bottom,
            n_fit_levels=extrapolate_n_levels,
            angle_degrees=True,
        )
        meso_density = _linear_extrapolate_profile_to_lower_bound(
            meso_density,
            rotor_bottom,
            n_fit_levels=extrapolate_n_levels,
            angle_degrees=False,
        )

    hybrid_U_hub = _interpolate_profile_at_height(hybrid_speed, Hub)
    hybrid_rel_dir_hub = wrap_180(_interpolate_profile_at_height(hybrid_rel_dir, Hub))
    hybrid_alpha = _compute_power_law_shear_alpha(hybrid_speed, Hub, diameter)
    hybrid_veer_interval = _compute_mean_interval_veer_rate(hybrid_rel_dir, Hub, diameter)
    hybrid_veer_top_bottom = _compute_top_bottom_veer_rate(hybrid_rel_dir, Hub, diameter)
    hybrid_REWS = _compute_rews(hybrid_speed, hybrid_rel_dir, meso_density, Hub, float(R), float(rho_std))
    U_lidar_HH_density_corrected = U_lidar_HH_raw * (meso_rho_hub / float(rho_std)) ** (1.0 / 3.0)

    # Unified scalar table and yaw definitions.
    unified = scada_ref.copy()
    unified["rho_hub"] = meso_rho_hub
    unified["U_lidar_HH_raw"] = U_lidar_HH_raw
    unified["U_lidar_HH_density_corrected"] = U_lidar_HH_density_corrected
    unified["hybrid_U_hub"] = hybrid_U_hub
    unified["hybrid_REWS"] = hybrid_REWS
    unified["hybrid_rel_dir_hub_deg"] = hybrid_rel_dir_hub
    unified["hybrid_alpha"] = hybrid_alpha
    unified["hybrid_veer_interval_deg_per_m"] = hybrid_veer_interval
    unified["hybrid_veer_top_bottom_deg_per_m"] = hybrid_veer_top_bottom
    unified["hybrid_V"] = hybrid_veer_interval * np.pi / 180.0 * float(R)
    unified["meso_U_HH"] = meso_U_HH
    unified["meso_dir_HH"] = meso_dir_HH
    unified["meso_alpha_shape"] = meso_alpha_shape
    unified["meso_veer_interval_deg_per_m"] = meso_veer_interval
    unified["meso_veer_top_bottom_deg_per_m"] = meso_veer_top_bottom
    unified["meso_V_interval"] = meso_veer_interval * np.pi / 180.0 * float(R)
    unified["lidar_hub_rel_dir_wrapped_deg"] = wrap_180(lidar_rel_dir_hub)
    unified["lidar_hub_rel_dir_center_deg"] = _robust_angle_center_deg(unified["lidar_hub_rel_dir_wrapped_deg"])
    unified["lidar_hub_rel_dir_centered_deg"] = wrap_180(unified["lidar_hub_rel_dir_wrapped_deg"] - unified["lidar_hub_rel_dir_center_deg"])
    unified["local_yaw_wrapped_deg"] = wrap_180(unified["local_yaw_deg"])
    unified["local_yaw_center_deg"] = _robust_angle_center_deg(unified["local_yaw_wrapped_deg"])
    unified["local_yaw_centered_deg"] = wrap_180(unified["local_yaw_wrapped_deg"] - unified["local_yaw_center_deg"])
    unified["lidar_minus_scada_yaw_deg"] = wrap_180(unified["lidar_hub_rel_dir_centered_deg"] - unified["local_yaw_centered_deg"])
    unified["valid_core_mesoscale_protocol"] = unified[["P_obs", "U_lidar_HH_raw", "hybrid_REWS", "hybrid_alpha", "hybrid_V", "rho_hub"]].notna().all(axis=1)

    # Keep all core-valid rows before field-level filters.
    table = unified.loc[unified["valid_core_mesoscale_protocol"].fillna(False)].copy()
    hybrid_speed = hybrid_speed.loc[table.index]
    hybrid_rel_dir = hybrid_rel_dir.loc[table.index]
    meso_density = meso_density.loc[table.index]
    lidar_ti = lidar_ti.loc[table.index] if lidar_ti is not None else None

    n = len(table)
    if n == 0:
        raise ValueError("No rows remained after constructing valid_core_mesoscale_protocol.")

    heights = np.asarray(hybrid_speed.columns, dtype=float)
    speed = hybrid_speed.to_numpy(dtype=float).T
    dir_rel = hybrid_rel_dir.to_numpy(dtype=float).T
    if lidar_ti is not None:
        # Keep ti_profiles aligned with the returned height grid.  The added
        # rotor-bottom mesoscale level does not have a direct lidar TI value.
        ti_prof_arr = lidar_ti.reindex(columns=hybrid_speed.columns).to_numpy(dtype=float).T
    else:
        ti_prof_arr = np.full_like(speed, np.nan)

    power = pd.to_numeric(table["P_obs"], errors="coerce").to_numpy(dtype=float)
    hubspeed = pd.to_numeric(table["hybrid_U_hub"], errors="coerce").to_numpy(dtype=float)
    Thubspeed = pd.to_numeric(table["turbine_hub_speed"], errors="coerce").to_numpy(dtype=float)
    alpha = pd.to_numeric(table["hybrid_alpha"], errors="coerce").to_numpy(dtype=float)
    dsrate = pd.to_numeric(table["hybrid_veer_interval_deg_per_m"], errors="coerce").to_numpy(dtype=float)
    pitch_deg = pd.to_numeric(table["pitch_deg"], errors="coerce").to_numpy(dtype=float)
    generator_rpm = pd.to_numeric(table["generator_rpm"], errors="coerce").to_numpy(dtype=float)
    omega_rad_s = generator_rpm * 2.0 * np.pi / 60.0
    tsr = omega_rad_s * float(R) / hubspeed
    ti = pd.to_numeric(table["turbine_TI_percent"], errors="coerce").to_numpy(dtype=float)
    rho_arr = pd.to_numeric(table["rho_hub"], errors="coerce").to_numpy(dtype=float)

    yaw_lidar = wrap_180(pd.to_numeric(table["lidar_hub_rel_dir_wrapped_deg"], errors="coerce").to_numpy(dtype=float))
    yaw_lidar_centered = pd.to_numeric(table["lidar_hub_rel_dir_centered_deg"], errors="coerce").to_numpy(dtype=float)
    yaw_scada = pd.to_numeric(table["local_yaw_deg"], errors="coerce").to_numpy(dtype=float)
    yaw_scada_centered = pd.to_numeric(table["local_yaw_centered_deg"], errors="coerce").to_numpy(dtype=float)
    yaw_lidar_minus_scada = pd.to_numeric(table["lidar_minus_scada_yaw_deg"], errors="coerce").to_numpy(dtype=float)
    yaw_options = {
        "lidar": yaw_lidar,
        "lidar_centered": yaw_lidar_centered,
        "scada": yaw_scada,
        "scada_centered": yaw_scada_centered,
        "lidar_minus_scada": yaw_lidar_minus_scada,
    }
    if yaw_mode not in yaw_options:
        raise ValueError(f"yaw_mode must be one of {list(yaw_options)}")
    hubdir = yaw_options[yaw_mode]

    data = {
        "turbine_name": turbine_name,
        "R": float(R),
        "Hub": float(Hub),
        "HubR": float(HubR) if np.isfinite(HubR) else np.nan,
        "B": int(B),
        "heights": heights.astype(float),
        "speed": speed,
        "dir_rel": dir_rel,
        "hubspeed": hubspeed,
        "hubdir": hubdir,
        "alpha": alpha,
        "dsrate": dsrate,
        "pitch_deg": pitch_deg,
        "omega": generator_rpm,
        "omega_rad_s": omega_rad_s,
        "rho": rho_arr,
        "flag_indx": np.ones(n, dtype=float),
        "Thubspeed": Thubspeed,
        "turbine_hub_speed": Thubspeed,
        "tsr": tsr,
        "ti": ti,
        "TI": ti,
        "power": power,
        "Pcurve_U": np.asarray([] if pcurve_u is None else pcurve_u, dtype=float),
        "Pcurve_P": np.asarray([] if pcurve_p is None else pcurve_p, dtype=float),
        "time": _numeric_time(table.index),
        "source_index": table.index.astype(str).to_numpy(),

        # Aliases expected downstream.
        "speed_profiles": speed,
        "dir_profiles": dir_rel,
        "ti_profiles": ti_prof_arr,
        "veer_deg_per_m": dsrate,
        "tsr_data": tsr,
        "alpha_data": alpha,
        "hubheight": float(Hub),
        "turbinePower": power,
        "nH": int(len(heights)),
        "nCases": int(n),

        # Mesoscale/hybrid diagnostics.
        "profile_source": "hybrid_lidar_scaled_mesoscale",
        "profile_extrapolated_to_rotor_bottom": bool(extrapolate_to_rotor_bottom),
        "profile_extrapolation_method": "linear_lowest_levels" if extrapolate_to_rotor_bottom else "none",
        "profile_extrapolation_n_levels": int(max(int(extrapolate_n_levels), 2)),
        "profile_extrapolation_lower_height": float(Hub) - float(R),
        "hybrid_REWS": pd.to_numeric(table["hybrid_REWS"], errors="coerce").to_numpy(dtype=float),
        "hybrid_U_hub": hubspeed,
        "hybrid_V": pd.to_numeric(table["hybrid_V"], errors="coerce").to_numpy(dtype=float),
        "meso_U_HH": pd.to_numeric(table["meso_U_HH"], errors="coerce").to_numpy(dtype=float),
        "meso_alpha_shape": pd.to_numeric(table["meso_alpha_shape"], errors="coerce").to_numpy(dtype=float),
        "meso_veer_interval_deg_per_m": pd.to_numeric(table["meso_veer_interval_deg_per_m"], errors="coerce").to_numpy(dtype=float),
        "U_lidar_HH_raw": pd.to_numeric(table["U_lidar_HH_raw"], errors="coerce").to_numpy(dtype=float),
        "U_lidar_HH_density_corrected": pd.to_numeric(table["U_lidar_HH_density_corrected"], errors="coerce").to_numpy(dtype=float),

        # Yaw diagnostics.
        "yaw_mode": yaw_mode,
        "yaw_lidar_hub_deg": yaw_lidar,
        "yaw_lidar_centered_deg": yaw_lidar_centered,
        "yaw_scada_nacelle_deg": yaw_scada,
        "yaw_scada_centered_deg": yaw_scada_centered,
        "yaw_lidar_minus_scada_deg": yaw_lidar_minus_scada,
        "yaw_lidar": yaw_lidar,
        "yaw_lidar_centered": yaw_lidar_centered,
        "yaw_scada": yaw_scada,
        "yaw_scada_centered": yaw_scada_centered,
        "yaw_lidar_minus_scada": yaw_lidar_minus_scada,
        "nacelle_heading_deg": pd.to_numeric(table["nacelle_heading_deg"], errors="coerce").to_numpy(dtype=float),
        "scada_wind_dir_deg": pd.to_numeric(table["scada_wind_dir_deg"], errors="coerce").to_numpy(dtype=float),
        "ambient_temp_c": pd.to_numeric(table["ambient_temp_c"], errors="coerce").to_numpy(dtype=float),

        "resample_native_scada_dt_seconds": float(time_diag["scada_native_dt"].total_seconds()),
        "resample_native_lidar_dt_seconds": float(time_diag["lidar_native_dt"].total_seconds()),
        "resample_native_meso_dt_seconds": float(time_diag["meso_native_dt"].total_seconds()),
        "resample_target_dt_seconds": float(time_diag["target_dt"].total_seconds()),
    }

    mask = _case_mask(data, filters)
    data = _apply_case_mask(data, mask)

    try:
        p_aero_kw = 0.5 * np.asarray(data["rho"]) * np.pi * float(R) ** 2 * np.asarray(data["hubspeed"]) ** 3 * 1e-3
        data["turbine_CP"] = np.asarray(data["power"]) / p_aero_kw
    except Exception:
        data["turbine_CP"] = np.full(int(data["nCases"]), np.nan)

    if return_intermediate:
        data["_intermediate"] = {
            "scada_raw": scada,
            "meso_raw": meso,
            "lidar_raw": lidar_raw,
            "inventory": inventory,
            "table_before_filter": table,
            "hybrid_speed_profile_before_filter": hybrid_speed,
            "hybrid_relative_direction_profile_before_filter": hybrid_rel_dir,
            "meso_density_profile_before_filter": meso_density,
            "resample_diagnostics": time_diag,
        }

    return data


def save_india_style_mat(data: Mapping, out_path: str | Path, struct_name: str = "out") -> None:
    if savemat is None:
        raise ImportError("scipy.io.savemat is unavailable in this environment.")
    out = {}
    skip = {"nH", "nCases", "speed_profiles", "dir_profiles", "ti_profiles", "turbinePower", "turbine_CP", "alpha_data", "tsr_data", "hubheight"}
    for k, v in data.items():
        if k.startswith("_") or k in skip:
            continue
        arr = np.asarray(v)
        if arr.ndim == 1 and k not in {"heights", "Pcurve_U", "Pcurve_P"}:
            out[k] = arr.reshape(1, -1)
        elif k == "heights":
            out[k] = arr.reshape(-1, 1)
        else:
            out[k] = v
    savemat(out_path, {struct_name: out})


__all__ = ["load_orsted_meso_data", "save_india_style_mat", "wrap_180"]
