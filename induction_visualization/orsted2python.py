"""Ørsted raw-data loader with India-style output schema.

This module mirrors the user-facing pattern of ``matlab2python.py``:

    import induction_visualization.orsted2python as o2p
    data = o2p.load_orsted_data(root, filters=mask)

Unlike earlier versions, this loader does *not* require the intermediate
``orsted_lidar_workflow_v5.ipynb`` CSV outputs.  It reads the individual SCADA
parquet files and ZX lidar CSV files directly, resamples all sources onto a
single time grid, assembles the India-style dictionary, and then applies the
requested filters.

Design choices are explicit and configurable:
- default lidar source: ``ZX_zxtm5052``;
- SCADA power ``ActPower_Value_mean`` is interpreted as kW;
- all turbine scalar variables are extracted from explicit hard-coded SCADA columns with no fallback;
- ``GenRpm_Value_mean`` is direct-drive rotor speed in rpm;
- TSR is computed as ``(rpm * 2*pi/60) * R / U_hub``;
- multiple yaw definitions are preserved as ``yaw_*`` arrays, and ``yaw_mode``
  only chooses which one is exposed as India-compatible ``hubdir``;
- for Ørsted, ``filters["avg_window"]`` is a *time window in minutes*.
  The requested averaging window must be no smaller than the coarsest native
  timestep among SCADA and the selected lidar source.  For 1-min SCADA and
  10-min lidar, use ``avg_window >= 10``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import re
import warnings

import numpy as np
import pandas as pd

try:
    from scipy.io import savemat
except Exception:  # pragma: no cover
    savemat = None


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULT_LIDAR_KEY = "ZX_zxtm5052"
DEFAULT_TURBINE_ID = "ROW01A04"

DEFAULT_DIAMETER = 154.0       # [m]
DEFAULT_R = DEFAULT_DIAMETER / 2.0
DEFAULT_HUB = 103.3            # [m]
DEFAULT_RHO = 1.225            # [kg/m^3]
DEFAULT_HUBR = np.nan
DEFAULT_B = 3

# Explicit SCADA column mapping requested for Ørsted A04.
# These are intentionally hard-coded: if a column is absent, the loader raises
# an error rather than falling back to another variable.
POWER_COL = "ActPower_Value_mean"              # turbine power [kW]
NACELLE_HEADING_COL = "NacelPos_Value_mean"    # turbine heading / nacelle position [deg]
SCADA_WIND_DIR_COL = "AcWindDr_Value_mean"     # turbine hub wind direction, absolute [deg]
TURBINE_HUB_SPEED_COL = "AcWindSp_AcWindSp_mean"  # turbine hub/nacelle wind speed [m/s]
GENERATOR_RPM_COL = "GenRpm_Value_mean"          # direct-drive rotor/generator speed [rpm]
PITCH_COL = "PitcPosA_Value_mean"                # pitch [deg]
TURBINE_TI_COL = "TurbEst_TurbEst_mean"          # turbine TI, as provided (not percent)
AMBIENT_TEMP_COL = "AmbieTmp_Value_mean"         # ambient temperature [deg C]

DEFAULT_NBL_FILES = {
    "WindCube_WI01030180": {
        "filename": "RaceBank_A04_WindCubeNBL_WI01030180.csv",
        "instrument": "WindCube",
        "device_id": "WI01030180",
        "subdir_kind": "windcube",
    },
    "WindCube_WI01030188": {
        "filename": "RaceBank_A04_WindCubeNBL_WI01030188.csv",
        "instrument": "WindCube",
        "device_id": "WI01030188",
        "subdir_kind": "windcube",
    },
    "ZX_zxtm5005": {
        "filename": "RaceBank_A04_ZXNBL_zxtm5005.csv",
        "instrument": "ZX",
        "device_id": "zxtm5005",
        "subdir_kind": "zx",
    },
    "ZX_zxtm5052": {
        "filename": "RaceBank_A04_ZXNBL_zxtm5052.csv",
        "instrument": "ZX",
        "device_id": "zxtm5052",
        "subdir_kind": "zx",
    },
}

# The email/header convention says the ZX RelativeDirection columns are already
# local/relative to turbine/nacelle frame.  This can be overridden by passing
# ``lidar_direction_is_already_relative=False``.
DEFAULT_LIDAR_DIRECTION_IS_ALREADY_RELATIVE = {
    "ZX_zxtm5005": True,
    "ZX_zxtm5052": True,
}

LIDAR_SCALAR_COLUMNS_TO_RETAIN = [
    "Bearing_ZephIR_mean",
    "P_mean",
    "RH_mean",
    "Roll_mean",
    "T_mean",
    "Tilt_mean",
    "WindDirection_Horizontal_MET_mean",
    "Horizontal_MET_mean",
    "Packets_WithRain_pct",
]


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def wrap_180(angle_deg):
    """Wrap degrees to [-180, 180)."""
    a = np.asarray(angle_deg, dtype=float)
    return (a + 180.0) % 360.0 - 180.0


def _as_float_array(x):
    return np.asarray(x, dtype=float)


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


def _ceil_timestamp(ts: pd.Timestamp, freq: pd.Timedelta) -> pd.Timestamp:
    return pd.Timestamp(ts).ceil(freq)


def _floor_timestamp(ts: pd.Timestamp, freq: pd.Timedelta) -> pd.Timestamp:
    return pd.Timestamp(ts).floor(freq)


def _circular_mean_deg_1d(x) -> float:
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan
    rad = np.deg2rad(x)
    mean_angle = np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
    return float(np.rad2deg(mean_angle) % 360.0)


def _circular_mean_signed_deg_1d(x) -> float:
    return float(wrap_180(_circular_mean_deg_1d(x)))


def _circular_mean_deg_df(df: pd.DataFrame, axis: int = 0) -> pd.Series:
    rad = np.deg2rad(df.astype(float))
    sin_mean = np.nanmean(np.sin(rad), axis=axis)
    cos_mean = np.nanmean(np.cos(rad), axis=axis)
    idx = df.columns if axis == 0 else df.index
    return pd.Series(np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360.0, index=idx)


def _circular_mean_signed_deg_df(df: pd.DataFrame, axis: int = 0) -> pd.Series:
    return wrap_180(_circular_mean_deg_df(df, axis=axis))


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


def _choose_existing_column(df: pd.DataFrame, candidates: Sequence[str], *, required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of the candidate columns exist: {list(candidates)}")
    return None


def _series_or_default(df: pd.DataFrame, candidates: Sequence[str], default, n: int) -> np.ndarray:
    c = _choose_existing_column(df, candidates, required=False)
    if c is None:
        if np.isscalar(default):
            return np.full(n, default, dtype=float)
        arr = np.asarray(default, dtype=float)
        if arr.size != n:
            raise ValueError(f"Default array has length {arr.size}, expected {n}")
        return arr
    return pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)

def _require_columns(df: pd.DataFrame, columns: Sequence[str], source_name: str) -> None:
    """Raise a clear error if any required hard-coded columns are absent."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required {source_name} columns: {missing}. "
            "orsted2python now uses explicit hard-coded Ørsted column names "
            "and does not fall back to alternative variables."
        )


# -----------------------------------------------------------------------------
# Raw file discovery and loading
# -----------------------------------------------------------------------------

def _resolve_raw_paths(
    root: str | Path,
    *,
    scada_dir: str | Path | None = None,
    nbl_dir: str | Path | None = None,
    zx_nbl_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve raw SCADA and lidar folders.

    Preferred usage is explicit:
        load_orsted_data(..., scada_dir=..., nbl_dir=..., zx_nbl_dir=...)

    Auto-discovery supports common layouts:
        <root>/hfdata ROW01
        <root>/LiDAR data/LiDAR on A04
        <root>/LiDAR data/ZX LiDAR on A04
    or direct root folders containing the files.
    """
    root = Path(root)

    if scada_dir is None:
        candidates = [
            root,
            root / "hfdata ROW01",
            root / "ROW" / "hfdata ROW01",
        ]
        scada_dir = next((p for p in candidates if p.exists() and any(p.glob("*.parquet"))), None)
    else:
        scada_dir = Path(scada_dir)

    if nbl_dir is None:
        candidates = [
            root,
            root / "LiDAR data" / "LiDAR on A04",
            root / "ROW" / "LiDAR data" / "LiDAR on A04",
        ]
        nbl_dir = next((p for p in candidates if p.exists() and (p / "RaceBank_A04_WindCubeNBL_WI01030180.csv").exists()), None)
    else:
        nbl_dir = Path(nbl_dir)

    if zx_nbl_dir is None:
        candidates = [
            root,
            root / "LiDAR data" / "ZX LiDAR on A04",
            root / "ROW" / "LiDAR data" / "ZX LiDAR on A04",
        ]
        zx_nbl_dir = next((p for p in candidates if p.exists() and (p / "RaceBank_A04_ZXNBL_zxtm5052.csv").exists()), None)
    else:
        zx_nbl_dir = Path(zx_nbl_dir)

    missing = []
    if scada_dir is None or not Path(scada_dir).exists():
        missing.append("scada_dir")
    if nbl_dir is None or not Path(nbl_dir).exists():
        missing.append("nbl_dir")
    if zx_nbl_dir is None or not Path(zx_nbl_dir).exists():
        missing.append("zx_nbl_dir")
    if missing:
        raise FileNotFoundError(
            "Could not resolve raw Ørsted input folders: "
            + ", ".join(missing)
            + ". Pass scada_dir=..., nbl_dir=..., and zx_nbl_dir=... explicitly."
        )

    return Path(scada_dir), Path(nbl_dir), Path(zx_nbl_dir)


def _list_scada_files(data_path: Path, turbine_id: str = DEFAULT_TURBINE_ID) -> list[Path]:
    files = sorted(Path(data_path).glob("*.parquet"))
    files = [p for p in files if turbine_id in p.name]
    if len(files) == 0:
        raise FileNotFoundError(f"No parquet files containing {turbine_id!r} found in {data_path}")
    parsed = []
    for p in files:
        m = re.search(r"_(\d{4})_(\d{1,2})\.parquet$", p.name)
        year = int(m.group(1)) if m else 9999
        month = int(m.group(2)) if m else 99
        parsed.append((year, month, p))
    return [p for _, _, p in sorted(parsed)]


def _load_single_scada_parquet(file_path: Path) -> pd.DataFrame:
    data_load = pd.read_parquet(file_path)
    time_columns = [c for c in data_load.columns if "Time" in str(c)]
    if len(time_columns) == 0:
        raise ValueError(f"No column containing 'Time' found in {file_path.name}")
    if len(time_columns) > 1:
        raise ValueError(f"More than one time column found in {file_path.name}: {time_columns}")
    time_col = time_columns[0]
    row_times = pd.to_datetime(data_load[time_col], errors="coerce")
    data_vars = data_load.drop(columns=time_columns).copy()
    data_vars = data_vars.loc[row_times.notna()].copy()
    row_times = row_times.loc[row_times.notna()]
    data_vars.index = pd.DatetimeIndex(row_times)
    data_vars.index.name = "Time"
    return data_vars


def _load_scada_parquet_dir(data_path: Path, turbine_id: str = DEFAULT_TURBINE_ID) -> pd.DataFrame:
    chunks = [_load_single_scada_parquet(p) for p in _list_scada_files(data_path, turbine_id=turbine_id)]
    out = pd.concat(chunks, axis=0).sort_index()
    return _standardize_time_index(out)


def _load_single_nbl_csv(path: Path) -> pd.DataFrame:
    df = _read_csv_flexible(path)
    if "Time" in df.columns:
        time_col = "Time"
    else:
        time_columns = [c for c in df.columns if "Time" in str(c)]
        if len(time_columns) != 1:
            raise ValueError(f"Could not identify a unique time column in {path.name}: {time_columns}")
        time_col = time_columns[0]
    row_times = pd.to_datetime(df[time_col], errors="coerce")
    df = df.loc[row_times.notna()].copy()
    row_times = row_times.loc[row_times.notna()]
    df = df.drop(columns=[time_col])
    df.index = pd.DatetimeIndex(row_times)
    df.index.name = "Time"
    return _standardize_time_index(df)


def _make_nbl_file_spec(nbl_dir: Path, zx_nbl_dir: Path) -> dict:
    spec = {}
    for key, meta in DEFAULT_NBL_FILES.items():
        base = zx_nbl_dir if meta["subdir_kind"] == "zx" else nbl_dir
        spec[key] = {
            "path": base / meta["filename"],
            "instrument": meta["instrument"],
            "device_id": meta["device_id"],
        }
    return spec


def _extract_r_distance_from_name(col: str) -> float | None:
    m = re.search(r"r(\d+(?:\.\d+)?)", str(col), flags=re.IGNORECASE)
    return None if m is None else float(m.group(1))


def _extract_h_height_from_name(col: str) -> float | None:
    m = re.search(r"h(\d+(?:\.\d+)?)", str(col), flags=re.IGNORECASE)
    return None if m is None else float(m.group(1))


def _extract_beam_number_from_name(col: str) -> int | None:
    m = re.search(r"Beam0?(\d+)", str(col), flags=re.IGNORECASE)
    return None if m is None else int(m.group(1))


def _classify_nbl_columns(columns, instrument: str) -> pd.DataFrame:
    rows = []
    for col in columns:
        col_str = str(col)
        col_lower = col_str.lower()
        r_distance = _extract_r_distance_from_name(col_str)
        h_height = _extract_h_height_from_name(col_str)
        beam = _extract_beam_number_from_name(col_str)

        is_relative_direction = ("relativedirection" in col_lower or "relative_direction" in col_lower)
        is_wind_speed = (
            "windspeed" in col_lower
            or "wind_speed" in col_lower
            or "horizontal_wind_speed" in col_lower
            or ("wind" in col_lower and "speed" in col_lower)
        )
        is_wind_direction = (
            is_relative_direction
            or "winddirection" in col_lower
            or "wind_direction" in col_lower
            or ("wind" in col_lower and ("dir" in col_lower or "direction" in col_lower))
        )
        is_ti = (
            col_lower == "ti"
            or col_lower.startswith("ti_")
            or "_ti_" in col_lower
            or "ti_h" in col_lower
            or "turbulence" in col_lower
        )

        rows.append({
            "column": col_str,
            "instrument": instrument,
            "r_distance_from_lidar_m": r_distance,
            "h_height_msl_m": h_height if instrument.lower() == "zx" else None,
            "beam": beam,
            "is_wind_speed": is_wind_speed,
            "is_wind_direction": is_wind_direction,
            "is_relative_direction": is_relative_direction,
            "is_ti": is_ti,
        })
    return pd.DataFrame(rows)


def _load_nbl_raw_files(nbl_dir: Path, zx_nbl_dir: Path, required_keys: Sequence[str]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    spec = _make_nbl_file_spec(nbl_dir, zx_nbl_dir)
    keys_to_load = sorted(set(required_keys))
    nbl_raw = {}
    inventories = []
    for key in keys_to_load:
        if key not in spec:
            raise KeyError(f"Unknown lidar key {key!r}. Available: {list(spec)}")
        path = spec[key]["path"]
        if not path.exists():
            raise FileNotFoundError(path)
        df = _load_single_nbl_csv(path)
        nbl_raw[key] = df
        inv = _classify_nbl_columns([c for c in df.columns if not str(c).startswith("__")], spec[key]["instrument"])
        inv.insert(0, "nbl_key", key)
        inventories.append(inv)
    all_inventory = pd.concat(inventories, axis=0, ignore_index=True)
    return nbl_raw, all_inventory


# -----------------------------------------------------------------------------
# Resampling
# -----------------------------------------------------------------------------

def _resample_numeric_to_grid(df: pd.DataFrame, grid_index: pd.DatetimeIndex, freq: pd.Timedelta) -> pd.DataFrame:
    return (
        df.select_dtypes(include=[np.number])
        .resample(freq, origin=grid_index[0], label="left", closed="left")
        .mean()
        .reindex(grid_index)
    )


def _resample_direction_to_grid(
    df: pd.DataFrame,
    columns: Sequence[str],
    grid_index: pd.DatetimeIndex,
    freq: pd.Timedelta,
    already_relative: bool = False,
) -> pd.DataFrame:
    out = pd.DataFrame(index=grid_index)
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"{col!r} not found in dataframe.")
        grouped = pd.to_numeric(df[col], errors="coerce").resample(freq, origin=grid_index[0], label="left", closed="left")
        out[col] = grouped.apply(_circular_mean_signed_deg_1d if already_relative else _circular_mean_deg_1d).reindex(grid_index)
    return out


def _resample_profile_numeric_to_grid(profile_df: pd.DataFrame, grid_index: pd.DatetimeIndex, freq: pd.Timedelta) -> pd.DataFrame:
    out = profile_df.resample(freq, origin=grid_index[0], label="left", closed="left").mean().reindex(grid_index)
    out.columns = np.asarray(out.columns, dtype=float)
    return out


def _resample_profile_direction_to_grid(
    profile_df: pd.DataFrame,
    grid_index: pd.DatetimeIndex,
    freq: pd.Timedelta,
    already_relative: bool = False,
) -> pd.DataFrame:
    out = pd.DataFrame(index=grid_index)
    func = _circular_mean_signed_deg_1d if already_relative else _circular_mean_deg_1d
    for col in profile_df.columns:
        grouped = pd.to_numeric(profile_df[col], errors="coerce").resample(freq, origin=grid_index[0], label="left", closed="left")
        out[float(col)] = grouped.apply(func).reindex(grid_index)
    return out


def _resample_lidar_scalar_columns_to_grid(
    df: pd.DataFrame,
    columns: Sequence[str],
    grid_index: pd.DatetimeIndex,
    freq: pd.Timedelta,
) -> pd.DataFrame:
    out = pd.DataFrame(index=grid_index)
    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        col_lower = str(col).lower()
        is_direction = any(token in col_lower for token in ["direction", "bearing", "yaw", "heading"])
        grouped = s.resample(freq, origin=grid_index[0], label="left", closed="left")
        out[col] = grouped.apply(_circular_mean_deg_1d).reindex(grid_index) if is_direction else grouped.mean().reindex(grid_index)
    return out


def _make_common_grid(
    scada: pd.DataFrame,
    lidar: pd.DataFrame,
    *,
    avg_window_minutes: float | None,
) -> tuple[pd.DatetimeIndex, pd.Timedelta, dict]:
    scada_dt = _infer_base_dt(scada.index)
    lidar_dt = _infer_base_dt(lidar.index)
    native_max = max(scada_dt, lidar_dt)

    if avg_window_minutes is None:
        target_dt = native_max
    else:
        target_dt = pd.Timedelta(minutes=float(avg_window_minutes))
        # Must not be smaller than any source timestep, i.e. must be >= coarsest native dt.
        if target_dt < native_max:
            raise ValueError(
                "For Ørsted, filters['avg_window'] is a time window in minutes and must be "
                "at least as large as the coarsest native timestep among SCADA and lidar.\n"
                f"SCADA native timestep: {scada_dt}\n"
                f"Lidar native timestep: {lidar_dt}\n"
                f"Requested avg_window: {target_dt}\n"
                f"Use avg_window >= {native_max / pd.Timedelta(minutes=1):.3g} minutes."
            )

    common_start = max(scada.index.min(), lidar.index.min()).ceil(target_dt)
    common_end = min(scada.index.max(), lidar.index.max()).floor(target_dt)
    if common_end <= common_start:
        raise ValueError(
            "No overlapping SCADA/lidar time window after snapping to target grid.\n"
            f"SCADA range: {scada.index.min()} to {scada.index.max()}\n"
            f"Lidar range: {lidar.index.min()} to {lidar.index.max()}\n"
            f"Target dt: {target_dt}"
        )
    grid = pd.date_range(start=common_start, end=common_end, freq=target_dt, inclusive="left")
    diagnostics = {
        "scada_native_dt": scada_dt,
        "lidar_native_dt": lidar_dt,
        "target_dt": target_dt,
        "common_start": common_start,
        "common_end": common_end,
        "n_windows": len(grid),
    }
    return grid, target_dt, diagnostics


# -----------------------------------------------------------------------------
# Profile extraction/features
# -----------------------------------------------------------------------------

def _is_mean_signal(col: str) -> bool:
    s = str(col).lower()
    if any(token in s for token in ["std", "_min", "_max", "minimum", "maximum"]):
        return False
    if "mean" in s or "avg" in s or "average" in s:
        return True
    return True


def _select_zx_profile_columns(
    all_inventory: pd.DataFrame,
    nbl_key: str,
    quantity: str,
    preferred_range_m: float | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    inv = all_inventory.copy()
    inv = inv[
        (inv["nbl_key"] == nbl_key)
        & (inv["instrument"].str.lower() == "zx")
        & (inv["h_height_msl_m"].notna())
    ].copy()

    if quantity == "speed":
        inv = inv[inv["is_wind_speed"]].copy()
    elif quantity == "direction":
        if "is_relative_direction" in inv.columns:
            rel = inv[inv["is_relative_direction"]].copy()
            inv = rel if not rel.empty else inv[inv["is_wind_direction"]].copy()
        else:
            inv = inv[inv["is_wind_direction"]].copy()
    elif quantity == "ti":
        inv = inv[inv["is_ti"]].copy()
    else:
        raise ValueError("quantity must be 'speed', 'direction', or 'ti'.")

    inv = inv[inv["column"].map(_is_mean_signal)].copy()
    if inv.empty:
        if strict:
            raise ValueError(f"No ZX {quantity} profile columns found for {nbl_key}.")
        return pd.DataFrame(columns=["column", "h_height_msl_m", "r_distance_from_lidar_m"])

    if preferred_range_m is not None and inv["r_distance_from_lidar_m"].notna().any():
        ranges = np.sort(inv["r_distance_from_lidar_m"].dropna().unique())
        nearest_range = ranges[np.argmin(np.abs(ranges - preferred_range_m))]
        inv = inv[inv["r_distance_from_lidar_m"] == nearest_range].copy()

    return inv[["column", "h_height_msl_m", "r_distance_from_lidar_m"]].sort_values(
        ["h_height_msl_m", "column"]
    )


def _build_zx_numeric_profile_from_columns(df: pd.DataFrame, selected_cols: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for h, group in selected_cols.groupby("h_height_msl_m"):
        cols = [c for c in group["column"].tolist() if c in df.columns]
        if len(cols) == 0:
            continue
        tmp = df[cols].apply(pd.to_numeric, errors="coerce")
        series = tmp[cols[0]] if len(cols) == 1 else tmp.mean(axis=1)
        series.name = float(h)
        pieces.append(series)
    if len(pieces) == 0:
        raise ValueError("No selected columns were present in the raw lidar dataframe.")
    profile = pd.concat(pieces, axis=1)
    return profile.reindex(sorted(profile.columns), axis=1)


def _build_zx_direction_profile_from_columns(
    df: pd.DataFrame,
    selected_cols: pd.DataFrame,
    already_relative: bool,
) -> pd.DataFrame:
    pieces = []
    for h, group in selected_cols.groupby("h_height_msl_m"):
        cols = [c for c in group["column"].tolist() if c in df.columns]
        if len(cols) == 0:
            continue
        tmp = df[cols].apply(pd.to_numeric, errors="coerce")
        if len(cols) == 1:
            series = tmp[cols[0]]
        else:
            series = _circular_mean_signed_deg_df(tmp, axis=1) if already_relative else _circular_mean_deg_df(tmp, axis=1)
        series.name = float(h)
        pieces.append(series)
    if len(pieces) == 0:
        raise ValueError("No selected direction columns were present in the raw lidar dataframe.")
    profile = pd.concat(pieces, axis=1)
    return profile.reindex(sorted(profile.columns), axis=1)


def _interpolate_profile_at_height(profile_df: pd.DataFrame, target_height: float) -> pd.Series:
    heights = np.asarray(profile_df.columns, dtype=float)
    values = profile_df.to_numpy(dtype=float)
    out = np.full(values.shape[0], np.nan)
    for i in range(values.shape[0]):
        row = values[i, :]
        ok = np.isfinite(row) & np.isfinite(heights)
        if ok.sum() >= 2:
            order = np.argsort(heights[ok])
            out[i] = np.interp(target_height, heights[ok][order], row[ok][order])
    return pd.Series(out, index=profile_df.index)


def _compute_top_bottom_veer_rate(rel_dir_profile_deg: pd.DataFrame, hub_height: float, diameter: float) -> pd.Series:
    z_bottom = hub_height - diameter / 2.0
    z_top = hub_height + diameter / 2.0
    theta_bottom = _interpolate_profile_at_height(rel_dir_profile_deg, z_bottom)
    theta_top = _interpolate_profile_at_height(rel_dir_profile_deg, z_top)
    return pd.Series(wrap_180(theta_top - theta_bottom) / diameter, index=rel_dir_profile_deg.index)


def _compute_power_law_shear_alpha(speed_profile: pd.DataFrame, hub_height: float, diameter: float) -> pd.Series:
    heights = np.asarray(speed_profile.columns, dtype=float)
    z_bottom = hub_height - diameter / 2.0
    z_top = hub_height + diameter / 2.0
    in_rotor = (heights >= z_bottom) & (heights <= z_top)
    z = heights[in_rotor]
    if len(z) < 2:
        raise ValueError("Need at least two profile heights inside rotor to fit shear.")
    x = np.log(z / hub_height)
    values = speed_profile.iloc[:, in_rotor].to_numpy(dtype=float)
    alpha = np.full(values.shape[0], np.nan)
    for i in range(values.shape[0]):
        u = values[i, :]
        ok = np.isfinite(u) & (u > 0) & np.isfinite(x)
        if ok.sum() >= 2:
            alpha[i] = np.polyfit(x[ok], np.log(u[ok]), deg=1)[0]
    return pd.Series(alpha, index=speed_profile.index, name="alpha")


def _rotor_layer_mean(profile_df: pd.DataFrame, hub_height: float, diameter: float) -> pd.Series:
    heights = np.asarray(profile_df.columns, dtype=float)
    in_rotor = (heights >= hub_height - diameter / 2.0) & (heights <= hub_height + diameter / 2.0)
    if not np.any(in_rotor):
        return pd.Series(np.nan, index=profile_df.index)
    return profile_df.iloc[:, in_rotor].mean(axis=1)


def _make_profile_features(
    speed_profile: pd.DataFrame,
    rel_dir_profile: pd.DataFrame,
    source_name: str,
    hub_height: float,
    diameter: float,
    ti_profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=speed_profile.index)
    out[f"{source_name}_U_hub"] = _interpolate_profile_at_height(speed_profile, hub_height)
    out[f"{source_name}_U_rotor_mean"] = _rotor_layer_mean(speed_profile, hub_height, diameter)
    out[f"{source_name}_alpha"] = _compute_power_law_shear_alpha(speed_profile, hub_height, diameter)

    z_bottom = hub_height - diameter / 2.0
    z_top = hub_height + diameter / 2.0
    out[f"{source_name}_rel_dir_hub_deg"] = wrap_180(_interpolate_profile_at_height(rel_dir_profile, hub_height))
    out[f"{source_name}_rel_dir_bottom_deg"] = wrap_180(_interpolate_profile_at_height(rel_dir_profile, z_bottom))
    out[f"{source_name}_rel_dir_top_deg"] = wrap_180(_interpolate_profile_at_height(rel_dir_profile, z_top))
    out[f"{source_name}_veer_rate_deg_per_m"] = _compute_top_bottom_veer_rate(rel_dir_profile, hub_height, diameter)
    out[f"{source_name}_V"] = out[f"{source_name}_veer_rate_deg_per_m"] * np.pi / 180.0 * (diameter / 2.0)

    if ti_profile is not None:
        out[f"{source_name}_TI_hub"] = _interpolate_profile_at_height(ti_profile, hub_height)
        out[f"{source_name}_TI_rotor_mean"] = _rotor_layer_mean(ti_profile, hub_height, diameter)
        # Historical convenience name used by some notebooks.
        out[f"{source_name}_T_mean"] = out[f"{source_name}_TI_hub"]

    return out


def _robust_angle_center_deg(angle_deg: pd.Series) -> float:
    x = pd.Series(wrap_180(angle_deg)).dropna()
    if len(x) == 0:
        return np.nan
    return float(np.nanmedian(x))


def _add_yaw_diagnostics(df: pd.DataFrame, lidar_key: str, center_yaw: bool = True) -> pd.DataFrame:
    out = df.copy()
    out["local_yaw_wrapped_deg"] = wrap_180(out["local_yaw_deg"])
    scada_center = _robust_angle_center_deg(out["local_yaw_wrapped_deg"]) if center_yaw else 0.0
    out["local_yaw_center_deg"] = scada_center
    out["local_yaw_centered_deg"] = wrap_180(out["local_yaw_wrapped_deg"] - scada_center)

    hub_col = f"{lidar_key}_rel_dir_hub_deg"
    out[f"{lidar_key}_rel_dir_hub_wrapped_deg"] = wrap_180(out[hub_col])
    lidar_center = _robust_angle_center_deg(out[f"{lidar_key}_rel_dir_hub_wrapped_deg"]) if center_yaw else 0.0
    out[f"{lidar_key}_rel_dir_hub_center_deg"] = lidar_center
    out[f"{lidar_key}_rel_dir_hub_centered_deg"] = wrap_180(out[f"{lidar_key}_rel_dir_hub_wrapped_deg"] - lidar_center)
    out[f"{lidar_key}_lidar_minus_scada_yaw_deg"] = wrap_180(
        out[f"{lidar_key}_rel_dir_hub_centered_deg"] - out["local_yaw_centered_deg"]
    )
    return out


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

def load_orsted_data(
    root: str | Path,
    filters: Mapping | None = None,
    *,
    lidar_key: str = DEFAULT_LIDAR_KEY,
    yaw_mode: str = "lidar",
    scada_dir: str | Path | None = None,
    nbl_dir: str | Path | None = None,
    zx_nbl_dir: str | Path | None = None,
    turbine_id: str = DEFAULT_TURBINE_ID,
    R: float = DEFAULT_R,
    Hub: float = DEFAULT_HUB,
    HubR: float = DEFAULT_HUBR,
    B: int = DEFAULT_B,
    rho: float | Sequence[float] = DEFAULT_RHO,
    turbine_name: str = "Racebank_A04",
    power_col: str | None = None,
    pcurve_u: Sequence[float] | None = None,
    pcurve_p: Sequence[float] | None = None,
    preferred_lidar_range_m: float | None = None,
    lidar_direction_is_already_relative: bool | None = None,
    center_yaw: bool = True,
    return_intermediate: bool = False,
) -> dict:
    """Load raw Ørsted SCADA/lidar data and return an India-style dictionary.

    Parameters
    ----------
    root : path-like
        Root directory used for auto-discovery.  For robustness, you may pass
        ``scada_dir``, ``nbl_dir``, and ``zx_nbl_dir`` explicitly.
    filters : mapping, optional
        Same filter style as ``matlab2python.py``.  Supports ``<field>_min``,
        ``<field>_max``, plus ``avg_window`` and ``window_min``.  For Ørsted,
        ``avg_window`` is a time window in minutes.
    lidar_key : str
        Selected ZX lidar. Default is ``ZX_zxtm5052``.
    yaw_mode : {"lidar", "lidar_centered", "scada", "scada_centered", "lidar_minus_scada"}
        Which yaw definition is exposed as India-compatible ``hubdir``.
    preferred_lidar_range_m : float, optional
        Select ZX range nearest this distance.  Defaults to ``3.5 * diameter``.
    lidar_direction_is_already_relative : bool, optional
        If False, selected lidar direction profiles are converted to relative
        direction by subtracting SCADA nacelle heading.
    return_intermediate : bool
        If True, include additional pandas DataFrames under ``data["_intermediate"]``.

    Returns
    -------
    dict
        India-style data dictionary with Ørsted-specific yaw diagnostics.
    """
    root = Path(root)
    filters = dict(filters or {})
    scada_dir, nbl_dir, zx_nbl_dir = _resolve_raw_paths(root, scada_dir=scada_dir, nbl_dir=nbl_dir, zx_nbl_dir=zx_nbl_dir)

    if lidar_key not in DEFAULT_NBL_FILES:
        raise KeyError(f"Unknown lidar_key {lidar_key!r}. Available: {list(DEFAULT_NBL_FILES)}")
    if DEFAULT_NBL_FILES[lidar_key]["instrument"].lower() != "zx":
        raise ValueError("This India-style engineering loader currently supports ZX lidar profile sources.")

    diameter = 2.0 * float(R)
    preferred_lidar_range_m = 3.5 * diameter if preferred_lidar_range_m is None else preferred_lidar_range_m
    if lidar_direction_is_already_relative is None:
        lidar_direction_is_already_relative = DEFAULT_LIDAR_DIRECTION_IS_ALREADY_RELATIVE.get(lidar_key, True)

    # 1. Load raw sources.
    scada = _load_scada_parquet_dir(scada_dir, turbine_id=turbine_id)
    nbl_raw, inventory = _load_nbl_raw_files(nbl_dir, zx_nbl_dir, required_keys=[lidar_key])
    lidar_raw = nbl_raw[lidar_key]

    # 2. Build common time grid directly from raw native timesteps.
    avg_window = filters.get("avg_window", None)
    grid, target_dt, time_diag = _make_common_grid(scada, lidar_raw, avg_window_minutes=avg_window)

    # 3. Resample SCADA to common grid.
    scada_num = _resample_numeric_to_grid(scada, grid_index=grid, freq=target_dt)
    scada_dir_df = _resample_direction_to_grid(
        scada,
        columns=[NACELLE_HEADING_COL, SCADA_WIND_DIR_COL],
        grid_index=grid,
        freq=target_dt,
        already_relative=False,
    )

    scada_ref = pd.DataFrame(index=grid)

    # Hard-coded SCADA extraction.  Do not fall back to substitute variables.
    if power_col is None:
        power_col = POWER_COL
    required_numeric_scada_cols = [
        power_col,
        TURBINE_HUB_SPEED_COL,
        GENERATOR_RPM_COL,
        PITCH_COL,
        TURBINE_TI_COL,
        AMBIENT_TEMP_COL,
    ]
    required_direction_scada_cols = [NACELLE_HEADING_COL, SCADA_WIND_DIR_COL]
    _require_columns(scada_num, required_numeric_scada_cols, "SCADA numeric")
    _require_columns(scada_dir_df, required_direction_scada_cols, "SCADA direction")

    # User-specified meanings:
    #   AcWindSp_AcWindSp_mean -> turbine hub speed
    #   AcWindDr_Value_mean    -> turbine hub wind direction, absolute
    #   GenRpm_Value_mean      -> direct-drive rotor/generator rpm
    #   PitcPosA_Value_mean    -> pitch [deg]
    #   TurbEst_TurbEst_mean   -> turbine TI, as provided (not percent)
    #   AmbieTmp_Value_mean    -> temperature [C]
    #   NacelPos_Value_mean    -> turbine/nacelle heading [deg]
    #   ActPower_Value_mean    -> turbine power [kW]
    scada_ref["P_obs"] = scada_num[power_col]
    scada_ref["scada_power"] = scada_num[power_col]
    scada_ref["turbine_hub_speed"] = scada_num[TURBINE_HUB_SPEED_COL]
    scada_ref["turbine_hub_direction_deg"] = scada_dir_df[SCADA_WIND_DIR_COL]
    scada_ref["generator_rpm"] = scada_num[GENERATOR_RPM_COL]
    scada_ref["pitch_deg"] = scada_num[PITCH_COL]
    scada_ref["turbine_TI"] = scada_num[TURBINE_TI_COL]
    scada_ref["ambient_temp_c"] = scada_num[AMBIENT_TEMP_COL]
    scada_ref["nacelle_heading_deg"] = scada_dir_df[NACELLE_HEADING_COL]
    scada_ref["scada_wind_dir_deg"] = scada_dir_df[SCADA_WIND_DIR_COL]
    scada_ref["local_yaw_deg"] = wrap_180(scada_ref["scada_wind_dir_deg"] - scada_ref["nacelle_heading_deg"])

    # 4. Extract and resample ZX profiles from raw lidar.
    speed_cols = _select_zx_profile_columns(inventory, lidar_key, "speed", preferred_range_m=preferred_lidar_range_m, strict=True)
    dir_cols = _select_zx_profile_columns(inventory, lidar_key, "direction", preferred_range_m=preferred_lidar_range_m, strict=True)
    ti_cols = _select_zx_profile_columns(inventory, lidar_key, "ti", preferred_range_m=preferred_lidar_range_m, strict=False)

    speed_raw = _build_zx_numeric_profile_from_columns(lidar_raw, speed_cols)
    dir_raw = _build_zx_direction_profile_from_columns(lidar_raw, dir_cols, already_relative=lidar_direction_is_already_relative)
    ti_raw = None if ti_cols.empty else _build_zx_numeric_profile_from_columns(lidar_raw, ti_cols)

    speed_profile = _resample_profile_numeric_to_grid(speed_raw, grid_index=grid, freq=target_dt)
    dir_profile = _resample_profile_direction_to_grid(
        dir_raw,
        grid_index=grid,
        freq=target_dt,
        already_relative=lidar_direction_is_already_relative,
    )
    ti_profile = None
    if ti_raw is not None:
        ti_profile = _resample_profile_numeric_to_grid(ti_raw, grid_index=grid, freq=target_dt)

    if lidar_direction_is_already_relative:
        rel_dir_profile = pd.DataFrame(wrap_180(dir_profile.to_numpy(dtype=float)), index=dir_profile.index, columns=dir_profile.columns)
    else:
        rel_dir_profile = pd.DataFrame(
            wrap_180(dir_profile.sub(scada_ref["nacelle_heading_deg"], axis=0).to_numpy(dtype=float)),
            index=dir_profile.index,
            columns=dir_profile.columns,
        )

    features = _make_profile_features(
        speed_profile,
        rel_dir_profile,
        source_name=lidar_key,
        hub_height=float(Hub),
        diameter=diameter,
        ti_profile=ti_profile,
    )

    lidar_scalar_grid = _resample_lidar_scalar_columns_to_grid(
        lidar_raw,
        columns=LIDAR_SCALAR_COLUMNS_TO_RETAIN,
        grid_index=grid,
        freq=target_dt,
    )
    for col in lidar_scalar_grid.columns:
        features[f"{lidar_key}_{col}"] = lidar_scalar_grid[col]

    # 5. Assemble unified scalar table and yaw diagnostics.
    table = scada_ref.join(features, how="inner")
    table = _add_yaw_diagnostics(table, lidar_key=lidar_key, center_yaw=center_yaw)

    # 6. Assemble India-style dictionary.
    n = len(table)
    if n == 0:
        raise ValueError("No samples remained after raw SCADA/lidar resampling and joining.")

    heights = np.asarray(speed_profile.columns, dtype=float)
    hubspeed = pd.to_numeric(table[f"{lidar_key}_U_hub"], errors="coerce").to_numpy(dtype=float)
    alpha = pd.to_numeric(table[f"{lidar_key}_alpha"], errors="coerce").to_numpy(dtype=float)
    dsrate = pd.to_numeric(table[f"{lidar_key}_veer_rate_deg_per_m"], errors="coerce").to_numpy(dtype=float)

    power = pd.to_numeric(table["P_obs"], errors="coerce").to_numpy(dtype=float)  # kW
    Thubspeed = pd.to_numeric(table["turbine_hub_speed"], errors="coerce").to_numpy(dtype=float)
    pitch_deg = pd.to_numeric(table["pitch_deg"], errors="coerce").to_numpy(dtype=float)
    generator_rpm = pd.to_numeric(table["generator_rpm"], errors="coerce").to_numpy(dtype=float)
    omega_rad_s = generator_rpm * 2.0 * np.pi / 60.0
    tsr = omega_rad_s * float(R) / hubspeed

    # Turbine TI is explicitly TurbEst_TurbEst_mean, as provided in the SCADA
    # file.  It is not converted to percent and is not replaced by lidar TI.
    ti = 100.0 * pd.to_numeric(table["turbine_TI"], errors="coerce").to_numpy(dtype=float)
    ambient_temp_c = pd.to_numeric(table["ambient_temp_c"], errors="coerce").to_numpy(dtype=float)
    turbine_hub_direction_deg = pd.to_numeric(table["turbine_hub_direction_deg"], errors="coerce").to_numpy(dtype=float)
    nacelle_heading_deg = pd.to_numeric(table["nacelle_heading_deg"], errors="coerce").to_numpy(dtype=float)

    yaw_lidar_hub_deg = wrap_180(pd.to_numeric(table[f"{lidar_key}_rel_dir_hub_deg"], errors="coerce").to_numpy(dtype=float))
    yaw_lidar_centered_deg = pd.to_numeric(table[f"{lidar_key}_rel_dir_hub_centered_deg"], errors="coerce").to_numpy(dtype=float)
    yaw_scada_nacelle_deg = pd.to_numeric(table["local_yaw_deg"], errors="coerce").to_numpy(dtype=float)
    yaw_scada_centered_deg = pd.to_numeric(table["local_yaw_centered_deg"], errors="coerce").to_numpy(dtype=float)
    yaw_lidar_minus_scada_deg = pd.to_numeric(table[f"{lidar_key}_lidar_minus_scada_yaw_deg"], errors="coerce").to_numpy(dtype=float)

    yaw_options = {
        "lidar": yaw_lidar_hub_deg,
        "lidar_centered": yaw_lidar_centered_deg,
        "scada": yaw_scada_nacelle_deg,
        "scada_centered": yaw_scada_centered_deg,
        "lidar_minus_scada": yaw_lidar_minus_scada_deg,
    }
    if yaw_mode not in yaw_options:
        raise ValueError(f"yaw_mode must be one of {list(yaw_options)}")
    hubdir = yaw_options[yaw_mode]

    rho_arr = np.full(n, float(rho), dtype=float) if np.isscalar(rho) else np.asarray(rho, dtype=float)
    if rho_arr.size != n:
        raise ValueError(f"rho array length {rho_arr.size} does not match number of cases {n}")

    speed = speed_profile.reindex(table.index).to_numpy(dtype=float).T
    dir_rel = rel_dir_profile.reindex(table.index).to_numpy(dtype=float).T
    ti_prof_arr = ti_profile.reindex(table.index).to_numpy(dtype=float).T if ti_profile is not None else np.full_like(speed, np.nan)

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
        "omega": generator_rpm,       # rpm
        "omega_rad_s": omega_rad_s,
        "rho": rho_arr,
        "flag_indx": np.ones(n, dtype=float),
        "Thubspeed": Thubspeed,
        "turbine_hub_speed": Thubspeed,
        "turbine_hub_direction_deg": turbine_hub_direction_deg,
        "nacelle_heading_deg": nacelle_heading_deg,
        "ambient_temp_c": ambient_temp_c,
        "tsr": tsr,
        "ti": ti,
        "power": power,
        "Pcurve_U": np.asarray([] if pcurve_u is None else pcurve_u, dtype=float),
        "Pcurve_P": np.asarray([] if pcurve_p is None else pcurve_p, dtype=float),
        "time": _numeric_time(table.index),

        # Aliases expected by downstream Python workflows
        "speed_profiles": speed,
        "dir_profiles": dir_rel,
        "ti_profiles": ti_prof_arr,
        "veer_deg_per_m": dsrate,
        "tsr_data": tsr,
        "alpha_data": alpha,
        "hubheight": float(Hub),
        "turbinePower": power,
        "TI": ti,
        "nH": int(len(heights)),
        "nCases": int(n),

        # Ørsted diagnostics
        "lidar_key": lidar_key,
        "yaw_mode": yaw_mode,
        "yaw_lidar_hub_deg": yaw_lidar_hub_deg,
        "yaw_lidar_centered_deg": yaw_lidar_centered_deg,
        "yaw_scada_nacelle_deg": yaw_scada_nacelle_deg,
        "yaw_scada_centered_deg": yaw_scada_centered_deg,
        "yaw_lidar_minus_scada_deg": yaw_lidar_minus_scada_deg,

        # Short yaw aliases so users can compare all definitions after one load.
        # ``yaw_mode`` only controls which one is copied into ``hubdir``.
        "yaw_lidar": yaw_lidar_hub_deg,
        "yaw_lidar_centered": yaw_lidar_centered_deg,
        "yaw_scada": yaw_scada_nacelle_deg,
        "yaw_scada_centered": yaw_scada_centered_deg,
        "yaw_lidar_minus_scada": yaw_lidar_minus_scada_deg,

        "nacelle_heading_deg": pd.to_numeric(table["nacelle_heading_deg"], errors="coerce").to_numpy(dtype=float),
        "scada_wind_dir_deg": pd.to_numeric(table["scada_wind_dir_deg"], errors="coerce").to_numpy(dtype=float),
        "source_index": table.index.astype(str).to_numpy(),
        "resample_native_scada_dt_seconds": float(time_diag["scada_native_dt"].total_seconds()),
        "resample_native_lidar_dt_seconds": float(time_diag["lidar_native_dt"].total_seconds()),
        "resample_target_dt_seconds": float(time_diag["target_dt"].total_seconds()),
    }

    # 7. Apply filters after unified structure is assembled.
    mask = _case_mask(data, filters)
    data = _apply_case_mask(data, mask)

    # 8. Derived CP, using power [kW] and Paero [kW].
    try:
        p_aero_kw = 0.5 * np.asarray(data["rho"]) * np.pi * float(R) ** 2 * np.asarray(data["hubspeed"]) ** 3 * 1e-3
        data["turbine_CP"] = np.asarray(data["power"]) / p_aero_kw
    except Exception:
        data["turbine_CP"] = np.full(int(data["nCases"]), np.nan)

    if return_intermediate:
        data["_intermediate"] = {
            "scada_raw": scada,
            "lidar_raw": lidar_raw,
            "inventory": inventory,
            "table_before_filter": table,
            "speed_profile": speed_profile,
            "relative_direction_profile": rel_dir_profile,
            "ti_profile": ti_profile,
            "selected_speed_columns": speed_cols,
            "selected_direction_columns": dir_cols,
            "selected_ti_columns": ti_cols,
            "resample_diagnostics": time_diag,
        }

    return data


# -----------------------------------------------------------------------------
# Optional MATLAB export
# -----------------------------------------------------------------------------

def save_india_style_mat(data: Mapping, out_path: str | Path, struct_name: str = "out") -> None:
    """Save a loaded Ørsted dictionary to an India-style MATLAB .mat file."""
    if savemat is None:
        raise ImportError("scipy.io.savemat is unavailable in this environment.")

    out = {}
    for k, v in data.items():
        if k.startswith("_"):
            continue
        if k in {"nH", "nCases", "speed_profiles", "dir_profiles", "ti_profiles", "turbinePower", "turbine_CP", "alpha_data", "tsr_data", "hubheight"}:
            continue
        arr = np.asarray(v)
        if arr.ndim == 1 and k not in {"heights", "Pcurve_U", "Pcurve_P"}:
            out[k] = arr.reshape(1, -1)
        elif k == "heights":
            out[k] = arr.reshape(-1, 1)
        else:
            out[k] = v
    savemat(out_path, {struct_name: out})


__all__ = [
    "load_orsted_data",
    "save_india_style_mat",
    "wrap_180",
]
