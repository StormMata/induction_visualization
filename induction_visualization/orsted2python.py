"""Ørsted processed-data loader with India-style output schema.

This module is intended to mirror the user-facing pattern of matlab2python.py:

    import orsted2python as o2p
    data = o2p.load_orsted_data(root, filters=mask)

It reads the CSV outputs produced by the Ørsted lidar workflow notebook and
returns a dictionary with the same downstream field names used by the India
wind-farm engineering-model workflow.

Default design choices encoded here are explicit and configurable:
- use ZX_zxtm5052 by default;
- power is interpreted as kW;
- generator speed is direct-drive rotor speed in rpm;
- TSR is computed as (rpm * 2*pi/60) * R / U_hub;
- several yaw definitions are preserved, and `yaw_mode` chooses which one is
  exposed as India-compatible `hubdir`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

try:
    from scipy.io import savemat
except Exception:  # pragma: no cover - scipy may not always be present
    savemat = None


# -----------------------------------------------------------------------------
# Constants / utilities
# -----------------------------------------------------------------------------

DEFAULT_LIDAR_KEY = "ZX_zxtm5052"
DEFAULT_DIAMETER = 154.0       # [m]
DEFAULT_R = DEFAULT_DIAMETER / 2.0
DEFAULT_HUB = 103.3            # [m]
DEFAULT_RHO = 1.225            # [kg/m^3]
DEFAULT_HUBR = np.nan
DEFAULT_B = 3

MAIN_INPUT_FILENAMES = (
    "unified_scada_zx_lidar_inputs.csv",
    "unified_scada_zx_lidar_inputs_filtered.csv",
    "unified_scada_zx_lidar_inputs_all_rows.csv",
)


def wrap_180(angle_deg):
    """Wrap degrees to [-180, 180)."""
    a = np.asarray(angle_deg, dtype=float)
    return (a + 180.0) % 360.0 - 180.0


def _as_1d(x):
    arr = np.asarray(x)
    return np.ravel(arr)


def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _read_time_indexed_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0)
    # Try to recover datetime index. If this fails, keep original index.
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df


def _find_main_input_file(root: Path) -> Path:
    for name in MAIN_INPUT_FILENAMES:
        p = root / name
        if p.exists():
            return p
    expected = ", ".join(MAIN_INPUT_FILENAMES)
    raise FileNotFoundError(f"Could not find any of {expected} in {root}")


def _height_columns_to_float(df: pd.DataFrame) -> np.ndarray:
    heights = []
    for c in df.columns:
        try:
            heights.append(float(c))
        except Exception as exc:
            raise ValueError(f"Profile column {c!r} cannot be converted to a height.") from exc
    return np.asarray(heights, dtype=float)


def _interp_profile_at_height(profile: pd.DataFrame, heights: np.ndarray, z: float) -> np.ndarray:
    out = []
    for _, row in profile.iterrows():
        y = row.to_numpy(dtype=float)
        m = np.isfinite(heights) & np.isfinite(y)
        if np.count_nonzero(m) < 2:
            out.append(np.nan)
        else:
            out.append(float(np.interp(z, heights[m], y[m])))
    return np.asarray(out)


def _circular_mean_deg(values: np.ndarray, axis=0):
    """NaN-aware circular mean in degrees."""
    vals = np.asarray(values, dtype=float)
    rad = np.deg2rad(vals)
    s = np.nanmean(np.sin(rad), axis=axis)
    c = np.nanmean(np.cos(rad), axis=axis)
    return np.rad2deg(np.arctan2(s, c))


def _numeric_time(index: pd.Index) -> np.ndarray:
    """Return a numeric time vector compatible with row-wise workflows."""
    if isinstance(index, pd.DatetimeIndex):
        # Seconds since Unix epoch as float. This preserves gaps and ordering.
        return index.view("int64").astype(float) / 1e9
    return np.arange(len(index), dtype=float)


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


def _load_selected_profiles(root: Path, lidar_key: str):
    speed_path = root / f"{lidar_key}_speed_profile.csv"
    dir_path = root / f"{lidar_key}_relative_direction_profile.csv"
    ti_path = root / f"{lidar_key}_ti_profile.csv"

    speed = _read_time_indexed_csv(speed_path)
    direction = _read_time_indexed_csv(dir_path)
    ti = _read_time_indexed_csv(ti_path) if ti_path.exists() else None

    heights = _height_columns_to_float(speed)
    dir_heights = _height_columns_to_float(direction)
    if not np.allclose(heights, dir_heights, equal_nan=True):
        raise ValueError("Speed and relative-direction profile heights do not match.")
    if ti is not None:
        ti_heights = _height_columns_to_float(ti)
        if not np.allclose(heights, ti_heights, equal_nan=True):
            raise ValueError("Speed and TI profile heights do not match.")

    return speed, direction, ti, heights


# -----------------------------------------------------------------------------
# Window averaging and filtering
# -----------------------------------------------------------------------------

PROFILE_KEYS = {"speed", "dir_rel", "speed_profiles", "dir_profiles", "ti_profiles"}
ANGLE_PROFILE_KEYS = {"dir_rel", "dir_profiles"}


def _average_blocks(data: dict, avg_window: int, window_min: float | None = None) -> dict:
    """Average contiguous samples in fixed-size blocks.

    This is intentionally simple and deterministic. It mirrors the spirit of the
    India loader's `avg_window` option but operates on the already assembled
    dictionary. Numeric case-wise fields are averaged by nanmean; direction
    profile fields are circular-averaged.
    """
    if avg_window is None or int(avg_window) <= 1:
        return data

    avg_window = int(avg_window)
    n = int(data["nCases"])
    keep_blocks = []
    time = np.asarray(data.get("time", np.arange(n)), dtype=float)

    for start in range(0, n - avg_window + 1, avg_window):
        stop = start + avg_window
        # Optional gap check: require total block span to be no larger than
        # avg_window*window_min minutes when time is seconds since epoch.
        if window_min is not None and np.isfinite(window_min) and len(time[start:stop]) > 1:
            span_seconds = np.nanmax(time[start:stop]) - np.nanmin(time[start:stop])
            max_span = float(avg_window) * float(window_min) * 60.0
            if np.isfinite(span_seconds) and span_seconds > max_span:
                continue
        keep_blocks.append(slice(start, stop))

    out = {}
    for key, val in data.items():
        arr = np.asarray(val)

        if key in {"nCases", "nH"}:
            continue

        # Profiles: nH x nCases
        if arr.ndim == 2 and arr.shape[1] == n:
            cols = []
            for sl in keep_blocks:
                block = arr[:, sl]
                if key in ANGLE_PROFILE_KEYS:
                    cols.append(_circular_mean_deg(block, axis=1))
                else:
                    cols.append(np.nanmean(block, axis=1))
            out[key] = np.column_stack(cols) if cols else arr[:, :0]

        # Case-wise vectors: length n
        elif arr.ndim == 1 and arr.size == n:
            vals = []
            for sl in keep_blocks:
                block = arr[sl]
                if key.lower() in {"hubdir", "yaw_lidar_hub_deg", "yaw_scada_nacelle_deg", "yaw_lidar_minus_nacelle_deg"}:
                    vals.append(_circular_mean_deg(block, axis=0))
                else:
                    vals.append(np.nanmean(block))
            out[key] = np.asarray(vals)

        else:
            out[key] = val

    out["nCases"] = len(keep_blocks)
    out["nH"] = int(data["nH"])
    return out


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
            mask &= arr >= float(val)
        elif key.endswith("_max"):
            field = key[:-4]
            if field not in data:
                raise KeyError(f"Filter refers to missing field: {field!r}")
            arr = np.asarray(data[field], dtype=float)
            mask &= arr <= float(val)
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
# Main public loader
# -----------------------------------------------------------------------------


def load_orsted_data(
    root: str | Path,
    filters: Mapping | None = None,
    *,
    lidar_key: str = DEFAULT_LIDAR_KEY,
    yaw_mode: str = "lidar",
    R: float = DEFAULT_R,
    Hub: float = DEFAULT_HUB,
    HubR: float = DEFAULT_HUBR,
    B: int = DEFAULT_B,
    rho: float | Sequence[float] = DEFAULT_RHO,
    turbine_name: str = "Racebank_A04",
    power_col: str | None = None,
    pcurve_u: Sequence[float] | None = None,
    pcurve_p: Sequence[float] | None = None,
) -> dict:
    """Load processed Ørsted lidar/SCADA outputs as an India-style dictionary.

    Parameters
    ----------
    root : path-like
        Directory containing the CSV outputs from `orsted_lidar_workflow_v5.ipynb`.
    filters : mapping, optional
        Same filter style as matlab2python.py. Supports `<field>_min`,
        `<field>_max`, plus `avg_window` and `window_min`.
    lidar_key : str
        Processed lidar key. Default is "ZX_zxtm5052".
    yaw_mode : {"lidar", "lidar_centered", "scada", "scada_centered", "lidar_minus_scada"}
        Which yaw definition is exposed as India-compatible `hubdir`.
    R, Hub, HubR, B : turbine constants
        Stored in the returned dictionary.
    rho : scalar or array
        Density in kg/m^3. If scalar, expanded to all cases.
    turbine_name : str
        Stored in the returned dictionary.
    power_col : str, optional
        Power column in the unified inputs table. If None, use P_obs if present,
        then scada_power. Values are assumed to be kW.
    pcurve_u, pcurve_p : optional sequences
        OEM power curve arrays to include as Pcurve_U/Pcurve_P. If omitted,
        empty arrays are returned so downstream code can explicitly detect that
        no curve was supplied.

    Returns
    -------
    dict
        India-style data dictionary with extra Ørsted yaw diagnostics preserved.
    """
    root = Path(root)
    filters = dict(filters or {})

    main_path = _find_main_input_file(root)
    table = _read_time_indexed_csv(main_path)

    speed_df, dir_df, ti_df, heights = _load_selected_profiles(root, lidar_key)

    # Align all tables on the same timestamps.
    common = table.index.intersection(speed_df.index).intersection(dir_df.index)
    if ti_df is not None:
        common = common.intersection(ti_df.index)
    common = common.sort_values()

    table = table.loc[common].copy()
    speed_df = speed_df.loc[common]
    dir_df = dir_df.loc[common]
    if ti_df is not None:
        ti_df = ti_df.loc[common]

    n = len(table)
    if n == 0:
        raise ValueError("No overlapping timestamps between unified inputs and selected lidar profiles.")

    # Core Ørsted columns.
    hubspeed = _series_or_default(table, [f"{lidar_key}_U_hub"], np.nan, n)
    alpha = _series_or_default(table, [f"{lidar_key}_alpha"], np.nan, n)
    dsrate = _series_or_default(table, [f"{lidar_key}_veer_rate_deg_per_m"], np.nan, n)
    lidar_rel_hub = _series_or_default(table, [f"{lidar_key}_rel_dir_hub_deg"], np.nan, n)

    # Power in kW.
    if power_col is None:
        power_col = _choose_existing_column(table, ["P_obs", "scada_power"], required=True)
    elif power_col not in table.columns:
        raise KeyError(f"Requested power_col={power_col!r} not found in input table.")
    power = pd.to_numeric(table[power_col], errors="coerce").to_numpy(dtype=float)

    # Turbine-estimated hub speed. Fall back in transparent order.
    Thubspeed = _series_or_default(
        table,
        ["turbine_est_wind_speed", "nacelle_wind_speed", "scada_wind_speed", "Thubspeed"],
        hubspeed,
        n,
    )

    # Pitch and rotor/generator speed.
    pitch_deg = _series_or_default(table, ["pitch_deg", "PitcPosA_Value_mean"], np.nan, n)
    generator_rpm = _series_or_default(table, ["generator_rpm", "GenRpm_Value_mean"], np.nan, n)
    omega_rad_s = generator_rpm * 2.0 * np.pi / 60.0
    tsr = omega_rad_s * float(R) / hubspeed

    # TI. Prefer scalar mean from lidar workflow; otherwise interpolate profile at hub.
    ti_candidates = [f"{lidar_key}_T_mean", f"{lidar_key}_TI_hub", "TI", "ti"]
    ti_col = _choose_existing_column(table, ti_candidates, required=False)
    if ti_col is not None:
        ti = pd.to_numeric(table[ti_col], errors="coerce").to_numpy(dtype=float)
    elif ti_df is not None:
        ti = _interp_profile_at_height(ti_df, heights, float(Hub))
    else:
        ti = np.full(n, np.nan)

    # Yaw definitions. Preserve everything available.
    yaw_lidar_hub_deg = wrap_180(lidar_rel_hub)
    yaw_lidar_centered_deg = _series_or_default(
        table,
        [f"{lidar_key}_rel_dir_hub_centered_deg"],
        yaw_lidar_hub_deg,
        n,
    )
    yaw_scada_nacelle_deg = _series_or_default(
        table,
        ["local_yaw_deg"],
        np.nan,
        n,
    )
    yaw_scada_centered_deg = _series_or_default(
        table,
        ["local_yaw_centered_deg"],
        yaw_scada_nacelle_deg,
        n,
    )
    yaw_lidar_minus_scada_deg = _series_or_default(
        table,
        [f"{lidar_key}_lidar_minus_scada_yaw_deg"],
        wrap_180(yaw_lidar_centered_deg - yaw_scada_centered_deg),
        n,
    )

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

    speed = speed_df.to_numpy(dtype=float).T       # nH x nCases
    dir_rel = dir_df.to_numpy(dtype=float).T       # nH x nCases
    ti_profile = ti_df.to_numpy(dtype=float).T if ti_df is not None else np.full_like(speed, np.nan)

    data = {
        # India-style raw/schema fields
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
        # Keep `omega` in rpm for compatibility with your note that GenRpm is direct rotor rpm.
        "omega": generator_rpm,
        "omega_rad_s": omega_rad_s,
        "rho": rho_arr,
        "flag_indx": np.ones(n, dtype=float),
        "Thubspeed": Thubspeed,
        "tsr": tsr,
        "ti": ti,
        "power": power,
        "Pcurve_U": np.asarray([] if pcurve_u is None else pcurve_u, dtype=float),
        "Pcurve_P": np.asarray([] if pcurve_p is None else pcurve_p, dtype=float),
        "time": _numeric_time(table.index),

        # Standardized aliases expected by downstream Python workflows
        "speed_profiles": speed,
        "dir_profiles": dir_rel,
        "ti_profiles": ti_profile,
        "veer_deg_per_m": dsrate,
        "tsr_data": tsr,
        "alpha_data": alpha,
        "hubheight": float(Hub),
        "turbinePower": power,
        "TI": ti,
        "nH": int(len(heights)),
        "nCases": int(n),

        # Ørsted-specific diagnostic fields
        "lidar_key": lidar_key,
        "yaw_mode": yaw_mode,
        "yaw_lidar_hub_deg": yaw_lidar_hub_deg,
        "yaw_lidar_centered_deg": yaw_lidar_centered_deg,
        "yaw_scada_nacelle_deg": yaw_scada_nacelle_deg,
        "yaw_scada_centered_deg": yaw_scada_centered_deg,
        "yaw_lidar_minus_scada_deg": yaw_lidar_minus_scada_deg,
        "nacelle_heading_deg": _series_or_default(table, ["nacelle_heading_deg"], np.nan, n),
        "scada_wind_dir_deg": _series_or_default(table, ["scada_wind_dir_deg"], np.nan, n),
        "source_index": table.index.astype(str).to_numpy(),
    }

    # Apply optional averaging before filters, matching the India-loader convention.
    avg_window = filters.get("avg_window", 1)
    window_min = filters.get("window_min", None)
    data = _average_blocks(data, int(avg_window), window_min=window_min)

    # Apply case filters.
    mask = _case_mask(data, filters)
    data = _apply_case_mask(data, mask)

    # Derived CP, in the same unit convention as India loader: power [kW], Paero [kW].
    try:
        p_aero_kw = 0.5 * np.asarray(data["rho"]) * np.pi * float(R) ** 2 * np.asarray(data["hubspeed"]) ** 3 * 1e-3
        data["turbine_CP"] = np.asarray(data["power"]) / p_aero_kw
    except Exception:
        data["turbine_CP"] = np.full(int(data["nCases"]), np.nan)

    return data


# -----------------------------------------------------------------------------
# Optional MATLAB export
# -----------------------------------------------------------------------------


def save_india_style_mat(data: Mapping, out_path: str | Path, struct_name: str = "out") -> None:
    """Save a loaded Ørsted dictionary to an India-style MATLAB .mat file.

    This is optional. The preferred Python workflow can use `load_orsted_data`
    directly, but this function is useful if you need a .mat file with an `out`
    struct for compatibility checks.
    """
    if savemat is None:
        raise ImportError("scipy.io.savemat is unavailable in this environment.")

    out = {}
    for k, v in data.items():
        if k in {"nH", "nCases", "speed_profiles", "dir_profiles", "ti_profiles", "turbinePower", "turbine_CP", "alpha_data", "tsr_data", "hubheight"}:
            # These are Python aliases/derived fields, not part of the original 23-field mat schema.
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
