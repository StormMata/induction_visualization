import numpy as np
from scipy.io import loadmat

def load_india_data(
    mat_path: str,
    filters: dict | None = None,
    *,
    rho: float = 1.225,
):
    """
    filters: dict with keys like:
        {'alpha_min': 0, 'alpha_max': 1, 'hubspeed_min': 6, 'hubspeed_max': 8, ...}
    Field names must match a case-indexed variable name in the returned `data`
    (typically a field in `out` like 'alpha', 'hubspeed', 'dsrate', 'ti', etc.).
    """

    m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "out" not in m:
        raise KeyError(f"'out' not found in {mat_path}. Keys: {list(m.keys())}")
    out = m["out"]

    # ----------------------------
    # Convert every field in 'out' to numpy 
    # ----------------------------
    data = {}
    fieldnames = getattr(out, "_fieldnames", None)
    if fieldnames is None:
        fieldnames = [k for k in dir(out) if not k.startswith("_")]

    data["_fieldnames"] = list(fieldnames)
    data["_mat_path"] = mat_path

    for name in fieldnames:
        try:
            val = getattr(out, name)
        except Exception:
            continue
        try:
            data[name] = np.asarray(val)
        except Exception:
            data[name] = val

    # ----------------------------
    # Standardized convenience pulls
    # ----------------------------
    def _as_float_1d(x, nm):
        if x is None:
            raise KeyError(f"Missing required field: out.{nm}")
        return np.asarray(x, dtype=float).reshape(-1)

    def _as_float_nd(x, nm):
        if x is None:
            raise KeyError(f"Missing required field: out.{nm}")
        return np.asarray(x, dtype=float)

    heights         = _as_float_1d(getattr(out, "heights", None), "heights")
    speed_profiles  = _as_float_nd(getattr(out, "speed", None),   "speed")
    dir_profiles    = _as_float_nd(getattr(out, "dir_rel", None), "dir_rel")

    pitch_deg       = _as_float_1d(getattr(out, "pitch_deg", None), "pitch_deg")
    alpha_data      = _as_float_1d(getattr(out, "alpha", None),     "alpha")
    tsr_data        = _as_float_1d(getattr(out, "tsr", None),       "tsr")
    dsrate          = _as_float_1d(getattr(out, "dsrate", None),    "dsrate")
    veer_deg_per_m  = dsrate.copy()

    hubheight       = _as_float_1d(getattr(out, "Hub", None),       "Hub")
    R               = _as_float_1d(getattr(out, "R", None),         "R")
    turbinePower    = _as_float_1d(getattr(out, "power", None),     "power")
    hubspeed        = _as_float_1d(getattr(out, "hubspeed", None),  "hubspeed")
    TI              = _as_float_1d(getattr(out, "ti", None),        "ti")

    turbine_CP = turbinePower / (0.5 * rho * (R**2) * np.pi * (hubspeed**3) * 1e-3)

    nH, nCases = speed_profiles.shape
    if dir_profiles.shape != (nH, nCases):
        raise ValueError(
            f"dir_profiles shape {dir_profiles.shape} does not match speed_profiles {(nH, nCases)}"
        )

    # Put the standardized names into the structure (and also common aliases if you like)
    data.update(
        heights=heights,
        speed_profiles=speed_profiles,
        dir_profiles=dir_profiles,
        pitch_deg=pitch_deg,
        veer_deg_per_m=veer_deg_per_m,
        tsr_data=tsr_data,
        alpha=alpha_data,         # <- expose as 'alpha' for filtering convenience
        alpha_data=alpha_data,    # <- keep your old name too
        dsrate=dsrate,
        Hub=hubheight,            # <- expose original-ish name too
        hubheight=hubheight,
        R=R,
        power=turbinePower,       # <- expose original-ish name too
        turbinePower=turbinePower,
        hubspeed=hubspeed,
        ti=TI,                    # <- expose original-ish name too
        TI=TI,
        turbine_CP=turbine_CP,
        nH=nH,
        nCases=nCases,
    )

    # NEW: optional pre-filter time-window averaging
    data, filters = _apply_time_window_averaging(data, filters)
    nCases = int(data["nCases"])
    nH = int(data["nH"])

    # ----------------------------
    # Build mask from filters (ONLY if provided)
    # ----------------------------
    if not filters:
        mask = np.ones(nCases, dtype=bool)
        case_idx = np.arange(nCases, dtype=int)
        data["filters"] = {} if filters is None else dict(filters)
        data["mask"] = mask
        data["case_idx"] = case_idx
        data["nCases_filtered"] = nCases
        return data

    mask = np.ones(nCases, dtype=bool)

    for key, bound in filters.items():
        if bound is None:
            continue
        if not isinstance(key, str) or ("_" not in key):
            raise ValueError(f"Filter key '{key}' must look like '<field>_min' or '<field>_max'.")

        field, suffix = key.rsplit("_", 1)
        if suffix not in ("min", "max"):
            raise ValueError(f"Filter key '{key}' must end in '_min' or '_max'.")

        if field not in data:
            raise KeyError(
                f"Filter refers to field '{field}', but it's not present in loaded data. "
                f"Available example fields: {sorted([k for k,v in data.items() if isinstance(v,np.ndarray)])[:25]}"
            )

        arr = data[field]
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Field '{field}' is not a numpy array (got {type(arr)}).")

        # Only allow case-indexed 1D arrays for filtering
        if arr.ndim != 1 or arr.shape[0] != nCases:
            raise ValueError(
                f"Field '{field}' must be 1D with length nCases={nCases} to be filterable. "
                f"Got shape {arr.shape}."
            )

        arr_f = np.asarray(arr, dtype=float)
        if suffix == "min":
            mask &= (arr_f >= float(bound))
        else:
            mask &= (arr_f <= float(bound))

    case_idx = np.where(mask)[0]
    if case_idx.size == 0:
        raise ValueError("No cases left after filtering. Relax bounds or verify units.")

    data["filters"] = dict(filters)
    data["mask"] = mask
    data["case_idx"] = case_idx

    # ----------------------------
    # Apply filtering to case-indexed arrays
    # ----------------------------
    for k, v in list(data.items()):
        if not isinstance(v, np.ndarray):
            continue
        if v.ndim == 1 and v.shape[0] == nCases:
            data[k] = v[case_idx]
        elif v.ndim == 2 and v.shape == (nH, nCases):
            data[k] = v[:, case_idx]

    data["nCases_filtered"] = case_idx.size
    return data

import numpy as np

def _apply_time_window_averaging(
    data: dict,
    filters: dict | None,
    *,
    time_field: str = "time",
    gap_factor: float = 5.0,
):
    """
    Optional pre-filter averaging over contiguous time windows.

    Expected special keys inside `filters`:
        avg_window : int >= 1
            Target number of timestamps per full averaging window.
        window_min : int >= 1, optional
            Minimum allowable size for a leftover partial window.
            Defaults to avg_window, meaning only full windows are kept.

    Behavior:
    - If avg_window is missing or == 1, data is returned unchanged.
    - Time gaps are detected from `data[time_field]` using the median positive
      sampling interval. A new contiguous segment starts whenever:
          dt <= 0, dt is non-finite, or dt > gap_factor * median_dt
    - Within each contiguous segment of length N:
          q, r = divmod(N, avg_window)
      keep q full windows of length avg_window, and keep the remainder r only if
      r >= window_min.

    Notes:
    - Assumes timestamps are already in dataset order.
    - Numeric case-indexed arrays are averaged with nanmean.
    - Non-numeric case-indexed arrays are reduced by taking the first value
      in each window.
    """

    if not filters:
        return data, filters

    # Pull out the special windowing options, but leave the original dict untouched.
    avg_window = filters.get("avg_window", None)
    window_min = filters.get("window_min", None)

    cleaned_filters = {
        k: v for k, v in filters.items()
        if k not in ("avg_window", "window_min")
    }

    # No windowing requested
    if avg_window is None:
        return data, cleaned_filters

    # Validate avg_window
    if not isinstance(avg_window, (int, np.integer)) or int(avg_window) < 1:
        raise ValueError(f"'avg_window' must be an integer >= 1. Got {avg_window!r}.")
    avg_window = int(avg_window)

    if avg_window == 1:
        return data, cleaned_filters

    # Default window_min: only keep full windows unless user explicitly allows smaller remainder windows
    if window_min is None:
        window_min = avg_window

    if not isinstance(window_min, (int, np.integer)) or int(window_min) < 1:
        raise ValueError(f"'window_min' must be an integer >= 1. Got {window_min!r}.")
    window_min = int(window_min)

    if window_min > avg_window:
        raise ValueError(
            f"'window_min' must be <= 'avg_window'. Got window_min={window_min}, avg_window={avg_window}."
        )

    # Need time data
    if time_field not in data:
        raise KeyError(
            f"Requested averaging by time, but '{time_field}' is not present in data."
        )

    time = np.asarray(data[time_field], dtype=float).reshape(-1)
    nCases = int(data["nCases"])

    if time.shape[0] != nCases:
        raise ValueError(
            f"'{time_field}' must be 1D with length nCases={nCases}. Got shape {time.shape}."
        )

    if nCases == 0:
        raise ValueError("No cases available for time-window averaging.")

    # --------------------------------
    # Detect contiguous time segments
    # --------------------------------
    if nCases == 1:
        segments = [(0, 1)]
        median_dt = np.nan
    else:
        dt = np.diff(time)
        pos_dt = dt[np.isfinite(dt) & (dt > 0)]

        if pos_dt.size == 0:
            # Fallback: treat every sample as its own segment if time is unusable
            segments = [(i, i + 1) for i in range(nCases)]
            median_dt = np.nan
        else:
            median_dt = float(np.median(pos_dt))

            # Break when time is non-increasing, non-finite, or there is a large gap
            breaks = (~np.isfinite(dt)) | (dt <= 0) | (dt > gap_factor * median_dt)

            starts = np.r_[0, np.where(breaks)[0] + 1]
            ends   = np.r_[np.where(breaks)[0] + 1, nCases]
            segments = list(zip(starts, ends))

    # --------------------------------
    # Build windows
    # --------------------------------
    windows = []
    dropped_tail_counts = []

    for start, end in segments:
        seg_len = end - start
        if seg_len <= 0:
            continue

        q, r = divmod(seg_len, avg_window)

        # Full windows of exact size avg_window
        for j in range(q):
            s = start + j * avg_window
            e = s + avg_window
            windows.append(np.arange(s, e, dtype=int))

        # Optional remainder window
        if r >= window_min:
            s = start + q * avg_window
            e = end
            windows.append(np.arange(s, e, dtype=int))
        elif r > 0:
            dropped_tail_counts.append(r)

    if len(windows) == 0:
        raise ValueError(
            "No averaging windows could be formed. "
            f"Try reducing avg_window={avg_window} or window_min={window_min}."
        )

    # --------------------------------
    # Aggregate case-indexed arrays
    # --------------------------------
    new_data = dict(data)  # shallow copy is fine; we replace arrays below
    nH = int(data["nH"])

    def _aggregate_1d(arr, idx_groups):
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.number):
            return np.array([np.nanmean(arr[idx]) for idx in idx_groups], dtype=float)
        else:
            return np.array([arr[idx[0]] for idx in idx_groups], dtype=arr.dtype)

    def _aggregate_2d_profiles(arr, idx_groups):
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.number):
            return np.column_stack([np.nanmean(arr[:, idx], axis=1) for idx in idx_groups])
        else:
            return np.column_stack([arr[:, idx[0]] for idx in idx_groups])

    for k, v in list(data.items()):
        if not isinstance(v, np.ndarray):
            continue

        # 1D case-indexed arrays
        if v.ndim == 1 and v.shape[0] == nCases:
            new_data[k] = _aggregate_1d(v, windows)

        # 2D profile arrays with cases along axis 1
        elif v.ndim == 2 and v.shape == (nH, nCases):
            new_data[k] = _aggregate_2d_profiles(v, windows)

    # Update bookkeeping
    new_nCases = len(windows)
    new_data["nCases"] = new_nCases
    new_data["avg_window_info"] = {
        "applied": True,
        "time_field": time_field,
        "avg_window": avg_window,
        "window_min": window_min,
        "gap_factor": gap_factor,
        "median_dt_days": median_dt,
        "n_original_cases": nCases,
        "n_segments": len(segments),
        "n_windows": new_nCases,
        "window_lengths": np.array([len(w) for w in windows], dtype=int),
        "window_start_idx": np.array([w[0] for w in windows], dtype=int),
        "window_end_idx": np.array([w[-1] for w in windows], dtype=int),
        "dropped_tail_counts": np.array(dropped_tail_counts, dtype=int),
    }

    return new_data, cleaned_filters