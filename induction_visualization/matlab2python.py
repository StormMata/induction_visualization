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
    # Convert every field in 'out' to numpy (best-effort)
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
    # Standardized convenience pulls (your named variables)
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