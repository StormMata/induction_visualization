import numpy as np
from scipy.io import loadmat

def load_india_data(
    mat_path: str,
    mask: np.ndarray | None = None,
    *,
    rho: float = 1.225,
):
    """
    Load a PadeOps-style .mat file and return a dict containing:
      - raw fields from out (best-effort numpy conversion)
      - standardized convenience variables (heights, speed_profiles, etc.)
      - turbine_CP
      - (optionally) filtered selection via mask

    If `mask` is None: keep all points (no filtering applied).
    If `mask` is provided: must be boolean shape (nCases,), and filtering is applied.
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

    data.update(
        heights=heights,
        speed_profiles=speed_profiles,
        dir_profiles=dir_profiles,
        pitch_deg=pitch_deg,
        veer_deg_per_m=veer_deg_per_m,
        tsr_data=tsr_data,
        alpha_data=alpha_data,
        dsrate=dsrate,
        hubheight=hubheight,
        R=R,
        turbinePower=turbinePower,
        hubspeed=hubspeed,
        TI=TI,
        turbine_CP=turbine_CP,
        nH=nH,
        nCases=nCases,
    )

    # ----------------------------
    # Masking / filtering (ONLY if mask is provided)
    # ----------------------------
    if mask is None:
        # Keep all points
        mask = np.ones(nCases, dtype=bool)
        case_idx = np.arange(nCases, dtype=int)
        data["mask"] = mask
        data["case_idx"] = case_idx
        data["nCases_filtered"] = nCases
        return data

    # Otherwise apply provided mask
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size != nCases:
        raise ValueError(f"mask has size {mask.size} but nCases is {nCases}")

    case_idx = np.where(mask)[0]
    if case_idx.size == 0:
        raise ValueError("No cases left after filtering.")

    data["mask"] = mask
    data["case_idx"] = case_idx

    # Apply to arrays that look case-indexed
    for k, v in list(data.items()):
        if not isinstance(v, np.ndarray):
            continue
        if v.ndim == 1 and v.shape[0] == nCases:
            data[k] = v[case_idx]
        elif v.ndim == 2 and v.shape == (nH, nCases):
            data[k] = v[:, case_idx]

    data["nCases_filtered"] = case_idx.size
    return data