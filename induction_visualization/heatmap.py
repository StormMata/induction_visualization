import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

fontsize = 12

plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize

plt.rcParams.update(
    {"text.usetex": True, "text.latex.preamble": r"\usepackage{amsfonts}"}
)

def _centers_to_edges(c):
    """Convert monotonically increasing bin centers -> edges (len = len(c)+1)."""
    c = np.asarray(c, float)
    if c.ndim != 1 or c.size < 1:
        raise ValueError("centers must be a 1D array with at least 1 element")
    if c.size == 1:
        w = 1.0
        return np.array([c[0] - w/2, c[0] + w/2], float)
    mid = 0.5 * (c[:-1] + c[1:])
    left  = c[0]  - (mid[0] - c[0])
    right = c[-1] + (c[-1] - mid[-1])
    return np.r_[left, mid, right]

def bin2d_from_centers(
        df,
        alpha_centers,
        veer_centers,
        val_col=None,
        agg="mean",
        min_count=1
):
    """
    Bin df[val_col] in (alpha, veer) using bins defined by *centers*.

    Returns
    -------
    Z : (nV, nA) aggregated values
    N : (nV, nA) counts in each bin
    alpha_edges, veer_edges : bin edges used
    """
    a = df["alpha"].to_numpy(float)
    v = df["veer"].to_numpy(float)
    y = df[val_col].to_numpy(float)

    m = np.isfinite(a) & np.isfinite(v) & np.isfinite(y)
    a, v, y = a[m], v[m], y[m]

    alpha_centers = np.asarray(alpha_centers, float)
    veer_centers  = np.asarray(veer_centers,  float)

    alpha_edges = _centers_to_edges(alpha_centers)
    veer_edges  = _centers_to_edges(veer_centers)

    nA = alpha_centers.size
    nV = veer_centers.size

    # bin indices
    ia = np.digitize(a, alpha_edges) - 1
    iv = np.digitize(v, veer_edges) - 1

    # keep only points inside the defined bins
    inside = (ia >= 0) & (ia < nA) & (iv >= 0) & (iv < nV)
    ia, iv, y = ia[inside], iv[inside], y[inside]

    # accumulate sums/counts or store lists
    N = np.zeros((nV, nA), dtype=int)

    if agg == "mean":
        S = np.zeros((nV, nA), dtype=float)
        np.add.at(S, (iv, ia), y)
        np.add.at(N, (iv, ia), 1)
        Z = np.full((nV, nA), np.nan, float)
        ok = N >= min_count
        Z[ok] = S[ok] / N[ok]
        return Z, N, alpha_edges, veer_edges

    elif agg == "count":
        np.add.at(N, (iv, ia), 1)
        Z = np.full((nV, nA), np.nan, float)
        ok = N >= min_count
        Z[ok] = N[ok].astype(float)
        return Z, N, alpha_edges, veer_edges

    else:
        bins = [[[] for _ in range(nA)] for __ in range(nV)]
        for k in range(y.size):
            bins[iv[k]][ia[k]].append(y[k])
            N[iv[k], ia[k]] += 1

        Z = np.full((nV, nA), np.nan, float)
        for j in range(nV):
            for i in range(nA):
                if N[j, i] >= min_count:
                    arr = np.asarray(bins[j][i], float)
                    if agg == "median":
                        Z[j, i] = np.median(arr)
                    elif callable(agg):
                        Z[j, i] = float(agg(arr))
                    else:
                        raise ValueError("agg must be 'mean', 'median', 'count', or a callable")
        return Z, N, alpha_edges, veer_edges

def binned_heatmap(
    Z, alpha_centers, veer_centers,
    alpha_edges=None, veer_edges=None,
    xlabel=None,
    ylabel=None,
    cbar_label=None,
    fmt="{:.2f}",
    annotate=True,
    cmap="RdBu_r",
    vcenter=None,
    vmin=None, vmax=None,
    linewidth=0.5,
    figsize=(7.5, 3.8), dpi=400,
    alpha=1,
    textcolor="auto",
    textcolor_thresh=0.4,
    label_fontsize=1,
    fontsize=fontsize,
    normalize_by=None,         
    normalize_mode="divide",   
    normalize_tol=None,
    aspect_eq=True,

):
    # TeX-like serif digits without requiring external LaTeX
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.family"] = "STIXGeneral"

    alpha_centers = np.asarray(alpha_centers, float)
    veer_centers  = np.asarray(veer_centers, float)
    Z = np.asarray(Z, float)

    def centers_to_edges(c):
        c = np.asarray(c, float)
        if c.size == 1:
            w = 1.0
            return np.array([c[0]-w/2, c[0]+w/2])
        mid = 0.5*(c[:-1] + c[1:])
        left  = c[0]  - (mid[0] - c[0])
        right = c[-1] + (c[-1] - mid[-1])
        return np.r_[left, mid, right]

    if alpha_edges is None:
        alpha_edges = centers_to_edges(alpha_centers)
    if veer_edges is None:
        veer_edges  = centers_to_edges(veer_centers)

    # ---------- NEW: normalization ----------
    def _resolve_bin_index(val, centers, name):
        """
        Accept either an integer index or a center value.
        - If int-like: used as index (supports negative indices).
        - Else: use closest center. If normalize_tol provided, enforce closeness.
        """
        # index path
        if isinstance(val, (int, np.integer)):
            idx = int(val)
            if idx < 0:
                idx = centers.size + idx
            if idx < 0 or idx >= centers.size:
                raise IndexError(f"{name} index {val} out of range for centers size {centers.size}.")
            return idx

        # value path -> nearest center
        v = float(val)
        idx = int(np.argmin(np.abs(centers - v)))
        if normalize_tol is not None:
            if np.abs(centers[idx] - v) > float(normalize_tol):
                raise ValueError(
                    f"{name} value {v} not within tol={normalize_tol} of any center "
                    f"(closest is {centers[idx]} at index {idx})."
                )
        return idx

    Z_plot = Z.copy()
    if normalize_by is not None:
        a_ref, v_ref = normalize_by  # a = alpha bin, b = v bin
        i_ref = _resolve_bin_index(a_ref, alpha_centers, "alpha")
        j_ref = _resolve_bin_index(v_ref, veer_centers, "veer")

        Z_ref = Z_plot[j_ref, i_ref]
        if not np.isfinite(Z_ref) or np.isclose(Z_ref, 0.0):
            raise ValueError(
                f"Reference bin Z[{j_ref},{i_ref}] is not finite or is ~0 (value={Z_ref}). "
                "Choose a different normalize_by."
            )

        Z_plot = Z_plot / Z_ref
        if normalize_mode == "relative":
            Z_plot = Z_plot - 1.0
        elif normalize_mode != "divide":
            raise ValueError("normalize_mode must be 'divide' or 'relative'.")

    # mask invalids after normalization
    Zm = np.ma.masked_invalid(Z_plot)

    cmap_obj = mpl.cm.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
    cmap_obj.set_bad("white")

    Z_finite = Z_plot[np.isfinite(Z_plot)]
    if Z_finite.size == 0:
        raise ValueError("Z has no finite values to set color limits.")

    vmin_eff = np.min(Z_finite) if vmin is None else vmin
    vmax_eff = np.max(Z_finite) if vmax is None else vmax

    if vcenter is not None:
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin_eff, vcenter=vcenter, vmax=vmax_eff)
    else:
        norm = mpl.colors.Normalize(vmin=vmin_eff, vmax=vmax_eff)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    mesh = ax.pcolormesh(
        alpha_edges, veer_edges, Zm,
        shading="flat",
        cmap=cmap_obj, norm=norm,
        edgecolors="k", linewidth=linewidth,
        alpha=alpha
    )

    ax.set_xticks(alpha_centers)
    ax.set_yticks(veer_centers)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

    def auto_text_color(rgba, thresh=textcolor_thresh):
        r, g, b, _ = rgba
        lum = 0.2126*r + 0.7152*g + 0.0722*b
        return "w" if lum < thresh else "k"

    if annotate:
        for j, v0 in enumerate(veer_centers):
            for i, a0 in enumerate(alpha_centers):
                val = Z_plot[j, i]
                if np.isfinite(val):
                    if textcolor == "auto":
                        rgba = cmap_obj(norm(val))
                        tc = auto_text_color(rgba)
                    else:
                        tc = textcolor

                    s = r"$\mathdefault{" + fmt.format(val) + "}$"
                    ax.text(a0, v0, s,
                            ha="center", va="center",
                            fontsize=label_fontsize * fontsize,
                            color=tc)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)

    ax.set_xlim(alpha_edges[0], alpha_edges[-1])
    ax.set_ylim(veer_edges[0], veer_edges[-1])

    if aspect_eq:
        ax.set_aspect('equal')

    fig.tight_layout()
    return fig, ax