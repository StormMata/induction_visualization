import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import shapiro

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


def _bin2d_lists_from_centers(df, alpha_centers, veer_centers, val_col):
    """
    Bin df[val_col] in (alpha, veer) using bins defined by centers.

    Returns
    -------
    bins : list[list[np.ndarray]]
        Nested list with shape (nV, nA). Each entry contains the values in that bin.
    N : (nV, nA) ndarray
        Number of samples in each bin.
    alpha_edges, veer_edges : ndarray
        Bin edges inferred from the supplied centers.
    """
    a = df["alpha"].to_numpy(float)
    v = df["veer"].to_numpy(float)
    y = df[val_col].to_numpy(float)

    m = np.isfinite(a) & np.isfinite(v) & np.isfinite(y)
    a, v, y = a[m], v[m], y[m]

    alpha_centers = np.asarray(alpha_centers, float)
    veer_centers = np.asarray(veer_centers, float)

    alpha_edges = _centers_to_edges(alpha_centers)
    veer_edges = _centers_to_edges(veer_centers)

    nA = alpha_centers.size
    nV = veer_centers.size

    ia = np.digitize(a, alpha_edges) - 1
    iv = np.digitize(v, veer_edges) - 1

    inside = (ia >= 0) & (ia < nA) & (iv >= 0) & (iv < nV)
    ia, iv, y = ia[inside], iv[inside], y[inside]

    bins = [[[] for _ in range(nA)] for __ in range(nV)]
    N = np.zeros((nV, nA), dtype=int)

    for k in range(y.size):
        bins[iv[k]][ia[k]].append(y[k])
        N[iv[k], ia[k]] += 1

    bins = [[np.asarray(bins[j][i], float) for i in range(nA)] for j in range(nV)]
    return bins, N, alpha_edges, veer_edges


def _aggregate_bins(bins, N, agg="mean", min_count=1):
    """Aggregate per-bin arrays into a 2D statistic array Z.

    Supported string values for ``agg`` are:
    - ``"mean"``
    - ``"median"``
    - ``"count"``
    - ``"shapiro"`` or ``"shapiro_p"``: Shapiro-Wilk p-value
    - ``"shapiro_w"``: Shapiro-Wilk test statistic
    """
    nV = len(bins)
    nA = len(bins[0]) if nV > 0 else 0
    Z = np.full((nV, nA), np.nan, float)

    for j in range(nV):
        for i in range(nA):
            if N[j, i] < min_count:
                continue

            arr = bins[j][i]
            if agg == "mean":
                Z[j, i] = np.mean(arr)
            elif agg == "median":
                Z[j, i] = np.median(arr)
            elif agg == "count":
                Z[j, i] = float(N[j, i])
            elif agg in ("shapiro", "shapiro_p", "shapiro_w"):
                if arr.size < 3:
                    Z[j, i] = np.nan
                    continue
                if np.allclose(arr, arr[0], equal_nan=False):
                    Z[j, i] = np.nan
                    continue
                try:
                    stat, pval = shapiro(arr)
                except Exception:
                    Z[j, i] = np.nan
                    continue
                Z[j, i] = pval if agg in ("shapiro", "shapiro_p") else stat
            elif callable(agg):
                Z[j, i] = float(agg(arr))
            else:
                raise ValueError(
                    "agg must be 'mean', 'median', 'count', 'shapiro', 'shapiro_p', 'shapiro_w', or a callable"
                )

    return Z


def bin2d_from_centers(
    df,
    alpha_centers,
    veer_centers,
    val_col=None,
    agg="mean",
    min_count=1,
):
    """
    Bin df[val_col] in (alpha, veer) using bins defined by *centers*.

    Returns
    -------
    Z : (nV, nA) aggregated values
    N : (nV, nA) counts in each bin
    alpha_edges, veer_edges : bin edges used
    """
    bins, N, alpha_edges, veer_edges = _bin2d_lists_from_centers(
        df=df,
        alpha_centers=alpha_centers,
        veer_centers=veer_centers,
        val_col=val_col,
    )
    Z = _aggregate_bins(bins, N, agg=agg, min_count=min_count)
    return Z, N, alpha_edges, veer_edges


def binned_heatmap(
    Z,
    alpha_centers,
    veer_centers,
    alpha_edges=None,
    veer_edges=None,
    xlabel=None,
    ylabel=None,
    cbar_label=None,
    fmt="{:.2f}",
    annotate=True,
    cmap="RdBu_r",
    vcenter=None,
    vmin=None,
    vmax=None,
    linewidth=0.5,
    figsize=(7.5, 3.8),
    dpi=400,
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
    veer_centers = np.asarray(veer_centers, float)
    Z = np.asarray(Z, float)

    def centers_to_edges(c):
        c = np.asarray(c, float)
        if c.size == 1:
            w = 1.0
            return np.array([c[0] - w / 2, c[0] + w / 2])
        mid = 0.5 * (c[:-1] + c[1:])
        left = c[0] - (mid[0] - c[0])
        right = c[-1] + (c[-1] - mid[-1])
        return np.r_[left, mid, right]

    if alpha_edges is None:
        alpha_edges = centers_to_edges(alpha_centers)
    if veer_edges is None:
        veer_edges = centers_to_edges(veer_centers)

    def _resolve_bin_index(val, centers, name):
        """
        Accept either an integer index or a center value.
        - If int-like: used as index (supports negative indices).
        - Else: use closest center. If normalize_tol provided, enforce closeness.
        """
        if isinstance(val, (int, np.integer)):
            idx = int(val)
            if idx < 0:
                idx = centers.size + idx
            if idx < 0 or idx >= centers.size:
                raise IndexError(f"{name} index {val} out of range for centers size {centers.size}.")
            return idx

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
        a_ref, v_ref = normalize_by
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
        alpha_edges,
        veer_edges,
        Zm,
        shading="flat",
        cmap=cmap_obj,
        norm=norm,
        edgecolors="k",
        linewidth=linewidth,
        alpha=alpha,
    )

    ax.set_xticks(alpha_centers)
    ax.set_yticks(veer_centers)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)

    def auto_text_color(rgba, thresh=textcolor_thresh):
        r, g, b, _ = rgba
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
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
                    ax.text(
                        a0,
                        v0,
                        s,
                        ha="center",
                        va="center",
                        fontsize=label_fontsize * fontsize,
                        color=tc,
                    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)

    ax.set_xlim(alpha_edges[0], alpha_edges[-1])
    ax.set_ylim(veer_edges[0], veer_edges[-1])

    if aspect_eq:
        ax.set_aspect("equal")

    fig.tight_layout()
    return fig, ax


def binned_heatmap_hist(
    df,
    alpha_centers,
    veer_centers,
    val_col,
    agg="mean",
    min_count=1,
    alpha_edges=None,
    veer_edges=None,
    xlabel=None,
    ylabel=None,
    cbar_label=None,
    cmap="RdBu_r",
    vcenter=None,
    vmin=None,
    vmax=None,
    linewidth=0.5,
    figsize=(7.5, 3.8),
    dpi=400,
    alpha=1,
    fontsize=fontsize,
    normalize_by=None,
    normalize_mode="divide",
    normalize_tol=None,
    aspect_eq=True,
    hist_bins=8,
    hist_color="k",
    hist_alpha=0.65,
    hist_rel_height=0.72,
    hist_rel_width=0.78,
    hist_bottom_pad=0.10,
    hist_range="global",
    hist_density=False,
    draw_bin_count=False,
    count_fontsize=0.65,
):
    """
    Plot a heatmap colored by a bin statistic and draw a miniature histogram
    inside each populated cell.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns "alpha", "veer", and val_col.
    alpha_centers, veer_centers : array-like
        Bin centers used to define the 2D binning.
    val_col : str
        Column whose distribution is plotted inside each cell.
    agg : {'mean', 'median', 'count', 'shapiro', 'shapiro_p', 'shapiro_w'} or callable, default 'mean'
        Statistic used for the heatmap color. 'shapiro' and 'shapiro_p'
        return the Shapiro-Wilk p-value in each bin; 'shapiro_w' returns
        the Shapiro-Wilk test statistic.
    min_count : int, default 1
        Minimum number of samples required for a bin to be shown.
    hist_bins : int or array-like, default 8
        Passed directly to numpy.histogram for the mini histograms.
    hist_range : {'global', 'bin'} or tuple, default 'global'
        Range used for each mini histogram. 'global' uses one shared value
        range for all cells, which makes the histograms visually comparable.
        'bin' rescales each histogram to its own local data range.
    hist_density : bool, default False
        If True, draw densities instead of raw counts in each mini histogram.

    Returns
    -------
    fig, ax, Z, N, bins
        Matplotlib figure/axes, aggregated heatmap values, bin counts, and the
        per-cell value arrays.
    """
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.family"] = "STIXGeneral"

    alpha_centers = np.asarray(alpha_centers, float)
    veer_centers = np.asarray(veer_centers, float)

    bins, N, alpha_edges_auto, veer_edges_auto = _bin2d_lists_from_centers(
        df=df,
        alpha_centers=alpha_centers,
        veer_centers=veer_centers,
        val_col=val_col,
    )

    if alpha_edges is None:
        alpha_edges = alpha_edges_auto
    if veer_edges is None:
        veer_edges = veer_edges_auto

    Z = _aggregate_bins(bins, N, agg=agg, min_count=min_count)

    fig, ax = binned_heatmap(
        Z=Z,
        alpha_centers=alpha_centers,
        veer_centers=veer_centers,
        alpha_edges=alpha_edges,
        veer_edges=veer_edges,
        xlabel=xlabel,
        ylabel=ylabel,
        cbar_label=cbar_label,
        annotate=False,
        cmap=cmap,
        vcenter=vcenter,
        vmin=vmin,
        vmax=vmax,
        linewidth=linewidth,
        figsize=figsize,
        dpi=dpi,
        alpha=alpha,
        fontsize=fontsize,
        normalize_by=normalize_by,
        normalize_mode=normalize_mode,
        normalize_tol=normalize_tol,
        aspect_eq=aspect_eq,
    )

    all_values = np.concatenate(
        [arr for row in bins for arr in row if arr.size > 0], axis=0
    ) if np.any(N > 0) else np.array([], float)

    if hist_range == "global":
        if all_values.size == 0:
            global_range = (0.0, 1.0)
        else:
            y0 = np.nanmin(all_values)
            y1 = np.nanmax(all_values)
            if np.isclose(y0, y1):
                pad = 0.5 if np.isclose(y0, 0.0) else 0.05 * abs(y0)
                y0 -= pad
                y1 += pad
            global_range = (y0, y1)
    elif isinstance(hist_range, tuple) and len(hist_range) == 2:
        global_range = tuple(map(float, hist_range))
    elif hist_range == "bin":
        global_range = None
    else:
        raise ValueError("hist_range must be 'global', 'bin', or a length-2 tuple.")

    for j in range(len(veer_centers)):
        yb0, yb1 = veer_edges[j], veer_edges[j + 1]
        cell_h = yb1 - yb0

        for i in range(len(alpha_centers)):
            if N[j, i] < min_count:
                continue

            xb0, xb1 = alpha_edges[i], alpha_edges[i + 1]
            cell_w = xb1 - xb0
            arr = bins[j][i]
            if arr.size == 0:
                continue

            if global_range is None:
                h0 = np.nanmin(arr)
                h1 = np.nanmax(arr)
                if np.isclose(h0, h1):
                    pad = 0.5 if np.isclose(h0, 0.0) else 0.05 * abs(h0)
                    h0 -= pad
                    h1 += pad
                hist_range_use = (h0, h1)
            else:
                hist_range_use = global_range

            counts, edges = np.histogram(arr, bins=hist_bins, range=hist_range_use, density=hist_density)
            if np.all(counts == 0):
                continue

            xpad = 0.5 * (1.0 - hist_rel_width) * cell_w
            hist_x0 = xb0 + xpad
            hist_w = hist_rel_width * cell_w

            ypad = 0.5 * (1.0 - hist_rel_height) * cell_h
            hist_y0 = yb0 + ypad
            hist_h = hist_rel_height * cell_h

            ax.add_patch(
                Rectangle(
                    (hist_x0, hist_y0),
                    hist_w,
                    hist_h,
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.35,
                    zorder=3,
                )
            )

            cmax = np.max(counts)
            if cmax <= 0:
                continue

            nb = len(counts)
            bar_gap = 0.08 * hist_w / max(nb, 1)
            bar_w = hist_w / max(nb, 1) - bar_gap

            for k, c in enumerate(counts):
                frac = c / cmax
                bx = hist_x0 + k * (hist_w / nb) + 0.5 * bar_gap
                bh = frac * hist_h
                ax.add_patch(
                    Rectangle(
                        (bx, hist_y0),
                        bar_w,
                        bh,
                        facecolor=hist_color,
                        edgecolor="none",
                        alpha=hist_alpha,
                        zorder=4,
                    )
                )

            ax.plot(
                [hist_x0, hist_x0 + hist_w],
                [hist_y0, hist_y0],
                color="k",
                linewidth=0.45,
                zorder=5,
            )

            if draw_bin_count:
                ax.text(
                    xb0 + 0.05 * cell_w,
                    yb1 - 0.06 * cell_h,
                    rf"$\mathdefault{{n={N[j, i]}}}$",
                    ha="left",
                    va="top",
                    fontsize=count_fontsize * fontsize,
                    color="k",
                    zorder=6,
                )

    return fig, ax, Z, N, bins