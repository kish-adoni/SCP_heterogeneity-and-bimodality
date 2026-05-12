import os
import re
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    stats = None
    SCIPY_AVAILABLE = False

try:
    from diptest import diptest as _hartigans_diptest
    DIPTEST_AVAILABLE = True
except Exception:
    _hartigans_diptest = None
    DIPTEST_AVAILABLE = False

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


EPSILON = 1e-12
CONDITION_A_COLOR = "#AA00D4FF"
CONDITION_B_COLOR = "#FF6600FF"
BOTH_SIG_COLOR = "#B279A2"
MEAN_SIG_COLOR = "#72B7B2"
HET_SIG_COLOR = "#F58518"
NOT_SIG_COLOR = "#9D9D9D"


def safe_divide(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def safe_positive_divide(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return np.nan
    return numerator / denominator


def compute_iqr(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    return float(q3 - q1)


def compute_mad(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def compute_normalised_mad(values):
    """Median absolute deviation normalised by the absolute median abundance.

    This gives a robust CV-like dispersion metric:
        median(|x_i - median(x)|) / |median(x)|
    The absolute median is used so the normalised heterogeneity remains
    non-negative if transformed abundance values have a negative median.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    med = float(np.median(arr))
    if pd.isna(med) or abs(med) <= EPSILON:
        return np.nan
    return float(compute_mad(arr) / abs(med))



def _clean_numeric_array(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def compute_kde_bimodality_score(values, min_n=10):
    """Fallback bimodality score based on KDE peak structure.

    Used only when the optional `diptest` package is not installed. The score is
    zero for distributions with fewer than two KDE peaks. Larger values indicate
    two reasonably prominent, well-separated peaks. It is not a p-value.
    """
    arr = _clean_numeric_array(values)
    if arr.size < int(min_n) or np.nanstd(arr) <= EPSILON:
        return np.nan, np.nan, 0, "KDE_peak_score"
    if not SCIPY_AVAILABLE:
        return np.nan, np.nan, 0, "not_available"
    try:
        kde = stats.gaussian_kde(arr)
        xmin = float(np.min(arr))
        xmax = float(np.max(arr))
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            return np.nan, np.nan, 0, "KDE_peak_score"
        grid = np.linspace(xmin, xmax, 256)
        dens = kde(grid)
        peaks = []
        for i in range(1, len(dens) - 1):
            if dens[i] > dens[i - 1] and dens[i] > dens[i + 1]:
                peaks.append((float(dens[i]), float(grid[i])))
        if len(peaks) < 2:
            return 0.0, np.nan, len(peaks), "KDE_peak_score"
        peaks.sort(reverse=True, key=lambda x: x[0])
        primary, secondary = peaks[0], peaks[1]
        i1 = int(np.argmin(np.abs(grid - primary[1])))
        i2 = int(np.argmin(np.abs(grid - secondary[1])))
        lo, hi = sorted((i1, i2))
        valley = float(np.min(dens[lo:hi + 1])) if hi > lo else min(primary[0], secondary[0])
        prominence = max(0.0, (secondary[0] - valley) / max(primary[0], EPSILON))
        iqr = compute_iqr(arr)
        separation = abs(primary[1] - secondary[1]) / max(iqr, EPSILON) if pd.notna(iqr) else 0.0
        score = float(prominence * separation)
        return score, np.nan, len(peaks), "KDE_peak_score"
    except Exception:
        return np.nan, np.nan, 0, "KDE_peak_score"


def compute_bimodality(values, min_n=10):
    """Return a bimodality statistic, p-value, number of peaks, and method name.

    If the optional `diptest` package is available, this uses Hartigan's dip
    test. Otherwise it falls back to a KDE peak-separation score so the GUI can
    still rank visibly bimodal distributions without an extra dependency.
    """
    arr = _clean_numeric_array(values)
    if arr.size < int(min_n) or np.nanstd(arr) <= EPSILON:
        return np.nan, np.nan, 0, "too_few_values"
    if DIPTEST_AVAILABLE:
        try:
            dip, pval = _hartigans_diptest(arr)
            return float(dip), float(pval), np.nan, "Hartigan_dip_test"
        except Exception:
            pass
    return compute_kde_bimodality_score(arr, min_n=min_n)



def _normal_pdf_1d(x, mean, sd):
    sd = max(float(sd), EPSILON)
    z = (x - float(mean)) / sd
    return np.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))


def fit_two_gaussian_mixture_1d(values, max_iter=200, tol=1e-6):
    """Fit a simple deterministic 2-component Gaussian mixture in 1D.

    This avoids a scikit-learn dependency while providing the diagnostics needed
    for strict bimodality filtering: mode fractions, separation, BIC improvement,
    and valley depth between the fitted components.
    """
    arr = _clean_numeric_array(values)
    n = arr.size
    if n < 4 or np.nanstd(arr) <= EPSILON:
        return None

    # One-component model.
    mean1 = float(np.mean(arr))
    sd1 = float(np.std(arr, ddof=0))
    sd1 = max(sd1, EPSILON)
    ll1 = float(np.sum(np.log(_normal_pdf_1d(arr, mean1, sd1) + EPSILON)))
    bic1 = float(2 * math.log(max(n, 1)) - 2 * ll1)  # k=2: mean + variance

    # Two-component initialisation from quartiles.
    means = np.percentile(arr, [25, 75]).astype(float)
    if abs(means[1] - means[0]) <= EPSILON:
        means = np.array([float(np.min(arr)), float(np.max(arr))])
    s0 = max(float(np.std(arr, ddof=0)), EPSILON)
    sds = np.array([s0, s0], dtype=float)
    weights = np.array([0.5, 0.5], dtype=float)

    prev_ll = -np.inf
    resp = np.zeros((n, 2), dtype=float)
    for _ in range(int(max_iter)):
        dens0 = weights[0] * _normal_pdf_1d(arr, means[0], sds[0])
        dens1 = weights[1] * _normal_pdf_1d(arr, means[1], sds[1])
        total = dens0 + dens1 + EPSILON
        resp[:, 0] = dens0 / total
        resp[:, 1] = dens1 / total
        nk = resp.sum(axis=0)
        if np.any(nk <= EPSILON):
            return None
        weights = nk / n
        means = (resp * arr[:, None]).sum(axis=0) / nk
        variances = (resp * (arr[:, None] - means[None, :]) ** 2).sum(axis=0) / nk
        sds = np.sqrt(np.maximum(variances, EPSILON))
        ll2 = float(np.sum(np.log(
            weights[0] * _normal_pdf_1d(arr, means[0], sds[0])
            + weights[1] * _normal_pdf_1d(arr, means[1], sds[1])
            + EPSILON
        )))
        if abs(ll2 - prev_ll) < tol:
            break
        prev_ll = ll2

    # Sort components by mean for reproducible outputs.
    order = np.argsort(means)
    means = means[order]
    sds = sds[order]
    weights = weights[order]
    resp = resp[:, order]
    labels = np.argmax(resp, axis=1)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        return None

    ll2 = float(np.sum(np.log(
        weights[0] * _normal_pdf_1d(arr, means[0], sds[0])
        + weights[1] * _normal_pdf_1d(arr, means[1], sds[1])
        + EPSILON
    )))
    bic2 = float(5 * math.log(max(n, 1)) - 2 * ll2)  # k=5: 2 means, 2 variances, 1 mixing weight
    bic_improvement = float(bic1 - bic2)  # positive means 2-component model is better.

    centre_sep = float(abs(means[1] - means[0]))
    iqr = compute_iqr(arr)
    sep_iqr = float(centre_sep / max(iqr, EPSILON)) if pd.notna(iqr) else np.nan

    # Valley depth: require a visible density valley between the two fitted modes.
    grid = np.linspace(float(means[0]), float(means[1]), 256)
    comp0 = weights[0] * _normal_pdf_1d(grid, means[0], sds[0])
    comp1 = weights[1] * _normal_pdf_1d(grid, means[1], sds[1])
    mixture = comp0 + comp1
    peak_left = float(weights[0] * _normal_pdf_1d(np.array([means[0]]), means[0], sds[0])[0])
    peak_right = float(weights[1] * _normal_pdf_1d(np.array([means[1]]), means[1], sds[1])[0])
    valley = float(np.min(mixture)) if mixture.size else np.nan
    reference_peak = min(peak_left, peak_right)
    valley_depth = float(1.0 - (valley / max(reference_peak, EPSILON))) if pd.notna(valley) else np.nan
    valley_depth = max(0.0, min(1.0, valley_depth)) if pd.notna(valley_depth) else np.nan

    small_n = int(np.min(counts))
    large_n = int(np.max(counts))
    small_fraction = float(small_n / n)
    large_fraction = float(large_n / n)

    return {
        "small_mode_fraction": small_fraction,
        "large_mode_fraction": large_fraction,
        "small_mode_n": small_n,
        "large_mode_n": large_n,
        "mode_centre_separation": centre_sep,
        "mode_separation_iqr": sep_iqr,
        "bic_one_component": bic1,
        "bic_two_component": bic2,
        "bic_improvement_2component_vs_1component": bic_improvement,
        "gmm_two_component_better": bool(bic_improvement > 0),
        "valley_depth": valley_depth,
        "component_means": means,
        "component_sds": sds,
        "component_weights": weights,
    }


def compute_two_mode_support(values, min_cells_per_mode=10):
    """Estimate two-mode support using a 2-component Gaussian mixture."""
    fit = fit_two_gaussian_mixture_1d(values)
    if fit is None:
        return np.nan, np.nan, 0, 0, np.nan, np.nan, np.nan, np.nan, False
    return (
        fit["small_mode_fraction"],
        fit["large_mode_fraction"],
        fit["small_mode_n"],
        fit["large_mode_n"],
        fit["mode_centre_separation"],
        fit["mode_separation_iqr"],
        fit["bic_improvement_2component_vs_1component"],
        fit["valley_depth"],
        fit["gmm_two_component_better"],
    )


def compute_balanced_bimodality(
    values,
    min_n=10,
    min_fraction_per_mode=0.15,
    min_cells_per_mode=10,
    min_mode_separation=0.5,
    max_dip_p_value=0.05,
    min_valley_depth=0.20,
    require_gmm_improvement=True,
):
    """Strict, balanced bimodality score.

    A protein is eligible only if it passes all enabled filters:
      - enough cells in both inferred modes
      - smaller mode fraction threshold
      - minimum separation between fitted mode centres
      - Hartigan dip-test p-value threshold when available
      - 2-component GMM improves over 1-component GMM by BIC
      - a visible valley between the two fitted modes

    The final score is raw_bimodality_score * smaller_mode_fraction for eligible
    proteins and 0 otherwise.
    """
    raw_score, pval, peaks, method = compute_bimodality(values, min_n=min_n)
    (
        small_frac,
        large_frac,
        small_n,
        large_n,
        centre_sep,
        sep_iqr,
        bic_improvement,
        valley_depth,
        gmm_better,
    ) = compute_two_mode_support(values, min_cells_per_mode=min_cells_per_mode)

    p_pass = True
    if method == "Hartigan_dip_test" and pd.notna(pval):
        p_pass = bool(pval <= float(max_dip_p_value))

    gmm_pass = True
    if require_gmm_improvement:
        gmm_pass = bool(gmm_better)

    passes = (
        pd.notna(raw_score)
        and pd.notna(small_frac)
        and small_n >= int(min_cells_per_mode)
        and small_frac >= float(min_fraction_per_mode)
        and pd.notna(centre_sep)
        and centre_sep >= float(min_mode_separation)
        and p_pass
        and gmm_pass
        and pd.notna(valley_depth)
        and valley_depth >= float(min_valley_depth)
    )
    balanced_score = float(raw_score * small_frac) if passes else 0.0 if pd.notna(raw_score) else np.nan
    return {
        "raw_score": raw_score,
        "p_value": pval,
        "peaks": peaks,
        "method": method,
        "small_mode_fraction": small_frac,
        "large_mode_fraction": large_frac,
        "small_mode_n": small_n,
        "large_mode_n": large_n,
        "mode_centre_separation": centre_sep,
        "minimum_mode_separation_required": float(min_mode_separation),
        "mode_separation_iqr": sep_iqr,
        "bic_improvement_2component_vs_1component": bic_improvement,
        "gmm_two_component_better": bool(gmm_better),
        "minimum_valley_depth_required": float(min_valley_depth),
        "valley_depth": valley_depth,
        "dip_p_value_threshold": float(max_dip_p_value),
        "passes_dip_p_value_filter": bool(p_pass),
        "passes_mode_support_filter": bool(passes),
        "balanced_score": balanced_score,
    }

def build_bimodality_analysis(
    df,
    protein_column,
    condition_a_keyword,
    condition_b_keyword,
    condition_a_label="Condition_A",
    condition_b_label="Condition_B",
    min_valid_n=10,
    min_fraction_per_mode=0.15,
    min_cells_per_mode=10,
    min_mode_separation=0.5,
    max_dip_p_value=0.05,
    min_valley_depth=0.20,
    require_gmm_improvement=True,
):
    cond_a_cols, cond_b_cols = get_condition_columns(
        df, protein_column, condition_a_keyword, condition_b_keyword
    )
    numeric_cols = cond_a_cols + cond_b_cols
    work = df[[protein_column] + numeric_cols].copy()
    work[numeric_cols] = work[numeric_cols].apply(pd.to_numeric, errors="coerce")

    rows = []
    for _, row in work.iterrows():
        a = pd.to_numeric(row[cond_a_cols], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(row[cond_b_cols], errors="coerce").to_numpy(dtype=float)
        a = _clean_numeric_array(a)
        b = _clean_numeric_array(b)
        bm_a = compute_balanced_bimodality(
            a,
            min_n=min_valid_n,
            min_fraction_per_mode=min_fraction_per_mode,
            min_cells_per_mode=min_cells_per_mode,
            min_mode_separation=min_mode_separation,
            max_dip_p_value=max_dip_p_value,
            min_valley_depth=min_valley_depth,
            require_gmm_improvement=require_gmm_improvement,
        )
        bm_b = compute_balanced_bimodality(
            b,
            min_n=min_valid_n,
            min_fraction_per_mode=min_fraction_per_mode,
            min_cells_per_mode=min_cells_per_mode,
            min_mode_separation=min_mode_separation,
            max_dip_p_value=max_dip_p_value,
            min_valley_depth=min_valley_depth,
            require_gmm_improvement=require_gmm_improvement,
        )
        score_a = bm_a["balanced_score"]
        score_b = bm_b["balanced_score"]
        raw_a = bm_a["raw_score"]
        raw_b = bm_b["raw_score"]
        p_a = bm_a["p_value"]
        p_b = bm_b["p_value"]
        peaks_a = bm_a["peaks"]
        peaks_b = bm_b["peaks"]
        method_a = bm_a["method"]
        method_b = bm_b["method"]
        max_score = np.nanmax([score_a, score_b]) if pd.notna(score_a) or pd.notna(score_b) else np.nan
        delta_score = score_b - score_a if pd.notna(score_a) and pd.notna(score_b) else np.nan
        abs_delta_score = abs(delta_score) if pd.notna(delta_score) else np.nan
        rows.append({
            "protein": row[protein_column],
            f"n_{condition_a_label}": int(a.size),
            f"n_{condition_b_label}": int(b.size),
            f"balanced_bimodality_score_{condition_a_label}": score_a,
            f"balanced_bimodality_score_{condition_b_label}": score_b,
            f"raw_bimodality_score_{condition_a_label}": raw_a,
            f"raw_bimodality_score_{condition_b_label}": raw_b,
            f"bimodality_p_value_{condition_a_label}": p_a,
            f"bimodality_p_value_{condition_b_label}": p_b,
            f"smaller_mode_fraction_{condition_a_label}": bm_a["small_mode_fraction"],
            f"smaller_mode_fraction_{condition_b_label}": bm_b["small_mode_fraction"],
            f"smaller_mode_n_{condition_a_label}": bm_a["small_mode_n"],
            f"smaller_mode_n_{condition_b_label}": bm_b["small_mode_n"],
            f"passes_mode_support_filter_{condition_a_label}": bm_a["passes_mode_support_filter"],
            f"passes_mode_support_filter_{condition_b_label}": bm_b["passes_mode_support_filter"],
            f"mode_separation_iqr_{condition_a_label}": bm_a["mode_separation_iqr"],
            f"mode_separation_iqr_{condition_b_label}": bm_b["mode_separation_iqr"],
            f"mode_centre_separation_{condition_a_label}": bm_a["mode_centre_separation"],
            f"mode_centre_separation_{condition_b_label}": bm_b["mode_centre_separation"],
            f"valley_depth_{condition_a_label}": bm_a["valley_depth"],
            f"valley_depth_{condition_b_label}": bm_b["valley_depth"],
            f"bic_improvement_2component_vs_1component_{condition_a_label}": bm_a["bic_improvement_2component_vs_1component"],
            f"bic_improvement_2component_vs_1component_{condition_b_label}": bm_b["bic_improvement_2component_vs_1component"],
            f"gmm_two_component_better_{condition_a_label}": bm_a["gmm_two_component_better"],
            f"gmm_two_component_better_{condition_b_label}": bm_b["gmm_two_component_better"],
            f"passes_dip_p_value_filter_{condition_a_label}": bm_a["passes_dip_p_value_filter"],
            f"passes_dip_p_value_filter_{condition_b_label}": bm_b["passes_dip_p_value_filter"],
            f"kde_peak_count_{condition_a_label}": peaks_a,
            f"kde_peak_count_{condition_b_label}": peaks_b,
            f"delta_balanced_bimodality_{condition_b_label}_minus_{condition_a_label}": delta_score,
            f"abs_delta_balanced_bimodality_{condition_b_label}_minus_{condition_a_label}": abs_delta_score,
            "max_balanced_bimodality_score": max_score,
            "bimodality_method": method_b if method_b == method_a else f"{method_a}/{method_b}",
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        for condition_label in (condition_a_label, condition_b_label):
            p_col = f"bimodality_p_value_{condition_label}"
            details = benjamini_hochberg_details(out[p_col])
            # Diagnostic columns for export: these make it clear why many/most
            # BH-FDR values can collapse to the same value near 1.
            out[f"bimodality_p_value_rank_{condition_label}"] = details["rank"]
            out[f"bimodality_bh_raw_adjusted_{condition_label}"] = details["raw_adjusted"]
            out[f"bimodality_q_value_bh_fdr_{condition_label}"] = details["q_value"]
    return out, cond_a_cols, cond_b_cols, work


def benjamini_hochberg_details(pvalues):
    """Return BH-FDR q-values plus diagnostic rank/raw-adjusted columns.

    rank is the p-value rank after sorting ascending. raw_adjusted is p*m/rank
    before the Benjamini-Hochberg monotonic cumulative-minimum step is applied.
    q_value is the final BH-FDR-adjusted p-value used elsewhere in the app.
    """
    series = pd.Series(pvalues, dtype=float)
    result = pd.DataFrame({
        "rank": pd.Series(np.nan, index=series.index, dtype=float),
        "raw_adjusted": pd.Series(np.nan, index=series.index, dtype=float),
        "q_value": pd.Series(np.nan, index=series.index, dtype=float),
    })

    valid = series.notna()
    if valid.sum() == 0:
        return result

    p = series.loc[valid].to_numpy(dtype=float)
    n = len(p)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]

    ranks_sorted = np.arange(1, n + 1, dtype=float)
    raw_adjusted_sorted = ranked * n / ranks_sorted
    q_sorted = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, raw_adjusted_sorted[i])
        q_sorted[i] = min(prev, 1.0)

    ranks = np.empty(n, dtype=float)
    raw_adjusted = np.empty(n, dtype=float)
    q_values = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    raw_adjusted[order] = raw_adjusted_sorted
    q_values[order] = q_sorted

    result.loc[valid, "rank"] = ranks
    result.loc[valid, "raw_adjusted"] = raw_adjusted
    result.loc[valid, "q_value"] = q_values
    return result


def benjamini_hochberg(pvalues):
    return benjamini_hochberg_details(pvalues)["q_value"]


def rank_descending_abs(series):
    return series.abs().rank(method="min", ascending=False)


def canonical_heterogeneity_metric(metric):
    """Map GUI labels to internal metric names."""
    metric_text = str(metric).strip().lower()
    if metric_text in {"sd", "standard deviation"}:
        return "SD"
    if metric_text in {"variance", "var"}:
        return "VARIANCE"
    if metric_text in {"mad", "median absolute deviation", "mad (not normalised)", "mad (not normalized)", "mad (raw median absolute deviation)"}:
        return "MAD"
    if metric_text in {"normalised mad", "normalized mad", "normalised mad (robust cv analogue)", "normalized mad (robust cv analogue)"}:
        return "NORMALISED_MAD"
    return "IQR"


def heterogeneity_metric_display_name(metric):
    canonical = canonical_heterogeneity_metric(metric)
    if canonical == "NORMALISED_MAD":
        return "Normalised MAD (robust CV analogue)"
    if canonical == "MAD":
        return "MAD (not normalised)"
    if canonical == "SD":
        return "SD"
    if canonical == "VARIANCE":
        return "Variance"
    return "IQR"


def guess_condition_keywords(columns):
    joined = " ".join(columns)
    defaults = []
    # Prefer control/untreated terms before treated terms so that "treated" is
    # not selected simply because it is a substring of "untreated".
    for candidate in ("U20S", "untreated", "control", "ctrl", "6TG", "treated"):
        if re.search(re.escape(candidate), joined, flags=re.IGNORECASE):
            defaults.append(candidate)

    has_u20s = any(re.search(r"u20s", c, flags=re.IGNORECASE) for c in columns)
    has_6tg = any(re.search(r"6tg", c, flags=re.IGNORECASE) for c in columns)
    has_untreated = any(re.search(r"untreated|control|ctrl", c, flags=re.IGNORECASE) for c in columns)
    if has_untreated and has_6tg:
        # Most common layout for this dataset: control/untreated versus 6TG.
        if any(re.search(r"untreated", c, flags=re.IGNORECASE) for c in columns):
            return "Untreated", "6TG"
        if any(re.search(r"control", c, flags=re.IGNORECASE) for c in columns):
            return "control", "6TG"
        return "ctrl", "6TG"
    if has_u20s and has_6tg:
        return "U20S", "6TG"

    if len(defaults) >= 2:
        return defaults[0], defaults[1]
    if len(defaults) == 1:
        return defaults[0], ""
    return "", ""


def _condition_keyword_matches_column(column_name, keyword):
    """Return True when a condition keyword matches a column safely.

    Matching is token-aware first, so a keyword like "treated" will not
    accidentally match the column name "Untreated_01". If token matching finds
    nothing for a keyword across all columns, get_condition_columns falls back
    to substring matching for backwards compatibility with older column names.
    """
    col = str(column_name).lower()
    kw = str(keyword).strip().lower()
    if not kw:
        return False
    tokens = [t for t in re.split(r"[^a-z0-9]+", col) if t]
    kw_tokens = [t for t in re.split(r"[^a-z0-9]+", kw) if t]
    if not kw_tokens:
        return False
    if len(kw_tokens) == 1:
        return kw_tokens[0] in tokens
    joined_tokens = " ".join(tokens)
    return " ".join(kw_tokens) in joined_tokens


def get_condition_columns(df, protein_column, condition_a_keyword, condition_b_keyword):
    if protein_column not in df.columns:
        raise ValueError(f"Protein column '{protein_column}' was not found in the file.")

    data_columns = [c for c in df.columns if c != protein_column]

    if not condition_a_keyword.strip():
        raise ValueError("Condition A keyword is empty.")
    if not condition_b_keyword.strip():
        raise ValueError("Condition B keyword is empty.")

    # Token-aware matching prevents Condition A keyword "treated" from also
    # capturing "Untreated" columns. This was the most likely cause of inverted
    # or misleading violin plots.
    cond_a_cols = [c for c in data_columns if _condition_keyword_matches_column(c, condition_a_keyword)]
    cond_b_cols = [c for c in data_columns if _condition_keyword_matches_column(c, condition_b_keyword)]

    # Backwards-compatible fallback for unusual column names without separators.
    if len(cond_a_cols) == 0:
        cond_a_cols = [c for c in data_columns if condition_a_keyword.lower() in str(c).lower()]
    if len(cond_b_cols) == 0:
        cond_b_cols = [c for c in data_columns if condition_b_keyword.lower() in str(c).lower()]

    if len(cond_a_cols) == 0:
        raise ValueError(f"No columns matched Condition A keyword '{condition_a_keyword}'.")
    if len(cond_b_cols) == 0:
        raise ValueError(f"No columns matched Condition B keyword '{condition_b_keyword}'.")

    overlap = sorted(set(cond_a_cols).intersection(cond_b_cols))
    if overlap:
        preview = ", ".join(map(str, overlap[:10]))
        more = "..." if len(overlap) > 10 else ""
        raise ValueError(
            "Some columns matched both condition keywords, which can invert or distort plots. "
            f"Please use more specific keywords. Overlapping columns: {preview}{more}"
        )

    return cond_a_cols, cond_b_cols


def summarise_two_conditions(
    df,
    protein_column,
    condition_a_keyword,
    condition_b_keyword,
    condition_a_label="Condition_A",
    condition_b_label="Condition_B",
    ddof=1,
):
    cond_a_cols, cond_b_cols = get_condition_columns(
        df, protein_column, condition_a_keyword, condition_b_keyword
    )

    numeric_cols = cond_a_cols + cond_b_cols
    work = df[[protein_column] + numeric_cols].copy()
    work[numeric_cols] = work[numeric_cols].apply(pd.to_numeric, errors="coerce")

    a_vals = work[cond_a_cols]
    b_vals = work[cond_b_cols]

    n_a = a_vals.notna().sum(axis=1)
    n_b = b_vals.notna().sum(axis=1)

    mean_a = a_vals.mean(axis=1)
    mean_b = b_vals.mean(axis=1)
    median_a = a_vals.median(axis=1)
    median_b = b_vals.median(axis=1)

    var_a = a_vals.var(axis=1, ddof=ddof)
    var_b = b_vals.var(axis=1, ddof=ddof)

    sd_a = a_vals.std(axis=1, ddof=ddof)
    sd_b = b_vals.std(axis=1, ddof=ddof)

    # Robust CV-style metric: standard deviation divided by the median abundance
    # rather than the mean abundance. This is less sensitive to outlying cells.
    cv_a = pd.Series(
        [safe_positive_divide(s, m) for s, m in zip(sd_a, median_a)],
        index=work.index,
        dtype=float,
    )
    cv_b = pd.Series(
        [safe_positive_divide(s, m) for s, m in zip(sd_b, median_b)],
        index=work.index,
        dtype=float,
    )

    cv2_a = cv_a ** 2
    cv2_b = cv_b ** 2

    var_over_mean_a = pd.Series(
        [safe_positive_divide(v, m) for v, m in zip(var_a, mean_a)],
        index=work.index,
        dtype=float,
    )
    var_over_mean_b = pd.Series(
        [safe_positive_divide(v, m) for v, m in zip(var_b, mean_b)],
        index=work.index,
        dtype=float,
    )

    delta_mean = mean_b - mean_a
    mean_ratio_b_over_a = pd.Series(
        [safe_positive_divide(mb, ma) for mb, ma in zip(mean_b, mean_a)],
        index=work.index,
        dtype=float,
    )

    log2_mean_ratio_b_over_a = pd.Series(index=work.index, dtype=float)
    positive_mask = (mean_a > 0) & (mean_b > 0)
    log2_mean_ratio_b_over_a.loc[positive_mask] = np.log2(
        mean_b.loc[positive_mask] / mean_a.loc[positive_mask]
    )
    log2_mean_ratio_b_over_a.loc[~positive_mask] = np.nan

    variance_ratio_b_over_a = pd.Series(
        [safe_divide(vb, va) for vb, va in zip(var_b, var_a)],
        index=work.index,
        dtype=float,
    )
    cv2_ratio_b_over_a = pd.Series(
        [safe_divide(vb, va) for vb, va in zip(cv2_b, cv2_a)],
        index=work.index,
        dtype=float,
    )
    var_over_mean_ratio_b_over_a = pd.Series(
        [safe_divide(vb, va) for vb, va in zip(var_over_mean_b, var_over_mean_a)],
        index=work.index,
        dtype=float,
    )

    summary_df = pd.DataFrame({
        "protein": work[protein_column],
        f"n_{condition_a_label}": n_a,
        f"n_{condition_b_label}": n_b,
        f"mean_{condition_a_label}": mean_a,
        f"mean_{condition_b_label}": mean_b,
        f"median_{condition_a_label}": median_a,
        f"median_{condition_b_label}": median_b,
        f"delta_mean_{condition_b_label}_minus_{condition_a_label}": delta_mean,
        f"mean_ratio_{condition_b_label}_over_{condition_a_label}": mean_ratio_b_over_a,
        f"log2_mean_ratio_{condition_b_label}_over_{condition_a_label}": log2_mean_ratio_b_over_a,
        f"variance_{condition_a_label}": var_a,
        f"variance_{condition_b_label}": var_b,
        f"variance_ratio_{condition_b_label}_over_{condition_a_label}": variance_ratio_b_over_a,
        f"sd_{condition_a_label}": sd_a,
        f"sd_{condition_b_label}": sd_b,
        f"cv_{condition_a_label}": cv_a,
        f"cv_{condition_b_label}": cv_b,
        f"cv2_{condition_a_label}": cv2_a,
        f"cv2_{condition_b_label}": cv2_b,
        f"cv2_ratio_{condition_b_label}_over_{condition_a_label}": cv2_ratio_b_over_a,
        f"var_over_mean_{condition_a_label}": var_over_mean_a,
        f"var_over_mean_{condition_b_label}": var_over_mean_b,
        f"var_over_mean_ratio_{condition_b_label}_over_{condition_a_label}": var_over_mean_ratio_b_over_a,
    })

    return summary_df, cond_a_cols, cond_b_cols


def build_next_step_analysis(
    df,
    protein_column,
    condition_a_keyword,
    condition_b_keyword,
    condition_a_label="Condition_A",
    condition_b_label="Condition_B",
    ddof=1,
    min_valid_n=3,
    heterogeneity_metric="IQR",
):
    cond_a_cols, cond_b_cols = get_condition_columns(
        df, protein_column, condition_a_keyword, condition_b_keyword
    )
    numeric_cols = cond_a_cols + cond_b_cols
    work = df[[protein_column] + numeric_cols].copy()
    work[numeric_cols] = work[numeric_cols].apply(pd.to_numeric, errors="coerce")

    rows = []
    heterogeneity_metric = canonical_heterogeneity_metric(heterogeneity_metric)

    for _, row in work.iterrows():
        a = row[cond_a_cols].to_numpy(dtype=float)
        b = row[cond_b_cols].to_numpy(dtype=float)
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]

        n_a = int(a.size)
        n_b = int(b.size)

        mean_a = float(np.mean(a)) if n_a else np.nan
        mean_b = float(np.mean(b)) if n_b else np.nan
        median_a = float(np.median(a)) if n_a else np.nan
        median_b = float(np.median(b)) if n_b else np.nan
        var_a = float(np.var(a, ddof=ddof)) if n_a > ddof else np.nan
        var_b = float(np.var(b, ddof=ddof)) if n_b > ddof else np.nan
        sd_a = float(np.std(a, ddof=ddof)) if n_a > ddof else np.nan
        sd_b = float(np.std(b, ddof=ddof)) if n_b > ddof else np.nan
        iqr_a = compute_iqr(a)
        iqr_b = compute_iqr(b)
        mad_a = compute_mad(a)
        mad_b = compute_mad(b)
        normalised_mad_a = compute_normalised_mad(a)
        normalised_mad_b = compute_normalised_mad(b)

        delta_mean = mean_b - mean_a if n_a and n_b else np.nan
        delta_median = median_b - median_a if n_a and n_b else np.nan

        log2_mean_ratio = np.nan
        if n_a and n_b and mean_a > 0 and mean_b > 0:
            log2_mean_ratio = float(np.log2(mean_b / mean_a))

        if heterogeneity_metric == "SD":
            het_a = sd_a
            het_b = sd_b
        elif heterogeneity_metric == "VARIANCE":
            het_a = var_a
            het_b = var_b
        elif heterogeneity_metric == "NORMALISED_MAD":
            het_a = normalised_mad_a
            het_b = normalised_mad_b
        elif heterogeneity_metric == "MAD":
            het_a = mad_a
            het_b = mad_b
        else:
            heterogeneity_metric = "IQR"
            het_a = iqr_a
            het_b = iqr_b

        delta_het = het_b - het_a if pd.notna(het_a) and pd.notna(het_b) else np.nan
        delta_normalised_mad = (
            normalised_mad_b - normalised_mad_a
            if pd.notna(normalised_mad_a) and pd.notna(normalised_mad_b)
            else np.nan
        )
        het_ratio = safe_divide(het_b, het_a)

        mean_p = np.nan
        variance_p = np.nan
        mean_test = "Welch_t_test" if SCIPY_AVAILABLE else "not_available"
        variance_test = "Brown_Forsythe" if SCIPY_AVAILABLE else "not_available"

        if SCIPY_AVAILABLE and n_a >= min_valid_n and n_b >= min_valid_n:
            try:
                mean_p = float(stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)
            except Exception:
                mean_p = np.nan
            try:
                variance_p = float(stats.levene(a, b, center="median").pvalue)
            except Exception:
                variance_p = np.nan

        rows.append({
            "protein": row[protein_column],
            f"n_{condition_a_label}": n_a,
            f"n_{condition_b_label}": n_b,
            f"mean_{condition_a_label}": mean_a,
            f"mean_{condition_b_label}": mean_b,
            f"median_{condition_a_label}": median_a,
            f"median_{condition_b_label}": median_b,
            f"variance_{condition_a_label}": var_a,
            f"variance_{condition_b_label}": var_b,
            f"sd_{condition_a_label}": sd_a,
            f"sd_{condition_b_label}": sd_b,
            f"iqr_{condition_a_label}": iqr_a,
            f"iqr_{condition_b_label}": iqr_b,
            f"mad_{condition_a_label}": mad_a,
            f"mad_{condition_b_label}": mad_b,
            f"normalised_mad_{condition_a_label}": normalised_mad_a,
            f"normalised_mad_{condition_b_label}": normalised_mad_b,
            f"delta_normalised_mad_{condition_b_label}_minus_{condition_a_label}": delta_normalised_mad,
            f"abs_delta_normalised_mad_{condition_b_label}_minus_{condition_a_label}": abs(delta_normalised_mad) if pd.notna(delta_normalised_mad) else np.nan,
            f"delta_mean_{condition_b_label}_minus_{condition_a_label}": delta_mean,
            f"abs_delta_mean_{condition_b_label}_minus_{condition_a_label}": abs(delta_mean) if pd.notna(delta_mean) else np.nan,
            f"delta_median_{condition_b_label}_minus_{condition_a_label}": delta_median,
            f"log2_mean_ratio_{condition_b_label}_over_{condition_a_label}": log2_mean_ratio,
            "heterogeneity_metric_used": heterogeneity_metric_display_name(heterogeneity_metric),
            f"heterogeneity_{condition_a_label}": het_a,
            f"heterogeneity_{condition_b_label}": het_b,
            f"delta_heterogeneity_{condition_b_label}_minus_{condition_a_label}": delta_het,
            f"abs_delta_heterogeneity_{condition_b_label}_minus_{condition_a_label}": abs(delta_het) if pd.notna(delta_het) else np.nan,
            f"heterogeneity_ratio_{condition_b_label}_over_{condition_a_label}": het_ratio,
            "mean_test": mean_test,
            "mean_p_value": mean_p,
            "heterogeneity_test": variance_test,
            "heterogeneity_p_value": variance_p,
        })

    analysis_df = pd.DataFrame(rows)
    analysis_df["mean_q_value_bh_fdr"] = benjamini_hochberg(analysis_df["mean_p_value"])
    analysis_df["heterogeneity_q_value_bh_fdr"] = benjamini_hochberg(analysis_df["heterogeneity_p_value"])
    analysis_df["rank_by_abs_mean_change"] = rank_descending_abs(
        analysis_df[f"delta_mean_{condition_b_label}_minus_{condition_a_label}"]
    )
    analysis_df["rank_by_abs_heterogeneity_change"] = rank_descending_abs(
        analysis_df[f"delta_heterogeneity_{condition_b_label}_minus_{condition_a_label}"]
    )

    analysis_df["passes_min_valid_n_for_tests"] = (
        (analysis_df[f"n_{condition_a_label}"] >= min_valid_n)
        & (analysis_df[f"n_{condition_b_label}"] >= min_valid_n)
    )

    mean_sorted = analysis_df.sort_values(
        by=["rank_by_abs_mean_change", "mean_q_value_bh_fdr", "protein"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    hetero_sorted = analysis_df.sort_values(
        by=["rank_by_abs_heterogeneity_change", "heterogeneity_q_value_bh_fdr", "protein"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return analysis_df, mean_sorted, hetero_sorted, cond_a_cols, cond_b_cols


def compute_pca_projection(wide_df, protein_column, cond_a_cols, cond_b_cols, top_variable_proteins=500):
    numeric_cols = cond_a_cols + cond_b_cols
    work = wide_df[[protein_column] + numeric_cols].copy()
    work[numeric_cols] = work[numeric_cols].apply(pd.to_numeric, errors="coerce")

    values = work[numeric_cols].to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("No numeric abundance columns were found for PCA.")

    valid_counts = np.sum(~np.isnan(values), axis=1)
    keep = valid_counts >= 2
    values = values[keep]
    proteins = work.loc[keep, protein_column].astype(str).to_numpy()

    if values.shape[0] < 2:
        raise ValueError("Not enough proteins with at least 2 valid values for PCA.")

    row_means = np.nanmean(values, axis=1, keepdims=True)
    filled = np.where(np.isnan(values), row_means, values)

    row_vars = np.nanvar(values, axis=1)
    order = np.argsort(-row_vars)
    if top_variable_proteins and top_variable_proteins > 0:
        order = order[: min(int(top_variable_proteins), len(order))]
    filled = filled[order]
    proteins = proteins[order]

    row_means = np.mean(filled, axis=1, keepdims=True)
    row_sds = np.std(filled, axis=1, ddof=1, keepdims=True)
    row_sds[row_sds < EPSILON] = 1.0
    standardized = (filled - row_means) / row_sds

    X = standardized.T  # cells x proteins
    X = X - np.mean(X, axis=0, keepdims=True)

    if X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError("Not enough data dimensions for PCA after filtering.")

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    scores = U[:, :2] * S[:2]

    eigenvalues = (S ** 2) / max(X.shape[0] - 1, 1)
    total_variance = np.sum(eigenvalues)
    explained = eigenvalues[:2] / total_variance if total_variance > 0 else np.array([np.nan, np.nan])

    cells = cond_a_cols + cond_b_cols
    conditions = (["A"] * len(cond_a_cols)) + (["B"] * len(cond_b_cols))
    result = pd.DataFrame({
        "cell": cells,
        "condition_code": conditions,
        "PC1": scores[:, 0],
        "PC2": scores[:, 1],
    })
    return result, explained, proteins


class MatplotlibTab:
    def __init__(self, parent, app=None, figsize=(9.0, 6.2)):
        self.app = app
        self.container = ttk.Frame(parent)
        self.controls = ttk.Frame(self.container)
        self.controls.pack(fill="x", padx=6, pady=(6, 4))

        self.figure_frame = ttk.Frame(self.container)
        self.figure_frame.pack(fill="both", expand=True)

        initial_dpi = 100
        if self.app is not None:
            try:
                initial_dpi = int(self.app.preview_dpi_var.get())
            except Exception:
                initial_dpi = 100

        self.figure = Figure(figsize=figsize, dpi=initial_dpi)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.figure_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.figure_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill="x")

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self.container, textvariable=self.status_var, anchor="w").pack(fill="x", padx=8, pady=(0, 6))

    def _get_preview_dpi(self):
        if self.app is not None:
            try:
                return int(self.app.preview_dpi_var.get())
            except Exception:
                pass
        return 100

    def _get_export_dpi(self):
        if self.app is not None:
            try:
                return int(self.app.export_dpi_var.get())
            except Exception:
                pass
        return 300

    def apply_size_and_dpi(self, width, height):
        width = max(float(width), 1.0)
        height = max(float(height), 1.0)
        self.figure.set_size_inches(width, height, forward=True)
        self.figure.set_dpi(self._get_preview_dpi())
        self.canvas.draw_idle()

    def set_status(self, text):
        self.status_var.set(text)

    def save_figure(self, default_name="plot.png"):
        path = filedialog.asksaveasfilename(
            title="Save plot image",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG image", "*.png"), ("PDF file", "*.pdf"), ("SVG file", "*.svg")],
        )
        if not path:
            return
        transparent = False
        if self.app is not None:
            try:
                transparent = bool(self.app.transparent_bg_var.get())
            except Exception:
                transparent = False
        self.figure.savefig(path, dpi=self._get_export_dpi(), bbox_inches="tight", transparent=transparent)
        self.set_status(f"Saved plot to: {path}")
        messagebox.showinfo("Done", f"Plot saved successfully.\n\n{path}")


class SingleCellSummaryGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Single-Cell Proteomics Summary + Visual Analysis")
        self._configure_window_geometry()

        self.df = None
        self.file_path = None

        self._build_ui()

    def _configure_window_geometry(self):
        """Size the app to the available screen first; scrolling handles any overflow."""
        desired_width = 1360
        desired_height = 900
        try:
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
        except Exception:
            screen_width = desired_width
            screen_height = desired_height

        width = min(desired_width, max(900, screen_width - 80))
        height = min(desired_height, max(650, screen_height - 120))
        x = max((screen_width - width) // 2, 0)
        y = max((screen_height - height) // 2, 0)

        self.geometry(f"{width}x{height}+{x}+{y}")
        self.minsize(min(1180, width), min(760, height))

    def _make_scrollable_root(self):
        """Create a root-level scrollable area for smaller screens and scaled displays."""
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, highlightthickness=0)
        v_scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(container, orient="horizontal", command=canvas.xview)

        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        scrollable_frame = ttk.Frame(canvas, padding=12)
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def _update_scrollregion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _fit_frame_width(event):
            requested_width = scrollable_frame.winfo_reqwidth()
            canvas.itemconfigure(canvas_window, width=max(event.width, requested_width))
            _update_scrollregion()

        scrollable_frame.bind("<Configure>", _update_scrollregion)
        canvas.bind("<Configure>", _fit_frame_width)

        def _on_mousewheel(event):
            if getattr(event, "num", None) == 4:
                canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(1, "units")
            else:
                delta = event.delta
                if abs(delta) < 120:
                    canvas.yview_scroll(-1 if delta > 0 else 1, "units")
                else:
                    canvas.yview_scroll(int(-delta / 120), "units")

        def _bind_mousewheel(_event=None):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_mousewheel(_event=None):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)

        self._root_scroll_canvas = canvas
        return scrollable_frame

    def _build_ui(self):
        outer = self._make_scrollable_root()
        title = ttk.Label(
            outer,
            text="Single-cell proteomics: summary, ranked analysis, and visualisation",
            font=("Segoe UI", 13, "bold"),
        )
        title.pack(anchor="w", pady=(0, 10))

        desc = ttk.Label(
            outer,
            text=(
                "Load a wide-format CSV where proteins are rows and single cells are columns. "
                "Use the first tabs to export tables, then use the plot tabs for a global protein-level view, "
                "ranked heterogeneity view, per-protein violin plots, and a cell-level PCA view."
            ),
            wraplength=1280,
            justify="left",
        )
        desc.pack(anchor="w", pady=(0, 12))

        file_frame = ttk.LabelFrame(outer, text="1) Input file", padding=10)
        file_frame.pack(fill="x", pady=(0, 10))

        self.file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(file_frame, text="Browse CSV", command=self.load_file).pack(side="left")

        settings_frame = ttk.LabelFrame(outer, text="2) File mapping and condition matching", padding=10)
        settings_frame.pack(fill="x", pady=(0, 10))

        grid = ttk.Frame(settings_frame)
        grid.pack(fill="x")

        self.protein_col_var = tk.StringVar()
        self.label_a_var = tk.StringVar(value="Untreated")
        self.label_b_var = tk.StringVar(value="6TG")
        self.keyword_a_var = tk.StringVar(value="U20S")
        self.keyword_b_var = tk.StringVar(value="6TG")
        self.ddof_var = tk.IntVar(value=1)
        self.min_valid_n_var = tk.IntVar(value=3)
        self.heterogeneity_metric_var = tk.StringVar(value="Normalised MAD (robust CV analogue)")

        self.q_threshold_var = tk.DoubleVar(value=0.05)
        self.effect_label_top_n_var = tk.IntVar(value=15)
        self.effect_label_method_var = tk.StringVar(value="Highest heterogeneity")
        self.hetero_top_n_var = tk.IntVar(value=20)
        self.violin_top_n_var = tk.IntVar(value=6)
        self.violin_manual_proteins_var = tk.StringVar(value="")
        self.hetero_sort_mode_var = tk.StringVar(value="Largest increase")
        self.violin_sort_mode_var = tk.StringVar(value="Largest increase")
        self.pca_top_variable_var = tk.IntVar(value=500)

        self.effect_view_mode_var = tk.StringVar(value="Group B - Group A")
        self.rank_view_mode_var = tk.StringVar(value="Group B - Group A")
        self.violin_view_mode_var = tk.StringVar(value="Group B - Group A")
        self.pca_view_mode_var = tk.StringVar(value="Both groups")

        self.preview_dpi_var = tk.IntVar(value=100)
        self.export_dpi_var = tk.IntVar(value=300)
        self.transparent_bg_var = tk.BooleanVar(value=False)
        self.condition_a_color_var = tk.StringVar(value=CONDITION_A_COLOR)
        self.condition_b_color_var = tk.StringVar(value=CONDITION_B_COLOR)
        self.title_font_var = tk.IntVar(value=14)
        self.axis_label_font_var = tk.IntVar(value=12)
        self.tick_font_var = tk.IntVar(value=10)
        self.legend_font_var = tk.IntVar(value=10)
        self.annotation_font_var = tk.IntVar(value=9)
        self.title_bold_var = tk.BooleanVar(value=False)
        self.title_italic_var = tk.BooleanVar(value=False)
        self.axis_bold_var = tk.BooleanVar(value=False)
        self.axis_italic_var = tk.BooleanVar(value=False)
        self.tick_bold_var = tk.BooleanVar(value=False)
        self.tick_italic_var = tk.BooleanVar(value=False)
        self.legend_bold_var = tk.BooleanVar(value=False)
        self.legend_italic_var = tk.BooleanVar(value=False)

        self.effect_width_var = tk.DoubleVar(value=9.6)
        self.effect_height_var = tk.DoubleVar(value=6.4)
        self.rank_width_var = tk.DoubleVar(value=9.6)
        self.rank_height_var = tk.DoubleVar(value=6.4)
        self.violin_width_var = tk.DoubleVar(value=10.0)
        self.violin_height_var = tk.DoubleVar(value=8.0)
        self.violin_max_panels_per_image_var = tk.IntVar(value=12)
        self.pca_width_var = tk.DoubleVar(value=9.6)
        self.pca_height_var = tk.DoubleVar(value=6.4)

        self.effect_title_text_var = tk.StringVar(value="")
        self.effect_xlabel_text_var = tk.StringVar(value="")
        self.effect_ylabel_text_var = tk.StringVar(value="")
        self.effect_show_xticks_var = tk.BooleanVar(value=True)
        self.effect_show_yticks_var = tk.BooleanVar(value=True)

        self.rank_title_text_var = tk.StringVar(value="")
        self.rank_xlabel_text_var = tk.StringVar(value="")
        self.rank_ylabel_text_var = tk.StringVar(value="")
        self.rank_show_xticks_var = tk.BooleanVar(value=True)
        self.rank_show_yticks_var = tk.BooleanVar(value=True)

        self.violin_title_text_var = tk.StringVar(value="")
        self.violin_xlabel_text_var = tk.StringVar(value="")
        self.violin_ylabel_text_var = tk.StringVar(value="")
        self.violin_show_xticks_var = tk.BooleanVar(value=True)
        self.violin_show_yticks_var = tk.BooleanVar(value=True)
        self.violin_fix_shared_yaxis_var = tk.BooleanVar(value=False)
        self.violin_dot_size_var = tk.DoubleVar(value=8.0)

        self.bimodality_top_n_var = tk.IntVar(value=6)
        self.bimodality_min_valid_n_var = tk.IntVar(value=10)
        self.bimodality_min_fraction_per_mode_var = tk.DoubleVar(value=0.15)
        self.bimodality_min_cells_per_mode_var = tk.IntVar(value=10)
        self.bimodality_min_mode_separation_var = tk.DoubleVar(value=0.5)
        self.bimodality_max_dip_p_value_var = tk.DoubleVar(value=0.05)
        self.bimodality_min_valley_depth_var = tk.DoubleVar(value=0.20)
        self.bimodality_require_gmm_improvement_var = tk.BooleanVar(value=True)
        self.bimodality_require_diptest_var = tk.BooleanVar(value=False)
        self.bimodality_sort_mode_var = tk.StringVar(value="Highest balanced bimodality in either condition")
        self.bimodality_fix_shared_yaxis_var = tk.BooleanVar(value=True)
        self.bimodality_dot_size_var = tk.DoubleVar(value=4.0)

        self.pca_title_text_var = tk.StringVar(value="")
        self.pca_xlabel_text_var = tk.StringVar(value="")
        self.pca_ylabel_text_var = tk.StringVar(value="")
        self.pca_show_xticks_var = tk.BooleanVar(value=True)
        self.pca_show_yticks_var = tk.BooleanVar(value=True)

        ttk.Label(grid, text="Protein ID column").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=4)
        self.protein_col_combo = ttk.Combobox(grid, textvariable=self.protein_col_var, state="readonly", width=40)
        self.protein_col_combo.grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(grid, text="Condition A label").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(grid, textvariable=self.label_a_var, width=20).grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(grid, text="Condition A keyword").grid(row=1, column=2, sticky="w", padx=(20, 10), pady=4)
        ttk.Entry(grid, textvariable=self.keyword_a_var, width=20).grid(row=1, column=3, sticky="w", pady=4)

        ttk.Label(grid, text="Condition B label").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(grid, textvariable=self.label_b_var, width=20).grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(grid, text="Condition B keyword").grid(row=2, column=2, sticky="w", padx=(20, 10), pady=4)
        ttk.Entry(grid, textvariable=self.keyword_b_var, width=20).grid(row=2, column=3, sticky="w", pady=4)

        ttk.Label(grid, text="Variance setting").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=4)
        ddof_row = ttk.Frame(grid)
        ddof_row.grid(row=3, column=1, columnspan=3, sticky="w", pady=4)
        ttk.Radiobutton(ddof_row, text="Sample variance (recommended)", variable=self.ddof_var, value=1).pack(side="left", padx=(0, 18))
        ttk.Radiobutton(ddof_row, text="Population variance", variable=self.ddof_var, value=0).pack(side="left")

        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(3, weight=1)

        plot_style_frame = ttk.LabelFrame(outer, text="3) Plot styling and export", padding=10)
        plot_style_frame.pack(fill="x", pady=(0, 10))
        self._build_plot_style_frame(plot_style_frame)

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)

        self.summary_tab = ttk.Frame(notebook, padding=10)
        self.analysis_tab = ttk.Frame(notebook, padding=10)
        notebook.add(self.summary_tab, text="Summary export")
        notebook.add(self.analysis_tab, text="Next-step analysis")

        self._build_summary_tab()
        self._build_analysis_tab()

        self.effect_map_tab = MatplotlibTab(notebook, app=self, figsize=(self.effect_width_var.get(), self.effect_height_var.get()))
        self.hetero_rank_tab = MatplotlibTab(notebook, app=self, figsize=(self.rank_width_var.get(), self.rank_height_var.get()))
        self.violin_tab = MatplotlibTab(notebook, app=self, figsize=(self.violin_width_var.get(), self.violin_height_var.get()))
        self.bimodality_tab = MatplotlibTab(notebook, app=self, figsize=(self.violin_width_var.get(), self.violin_height_var.get()))
        self.pca_tab = MatplotlibTab(notebook, app=self, figsize=(self.pca_width_var.get(), self.pca_height_var.get()))
        notebook.add(self.effect_map_tab.container, text="Plot 1: effect map")
        notebook.add(self.hetero_rank_tab.container, text="Plot 2: heterogeneity ranking")
        notebook.add(self.violin_tab.container, text="Plot 3: top protein violins")
        notebook.add(self.bimodality_tab.container, text="Plot 4: bimodality violins")
        notebook.add(self.pca_tab.container, text="Plot 5: PCA of cells")

        self._build_effect_map_tab()
        self._build_hetero_rank_tab()
        self._build_violin_tab()
        self._build_bimodality_tab()
        self._build_pca_tab()

    def _build_plot_style_frame(self, parent):
        row1 = ttk.Frame(parent)
        row1.pack(fill="x", pady=(0, 8))
        ttk.Label(row1, text="Preview DPI").pack(side="left", padx=(0, 6))
        ttk.Spinbox(row1, from_=50, to=600, textvariable=self.preview_dpi_var, width=7).pack(side="left", padx=(0, 14))
        ttk.Label(row1, text="Export DPI").pack(side="left", padx=(0, 6))
        ttk.Spinbox(row1, from_=72, to=1200, textvariable=self.export_dpi_var, width=7).pack(side="left", padx=(0, 18))
        ttk.Checkbutton(row1, text="Transparent background", variable=self.transparent_bg_var).pack(side="left", padx=(0, 18))
        ttk.Label(row1, text="Group A color").pack(side="left", padx=(0, 6))
        ttk.Entry(row1, textvariable=self.condition_a_color_var, width=12).pack(side="left", padx=(0, 12))
        ttk.Label(row1, text="Group B color").pack(side="left", padx=(0, 6))
        ttk.Entry(row1, textvariable=self.condition_b_color_var, width=12).pack(side="left", padx=(0, 18))

        ttk.Label(row1, text="Title font").pack(side="left", padx=(0, 6))
        ttk.Spinbox(row1, from_=6, to=40, textvariable=self.title_font_var, width=5).pack(side="left", padx=(0, 10))
        ttk.Label(row1, text="Axis label font").pack(side="left", padx=(0, 6))
        ttk.Spinbox(row1, from_=6, to=40, textvariable=self.axis_label_font_var, width=5).pack(side="left", padx=(0, 10))
        ttk.Label(row1, text="Tick font").pack(side="left", padx=(0, 6))
        ttk.Spinbox(row1, from_=6, to=40, textvariable=self.tick_font_var, width=5).pack(side="left", padx=(0, 10))
        ttk.Label(row1, text="Legend font").pack(side="left", padx=(0, 6))
        ttk.Spinbox(row1, from_=6, to=40, textvariable=self.legend_font_var, width=5).pack(side="left", padx=(0, 10))
        ttk.Label(row1, text="Annotation font").pack(side="left", padx=(0, 6))
        ttk.Spinbox(row1, from_=6, to=40, textvariable=self.annotation_font_var, width=5).pack(side="left")

        row2 = ttk.Frame(parent)
        row2.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(row2, text="Title bold", variable=self.title_bold_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(row2, text="Title italic", variable=self.title_italic_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(row2, text="Axis bold", variable=self.axis_bold_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(row2, text="Axis italic", variable=self.axis_italic_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(row2, text="Tick bold", variable=self.tick_bold_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(row2, text="Tick italic", variable=self.tick_italic_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(row2, text="Legend bold", variable=self.legend_bold_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(row2, text="Legend italic", variable=self.legend_italic_var).pack(side="left")

        row3 = ttk.Frame(parent)
        row3.pack(fill="x", pady=(0, 8))
        controls = [
            ("Effect map W", self.effect_width_var), ("H", self.effect_height_var),
            ("Ranking W", self.rank_width_var), ("H", self.rank_height_var),
            ("Violins W", self.violin_width_var), ("H", self.violin_height_var),
            ("PCA W", self.pca_width_var), ("H", self.pca_height_var),
        ]
        for label, var in controls:
            ttk.Label(row3, text=label).pack(side="left", padx=(0, 6))
            ttk.Spinbox(row3, from_=2.0, to=30.0, increment=0.2, textvariable=var, width=6).pack(side="left", padx=(0, 12))
        ttk.Label(row3, text="Max violin panels/image").pack(side="left", padx=(0, 6))
        ttk.Spinbox(row3, from_=1, to=200, textvariable=self.violin_max_panels_per_image_var, width=6).pack(side="left")

        detail = ttk.LabelFrame(parent, text="Per-plot text overrides and tick labels", padding=8)
        detail.pack(fill="x", pady=(4, 0))

        self._add_plot_override_row(detail, 0, "Effect map", self.effect_title_text_var, self.effect_xlabel_text_var, self.effect_ylabel_text_var, self.effect_show_xticks_var, self.effect_show_yticks_var)
        self._add_plot_override_row(detail, 1, "Ranking", self.rank_title_text_var, self.rank_xlabel_text_var, self.rank_ylabel_text_var, self.rank_show_xticks_var, self.rank_show_yticks_var)
        self._add_plot_override_row(detail, 2, "Violins", self.violin_title_text_var, self.violin_xlabel_text_var, self.violin_ylabel_text_var, self.violin_show_xticks_var, self.violin_show_yticks_var)
        self._add_plot_override_row(detail, 3, "PCA", self.pca_title_text_var, self.pca_xlabel_text_var, self.pca_ylabel_text_var, self.pca_show_xticks_var, self.pca_show_yticks_var)

        helper = ttk.Label(
            parent,
            text=(
                "Font sizes are global across all plots. Figure widths and heights are controlled per plot tab. "
                "You can optionally override the title and axis labels for each plot and hide tick labels. "
                "For violin export, the max-panels setting splits large selections into multiple saved images."
            ),
            wraplength=1200,
            justify="left",
        )
        helper.pack(anchor="w", pady=(8, 0))

    def _add_plot_override_row(self, parent, row, label_text, title_var, xlabel_var, ylabel_var, show_xticks_var, show_yticks_var):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Label(parent, text="Title").grid(row=row, column=1, sticky="e", padx=(0, 4), pady=4)
        ttk.Entry(parent, textvariable=title_var, width=28).grid(row=row, column=2, sticky="ew", padx=(0, 10), pady=4)
        ttk.Label(parent, text="X label").grid(row=row, column=3, sticky="e", padx=(0, 4), pady=4)
        ttk.Entry(parent, textvariable=xlabel_var, width=18).grid(row=row, column=4, sticky="ew", padx=(0, 10), pady=4)
        ttk.Label(parent, text="Y label").grid(row=row, column=5, sticky="e", padx=(0, 4), pady=4)
        ttk.Entry(parent, textvariable=ylabel_var, width=18).grid(row=row, column=6, sticky="ew", padx=(0, 10), pady=4)
        ttk.Checkbutton(parent, text="Show X ticks", variable=show_xticks_var).grid(row=row, column=7, sticky="w", padx=(0, 8), pady=4)
        ttk.Checkbutton(parent, text="Show Y ticks", variable=show_yticks_var).grid(row=row, column=8, sticky="w", pady=4)
        for col in (2, 4, 6):
            parent.grid_columnconfigure(col, weight=1)

    def _normalize_color_string(self, value, fallback):
        s = str(value).strip()
        if not s:
            return fallback
        lower = s.lower()
        if lower.startswith("rgba"):
            s = s[4:].strip()
        s = s.strip().strip("()[]{}")
        s = s.replace(",", "").replace(" ", "")
        if s.startswith("#"):
            s = s[1:]
        if len(s) in (6, 8):
            try:
                int(s, 16)
                return f"#{s.upper()}"
            except Exception:
                return fallback
        return fallback

    def _condition_a_color(self):
        return self._normalize_color_string(self.condition_a_color_var.get(), CONDITION_A_COLOR)

    def _condition_b_color(self):
        return self._normalize_color_string(self.condition_b_color_var.get(), CONDITION_B_COLOR)

    def _font_size(self, which):
        mapping = {
            "title": self.title_font_var,
            "axis": self.axis_label_font_var,
            "tick": self.tick_font_var,
            "legend": self.legend_font_var,
            "annotation": self.annotation_font_var,
        }
        try:
            return int(mapping[which].get())
        except Exception:
            return {"title": 14, "axis": 12, "tick": 10, "legend": 10, "annotation": 9}[which]

    def _font_kwargs(self, which):
        if which == "title":
            bold = bool(self.title_bold_var.get())
            italic = bool(self.title_italic_var.get())
        elif which == "axis":
            bold = bool(self.axis_bold_var.get())
            italic = bool(self.axis_italic_var.get())
        elif which == "tick":
            bold = bool(self.tick_bold_var.get())
            italic = bool(self.tick_italic_var.get())
        elif which == "legend":
            bold = bool(self.legend_bold_var.get())
            italic = bool(self.legend_italic_var.get())
        else:
            bold = False
            italic = False
        return {
            "fontsize": self._font_size(which),
            "fontweight": "bold" if bold else "normal",
            "fontstyle": "italic" if italic else "normal",
        }

    def _prepare_standard_figure(self, tab, width_var, height_var):
        tab.apply_size_and_dpi(float(width_var.get()), float(height_var.get()))
        tab.figure.clear()
        return tab.figure

    def _get_plot_override_bundle(self, plot_key):
        mapping = {
            "effect": (self.effect_title_text_var, self.effect_xlabel_text_var, self.effect_ylabel_text_var, self.effect_show_xticks_var, self.effect_show_yticks_var),
            "rank": (self.rank_title_text_var, self.rank_xlabel_text_var, self.rank_ylabel_text_var, self.rank_show_xticks_var, self.rank_show_yticks_var),
            "violin": (self.violin_title_text_var, self.violin_xlabel_text_var, self.violin_ylabel_text_var, self.violin_show_xticks_var, self.violin_show_yticks_var),
            "pca": (self.pca_title_text_var, self.pca_xlabel_text_var, self.pca_ylabel_text_var, self.pca_show_xticks_var, self.pca_show_yticks_var),
        }
        return mapping[plot_key]

    def _style_axes_text(self, ax, plot_key, xlabel=None, ylabel=None, title=None, apply_title_override=True):
        title_var, xlabel_var, ylabel_var, show_xticks_var, show_yticks_var = self._get_plot_override_bundle(plot_key)
        final_title = (str(title_var.get()).strip() if apply_title_override else "") or title
        final_xlabel = str(xlabel_var.get()).strip() or xlabel
        final_ylabel = str(ylabel_var.get()).strip() or ylabel

        if final_xlabel is not None:
            ax.set_xlabel(final_xlabel, **self._font_kwargs("axis"))
        if final_ylabel is not None:
            ax.set_ylabel(final_ylabel, **self._font_kwargs("axis"))
        if final_title is not None:
            ax.set_title(final_title, **self._font_kwargs("title"))

        ax.tick_params(axis="x", labelbottom=bool(show_xticks_var.get()))
        ax.tick_params(axis="y", labelleft=bool(show_yticks_var.get()))
        for label in ax.get_xticklabels():
            label.set_fontsize(self._font_size("tick"))
            label.set_fontweight("bold" if self.tick_bold_var.get() else "normal")
            label.set_fontstyle("italic" if self.tick_italic_var.get() else "normal")
        for label in ax.get_yticklabels():
            label.set_fontsize(self._font_size("tick"))
            label.set_fontweight("bold" if self.tick_bold_var.get() else "normal")
            label.set_fontstyle("italic" if self.tick_italic_var.get() else "normal")

    def _style_legend(self, legend):
        if legend is None:
            return
        for text in legend.get_texts():
            text.set_fontsize(self._font_size("legend"))
            text.set_fontweight("bold" if self.legend_bold_var.get() else "normal")
            text.set_fontstyle("italic" if self.legend_italic_var.get() else "normal")

    def _add_spaced_labels(
        self,
        ax,
        label_df,
        x_col,
        y_col,
        label_col,
        fontsize=None,
        min_sep_px=12,
        x_offset_px=18,
        side_mode="auto",
        column_inset_px=8,
    ):
        if label_df is None or label_df.empty:
            return

        if fontsize is None:
            fontsize = self._font_size("annotation")

        ax.figure.canvas.draw()
        bbox = ax.get_window_extent()
        x_mid = 0.5 * (ax.get_xlim()[0] + ax.get_xlim()[1])

        work = label_df[[x_col, y_col, label_col]].copy()
        work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
        work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
        work = work.dropna(subset=[x_col, y_col])
        if work.empty:
            return

        if side_mode == "right":
            work["_side"] = "right"
            sides = ("right",)
        elif side_mode == "left":
            work["_side"] = "left"
            sides = ("left",)
        else:
            work["_side"] = np.where(work[x_col] >= x_mid, "right", "left")
            sides = ("right", "left")

        for side in sides:
            sub = work[work["_side"] == side].copy()
            if sub.empty:
                continue

            disp = ax.transData.transform(sub[[x_col, y_col]].to_numpy())
            sub["_disp_x"] = disp[:, 0]
            sub["_disp_y"] = disp[:, 1]
            sub = sub.sort_values(["_disp_y", x_col], ascending=[False, False]).reset_index(drop=True)

            n = len(sub)
            top_limit = bbox.y1 - 8
            bottom_limit = bbox.y0 + 8

            if n == 1:
                target_y = [0.5 * (top_limit + bottom_limit)]
            else:
                span = top_limit - bottom_limit
                requested_span = min_sep_px * (n - 1)
                if requested_span > span and n > 1:
                    min_sep_eff = span / max(n - 1, 1)
                else:
                    min_sep_eff = min_sep_px
                actual_span = min(span, min_sep_eff * (n - 1))
                start_y = top_limit
                end_y = top_limit - actual_span
                target_y = np.linspace(start_y, end_y, n).tolist()

            for (_, row), y_disp in zip(sub.iterrows(), target_y):
                if side == "right":
                    if side_mode == "right":
                        tx_disp = bbox.x1 - column_inset_px
                        ha = "right"
                    else:
                        tx_disp = min(row["_disp_x"] + x_offset_px, bbox.x1 - 4)
                        ha = "left"
                else:
                    if side_mode == "left":
                        tx_disp = bbox.x0 + column_inset_px
                        ha = "left"
                    else:
                        tx_disp = max(row["_disp_x"] - x_offset_px, bbox.x0 + 4)
                        ha = "right"

                tx_data, ty_data = ax.transData.inverted().transform((tx_disp, y_disp))
                ax.annotate(
                    str(row[label_col]),
                    xy=(row[x_col], row[y_col]),
                    xytext=(tx_data, ty_data),
                    textcoords="data",
                    ha=ha,
                    va="center",
                    fontsize=fontsize,
                    alpha=0.95,
                    arrowprops=dict(arrowstyle="-", color="0.45", lw=0.6, alpha=0.8, shrinkA=0, shrinkB=2),
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                    clip_on=False,
                )

    def _add_standard_labels(self, ax, label_df, x_col, y_col, label_col, fontsize=None, x_offset_pts=5, y_offset_pts=4):
        if label_df is None or label_df.empty:
            return
        if fontsize is None:
            fontsize = self._font_size("annotation")

        work = label_df[[x_col, y_col, label_col]].copy()
        work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
        work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
        work = work.dropna(subset=[x_col, y_col])
        if work.empty:
            return

        for _, row in work.iterrows():
            ax.annotate(
                str(row[label_col]),
                xy=(row[x_col], row[y_col]),
                xytext=(x_offset_pts, y_offset_pts),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=fontsize,
                alpha=0.95,
            )

    def _compute_local_residual_score(self, x_values, y_values):
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        score = np.full(len(x), np.nan, dtype=float)
        expected = np.full(len(x), np.nan, dtype=float)
        residual = np.full(len(x), np.nan, dtype=float)
        if valid.sum() < 8:
            return score, expected, residual

        idx = np.where(valid)[0]
        xv = x[valid]
        yv = y[valid]
        order = np.argsort(xv, kind='mergesort')
        xv = xv[order]
        yv = yv[order]
        idx_sorted = idx[order]

        n = len(xv)
        window = max(31, int(round(np.sqrt(n))))
        window = min(window, n)
        if window % 2 == 0:
            window = max(1, window - 1)
        half = window // 2

        for i in range(n):
            left = max(0, i - half)
            right = min(n, i + half + 1)
            if right - left < window:
                if left == 0:
                    right = min(n, window)
                elif right == n:
                    left = max(0, n - window)
            x_local = xv[left:right]
            y_local = yv[left:right]
            if len(y_local) == 0:
                continue
            exp = float(np.median(y_local))
            res = float(yv[i] - exp)
            mad = float(np.median(np.abs(y_local - exp)))
            scale = 1.4826 * mad
            if (not np.isfinite(scale)) or scale <= 0:
                scale = float(np.std(y_local, ddof=1)) if len(y_local) > 1 else np.nan
            if (not np.isfinite(scale)) or scale <= 0:
                scale = 1.0
            expected[idx_sorted[i]] = exp
            residual[idx_sorted[i]] = res
            score[idx_sorted[i]] = res / scale

        return score, expected, residual

    def _parse_manual_protein_list(self, text_value):
        if text_value is None:
            return []
        raw = str(text_value).replace(";", "\n").replace(",", "\n")
        items = []
        seen = set()
        for part in raw.splitlines():
            item = part.strip()
            if item and item not in seen:
                items.append(item)
                seen.add(item)
        return items

    def _difference_view_options(self):
        return ["Group A only", "Group B only", "Group B - Group A", "Group A - Group B"]

    def _effect_label_method_options(self):
        return [
            "Highest heterogeneity",
            "Mean-adjusted heterogeneity",
            "Largest mean",
            "Balanced mean + heterogeneity",
        ]

    def _is_difference_view(self, view_mode):
        return view_mode in ("Group B - Group A", "Group A - Group B")

    def _get_metric_series_for_view(self, analysis_df, metric_name, label_a, label_b, view_mode):
        metric_name = str(metric_name).lower()
        if metric_name not in {"mean", "heterogeneity"}:
            raise ValueError(f"Unsupported metric: {metric_name}")

        if view_mode == "Group A only":
            col = f"{metric_name}_{label_a}"
            return analysis_df[col].astype(float), f"{metric_name} ({label_a})", False, label_a, self._condition_a_color()

        if view_mode == "Group B only":
            col = f"{metric_name}_{label_b}"
            return analysis_df[col].astype(float), f"{metric_name} ({label_b})", False, label_b, self._condition_b_color()

        # For the difference effect map, use the currently selected heterogeneity
        # metric. If MAD (not normalised) is selected, the plotted y-axis is the
        # directional change in raw median absolute deviation.
        if metric_name == "heterogeneity":
            selected_heterogeneity_metric = canonical_heterogeneity_metric(self.heterogeneity_metric_var.get())
            if selected_heterogeneity_metric == "NORMALISED_MAD":
                delta_col = f"delta_normalised_mad_{label_b}_minus_{label_a}"
                axis_metric_label = "normalised MAD"
            elif selected_heterogeneity_metric == "MAD":
                delta_col = f"delta_heterogeneity_{label_b}_minus_{label_a}"
                axis_metric_label = "MAD"
            else:
                delta_col = f"delta_{metric_name}_{label_b}_minus_{label_a}"
                axis_metric_label = metric_name
        else:
            delta_col = f"delta_{metric_name}_{label_b}_minus_{label_a}"
            axis_metric_label = metric_name

        if view_mode == "Group A - Group B":
            return (-analysis_df[delta_col].astype(float)), f"{axis_metric_label} ({label_a} - {label_b})", True, f"{label_a} - {label_b}", None

        return analysis_df[delta_col].astype(float), f"{axis_metric_label} ({label_b} - {label_a})", True, f"{label_b} - {label_a}", None

    def _select_top_heterogeneity_rows(self, analysis_df, label_a, label_b, top_n, mode, view_mode):
        subset = analysis_df.copy()
        values, _, is_difference, _, _ = self._get_metric_series_for_view(
            subset, "heterogeneity", label_a, label_b, view_mode
        )
        subset["_plot_value"] = values
        subset = subset[subset["_plot_value"].notna()].copy()

        if subset.empty:
            return subset

        mode = str(mode)
        top_n = int(top_n)

        if mode == "Highest and lowest heterogeneity":
            high = subset.sort_values(by=["_plot_value", "protein"], ascending=[False, True]).head(top_n).copy()
            low = subset.sort_values(by=["_plot_value", "protein"], ascending=[True, True]).head(top_n).copy()
            high["_selection_group"] = "highest"
            low["_selection_group"] = "lowest"
            combined = pd.concat([high, low], axis=0, ignore_index=False)
            combined = combined[~combined.index.duplicated(keep="first")].copy()
            return combined

        if is_difference:
            if mode in {"Highest heterogeneity", "Largest increase"}:
                subset = subset.sort_values(by=["_plot_value", "protein"], ascending=[False, True])
            elif mode in {"Lowest heterogeneity", "Largest decrease"}:
                subset = subset.sort_values(by=["_plot_value", "protein"], ascending=[True, True])
            else:
                subset["_sort_abs"] = subset["_plot_value"].abs()
                subset = subset.sort_values(by=["_sort_abs", "protein"], ascending=[False, True])
        else:
            if mode in {"Lowest heterogeneity", "Largest decrease"}:
                subset = subset.sort_values(by=["_plot_value", "protein"], ascending=[True, True])
            elif mode == "Largest absolute change":
                subset["_sort_abs"] = subset["_plot_value"].abs()
                subset = subset.sort_values(by=["_sort_abs", "protein"], ascending=[False, True])
            else:
                subset = subset.sort_values(by=["_plot_value", "protein"], ascending=[False, True])

        return subset.head(top_n).copy()

    def _build_summary_tab(self):
        button_row = ttk.Frame(self.summary_tab)
        button_row.pack(fill="x", pady=(0, 10))
        ttk.Button(button_row, text="Auto-detect keywords", command=self.autodetect_keywords).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="Preview matched columns", command=self.preview_matches).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="Export summary CSV", command=self.export_summary_csv).pack(side="left")

        output_frame = ttk.LabelFrame(self.summary_tab, text="Summary tab output / preview", padding=10)
        output_frame.pack(fill="both", expand=True)

        self.info_text = tk.Text(output_frame, wrap="word", height=20)
        self.info_text.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(output_frame, orient="vertical", command=self.info_text.yview)
        scroll.pack(side="right", fill="y")
        self.info_text.configure(yscrollcommand=scroll.set)

        self._write_info(
            "Ready.\n\n"
            "Suggested workflow:\n"
            "1. Browse to your CSV.\n"
            "2. Check the protein column and condition keywords.\n"
            "3. Click 'Preview matched columns' to confirm the cells in each group.\n"
            "4. Use this tab to export the base summary CSV.\n"
            "5. Use the next tabs to export ranked tables and generate plots.\n"
        )

    def _build_analysis_tab(self):
        settings_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis settings", padding=10)
        settings_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(settings_frame, text="Minimum valid cells per condition for statistical tests").grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=4
        )
        ttk.Spinbox(settings_frame, from_=2, to=999, textvariable=self.min_valid_n_var, width=8).grid(
            row=0, column=1, sticky="w", pady=4
        )

        ttk.Label(settings_frame, text="Heterogeneity metric used for ranking").grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=4
        )
        metric_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.heterogeneity_metric_var,
            state="readonly",
            values=["Normalised MAD (robust CV analogue)", "MAD (not normalised)", "IQR", "SD", "Variance"],
            width=28,
        )
        metric_combo.grid(row=1, column=1, sticky="w", pady=4)

        helper = ttk.Label(
            settings_frame,
            text=(
                "Normalised MAD or IQR are usually the safest first choices for transformed single-cell abundances, "
                "especially when some values are negative or the mean is close to zero."
            ),
            wraplength=980,
            justify="left",
        )
        helper.grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))

        if not SCIPY_AVAILABLE:
            warn = ttk.Label(
                settings_frame,
                text="SciPy is not installed, so p-values and FDR fields will be left blank. Ranking exports will still work.",
                foreground="#a33",
                wraplength=980,
                justify="left",
            )
            warn.grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))

        button_row = ttk.Frame(self.analysis_tab)
        button_row.pack(fill="x", pady=(0, 10))
        ttk.Button(button_row, text="Preview next-step analysis", command=self.preview_next_step).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="Export combined analysis CSV", command=self.export_next_step_combined_csv).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="Export ranked CSV set", command=self.export_ranked_csv_set).pack(side="left")

        output_frame = ttk.LabelFrame(self.analysis_tab, text="Next-step tab output / preview", padding=10)
        output_frame.pack(fill="both", expand=True)

        self.analysis_info_text = tk.Text(output_frame, wrap="word", height=20)
        self.analysis_info_text.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(output_frame, orient="vertical", command=self.analysis_info_text.yview)
        scroll.pack(side="right", fill="y")
        self.analysis_info_text.configure(yscrollcommand=scroll.set)

        self._write_analysis_info(
            "This tab builds the next analysis stage.\n\n"
            "It exports a per-protein table with:\n"
            "- change in mean abundance\n"
            "- change in median abundance\n"
            "- change in heterogeneity (normalised MAD, IQR, SD, or variance)\n"
            "- ranks for strongest mean change and strongest heterogeneity change\n"
            "- optional p-values and BH-FDR q-values\n\n"
            "Recommended default: use IQR for heterogeneity ranking."
        )

    def _build_effect_map_tab(self):
        c = self.effect_map_tab.controls
        ttk.Label(c, text="View mode").pack(side="left", padx=(0, 6))
        ttk.Combobox(
            c,
            textvariable=self.effect_view_mode_var,
            state="readonly",
            values=self._difference_view_options(),
            width=18,
        ).pack(side="left", padx=(0, 12))
        ttk.Label(c, text="Q-value threshold").pack(side="left", padx=(0, 6))
        ttk.Spinbox(c, from_=0.001, to=1.0, increment=0.01, textvariable=self.q_threshold_var, width=8).pack(side="left", padx=(0, 12))
        ttk.Label(c, text="Label top proteins").pack(side="left", padx=(0, 6))
        ttk.Spinbox(c, from_=0, to=100, textvariable=self.effect_label_top_n_var, width=6).pack(side="left", padx=(0, 12))
        ttk.Label(c, text="Label by").pack(side="left", padx=(0, 6))
        ttk.Combobox(
            c,
            textvariable=self.effect_label_method_var,
            state="readonly",
            values=self._effect_label_method_options(),
            width=28,
        ).pack(side="left", padx=(0, 12))
        ttk.Button(c, text="Generate effect map", command=self.plot_effect_map).pack(side="left", padx=(0, 8))
        ttk.Button(c, text="Save image", command=lambda: self.effect_map_tab.save_figure("effect_map.png")).pack(side="left")
        self.effect_map_tab.set_status("Plot proteins as points using either a single group view or a directional group-difference view.")

    def _build_hetero_rank_tab(self):
        c = self.hetero_rank_tab.controls
        ttk.Label(c, text="View mode").pack(side="left", padx=(0, 6))
        ttk.Combobox(
            c,
            textvariable=self.rank_view_mode_var,
            state="readonly",
            values=self._difference_view_options(),
            width=18,
        ).pack(side="left", padx=(0, 12))
        ttk.Label(c, text="Ranking mode").pack(side="left", padx=(0, 6))
        ttk.Combobox(
            c,
            textvariable=self.hetero_sort_mode_var,
            state="readonly",
            values=["Highest heterogeneity", "Lowest heterogeneity", "Highest and lowest heterogeneity", "Largest increase", "Largest absolute change", "Largest decrease"],
            width=24,
        ).pack(side="left", padx=(0, 12))
        ttk.Label(c, text="Top proteins").pack(side="left", padx=(0, 6))
        ttk.Spinbox(c, from_=1, to=200, textvariable=self.hetero_top_n_var, width=6).pack(side="left", padx=(0, 12))
        ttk.Button(c, text="Generate ranking plot", command=self.plot_heterogeneity_ranking).pack(side="left", padx=(0, 8))
        ttk.Button(c, text="Save image", command=lambda: self.hetero_rank_tab.save_figure("heterogeneity_ranking.png")).pack(side="left")
        self.hetero_rank_tab.set_status("Rank proteins by within-group heterogeneity or by directional heterogeneity change between groups.")

    def _build_violin_tab(self):
        c = self.violin_tab.controls
        ttk.Label(c, text="View mode").pack(side="left", padx=(0, 6))
        ttk.Combobox(
            c,
            textvariable=self.violin_view_mode_var,
            state="readonly",
            values=self._difference_view_options(),
            width=18,
        ).pack(side="left", padx=(0, 12))
        ttk.Label(c, text="Protein selection mode").pack(side="left", padx=(0, 6))
        ttk.Combobox(
            c,
            textvariable=self.violin_sort_mode_var,
            state="readonly",
            values=["Largest increase", "Largest absolute change", "Largest decrease"],
            width=22,
        ).pack(side="left", padx=(0, 12))
        ttk.Label(c, text="Top proteins").pack(side="left", padx=(0, 6))
        ttk.Spinbox(c, from_=1, to=20, textvariable=self.violin_top_n_var, width=6).pack(side="left", padx=(0, 12))
        ttk.Label(c, text="Manual proteins").pack(side="left", padx=(0, 6))
        ttk.Entry(c, textvariable=self.violin_manual_proteins_var, width=42).pack(side="left", padx=(0, 12), fill="x", expand=True)
        ttk.Checkbutton(
            c,
            text="Use same y-axis across all violin plots",
            variable=self.violin_fix_shared_yaxis_var,
        ).pack(side="left", padx=(0, 12))
        ttk.Label(c, text="Dot size").pack(side="left", padx=(0, 4))
        ttk.Spinbox(c, from_=0.5, to=30.0, increment=0.5, textvariable=self.violin_dot_size_var, width=5).pack(side="left", padx=(0, 12))
        ttk.Button(c, text="Generate violin plots", command=self.plot_top_violin_panels).pack(side="left", padx=(0, 8))
        ttk.Button(c, text="Save image(s)", command=self.save_violin_panel_images).pack(side="left")
        self.violin_tab.set_status("Inspect top proteins either within one group alone or side-by-side under directional group-difference selection. Enter comma- or semicolon-separated proteins to override the automatic top-N list.")

    def _build_bimodality_tab(self):
        c = self.bimodality_tab.controls
        ttk.Label(c, text="Ranking mode").pack(side="left", padx=(0, 6))
        ttk.Combobox(
            c,
            textvariable=self.bimodality_sort_mode_var,
            state="readonly",
            values=["Highest balanced bimodality in either condition", "Largest increase in balanced bimodality (B-A)", "Largest decrease in balanced bimodality (A-B)", "Largest absolute change in balanced bimodality (|B-A|)", "Highest balanced bimodality in Group A", "Highest balanced bimodality in Group B"],
            width=46,
        ).pack(side="left", padx=(0, 12))
        ttk.Label(c, text="Top proteins").pack(side="left", padx=(0, 6))
        ttk.Spinbox(c, from_=1, to=50, textvariable=self.bimodality_top_n_var, width=6).pack(side="left", padx=(0, 12))
        ttk.Label(c, text="Min cells/condition").pack(side="left", padx=(0, 6))
        ttk.Spinbox(c, from_=3, to=500, textvariable=self.bimodality_min_valid_n_var, width=6).pack(side="left", padx=(0, 12))
        ttk.Label(c, text="Min fraction/mode").pack(side="left", padx=(0, 6))
        ttk.Spinbox(c, from_=0.01, to=0.49, increment=0.01, textvariable=self.bimodality_min_fraction_per_mode_var, width=5).pack(side="left", padx=(0, 12))
        ttk.Label(c, text="Min cells/mode").pack(side="left", padx=(0, 6))
        ttk.Spinbox(c, from_=1, to=250, textvariable=self.bimodality_min_cells_per_mode_var, width=5).pack(side="left", padx=(0, 12))
        ttk.Label(c, text="Min mode separation").pack(side="left", padx=(0, 6))
        ttk.Spinbox(c, from_=0.0, to=10.0, increment=0.1, textvariable=self.bimodality_min_mode_separation_var, width=5).pack(side="left", padx=(0, 12))
        ttk.Label(c, text="Max dip p").pack(side="left", padx=(0, 6))
        ttk.Spinbox(c, from_=0.001, to=1.0, increment=0.01, textvariable=self.bimodality_max_dip_p_value_var, width=5).pack(side="left", padx=(0, 12))
        ttk.Label(c, text="Min valley depth").pack(side="left", padx=(0, 6))
        ttk.Spinbox(c, from_=0.0, to=1.0, increment=0.05, textvariable=self.bimodality_min_valley_depth_var, width=5).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            c,
            text="Require 2-component GMM better than 1-component",
            variable=self.bimodality_require_gmm_improvement_var,
        ).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            c,
            text="Require Hartigan dip test",
            variable=self.bimodality_require_diptest_var,
        ).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            c,
            text="Use same y-axis across all plots",
            variable=self.bimodality_fix_shared_yaxis_var,
        ).pack(side="left", padx=(0, 12))
        ttk.Label(c, text="Dot size").pack(side="left", padx=(0, 4))
        ttk.Spinbox(c, from_=0.5, to=30.0, increment=0.5, textvariable=self.bimodality_dot_size_var, width=5).pack(side="left", padx=(0, 12))
        ttk.Button(c, text="Generate bimodality violin plots", command=self.plot_bimodality_violins).pack(side="left", padx=(0, 8))
        ttk.Button(c, text="Generate bimodality effect map", command=self.plot_bimodality_effect_map).pack(side="left", padx=(0, 8))
        ttk.Button(c, text="Export bimodality table", command=self.export_bimodality_table).pack(side="left", padx=(0, 8))
        ttk.Button(c, text="Save image", command=lambda: self.bimodality_tab.save_figure("bimodality_violins_or_effect_map.png")).pack(side="left")
        method_note = "Hartigan dip test" if DIPTEST_AVAILABLE else "KDE peak-separation fallback (warning: install `diptest` for Hartigan dip p-values)"
        self.bimodality_tab.set_status(f"Rank proteins by strict balanced within-condition bimodality using {method_note}. Proteins must pass mode support, mode separation, dip p-value, GMM improvement, and valley-depth filters.")

    def _build_pca_tab(self):
        c = self.pca_tab.controls
        ttk.Label(c, text="View mode").pack(side="left", padx=(0, 6))
        ttk.Combobox(
            c,
            textvariable=self.pca_view_mode_var,
            state="readonly",
            values=["Both groups", "Group A only", "Group B only"],
            width=14,
        ).pack(side="left", padx=(0, 12))
        ttk.Label(c, text="Top variable proteins used for PCA").pack(side="left", padx=(0, 6))
        ttk.Spinbox(c, from_=10, to=10000, textvariable=self.pca_top_variable_var, width=8).pack(side="left", padx=(0, 12))
        ttk.Button(c, text="Generate PCA plot", command=self.plot_pca).pack(side="left", padx=(0, 8))
        ttk.Button(c, text="Save image", command=lambda: self.pca_tab.save_figure("pca_cells.png")).pack(side="left")
        self.pca_tab.set_status("Plot both groups together or visualise either group alone in PCA space. Directional subtraction is not used for PCA.")

    def _write_info(self, text, append=False):
        self.info_text.configure(state="normal")
        if not append:
            self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, text)
        self.info_text.see(tk.END)
        self.info_text.configure(state="normal")

    def _write_analysis_info(self, text, append=False):
        self.analysis_info_text.configure(state="normal")
        if not append:
            self.analysis_info_text.delete("1.0", tk.END)
        self.analysis_info_text.insert(tk.END, text)
        self.analysis_info_text.see(tk.END)
        self.analysis_info_text.configure(state="normal")

    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Select the input CSV",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path, sep=None, engine="python")
        except Exception as exc:
            messagebox.showerror("Load error", f"Could not read file:\n\n{exc}")
            return

        self.df = df
        self.file_path = file_path
        self.file_var.set(file_path)

        columns = list(df.columns)
        self.protein_col_combo["values"] = columns
        if columns:
            self.protein_col_var.set(columns[0])

        kw_a, kw_b = guess_condition_keywords(columns[1:] if len(columns) > 1 else columns)
        if kw_a:
            self.keyword_a_var.set(kw_a)
        if kw_b:
            self.keyword_b_var.set(kw_b)

        if self.keyword_a_var.get().strip().lower() == "u20s":
            self.label_a_var.set("Untreated")
        if self.keyword_b_var.get().strip().lower() == "6tg":
            self.label_b_var.set("6TG")

        values = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce") if df.shape[1] > 1 else pd.DataFrame()
        negative_count = int((values < 0).sum().sum()) if not values.empty else 0

        load_msg = (
            f"Loaded file:\n{file_path}\n\n"
            f"Rows: {df.shape[0]:,}\n"
            f"Columns: {df.shape[1]:,}\n"
            f"Default protein column: {self.protein_col_var.get()}\n"
            f"Negative numeric values detected: {negative_count:,}\n\n"
            "Check the matched columns before exporting or plotting."
        )
        self._write_info(load_msg)
        self._write_analysis_info(
            f"Loaded file and ready for ranked analysis and plots.\n\n"
            f"Rows: {df.shape[0]:,}\n"
            f"Columns: {df.shape[1]:,}\n"
            f"Negative numeric values detected: {negative_count:,}\n\n"
            "Because transformed datasets often contain negative values, normalised MAD or IQR are generally better default heterogeneity metrics than CV."
        )
        self.effect_map_tab.set_status("File loaded. Generate the effect map when ready.")
        self.hetero_rank_tab.set_status("File loaded. Generate the heterogeneity ranking plot when ready.")
        self.violin_tab.set_status("File loaded. Generate the violin panel when ready.")
        self.pca_tab.set_status("File loaded. Generate the PCA plot when ready.")

    def autodetect_keywords(self):
        if self.df is None:
            messagebox.showinfo("No file loaded", "Load a file first.")
            return

        columns = list(self.df.columns)
        kw_a, kw_b = guess_condition_keywords(columns[1:] if len(columns) > 1 else columns)
        self.keyword_a_var.set(kw_a)
        self.keyword_b_var.set(kw_b)

        if kw_a.lower() == "u20s":
            self.label_a_var.set("Untreated")
        else:
            self.label_a_var.set("Condition_A")
        if kw_b.lower() == "6tg":
            self.label_b_var.set("6TG")
        else:
            self.label_b_var.set("Condition_B")

        msg = (
            "Auto-detection complete.\n\n"
            f"Condition A keyword: {kw_a or '[not found]'}\n"
            f"Condition B keyword: {kw_b or '[not found]'}\n\n"
            "You can edit both labels and keywords manually."
        )
        self._write_info(msg)
        self._write_analysis_info(msg)

    def preview_matches(self):
        if self.df is None:
            messagebox.showinfo("No file loaded", "Load a file first.")
            return

        protein_col = self.protein_col_var.get().strip()
        kw_a = self.keyword_a_var.get().strip()
        kw_b = self.keyword_b_var.get().strip()
        label_a = self.label_a_var.get().strip() or "Condition_A"
        label_b = self.label_b_var.get().strip() or "Condition_B"

        try:
            _, cols_a, cols_b = summarise_two_conditions(
                self.df,
                protein_col,
                kw_a,
                kw_b,
                label_a,
                label_b,
                ddof=self.ddof_var.get(),
            )
        except Exception as exc:
            messagebox.showerror("Match error", str(exc))
            return

        preview = (
            f"Matched columns preview\n\n"
            f"{label_a}: {len(cols_a)} cell columns matched using keyword '{kw_a}'\n"
            + "\n".join(cols_a[:25])
            + ("\n..." if len(cols_a) > 25 else "")
            + "\n\n"
            f"{label_b}: {len(cols_b)} cell columns matched using keyword '{kw_b}'\n"
            + "\n".join(cols_b[:25])
            + ("\n..." if len(cols_b) > 25 else "")
        )
        self._write_info(preview)

    def _get_shared_inputs(self):
        if self.df is None:
            raise ValueError("Load a file first.")

        protein_col = self.protein_col_var.get().strip()
        kw_a = self.keyword_a_var.get().strip()
        kw_b = self.keyword_b_var.get().strip()
        label_a = self.label_a_var.get().strip() or "Condition_A"
        label_b = self.label_b_var.get().strip() or "Condition_B"
        return protein_col, kw_a, kw_b, label_a, label_b

    def _get_wide_numeric_work(self):
        protein_col, kw_a, kw_b, _, _ = self._get_shared_inputs()
        cond_a_cols, cond_b_cols = get_condition_columns(self.df, protein_col, kw_a, kw_b)
        numeric_cols = cond_a_cols + cond_b_cols
        work = self.df[[protein_col] + numeric_cols].copy()
        work[numeric_cols] = work[numeric_cols].apply(pd.to_numeric, errors="coerce")
        return work, cond_a_cols, cond_b_cols

    def export_summary_csv(self):
        try:
            protein_col, kw_a, kw_b, label_a, label_b = self._get_shared_inputs()
            summary_df, cols_a, cols_b = summarise_two_conditions(
                self.df,
                protein_col,
                kw_a,
                kw_b,
                label_a,
                label_b,
                ddof=self.ddof_var.get(),
            )
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            return

        default_name = "single_cell_proteomics_summary.csv"
        if self.file_path:
            stem = os.path.splitext(os.path.basename(self.file_path))[0]
            default_name = f"{stem}_summary.csv"

        output_path = filedialog.asksaveasfilename(
            title="Save summary CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv")]
        )
        if not output_path:
            return

        try:
            summary_df.to_csv(output_path, index=False)
        except Exception as exc:
            messagebox.showerror("Save error", f"Could not save CSV:\n\n{exc}")
            return

        self._write_info(
            f"Export complete.\n\n"
            f"Saved to:\n{output_path}\n\n"
            f"{label_a}: {len(cols_a)} columns matched\n"
            f"{label_b}: {len(cols_b)} columns matched\n\n"
            f"Output rows (proteins): {summary_df.shape[0]:,}\n"
            f"Output columns (summary metrics): {summary_df.shape[1]:,}\n\n"
            "The CSV includes means, medians, variances, SD, median-based CV, median-based CV², variance/mean, and simple between-condition ratios."
        )
        messagebox.showinfo("Done", f"Summary CSV exported successfully.\n\n{output_path}")

    def _run_next_step_analysis(self):
        protein_col, kw_a, kw_b, label_a, label_b = self._get_shared_inputs()
        return build_next_step_analysis(
            self.df,
            protein_col,
            kw_a,
            kw_b,
            label_a,
            label_b,
            ddof=self.ddof_var.get(),
            min_valid_n=int(self.min_valid_n_var.get()),
            heterogeneity_metric=self.heterogeneity_metric_var.get(),
        )

    def preview_next_step(self):
        try:
            analysis_df, mean_sorted, hetero_sorted, cols_a, cols_b = self._run_next_step_analysis()
            _, kw_a, kw_b, label_a, label_b = self._get_shared_inputs()
        except Exception as exc:
            messagebox.showerror("Preview error", str(exc))
            return

        top_mean = mean_sorted[[
            "protein",
            f"delta_mean_{label_b}_minus_{label_a}",
            "mean_p_value",
            "mean_q_value_bh_fdr",
        ]].head(10)

        hetero_delta_col = f"delta_heterogeneity_{label_b}_minus_{label_a}"
        if canonical_heterogeneity_metric(self.heterogeneity_metric_var.get()) == "NORMALISED_MAD":
            hetero_delta_col = f"delta_normalised_mad_{label_b}_minus_{label_a}"

        top_hetero = hetero_sorted[[
            "protein",
            hetero_delta_col,
            "heterogeneity_p_value",
            "heterogeneity_q_value_bh_fdr",
        ]].head(10).rename(columns={
            hetero_delta_col: f"delta_normalised_MAD_{label_b}_minus_{label_a}"
            if canonical_heterogeneity_metric(self.heterogeneity_metric_var.get()) == "NORMALISED_MAD"
            else hetero_delta_col
        })

        preview = (
            f"Next-step analysis preview\n\n"
            f"{label_a}: {len(cols_a)} matched columns using '{kw_a}'\n"
            f"{label_b}: {len(cols_b)} matched columns using '{kw_b}'\n"
            f"Proteins analysed: {analysis_df.shape[0]:,}\n"
            f"Heterogeneity metric used for ranking: {heterogeneity_metric_display_name(self.heterogeneity_metric_var.get())}\n"
            f"Minimum valid cells per condition for tests: {self.min_valid_n_var.get()}\n"
            f"SciPy available for p-values: {'Yes' if SCIPY_AVAILABLE else 'No'}\n\n"
            "Top proteins by absolute mean change:\n"
            f"{top_mean.to_string(index=False)}\n\n"
            "Top proteins by absolute heterogeneity change:\n"
            f"{top_hetero.to_string(index=False)}"
        )
        self._write_analysis_info(preview)

    def export_next_step_combined_csv(self):
        try:
            analysis_df, _, _, cols_a, cols_b = self._run_next_step_analysis()
            _, _, _, label_a, label_b = self._get_shared_inputs()
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            return

        default_name = "single_cell_proteomics_next_step_analysis.csv"
        if self.file_path:
            stem = os.path.splitext(os.path.basename(self.file_path))[0]
            default_name = f"{stem}_next_step_analysis.csv"

        output_path = filedialog.asksaveasfilename(
            title="Save combined next-step analysis CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv")],
        )
        if not output_path:
            return

        try:
            analysis_df.to_csv(output_path, index=False)
        except Exception as exc:
            messagebox.showerror("Save error", f"Could not save CSV:\n\n{exc}")
            return

        self._write_analysis_info(
            f"Combined next-step analysis CSV exported.\n\n"
            f"Saved to:\n{output_path}\n\n"
            f"{label_a}: {len(cols_a)} matched columns\n"
            f"{label_b}: {len(cols_b)} matched columns\n"
            f"Output rows (proteins): {analysis_df.shape[0]:,}\n"
            f"Output columns: {analysis_df.shape[1]:,}\n\n"
            "This file includes mean-change ranking, heterogeneity-change ranking, p-values, and FDR q-values when available."
        )
        messagebox.showinfo("Done", f"Combined analysis CSV exported successfully.\n\n{output_path}")

    def export_ranked_csv_set(self):
        try:
            analysis_df, mean_sorted, hetero_sorted, cols_a, cols_b = self._run_next_step_analysis()
            _, _, _, label_a, label_b = self._get_shared_inputs()
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            return

        initial_dir = os.path.dirname(self.file_path) if self.file_path else os.getcwd()
        out_dir = filedialog.askdirectory(title="Choose folder for ranked CSV set", initialdir=initial_dir)
        if not out_dir:
            return

        stem = "single_cell_proteomics"
        if self.file_path:
            stem = os.path.splitext(os.path.basename(self.file_path))[0]

        combined_path = os.path.join(out_dir, f"{stem}_next_step_analysis.csv")
        mean_path = os.path.join(out_dir, f"{stem}_ranked_by_mean_change.csv")
        hetero_path = os.path.join(out_dir, f"{stem}_ranked_by_heterogeneity_change.csv")

        try:
            analysis_df.to_csv(combined_path, index=False)
            mean_sorted.to_csv(mean_path, index=False)
            hetero_sorted.to_csv(hetero_path, index=False)
        except Exception as exc:
            messagebox.showerror("Save error", f"Could not save one or more CSV files:\n\n{exc}")
            return

        self._write_analysis_info(
            f"Ranked CSV set exported.\n\n"
            f"Folder:\n{out_dir}\n\n"
            f"Files written:\n"
            f"- {os.path.basename(combined_path)}\n"
            f"- {os.path.basename(mean_path)}\n"
            f"- {os.path.basename(hetero_path)}\n\n"
            f"{label_a}: {len(cols_a)} matched columns\n"
            f"{label_b}: {len(cols_b)} matched columns\n\n"
            "Use the mean-ranked file to find proteins with the largest average abundance shifts.\n"
            "Use the heterogeneity-ranked file to find proteins whose cell-to-cell spread changes most after treatment."
        )
        messagebox.showinfo("Done", f"Ranked CSV set exported successfully.\n\nFolder:\n{out_dir}")


    def _select_effect_map_labels(self, plot_df, label_n, is_difference):
        label_work = plot_df.copy()
        method = str(self.effect_label_method_var.get()).strip()

        if label_n <= 0 or label_work.empty:
            return label_work.iloc[0:0].copy()

        if is_difference:
            if method == "Highest heterogeneity":
                # In the B-minus-A effect map, _y is the directional change in
                # heterogeneity. Label the proteins with the largest absolute
                # changes so that strong decreases are labelled as well as
                # strong increases. The plotted y values remain directional.
                label_work["_abs_y"] = label_work["_y"].abs()
                label_work["_abs_x"] = label_work["_x"].abs()
                label_df = label_work.sort_values(
                    by=["_abs_y", "_abs_x", "_y", "_x"],
                    ascending=[False, False, False, False],
                ).head(label_n)
            elif method == "Mean-adjusted heterogeneity":
                label_work["_abs_y"] = label_work["_y"].abs()
                label_work["_abs_x"] = label_work["_x"].abs()
                label_df = label_work.sort_values(
                    by=["_abs_y", "_abs_x", "_y", "_x"],
                    ascending=[False, False, False, False],
                ).head(label_n)
            elif method == "Largest mean":
                label_df = label_work.sort_values(
                    by=["_x", "_y"],
                    ascending=[False, False],
                ).head(label_n)
            else:
                label_work["_rank_x"] = label_work["_x"].abs().rank(method="average", pct=True)
                label_work["_rank_y"] = label_work["_y"].abs().rank(method="average", pct=True)
                label_work["combined_rank"] = label_work["_rank_x"] + label_work["_rank_y"]
                label_df = label_work.sort_values(
                    by=["combined_rank", "_y", "_x"],
                    ascending=[False, False, False],
                ).head(label_n)
            return label_df

        # Single-group map
        if method == "Highest heterogeneity":
            return label_work.sort_values(
                by=["_y", "_x"],
                ascending=[False, False],
            ).head(label_n)

        if method == "Largest mean":
            return label_work.sort_values(
                by=["_x", "_y"],
                ascending=[False, False],
            ).head(label_n)

        if method == "Balanced mean + heterogeneity":
            label_work["_rank_x"] = label_work["_x"].rank(method="average", pct=True)
            label_work["_rank_y"] = label_work["_y"].rank(method="average", pct=True)
            label_work["combined_rank"] = label_work["_rank_x"] + label_work["_rank_y"]
            return label_work.sort_values(
                by=["combined_rank", "_y", "_x"],
                ascending=[False, False, False],
            ).head(label_n)

        # Default and recommended refined option for single-group:
        residual_score, expected_het, residual_het = self._compute_local_residual_score(
            label_work["_x"].to_numpy(dtype=float),
            label_work["_y"].to_numpy(dtype=float),
        )
        label_work["expected_heterogeneity_for_mean"] = expected_het
        label_work["heterogeneity_residual"] = residual_het
        label_work["heterogeneity_residual_score"] = residual_score
        label_df = label_work.dropna(subset=["heterogeneity_residual_score"])
        return label_df.sort_values(
            by=["heterogeneity_residual_score", "heterogeneity_residual", "_y", "_x"],
            ascending=[False, False, False, False],
        ).head(label_n)

    def plot_effect_map(self):
        try:
            analysis_df, _, _, cols_a, cols_b = self._run_next_step_analysis()
            _, _, _, label_a, label_b = self._get_shared_inputs()
        except Exception as exc:
            messagebox.showerror("Plot error", str(exc))
            return

        view_mode = self.effect_view_mode_var.get()
        mean_series, mean_label, is_difference, view_desc, single_color = self._get_metric_series_for_view(
            analysis_df, "mean", label_a, label_b, view_mode
        )
        het_series, het_label, _, _, _ = self._get_metric_series_for_view(
            analysis_df, "heterogeneity", label_a, label_b, view_mode
        )

        plot_df = analysis_df[["protein", "mean_q_value_bh_fdr", "heterogeneity_q_value_bh_fdr"]].copy()
        plot_df["_x"] = mean_series
        plot_df["_y"] = het_series
        plot_df = plot_df.dropna(subset=["_x", "_y"])
        if plot_df.empty:
            messagebox.showinfo("No data", "No proteins had valid mean and heterogeneity values to plot.")
            return

        fig = self._prepare_standard_figure(self.effect_map_tab, self.effect_width_var, self.effect_height_var)
        ax = fig.add_subplot(111)

        if is_difference:
            q_thr = float(self.q_threshold_var.get())
            mean_sig = plot_df["mean_q_value_bh_fdr"] <= q_thr
            het_sig = plot_df["heterogeneity_q_value_bh_fdr"] <= q_thr
            plot_df["category"] = np.where(
                mean_sig & het_sig, "Both significant",
                np.where(mean_sig, "Mean significant", np.where(het_sig, "Heterogeneity significant", "Not significant"))
            )

            category_style = {
                "Not significant": (NOT_SIG_COLOR, 20, 0.55),
                "Mean significant": (MEAN_SIG_COLOR, 34, 0.80),
                "Heterogeneity significant": (HET_SIG_COLOR, 34, 0.80),
                "Both significant": (BOTH_SIG_COLOR, 42, 0.88),
            }

            for category in ["Not significant", "Mean significant", "Heterogeneity significant", "Both significant"]:
                sub = plot_df[plot_df["category"] == category]
                if sub.empty:
                    continue
                color, size, alpha = category_style[category]
                ax.scatter(
                    sub["_x"],
                    sub["_y"],
                    s=size,
                    alpha=alpha,
                    color=color,
                    edgecolors="none",
                    label=f"{category} (n={len(sub)})",
                )
        else:
            ax.scatter(
                plot_df["_x"],
                plot_df["_y"],
                s=24,
                alpha=0.72,
                color=single_color,
                edgecolors="none",
                label=f"{view_desc} proteins (n={len(plot_df)})",
            )

        label_n = int(self.effect_label_top_n_var.get())
        if label_n > 0:
            label_df = self._select_effect_map_labels(plot_df, label_n, is_difference)
            self._add_standard_labels(
                ax,
                label_df=label_df,
                x_col="_x",
                y_col="_y",
                label_col="protein",
                fontsize=self._font_size("annotation"),
            )

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.35)

        # Keep single-group effect maps directly comparable between Group A and Group B.
        # Difference mode can legitimately contain negative heterogeneity values, so only
        # enforce this fixed 0–3.5 scale for the individual group views.
        if not is_difference:
            ax.set_ylim(0, 3.5)

        effect_title = (
            f"Protein-level effect map: {view_desc}"
            if is_difference
            else f"Protein mean–heterogeneity landscape: {view_desc}"
        )
        self._style_axes_text(
            ax,
            "effect",
            xlabel=mean_label,
            ylabel=het_label,
            title=effect_title,
        )
        ax.grid(alpha=0.18)
        legend = ax.legend(frameon=False, loc="best")
        self._style_legend(legend)
        fig.tight_layout()
        self.effect_map_tab.canvas.draw_idle()
        if is_difference:
            self.effect_map_tab.set_status(
                f"Effect map generated for {view_desc}. Significance colours use q ≤ {float(self.q_threshold_var.get()):.3g}. "
                f"Labels in difference mode are selected by the largest absolute y-axis changes. "
                f"Matched cells: {label_a}={len(cols_a)}, {label_b}={len(cols_b)}."
            )
        else:
            self.effect_map_tab.set_status(
                f"Effect map generated for {view_desc} alone. In single-group mode labelled proteins are chosen by excess heterogeneity relative to proteins with similar mean abundance, rather than by raw extremeness. "
                f"Matched cells: {label_a}={len(cols_a)}, {label_b}={len(cols_b)}."
            )

    def plot_heterogeneity_ranking(self):
        try:
            analysis_df, _, _, cols_a, cols_b = self._run_next_step_analysis()
            _, _, _, label_a, label_b = self._get_shared_inputs()
        except Exception as exc:
            messagebox.showerror("Plot error", str(exc))
            return

        mode = self.hetero_sort_mode_var.get()
        view_mode = self.rank_view_mode_var.get()
        top_n = int(self.hetero_top_n_var.get())
        top_df = self._select_top_heterogeneity_rows(analysis_df, label_a, label_b, top_n, mode, view_mode)
        if top_df.empty:
            messagebox.showinfo("No data", "No proteins had valid heterogeneity values to rank.")
            return

        _, x_label, is_difference, view_desc, single_color = self._get_metric_series_for_view(
            analysis_df, "heterogeneity", label_a, label_b, view_mode
        )

        combined_high_low = mode == "Highest and lowest heterogeneity"

        if combined_high_low:
            high_df = top_df[top_df.get("_selection_group", "") == "highest"].copy()
            low_df = top_df[top_df.get("_selection_group", "") == "lowest"].copy()
            high_df = high_df.sort_values(by=["_plot_value", "protein"], ascending=[False, True])
            low_df = low_df.sort_values(by=["_plot_value", "protein"], ascending=[True, True])
            gap = 1.5 if (not high_df.empty and not low_df.empty) else 0.0
            plot_parts = []
            if not high_df.empty:
                high_df["_ypos"] = np.arange(len(high_df), dtype=float)
                plot_parts.append(high_df)
            if not low_df.empty:
                start = float(len(high_df)) + gap
                low_df["_ypos"] = np.arange(start, start + len(low_df), dtype=float)
                plot_parts.append(low_df)
            plot_df = pd.concat(plot_parts, axis=0).copy() if plot_parts else top_df.copy()
            if is_difference:
                colors = [self._condition_b_color() if v >= 0 else self._condition_a_color() for v in plot_df["_plot_value"].to_numpy(dtype=float)]
            else:
                colors = [single_color if g == "highest" else NOT_SIG_COLOR for g in plot_df["_selection_group"].tolist()]
        else:
            if is_difference:
                plot_df = top_df.sort_values(by=["_plot_value", "protein"], ascending=[False, True]).copy()
                colors = [self._condition_b_color() if v >= 0 else self._condition_a_color() for v in plot_df["_plot_value"].to_numpy(dtype=float)]
            else:
                ascending = mode in {"Lowest heterogeneity", "Largest decrease"}
                plot_df = top_df.sort_values(by=["_plot_value", "protein"], ascending=[ascending, True]).copy()
                colors = [single_color] * len(plot_df)
            plot_df["_ypos"] = np.arange(len(plot_df), dtype=float)

        fig = self._prepare_standard_figure(self.hetero_rank_tab, self.rank_width_var, self.rank_height_var)
        ax = fig.add_subplot(111)

        names = plot_df["protein"].astype(str).tolist()
        values = plot_df["_plot_value"].to_numpy(dtype=float)
        ypos = plot_df["_ypos"].to_numpy(dtype=float)

        for y, v in zip(ypos, values):
            ax.hlines(y, 0, v, color="#BFBFBF", linewidth=1.5, zorder=1)
        ax.scatter(values, ypos, c=colors, s=42, alpha=0.9, edgecolors="none", zorder=2)

        if combined_high_low and (plot_df["_selection_group"] == "lowest").any() and (plot_df["_selection_group"] == "highest").any():
            split_y = float(plot_df.loc[plot_df["_selection_group"] == "highest", "_ypos"].max()) + 0.75
            ax.axhline(split_y, color="#D0D0D0", linewidth=1.0, linestyle="--", zorder=0)

        ax.axvline(0, color="black", linewidth=0.9, alpha=0.6)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        if is_difference:
            rank_title = f"Proteins ranked by heterogeneity change: {view_desc} | {mode}"
        else:
            if combined_high_low:
                rank_title = f"Proteins ranked by within-group heterogeneity: {view_desc} | Highest and lowest heterogeneity"
            else:
                rank_title = f"Proteins ranked by within-group heterogeneity: {view_desc} | {mode}"
        self._style_axes_text(
            ax,
            "rank",
            xlabel=x_label,
            ylabel=None,
            title=rank_title,
        )
        ax.grid(axis="x", alpha=0.18)
        fig.tight_layout()
        self.hetero_rank_tab.canvas.draw_idle()
        if combined_high_low:
            self.hetero_rank_tab.set_status(
                f"Ranking plot generated for highest and lowest heterogeneity proteins using {view_desc}. "
                f"Matched cells: {label_a}={len(cols_a)}, {label_b}={len(cols_b)}."
            )
        else:
            self.hetero_rank_tab.set_status(
                f"Ranking plot generated for top {len(plot_df)} proteins using {view_desc}. "
                f"Matched cells: {label_a}={len(cols_a)}, {label_b}={len(cols_b)}."
            )

    def _resolve_violin_selection(self):
        analysis_df, _, _, cols_a, cols_b = self._run_next_step_analysis()
        work, cond_a_cols, cond_b_cols = self._get_wide_numeric_work()
        protein_col, _, _, label_a, label_b = self._get_shared_inputs()

        mode = self.violin_sort_mode_var.get()
        view_mode = self.violin_view_mode_var.get()
        top_n = int(self.violin_top_n_var.get())
        manual_names = self._parse_manual_protein_list(self.violin_manual_proteins_var.get())

        values, _, _, _, _ = self._get_metric_series_for_view(analysis_df, "heterogeneity", label_a, label_b, view_mode)
        analysis_work = analysis_df.copy()
        analysis_work["_plot_value"] = values
        analysis_work = analysis_work[analysis_work["_plot_value"].notna()].copy()
        analysis_work["_protein_key"] = analysis_work["protein"].astype(str)

        if manual_names:
            available = {p: p for p in analysis_work["_protein_key"].tolist()}
            available_lower = {p.lower(): p for p in analysis_work["_protein_key"].tolist()}
            chosen = []
            missing = []
            for name in manual_names:
                actual = available.get(name)
                if actual is None:
                    actual = available_lower.get(name.lower())
                if actual is None:
                    missing.append(name)
                elif actual not in chosen:
                    chosen.append(actual)
            top_df = analysis_work.set_index("_protein_key").loc[chosen].reset_index(drop=True) if chosen else analysis_work.iloc[0:0].copy()
        else:
            top_df = self._select_top_heterogeneity_rows(analysis_df, label_a, label_b, top_n, mode, view_mode)
            missing = []

        if top_df.empty:
            raise ValueError("No proteins had valid heterogeneity values for violin plotting.")

        proteins = top_df["protein"].astype(str).tolist()
        plot_value_lookup = dict(zip(top_df["protein"].astype(str), pd.to_numeric(top_df["_plot_value"], errors="coerce")))
        work_indexed = work.set_index(work[protein_col].astype(str), drop=False)
        is_difference = self._is_difference_view(view_mode)

        return {
            "analysis_df": analysis_df,
            "work": work,
            "cond_a_cols": cond_a_cols,
            "cond_b_cols": cond_b_cols,
            "protein_col": protein_col,
            "label_a": label_a,
            "label_b": label_b,
            "cols_a": cols_a,
            "cols_b": cols_b,
            "mode": mode,
            "view_mode": view_mode,
            "manual_names": manual_names,
            "missing": missing,
            "proteins": proteins,
            "plot_value_lookup": plot_value_lookup,
            "work_indexed": work_indexed,
            "is_difference": is_difference,
        }

    def _values_for_violin_protein(self, work_indexed, protein, cond_a_cols, cond_b_cols, view_mode, is_difference):
        if protein not in work_indexed.index:
            return []
        row = work_indexed.loc[protein]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        a = pd.to_numeric(row[cond_a_cols], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(row[cond_b_cols], errors="coerce").to_numpy(dtype=float)
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if is_difference:
            return [a, b]
        if view_mode == "Group A only":
            return [a]
        return [b]

    def _shared_violin_ylim(self, proteins, ctx):
        if not self.violin_fix_shared_yaxis_var.get():
            return None

        # Use the complete selected protein set stored in ctx, rather than only
        # the current page/chunk. This keeps y-axis limits identical across all
        # violin panels in the preview and across all saved split-image exports.
        proteins_for_axis = ctx.get("proteins", proteins)

        arrays = []
        for protein in proteins_for_axis:
            arrays.extend(
                self._values_for_violin_protein(
                    ctx["work_indexed"], protein, ctx["cond_a_cols"], ctx["cond_b_cols"],
                    ctx["view_mode"], ctx["is_difference"]
                )
            )
        arrays = [arr for arr in arrays if arr.size]
        if not arrays:
            return None
        vals = np.concatenate(arrays)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        ymin = float(np.min(vals))
        ymax = float(np.max(vals))
        pad = max(abs(ymin) * 0.05, 0.5) if ymin == ymax else (ymax - ymin) * 0.08
        return ymin - pad, ymax + pad

    def _draw_violin_panels_to_figure(self, fig, proteins, ctx, export_mode=False, page_index=None, total_pages=None):
        work_indexed = ctx["work_indexed"]
        protein_col = ctx["protein_col"]
        cond_a_cols = ctx["cond_a_cols"]
        cond_b_cols = ctx["cond_b_cols"]
        label_a = ctx["label_a"]
        label_b = ctx["label_b"]
        view_mode = ctx["view_mode"]
        is_difference = ctx["is_difference"]
        manual_names = ctx["manual_names"]
        plot_value_lookup = ctx["plot_value_lookup"]

        fig.clear()
        n = len(proteins)
        ncols = 2 if n > 1 else 1
        nrows = math.ceil(n / ncols)
        base_width = float(self.violin_width_var.get())
        base_height = float(self.violin_height_var.get())
        fig_height = max(base_height, nrows * 3.0)
        fig.set_size_inches(base_width, fig_height, forward=True)
        if not export_mode:
            fig.set_dpi(int(self.preview_dpi_var.get()))

        rng = np.random.default_rng(42)
        shared_ylim = self._shared_violin_ylim(proteins, ctx)

        for idx, protein in enumerate(proteins, start=1):
            ax = fig.add_subplot(nrows, ncols, idx)
            if protein not in work_indexed.index:
                ax.text(0.5, 0.5, f"Protein not found:\n{protein}", ha="center", va="center")
                ax.axis("off")
                continue

            values_arr = self._values_for_violin_protein(
                work_indexed, protein, cond_a_cols, cond_b_cols, view_mode, is_difference
            )

            if is_difference:
                positions = [1, 2]
                violin_colors = [self._condition_a_color(), self._condition_b_color()]
                xtick_labels = [label_a, label_b]
                value_label = "Δhet"
            elif view_mode == "Group A only":
                positions = [1]
                violin_colors = [self._condition_a_color()]
                xtick_labels = [label_a]
                value_label = "Het"
            else:
                positions = [1]
                violin_colors = [self._condition_b_color()]
                xtick_labels = [label_b]
                value_label = "Het"

            if not any(arr.size for arr in values_arr):
                ax.text(0.5, 0.5, f"No values for\n{protein}", ha="center", va="center")
                ax.axis("off")
                continue

            parts = ax.violinplot(values_arr, positions=positions, widths=0.75, showmeans=False, showmedians=False, showextrema=False)
            for body, color in zip(parts["bodies"], violin_colors):
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.35)

            for pos, arr, color in zip(positions, values_arr, violin_colors):
                if arr.size:
                    jitter = rng.uniform(-0.08, 0.08, size=arr.size)
                    dot_size = max(0.1, float(self.violin_dot_size_var.get()))
                    ax.scatter(np.full(arr.size, pos) + jitter, arr, s=dot_size, alpha=0.75, color=color, edgecolors="none")
                    med = np.median(arr)
                    q1 = np.percentile(arr, 25)
                    q3 = np.percentile(arr, 75)
                    ax.hlines(med, pos - 0.20, pos + 0.20, color="black", linewidth=1.4)
                    ax.vlines(pos, q1, q3, color="black", linewidth=1.2)

            plot_value = plot_value_lookup.get(protein, np.nan)
            ax.set_xticks(positions)
            ax.set_xticklabels(xtick_labels, rotation=0)
            panel_title = f"{protein} ({plot_value:.2f})" if np.isfinite(plot_value) else f"{protein}"
            self._style_axes_text(
                ax,
                "violin",
                xlabel=None,
                ylabel="Abundance",
                title=panel_title,
                apply_title_override=False,
            )
            if shared_ylim is not None:
                ax.set_ylim(shared_ylim)

        if is_difference:
            title = f"Top proteins by heterogeneity: {view_mode.lower()} side-by-side cell distributions"
        else:
            title = f"Top proteins by heterogeneity within {label_a if view_mode == 'Group A only' else label_b}: single-group cell distributions"
        if manual_names:
            title = f"User-selected proteins: {view_mode.lower()} distributions"
        if total_pages and total_pages > 1 and page_index is not None:
            title = f"{title} (page {page_index + 1} of {total_pages})"

        title_override = str(self.violin_title_text_var.get()).strip()
        final_title = title_override or title
        fig.suptitle(final_title, **self._font_kwargs("title"), y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])

    def plot_top_violin_panels(self):
        try:
            ctx = self._resolve_violin_selection()
        except Exception as exc:
            messagebox.showerror("Plot error", str(exc))
            return

        fig = self.violin_tab.figure
        self._draw_violin_panels_to_figure(fig, ctx["proteins"], ctx, export_mode=False)
        self.violin_tab.canvas.draw_idle()

        proteins = ctx["proteins"]
        manual_names = ctx["manual_names"]
        missing = ctx["missing"]
        label_a = ctx["label_a"]
        label_b = ctx["label_b"]
        cols_a = ctx["cols_a"]
        cols_b = ctx["cols_b"]
        view_mode = ctx["view_mode"]

        status = (
            f"Generated violin panels for {len(proteins)} proteins using {view_mode}. "
            f"Matched cells: {label_a}={len(cols_a)}, {label_b}={len(cols_b)}."
        )
        if self.violin_fix_shared_yaxis_var.get():
            status += " Shared y-axis across all selected violin plots enabled."
        if manual_names:
            status += f" Manual protein override used ({len(proteins)} found"
            if missing:
                status += f", {len(missing)} not found"
            status += ")."
        self.violin_tab.set_status(status)

        if manual_names and missing:
            messagebox.showwarning(
                "Some proteins were not found",
                "The following user-entered proteins were not found in the analysis table\n\n"
                + "\n".join(missing[:20])
                + ("\n..." if len(missing) > 20 else "")
            )

    def save_violin_panel_images(self):
        try:
            ctx = self._resolve_violin_selection()
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))
            return

        proteins = ctx["proteins"]
        max_per = max(1, int(self.violin_max_panels_per_image_var.get()))
        path = filedialog.asksaveasfilename(
            title="Save violin panel image(s)",
            defaultextension=".png",
            initialfile="top_protein_violins.png",
            filetypes=[("PNG image", "*.png"), ("PDF file", "*.pdf"), ("SVG file", "*.svg")],
        )
        if not path:
            return

        root, ext = os.path.splitext(path)
        ext = ext or ".png"
        chunks = [proteins[i:i + max_per] for i in range(0, len(proteins), max_per)]
        saved_paths = []
        for idx, chunk in enumerate(chunks):
            fig = Figure(figsize=(float(self.violin_width_var.get()), float(self.violin_height_var.get())), dpi=int(self.export_dpi_var.get()))
            self._draw_violin_panels_to_figure(fig, chunk, ctx, export_mode=True, page_index=idx, total_pages=len(chunks))
            out_path = path if len(chunks) == 1 else f"{root}_part{idx + 1:02d}{ext}"
            fig.savefig(out_path, dpi=int(self.export_dpi_var.get()), bbox_inches="tight", transparent=bool(self.transparent_bg_var.get()))
            saved_paths.append(out_path)

        self.violin_tab.set_status(f"Saved {len(saved_paths)} violin image(s).")
        preview = "\n".join(saved_paths[:8])
        if len(saved_paths) > 8:
            preview += "\n..."
        messagebox.showinfo("Done", f"Saved {len(saved_paths)} violin image(s).\n\n{preview}")

    def _run_bimodality_analysis(self):
        if self.df is None:
            raise ValueError("Please load a CSV file first.")
        if self.bimodality_require_diptest_var.get() and not DIPTEST_AVAILABLE:
            raise ValueError("Hartigan dip test is required, but the optional Python package 'diptest' is not installed. Install it with: pip install diptest")
        protein_col, kw_a, kw_b, label_a, label_b = self._get_shared_inputs()
        return build_bimodality_analysis(
            self.df,
            protein_col,
            kw_a,
            kw_b,
            condition_a_label=label_a,
            condition_b_label=label_b,
            min_valid_n=int(self.bimodality_min_valid_n_var.get()),
            min_fraction_per_mode=float(self.bimodality_min_fraction_per_mode_var.get()),
            min_cells_per_mode=int(self.bimodality_min_cells_per_mode_var.get()),
            min_mode_separation=float(self.bimodality_min_mode_separation_var.get()),
            max_dip_p_value=float(self.bimodality_max_dip_p_value_var.get()),
            min_valley_depth=float(self.bimodality_min_valley_depth_var.get()),
            require_gmm_improvement=bool(self.bimodality_require_gmm_improvement_var.get()),
        )

    def _resolve_bimodality_selection(self):
        bimodality_df, cond_a_cols, cond_b_cols, work = self._run_bimodality_analysis()
        protein_col, _, _, label_a, label_b = self._get_shared_inputs()
        mode = self.bimodality_sort_mode_var.get()
        top_n = int(self.bimodality_top_n_var.get())

        score_a_col = f"balanced_bimodality_score_{label_a}"
        score_b_col = f"balanced_bimodality_score_{label_b}"
        delta_col = f"delta_balanced_bimodality_{label_b}_minus_{label_a}"

        df = bimodality_df.copy()
        if mode == "Largest increase in balanced bimodality (B-A)":
            df["_plot_value"] = pd.to_numeric(df[delta_col], errors="coerce")
            df = df[df["_plot_value"].notna()].sort_values(["_plot_value", "protein"], ascending=[False, True])
            value_label = f"Δ balanced bimodality ({label_b} - {label_a})"
        elif mode == "Largest decrease in balanced bimodality (A-B)":
            # Decrease from A to B means proteins that are more bimodal in Group A than Group B.
            # Since delta_col is B - A, multiply by -1 so larger positive values represent larger decreases.
            df["_plot_value"] = -pd.to_numeric(df[delta_col], errors="coerce")
            df = df[df["_plot_value"].notna()].sort_values(["_plot_value", "protein"], ascending=[False, True])
            value_label = f"Decrease in balanced bimodality ({label_a} - {label_b})"
        elif mode == "Largest absolute change in balanced bimodality (|B-A|)":
            abs_delta_col = f"abs_delta_balanced_bimodality_{label_b}_minus_{label_a}"
            df["_plot_value"] = pd.to_numeric(df[abs_delta_col], errors="coerce")
            df = df[df["_plot_value"].notna()].sort_values(["_plot_value", "protein"], ascending=[False, True])
            value_label = f"|Δ balanced bimodality| ({label_b} - {label_a})"
        elif mode == "Highest balanced bimodality in Group A":
            df["_plot_value"] = pd.to_numeric(df[score_a_col], errors="coerce")
            df = df[df["_plot_value"].notna()].sort_values(["_plot_value", "protein"], ascending=[False, True])
            value_label = f"Balanced bimodality score ({label_a})"
        elif mode == "Highest balanced bimodality in Group B":
            df["_plot_value"] = pd.to_numeric(df[score_b_col], errors="coerce")
            df = df[df["_plot_value"].notna()].sort_values(["_plot_value", "protein"], ascending=[False, True])
            value_label = f"Balanced bimodality score ({label_b})"
        else:
            df["_plot_value"] = pd.to_numeric(df["max_balanced_bimodality_score"], errors="coerce")
            df = df[df["_plot_value"].notna()].sort_values(["_plot_value", "protein"], ascending=[False, True])
            value_label = "Max balanced bimodality score across conditions"

        df = df[pd.to_numeric(df["_plot_value"], errors="coerce") > 0].copy()
        top_df = df.head(top_n).copy()
        if top_df.empty:
            raise ValueError("No proteins passed the strict bimodality filters. Try lowering min mode separation, min valley depth, min fraction/mode, min cells/mode, or max dip p-value.")

        work_indexed = work.set_index(work[protein_col].astype(str), drop=False)
        return {
            "bimodality_df": bimodality_df,
            "top_df": top_df,
            "work": work,
            "work_indexed": work_indexed,
            "protein_col": protein_col,
            "cond_a_cols": cond_a_cols,
            "cond_b_cols": cond_b_cols,
            "label_a": label_a,
            "label_b": label_b,
            "mode": mode,
            "value_label": value_label,
            "proteins": top_df["protein"].astype(str).tolist(),
            "plot_value_lookup": dict(zip(top_df["protein"].astype(str), pd.to_numeric(top_df["_plot_value"], errors="coerce"))),
        }

    def _bimodality_shared_ylim(self, proteins, ctx):
        if not self.bimodality_fix_shared_yaxis_var.get():
            return None
        arrays = []
        for protein in ctx.get("proteins", proteins):
            arrays.extend(self._values_for_violin_protein(
                ctx["work_indexed"], protein, ctx["cond_a_cols"], ctx["cond_b_cols"], "Group B - Group A", True
            ))
        arrays = [arr for arr in arrays if arr.size]
        if not arrays:
            return None
        vals = np.concatenate(arrays)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        ymin = float(np.min(vals))
        ymax = float(np.max(vals))
        pad = max(0.5, (ymax - ymin) * 0.08) if ymax > ymin else 0.5
        return ymin - pad, ymax + pad

    def _draw_bimodality_violins_to_figure(self, fig, proteins, ctx):
        fig.clear()
        n = len(proteins)
        ncols = 2 if n > 1 else 1
        nrows = math.ceil(n / ncols)
        base_width = float(self.violin_width_var.get())
        base_height = float(self.violin_height_var.get())
        fig_height = max(base_height, nrows * 3.0)
        fig.set_size_inches(base_width, fig_height, forward=True)
        fig.set_dpi(int(self.preview_dpi_var.get()))

        rng = np.random.default_rng(42)
        shared_ylim = self._bimodality_shared_ylim(proteins, ctx)
        work_indexed = ctx["work_indexed"]
        label_a = ctx["label_a"]
        label_b = ctx["label_b"]
        lookup = ctx["plot_value_lookup"]

        for idx, protein in enumerate(proteins, start=1):
            ax = fig.add_subplot(nrows, ncols, idx)
            values_arr = self._values_for_violin_protein(
                work_indexed, protein, ctx["cond_a_cols"], ctx["cond_b_cols"], "Group B - Group A", True
            )
            if not any(arr.size for arr in values_arr):
                ax.text(0.5, 0.5, f"No values for\n{protein}", ha="center", va="center")
                ax.axis("off")
                continue
            positions = [1, 2]
            colors = [self._condition_a_color(), self._condition_b_color()]
            parts = ax.violinplot(values_arr, positions=positions, widths=0.75, showmeans=False, showmedians=False, showextrema=False)
            for body, color in zip(parts["bodies"], colors):
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.35)
            for pos, arr, color in zip(positions, values_arr, colors):
                if arr.size:
                    jitter = rng.uniform(-0.08, 0.08, size=arr.size)
                    dot_size = max(0.1, float(self.bimodality_dot_size_var.get()))
                    ax.scatter(np.full(arr.size, pos) + jitter, arr, s=dot_size, alpha=0.75, color=color, edgecolors="none")
                    med = np.median(arr)
                    q1 = np.percentile(arr, 25)
                    q3 = np.percentile(arr, 75)
                    ax.hlines(med, pos - 0.20, pos + 0.20, color="black", linewidth=1.4)
                    ax.vlines(pos, q1, q3, color="black", linewidth=1.2)
            plot_value = lookup.get(protein, np.nan)
            title = f"{protein} ({plot_value:.3f})" if np.isfinite(plot_value) else str(protein)
            self._style_axes_text(ax, "violin", xlabel=None, ylabel="Abundance", title=title, apply_title_override=False)
            ax.set_xticks(positions)
            ax.set_xticklabels([label_a, label_b])
            if shared_ylim is not None:
                ax.set_ylim(shared_ylim)

        method = "Hartigan dip test" if DIPTEST_AVAILABLE else "KDE peak-separation fallback"
        fig.suptitle(f"Top proteins by strict balanced bimodality: {ctx['mode']} ({method}; min separation={float(self.bimodality_min_mode_separation_var.get()):.2f}; min valley={float(self.bimodality_min_valley_depth_var.get()):.2f})", **self._font_kwargs("title"), y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])

    def plot_bimodality_violins(self):
        try:
            ctx = self._resolve_bimodality_selection()
        except Exception as exc:
            messagebox.showerror("Bimodality error", str(exc))
            return
        self._draw_bimodality_violins_to_figure(self.bimodality_tab.figure, ctx["proteins"], ctx)
        self.bimodality_tab.canvas.draw_idle()
        method = "Hartigan dip test" if DIPTEST_AVAILABLE else "KDE peak-separation fallback; install `diptest` for Hartigan dip p-values"
        warning = "" if DIPTEST_AVAILABLE else " WARNING: KDE fallback is approximate; tick 'Require Hartigan dip test' after installing diptest for stricter analysis."
        self.bimodality_tab.set_status(
            f"Generated bimodality violins for {len(ctx['proteins'])} proteins using {ctx['mode']}. "
            f"Method: {method}. Min separation={float(self.bimodality_min_mode_separation_var.get()):.2f}; min valley={float(self.bimodality_min_valley_depth_var.get()):.2f}; max dip p={float(self.bimodality_max_dip_p_value_var.get()):.3g}. "
            f"Matched cells: {ctx['label_a']}={len(ctx['cond_a_cols'])}, {ctx['label_b']}={len(ctx['cond_b_cols'])}." + warning
        )


    def plot_bimodality_effect_map(self):
        """Plot a bimodality effect map analogous to the Group B - Group A effect map.

        X-axis: directional change in mean log2 abundance (Group B - Group A).
        Y-axis: directional change in strict balanced bimodality score (Group B - Group A).
        Therefore, proteins that lose bimodality upon treatment have negative y values.
        """
        try:
            bimodality_df, cond_a_cols, cond_b_cols, work = self._run_bimodality_analysis()
            protein_col, _, _, label_a, label_b = self._get_shared_inputs()
        except Exception as exc:
            messagebox.showerror("Bimodality effect-map error", str(exc))
            return

        delta_bimodality_col = f"delta_balanced_bimodality_{label_b}_minus_{label_a}"
        if delta_bimodality_col not in bimodality_df.columns:
            messagebox.showerror("Bimodality effect-map error", f"Could not find column: {delta_bimodality_col}")
            return

        mean_a = work[cond_a_cols].mean(axis=1, skipna=True) if cond_a_cols else pd.Series(np.nan, index=work.index)
        mean_b = work[cond_b_cols].mean(axis=1, skipna=True) if cond_b_cols else pd.Series(np.nan, index=work.index)
        mean_df = pd.DataFrame({
            "protein": work[protein_col].astype(str),
            "_x": pd.to_numeric(mean_b - mean_a, errors="coerce"),
        })

        plot_df = bimodality_df[["protein", delta_bimodality_col]].copy()
        plot_df["protein"] = plot_df["protein"].astype(str)
        plot_df = plot_df.merge(mean_df, on="protein", how="left")
        plot_df["_y"] = pd.to_numeric(plot_df[delta_bimodality_col], errors="coerce")
        plot_df = plot_df.dropna(subset=["_x", "_y"])

        if plot_df.empty:
            messagebox.showinfo("No data", "No proteins had valid mean-change and bimodality-change values to plot.")
            return

        fig = self._prepare_standard_figure(self.bimodality_tab, self.effect_width_var, self.effect_height_var)
        ax = fig.add_subplot(111)

        increased = plot_df["_y"] > 0
        decreased = plot_df["_y"] < 0
        unchanged = ~(increased | decreased)

        if unchanged.any():
            sub = plot_df[unchanged]
            ax.scatter(sub["_x"], sub["_y"], s=20, alpha=0.45, color=NOT_SIG_COLOR, edgecolors="none", label=f"No bimodality change (n={len(sub)})")
        if increased.any():
            sub = plot_df[increased]
            ax.scatter(sub["_x"], sub["_y"], s=32, alpha=0.78, color=self._condition_b_color(), edgecolors="none", label=f"Increased bimodality in {label_b} (n={len(sub)})")
        if decreased.any():
            sub = plot_df[decreased]
            ax.scatter(sub["_x"], sub["_y"], s=32, alpha=0.78, color=self._condition_a_color(), edgecolors="none", label=f"Decreased bimodality in {label_b} (n={len(sub)})")

        label_n = int(self.effect_label_top_n_var.get()) if hasattr(self, "effect_label_top_n_var") else 15
        if label_n > 0:
            label_df = plot_df.copy()
            label_df["_abs_y"] = label_df["_y"].abs()
            label_df["_abs_x"] = label_df["_x"].abs()
            label_df = label_df.sort_values(["_abs_y", "_abs_x", "protein"], ascending=[False, False, True]).head(label_n)
            self._add_standard_labels(
                ax,
                label_df=label_df,
                x_col="_x",
                y_col="_y",
                label_col="protein",
                fontsize=self._font_size("annotation"),
            )

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.35)
        self._style_axes_text(
            ax,
            "effect",
            xlabel=f"Mean abundance change ({label_b} - {label_a}; log2 intensity)",
            ylabel=f"Δ balanced bimodality ({label_b} - {label_a})",
            title=f"Bimodality effect map: {label_b} - {label_a}",
            apply_title_override=False,
        )
        ax.grid(alpha=0.18)
        legend = ax.legend(frameon=False, loc="best")
        self._style_legend(legend)
        fig.tight_layout()
        self.bimodality_tab.canvas.draw_idle()
        self.bimodality_tab.set_status(
            f"Generated bimodality effect map. X = mean abundance change ({label_b} - {label_a}); "
            f"Y = directional change in strict balanced bimodality ({label_b} - {label_a}). "
            f"Negative y values indicate reduced bimodality in {label_b}. Labels are top {label_n} by |Δ bimodality|."
        )

    def export_bimodality_table(self):
        try:
            bimodality_df, _, _, _ = self._run_bimodality_analysis()
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            return
        path = filedialog.asksaveasfilename(
            title="Save bimodality table",
            defaultextension=".csv",
            initialfile="bimodality_analysis.csv",
            filetypes=[("CSV file", "*.csv")],
        )
        if not path:
            return
        bimodality_df.to_csv(path, index=False)
        self.bimodality_tab.set_status(f"Saved bimodality table to: {path}")
        messagebox.showinfo("Done", f"Bimodality table saved successfully.\n\n{path}")

    def plot_pca(self):
        try:
            work, cond_a_cols, cond_b_cols = self._get_wide_numeric_work()
            protein_col, _, _, label_a, label_b = self._get_shared_inputs()
            view_mode = self.pca_view_mode_var.get()

            pca_a_cols = cond_a_cols
            pca_b_cols = cond_b_cols
            if view_mode == "Group A only":
                pca_b_cols = []
            elif view_mode == "Group B only":
                pca_a_cols = []

            pca_df, explained, used_proteins = compute_pca_projection(
                work,
                protein_col,
                pca_a_cols,
                pca_b_cols,
                top_variable_proteins=int(self.pca_top_variable_var.get()),
            )
        except Exception as exc:
            messagebox.showerror("Plot error", str(exc))
            return

        pca_df["condition_label"] = np.where(pca_df["condition_code"] == "A", label_a, label_b)

        fig = self._prepare_standard_figure(self.pca_tab, self.pca_width_var, self.pca_height_var)
        ax = fig.add_subplot(111)

        if view_mode == "Both groups":
            plot_groups = [(label_a, self._condition_a_color()), (label_b, self._condition_b_color())]
        elif view_mode == "Group A only":
            plot_groups = [(label_a, self._condition_a_color())]
        else:
            plot_groups = [(label_b, self._condition_b_color())]

        for cond_label, color in plot_groups:
            sub = pca_df[pca_df["condition_label"] == cond_label]
            if sub.empty:
                continue
            ax.scatter(sub["PC1"], sub["PC2"], s=30, alpha=0.75, color=color, edgecolors="none", label=f"{cond_label} (n={len(sub)})")

        pc1_pct = explained[0] * 100 if np.isfinite(explained[0]) else np.nan
        pc2_pct = explained[1] * 100 if np.isfinite(explained[1]) else np.nan
        self._style_axes_text(
            ax,
            "pca",
            xlabel=f"PC1 ({pc1_pct:.1f}% variance)",
            ylabel=f"PC2 ({pc2_pct:.1f}% variance)",
            title=f"Cell-level PCA: {view_mode}",
        )
        legend = ax.legend(frameon=False, loc="best")
        self._style_legend(legend)
        fig.tight_layout()
        self.pca_tab.canvas.draw_idle()
        self.pca_tab.set_status(
            f"PCA generated for {view_mode} using the top {len(used_proteins)} variable proteins. "
            f"Matched cells: {label_a}={len(cond_a_cols)}, {label_b}={len(cond_b_cols)}."
        )


def main():
    app = SingleCellSummaryGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
