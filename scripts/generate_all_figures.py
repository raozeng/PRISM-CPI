#!/usr/bin/env python3
"""
Generates all manuscript figures (figures/fig1..fig10.{pdf,png}, graphical_abstract.{pdf,png})
and LaTeX tables (figures/table1..table5.tex) from the evaluation outputs in ./results/.
"""

import csv
import gzip
import sys
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scipy import stats

# Ensure writable Matplotlib cache directory before importing matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_prism_cpi_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch, Rectangle, Ellipse, Polygon, PathPatch
from matplotlib.path import Path as MplPath
import matplotlib.patheffects as patheffects
import matplotlib.gridspec as gridspec
import numpy as np

# =============================================================================
# Matplotlib rcParams and Color Palette
# =============================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9.5,
    "axes.titlesize": 14.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "figure.titlesize": 13.0,
    "figure.titleweight": "bold",
    "text.color": "#2B3542",
    "axes.labelcolor": "#2B3542",
    "axes.edgecolor": "#526071",
    "xtick.color": "#2B3542",
    "ytick.color": "#2B3542",
    "axes.linewidth": 0.85,
    "grid.color": "#DCE2E8",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.45,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})

# Exact Color Palette (Berry Rose, Oxford Navy, Forest Teal, Royal Purple, Warm Amber)
PALETTE = {
    "prism": "#BC6E80",        # Family E: Berry Rose Primary Bar / Marker Fill (S=0.42, V=0.74)
    "prism_dark": "#914153",   # Family E: Deep Berry Burgundy Header / Bold Text / Curve (S=0.55, V=0.57)
    "prism_light": "#F2E2E6",  # Family E: Soft Rose Highlight Pill Fill
    "destin": "#4E769E",       # Family A: Oxford Steel Blue Bar / Line Fill (S=0.50, V=0.62)
    "destin_dark": "#2E4D72",  # Family A: Deep Oxford Navy Header / Bold Text (S=0.60, V=0.45)
    "destin_light": "#F4F8FC", # Family A: Soft Navy Card Tint
    "graphban": "#43877E",     # Family C: Forest Pine Teal Bar / Line Fill (S=0.50, V=0.53)
    "graphban_dark": "#2D6B64",# Family C: Deep Forest Teal Header / Accent Curve (S=0.58, V=0.42)
    "drugban": "#7862A8",      # Family D: Royal Iris Purple Bar / Line Fill (S=0.42, V=0.66)
    "drugban_dark": "#5E468A", # Family D: Deep Royal Purple Header / Accent Curve (S=0.49, V=0.54)
    "fusiondti": "#5186A8",    # Supporting Cerulean Blue from Contact Map (S=0.52, V=0.66)
    "graphsformer": "#C28B4E", # Family B: Warm Amber Bronze Bar / Line Fill (S=0.60, V=0.76)
    "amber_dark": "#9D6833",   # Family B: Deep Amber Bronze Header / Accent Curve (S=0.67, V=0.62)
    "pocketdta": "#76A89D",    # Supporting Seafoam Sage from Contact Map (S=0.30, V=0.66)
    "other_base": "#9AA6B5",   # Supporting Cool Silver-Slate Inactive Bar (S=0.15, V=0.71)
    "mod1_bg": "#F4F8FC",      # Family A Module 1 panel tint
    "mod1_sub": "#DFECF3",     # Family A Module 1 subheader fill
    "mod1_border": "#88A6C4",  # Family A Module 1 border
    "mod1_text": "#2E4D72",    # Family A Module 1 dark text
    "mod2_bg": "#F2FAF8",      # Family C Module 2 panel tint
    "mod2_sub": "#D6ECE8",     # Family C Module 2 subheader fill
    "mod2_border": "#84B5AE",  # Family C Module 2 border
    "mod2_text": "#2D6B64",    # Family C Module 2 dark text
    "mod3_bg": "#F6F4FC",      # Family D Module 3 panel tint
    "mod3_sub": "#E5E0F5",     # Family D Module 3 subheader fill
    "mod3_border": "#A599C8",  # Family D Module 3 border
    "mod3_text": "#5E468A",    # Family D Module 3 dark text
    "ochre_bg": "#FAF5EA",     # Family B Warm Amber panel tint
    "ochre_sub": "#F4EBD7",    # Family B Warm Sand subheader fill
    "ochre_border": "#CFAF82", # Family B Warm Amber border
    "ochre_text": "#9D6833",   # Family B Deep Amber Bronze text
    "terra_bg": "#FAF3F5",     # Family E Berry Rose card tint
    "terra_sub": "#F2E2E6",    # Family E Berry Rose subheader fill
    "terra_border": "#C99AA5", # Family E Berry Rose soft border
    "terra_text": "#914153",   # Family E Deep Berry Burgundy text
    "slate_dark": "#2B3542",   # Neutral Typography & Spines
    "slate_mid": "#526071",    # Muted Mid Slate
    "slate_light": "#EDF2F7",  # Soft Neutral Fill
    "err_color": "#3E4C59",    # Error bar color
    "frame_border": "#CBD5E1", # Subtle Legend/Colorbar Frame
}

REF_CONTACT_CMAP = LinearSegmentedColormap.from_list(
    "ref_contact",
    ["#F5F9EE", "#DBE9D8", "#8BBCC3", "#5186A8", "#2E4D72"],
    N=256,
)

REF_ROSE_CMAP = LinearSegmentedColormap.from_list(
    "ref_rose",
    ["#FDF7F8", "#F2DFE4", "#DDAAB7", "#BC6E80", "#914153"],
    N=256,
)

REF_PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "ref_purple",
    ["#F7F5FC", "#E3DDF3", "#B1A3D4", "#7862A8", "#5E468A"],
    N=256,
)

REF_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "ref_blue",
    ["#F4F8FC", "#DCE8F4", "#9BB8D4", "#5B7C9B", "#2E4D72"],
    N=256,
)

REF_RBF_CMAP = LinearSegmentedColormap.from_list(
    "ref_rbf",
    ["#8BBCC3", "#5186A8", "#43877E", "#2E4D72"],
    N=256,
)

REF_SHAP_CMAP = LinearSegmentedColormap.from_list(
    "ref_shap_navy_rose",
    ["#2E4D72", "#5186A8", "#DCE2E8", "#DDAAB7", "#914153"],
    N=256,
)

ERROR_KW_SLENDER = dict(elinewidth=0.85, ecolor=PALETTE["err_color"], capthick=0.85)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
RAW_RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"


def _clamp_color_saturation(color: Any, max_sat: float = 0.68) -> Any:
    """Preserves all color palette colors while capping accidental neon S > max_sat."""
    if color is None or (isinstance(color, str) and color.lower() == "none"):
        return color
    try:
        r, g, b, a = mcolors.to_rgba(color)
    except Exception:
        return color
    if a == 0.0:
        return "none"
    h, s, v = mcolors.rgb_to_hsv(np.array([r, g, b], dtype=float))
    if s <= max_sat:
        return color
    rgb_new = mcolors.hsv_to_rgb(np.array([h, max(0.0, max_sat - 0.010), v]))
    return (float(rgb_new[0]), float(rgb_new[1]), float(rgb_new[2]), float(a))


def _clamp_colormap_saturation(cmap: Any, max_sat: float = 0.68) -> Any:
    """Clamps all LUT entries of a Matplotlib colormap so HSV saturation S <= max_sat."""
    if cmap is None:
        return cmap
    lut = cmap(np.linspace(0.0, 1.0, 256))
    hsv = mcolors.rgb_to_hsv(lut[:, :3])
    if float(np.max(hsv[:, 1])) <= max_sat:
        return cmap
    hsv[:, 1] = np.minimum(hsv[:, 1], max_sat - 0.010)
    lut_clamped = np.column_stack([mcolors.hsv_to_rgb(hsv), lut[:, 3]])
    return LinearSegmentedColormap.from_list(f"{cmap.name}_ref", lut_clamped, N=256)


def style_borderless_legend(leg: Any) -> Any:
    """Formats legend frame and patch handles."""
    if leg is None:
        return None
    if leg.get_frame_on():
        frame = leg.get_frame()
        frame.set_facecolor("#FFFFFF")
        frame.set_edgecolor(PALETTE["frame_border"])
        frame.set_linewidth(0.6)
    handles = getattr(leg, "legend_handles", None)
    if handles is None:
        handles = getattr(leg, "legendHandles", [])
    for h in list(handles) + list(leg.get_patches()):
        if isinstance(h, Patch):
            h.set_edgecolor("none")
            h.set_linewidth(0.0)
    return leg


def apply_figure_style(fig: plt.Figure, max_sat: float = 0.68) -> None:
    """Normalizes bar styles, legend frames, and subplot title sizes before saving."""
    from matplotlib.ticker import FixedFormatter, FixedLocator

    all_legends = list(fig.legends)
    saved_fixed_ticks: List[Tuple[plt.Axes, str, Any, List[str]]] = []
    for ax in fig.axes:
        for axis_name, axis_obj in (("x", ax.xaxis), ("y", ax.yaxis)):
            loc_obj = axis_obj.get_major_locator()
            fmt_obj = axis_obj.get_major_formatter()
            if isinstance(loc_obj, FixedLocator) and isinstance(fmt_obj, FixedFormatter):
                saved_fixed_ticks.append((ax, axis_name, list(loc_obj.locs), list(fmt_obj.seq)))

    for ax in fig.axes:
        ax.set_facecolor(_clamp_color_saturation(ax.get_facecolor(), max_sat))

        # 1. Format bar containers and error bars
        for container in ax.containers:
            if isinstance(container, BarContainer):
                for rect in container.patches:
                    rect.set_edgecolor("none")
                    rect.set_linewidth(0.0)
            elif isinstance(container, ErrorbarContainer):
                _, caplines, barlinecols = container
                for cap in caplines:
                    cap.set_color(_clamp_color_saturation(cap.get_color(), max_sat))
                for blc in barlinecols:
                    ecs = blc.get_edgecolor()
                    if ecs is not None and len(ecs) > 0:
                        blc.set_edgecolor([_clamp_color_saturation(c, max_sat) for c in ecs])

        # 2. Spines, ticks, axis labels, and titles
        for sp in ax.spines.values():
            sp.set_edgecolor(_clamp_color_saturation(sp.get_edgecolor(), max_sat))
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_color(_clamp_color_saturation(tl.get_color(), max_sat))
        for tk in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
            tk.set_color(_clamp_color_saturation(tk.get_color(), max_sat))
        ax.xaxis.label.set_color(_clamp_color_saturation(ax.xaxis.label.get_color(), max_sat))
        ax.yaxis.label.set_color(_clamp_color_saturation(ax.yaxis.label.get_color(), max_sat))
        ax.title.set_color(_clamp_color_saturation(ax.title.get_color(), max_sat))
        if hasattr(ax, "_left_title"):
            ax._left_title.set_color(_clamp_color_saturation(ax._left_title.get_color(), max_sat))

        # 3. Patches, Lines, Collections, Images, and Texts/Annotations
        for p in ax.patches:
            fc = p.get_facecolor()
            if fc is not None and len(fc) == 4 and fc[3] > 0.0:
                p.set_facecolor(_clamp_color_saturation(fc, max_sat))
            ec = p.get_edgecolor()
            if ec is not None and len(ec) == 4 and ec[3] > 0.0 and p.get_linewidth() > 0.0:
                p.set_edgecolor(_clamp_color_saturation(ec, max_sat))

        for ln in ax.lines:
            ln.set_color(_clamp_color_saturation(ln.get_color(), max_sat))
            mfc = ln.get_markerfacecolor()
            if mfc is not None and not (isinstance(mfc, str) and mfc.lower() == "none"):
                ln.set_markerfacecolor(_clamp_color_saturation(mfc, max_sat))
            mec = ln.get_markeredgecolor()
            if mec is not None and not (isinstance(mec, str) and mec.lower() == "none"):
                ln.set_markeredgecolor(_clamp_color_saturation(mec, max_sat))

        for col in ax.collections:
            if col.get_array() is not None and col.get_cmap() is not None:
                orig_cmap = col.get_cmap()
                clamped_cmap = _clamp_colormap_saturation(orig_cmap, max_sat)
                if clamped_cmap is not orig_cmap:
                    col.set_cmap(clamped_cmap)
            fcs = col.get_facecolor()
            if fcs is not None and len(fcs) > 0:
                col.set_facecolor([_clamp_color_saturation(c, max_sat) for c in fcs])
            ecs = col.get_edgecolor()
            if ecs is not None and len(ecs) > 0:
                col.set_edgecolor([_clamp_color_saturation(c, max_sat) for c in ecs])

        for im in ax.images:
            orig_im_cmap = im.get_cmap()
            clamped_im_cmap = _clamp_colormap_saturation(orig_im_cmap, max_sat)
            if clamped_im_cmap is not orig_im_cmap:
                im.set_cmap(clamped_im_cmap)

        for txt in ax.texts:
            txt.set_color(_clamp_color_saturation(txt.get_color(), max_sat))
            bb = txt.get_bbox_patch()
            if bb is not None:
                bfc = bb.get_facecolor()
                if bfc is not None and len(bfc) == 4 and bfc[3] > 0.0:
                    bb.set_facecolor(_clamp_color_saturation(bfc, max_sat))
                bec = bb.get_edgecolor()
                if bec is not None and len(bec) == 4 and bec[3] > 0.0 and bb.get_linewidth() > 0.0:
                    bb.set_edgecolor(_clamp_color_saturation(bec, max_sat))
                else:
                    bb.set_edgecolor("none")
                    bb.set_linewidth(0.0)
            ap = getattr(txt, "arrow_patch", None)
            if ap is not None:
                afc = ap.get_facecolor()
                if afc is not None and len(afc) == 4 and afc[3] > 0.0:
                    ap.set_facecolor(_clamp_color_saturation(afc, max_sat))
                aec = ap.get_edgecolor()
                if aec is not None and len(aec) == 4 and aec[3] > 0.0:
                    ap.set_edgecolor(_clamp_color_saturation(aec, max_sat))

        if ax.get_legend() is not None:
            all_legends.append(ax.get_legend())

    # Restore any custom FixedLocator/FixedFormatter ticks (e.g., inset colorbar Low/Mid/High)
    for ax, axis_name, locs, labels in saved_fixed_ticks:
        if axis_name == "x":
            ax.set_xticks(locs)
            ax.set_xticklabels(labels)
        else:
            ax.set_yticks(locs)
            ax.set_yticklabels(labels)

    for txt in fig.texts:
        txt.set_color(_clamp_color_saturation(txt.get_color(), max_sat))

    # 4. Strip borders from all legend bar/patch handles and clamp legend colors
    for leg in all_legends:
        style_borderless_legend(leg)
        for p in leg.get_patches():
            p.set_facecolor(_clamp_color_saturation(p.get_facecolor(), max_sat))
        for ln in leg.get_lines():
            ln.set_color(_clamp_color_saturation(ln.get_color(), max_sat))
        for t in leg.get_texts():
            t.set_color(_clamp_color_saturation(t.get_color(), max_sat))

    # 5. Maximize subplot title font sizes (A-F: 13.2-15.2 pt bold; A1-A4: up to 13.2 pt bold)
    #    Exclude narrow inset/colorbar/strip axes (ax_bbox.width < fig_w_px * 0.22)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w_px = fig.get_window_extent(renderer).width
    title_groups: Dict[str, List[Tuple[plt.Axes, Any, float, str]]] = {}
    for ax in fig.axes:
        ax_bbox = ax.get_window_extent(renderer)
        loc_used = "left" if ax.get_title(loc="left").strip() else ("center" if ax.get_title(loc="center").strip() else "")
        if not loc_used:
            continue
        t_str = ax.get_title(loc=loc_used).strip()
        t_obj = ax._left_title if loc_used == "left" else ax.title
        if re.match(r"^[A-Z]\d+\s+", t_str):
            if ax_bbox.width < fig_w_px * 0.16:
                continue
            title_groups.setdefault("subgrid", []).append((ax, t_obj, ax_bbox.width, loc_used))
        elif re.match(r"^[A-F]\s+", t_str):
            if ax_bbox.width < fig_w_px * 0.22:
                continue
            title_groups.setdefault("main", []).append((ax, t_obj, ax_bbox.width, loc_used))

    for g_key, items in title_groups.items():
        max_cap_pt = 13.2 if g_key == "subgrid" else 15.2
        min_floor_pt = 11.5
        group_best_fs = max_cap_pt
        for ax, t_obj, ax_w_px, _ in items:
            t_obj.set_fontweight("bold")
            lo, hi, best_fs = min_floor_pt, max_cap_pt, min_floor_pt
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                t_obj.set_fontsize(mid)
                t_w_px = t_obj.get_window_extent(renderer).width
                if t_w_px <= ax_w_px * 0.982:
                    best_fs = mid
                    lo = mid
                else:
                    hi = mid
            group_best_fs = min(group_best_fs, best_fs)
        final_fs = round(group_best_fs, 1)
        for ax, t_obj, _, loc_used in items:
            t_str = ax.get_title(loc=loc_used)
            pad_val = 7.5 if g_key == "subgrid" else (10.2 if final_fs >= 13.0 else 9.5)
            ax.set_title(t_str, loc=loc_used, fontsize=final_fs, fontweight="bold", pad=pad_val, color=t_obj.get_color())
    fig.canvas.draw()


def load_evaluation_data() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Loads evaluation metrics, training curves, SHAP attributions, and 3D docking data from ./results/."""
    with open(RAW_RESULTS_DIR / "metrics_summary.json", "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    tr_json_path = RAW_RESULTS_DIR / "logs" / "training_runs.json"
    tr_gz_path = RAW_RESULTS_DIR / "logs" / "training_runs.json.gz"
    if tr_json_path.exists():
        with open(tr_json_path, "r", encoding="utf-8") as tf:
            training_runs = json.load(tf)
    else:
        with gzip.open(tr_gz_path, "rt", encoding="utf-8") as tf:
            training_runs = json.load(tf)

    with open(RAW_RESULTS_DIR / "shap_attribution_summary.json", "r", encoding="utf-8") as sf:
        shap_data = json.load(sf)

    with open(RAW_RESULTS_DIR / "docking_network_validation.json", "r", encoding="utf-8") as df:
        dock_data = json.load(df)

    docking_30_rows: List[Dict[str, Any]] = []
    csv_30_path = RAW_RESULTS_DIR / "tables" / "docking_validation_30complexes.csv"
    if csv_30_path.exists():
        with open(csv_30_path, "r", encoding="utf-8") as cf:
            docking_30_rows = list(csv.DictReader(cf))

    return metrics_data, training_runs, shap_data, dock_data, docking_30_rows


def load_trimmed_image(
    png_path: Path,
    pad: int = 14,
    top_pad: int = 70,
    bottom_pad: int = 18,
    min_aspect_wh: float = 0.0,
    max_sat: float = 0.64,
) -> np.ndarray:
    """Reads a raster PNG, composites RGBA over white, auto-crops outer whitespace, and clamps HSV saturation."""
    raw = plt.imread(str(png_path)).astype(np.float32)
    if raw.max() > 1.0:
        raw = raw / 255.0
    if raw.ndim == 3 and raw.shape[-1] == 4:
        alpha = raw[..., 3:4]
        rgb = raw[..., :3] * alpha + (1.0 - alpha)
    elif raw.ndim == 2:
        rgb = np.stack([raw, raw, raw], axis=-1)
    else:
        rgb = raw[..., :3]

    mask = np.any(rgb < 0.985, axis=-1)
    if np.any(mask):
        ys, xs = np.where(mask)
        r0 = max(0, int(ys.min()) - pad)
        r1 = min(rgb.shape[0], int(ys.max()) + pad + 1)
        c0 = max(0, int(xs.min()) - pad)
        c1 = min(rgb.shape[1], int(xs.max()) + pad + 1)
        crop = rgb[r0:r1, c0:c1].copy()
    else:
        crop = rgb.copy()

    # Optional gentle horizontal centering on pure white for portrait 2D structures
    h_c, w_c = crop.shape[:2]
    if min_aspect_wh > 0.0 and (w_c / max(1, h_c)) < min_aspect_wh:
        target_w = int(round(h_c * min_aspect_wh))
        pad_left = (target_w - w_c) // 2
        pad_right = target_w - w_c - pad_left
        crop = np.pad(crop, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=1.0)

    if top_pad > 0 or bottom_pad > 0:
        crop = np.pad(crop, ((top_pad, bottom_pad), (0, 0), (0, 0)), mode="constant", constant_values=1.0)

    hsv = mcolors.rgb_to_hsv(np.clip(crop, 0.0, 1.0))
    hsv[..., 1] = np.minimum(hsv[..., 1], max_sat)
    return np.clip(mcolors.hsv_to_rgb(hsv), 0.0, 1.0)


def save_dual_format(fig: plt.Figure, stem: str, dpi: int = 320, pad_inches: float = 0.01) -> None:
    """Enforces borderless color palette styling and saves both vector PDF and high-DPI PNG into FIGURES_DIR."""
    apply_figure_style(fig, max_sat=0.68)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    png_path = FIGURES_DIR / f"{stem}.png"
    fig.savefig(pdf_path, format="pdf", dpi=dpi, bbox_inches="tight", pad_inches=pad_inches, facecolor="#FFFFFF")
    fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight", pad_inches=pad_inches, facecolor="#FFFFFF")
    plt.close(fig)
    print(f"[GENERATED] {pdf_path.name} ({pdf_path.stat().st_size // 1024} KB) & {png_path.name} ({png_path.stat().st_size // 1024} KB)")


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str = "#FFFFFF",
    ec: str = "#526071",
    lw: float = 1.25,
    fontsize: float = 8.2,
    fontweight: str = "normal",
    text_color: str = "#2B3542",
    boxstyle: str = "round,pad=0.010,rounding_size=0.016",
    linestyle: str = "-",
    linespacing: float = 1.36,
    zorder: int = 3,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=boxstyle,
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            x + w / 2.0,
            y + h / 2.0,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=fontweight,
            color=text_color,
            zorder=zorder + 1,
            linespacing=linespacing,
        )
    return patch


def draw_arrow(
    ax: plt.Axes,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "#526071",
    lw: float = 1.35,
    style: str = "-|>",
    mutation_scale: float = 12.0,
    rad: float = 0.0,
    linestyle: str = "-",
    label: str = "",
    label_pos: Tuple[float, float] = (0.0, 0.0),
    fontsize: float = 7.6,
    zorder: int = 4,
) -> FancyArrowPatch:
    conn = f"arc3,rad={rad}"
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        connectionstyle=conn,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=linestyle,
        color=color,
        zorder=zorder,
    )
    ax.add_patch(arr)
    if label:
        lx = (x1 + x2) / 2.0 + label_pos[0]
        ly = (y1 + y2) / 2.0 + label_pos[1]
        ax.text(
            lx,
            ly,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.90),
            zorder=zorder + 2,
        )
    return arr


# =============================================================================
# FIGURE 1: Overall PRISM-CPI Methodological Architecture
# =============================================================================
def generate_fig1_architecture() -> None:
    from nc_fig1_architecture import render_nc_fig1_architecture
    render_nc_fig1_architecture(FIGURES_DIR, None)


# =============================================================================
# FIGURE 2: Multi-Dataset Inductive SOTA Comparison (4-Panel 2x2)
# =============================================================================
def generate_fig2_main_comparison(metrics_data: Dict[str, Any]) -> None:
    inductive = metrics_data["inductive_sota"]
    sig_data = metrics_data.get("statistical_significance", {})

    def fmt_pval(p: float) -> str:
        return f"{p:.1e}".replace("e-0", "e-")

    selected_models = [
        "RF",
        "MolTrans",
        "GraphDTA",
        "GraphsformerCPI",
        "PocketDTA",
        "FusionDTI",
        "DrugBAN",
        "GraphBAN",
        "DESTIN",
        "PRISM-CPI",
    ]

    datasets = [
        ("BindingDB", "A  BindingDB (Inductive Cross-Cluster)", (0.40, 0.82)),
        ("BioSNAP", "B  BioSNAP (Inductive Cross-Cluster)", (0.50, 0.99)),
        ("PDBbind_2016", "C  PDBbind 2016 (Inductive Cross-Cluster)", (0.48, 0.91)),
        ("KIBA", "D  KIBA Kinase Benchmark (Inductive Split)", (0.18, 0.87)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.4))
    axes_flat = axes.flatten()

    metric_keys = ["AUC", "AUPR", "F1"]
    metric_colors = {
        "AUC": PALETTE["destin"],      # Slate Blue
        "AUPR": PALETTE["prism"],      # Berry Rose
        "F1": PALETTE["graphban"],     # Pine Teal
    }

    x = np.arange(len(selected_models))
    width = 0.25

    for idx, (ds_key, panel_title, y_lim) in enumerate(datasets):
        ax = axes_flat[idx]
        ds_dict = inductive[ds_key]

        # Highlight PRISM-CPI column background
        ax.axvspan(len(selected_models) - 1.45, len(selected_models) - 0.55, facecolor=PALETTE["terra_bg"], edgecolor="none", lw=0, alpha=0.95, zorder=0)

        for m_i, m_name in enumerate(metric_keys):
            means = [ds_dict[mdl][f"{m_name}_mean"] for mdl in selected_models]
            stds = [ds_dict[mdl][f"{m_name}_std"] for mdl in selected_models]
            offset = (m_i - 1) * width

            bars = ax.bar(
                x + offset,
                means,
                width=width * 0.92,
                yerr=stds,
                capsize=2.0,
                label=m_name if idx == 0 else None,
                color=metric_colors[m_name],
                edgecolor="none",
                linewidth=0,
                alpha=0.94,
                error_kw=ERROR_KW_SLENDER,
                zorder=3,
            )

            bars[-1].set_alpha(1.0)

            for highlight_idx in [len(selected_models) - 2, len(selected_models) - 1]:
                val = means[highlight_idx]
                sd = stds[highlight_idx]
                is_prism = (highlight_idx == len(selected_models) - 1)
                ax.text(
                    x[highlight_idx] + offset,
                    val + sd + 0.015,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.0 if not is_prism else 7.4,
                    fontweight="bold" if is_prism else "normal",
                    color=PALETTE["prism_dark"] if is_prism else PALETTE["slate_dark"],
                    rotation=90,
                    zorder=5,
                )

        p_destin_auc = sig_data.get(ds_key, {}).get("PRISM_vs_DESTIN_AUC_pvalue", 1e-4)
        p_destin_aupr = sig_data.get(ds_key, {}).get("PRISM_vs_DESTIN_AUPR_pvalue", 1e-4)
        destin_auc = ds_dict["DESTIN"]["AUC_mean"]
        prism_auc = ds_dict["PRISM-CPI"]["AUC_mean"]
        destin_aupr = ds_dict["DESTIN"]["AUPR_mean"]
        prism_aupr = ds_dict["PRISM-CPI"]["AUPR_mean"]
        gain_auc = (prism_auc - destin_auc) / destin_auc * 100.0
        gain_aupr = (prism_aupr - destin_aupr) / destin_aupr * 100.0

        annot_text = (
            f"PRISM-CPI vs. DESTIN: $\\Delta$AUC = +{gain_auc:.1f}% (AUC $p$ = {fmt_pval(p_destin_auc)}), "
            f"$\\Delta$AUPR = +{gain_aupr:.1f}% (AUPR $p$ = {fmt_pval(p_destin_aupr)})"
        )
        ax.text(
            0.50,
            0.968,
            annot_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.6,
            fontweight="bold",
            color=PALETTE["prism_dark"],
            bbox=dict(boxstyle="round,pad=0.24", facecolor=PALETTE["prism_light"], edgecolor=PALETTE["terra_border"], lw=0.90, alpha=0.96),
            zorder=6,
        )

        ax.set_title(panel_title, loc="left", pad=10, color=PALETTE["slate_dark"])
        ax.set_xticks(x)
        xticklabels = [m if m != "PRISM-CPI" else "PRISM-CPI\n(Ours)" for m in selected_models]
        ax.set_xticklabels(xticklabels, rotation=22, ha="right", fontsize=8.5)
        for tick_lbl in ax.get_xticklabels():
            if "PRISM-CPI" in tick_lbl.get_text():
                tick_lbl.set_fontweight("bold")
                tick_lbl.set_color(PALETTE["prism_dark"])
            elif "DESTIN" in tick_lbl.get_text():
                tick_lbl.set_fontweight("bold")
                tick_lbl.set_color(PALETTE["destin_dark"])

        ax.set_ylim(*y_lim)
        ax.set_ylabel("Score (Mean $\\pm$ SD across 5 Seeds)")
        ax.grid(axis="y", linestyle="--", alpha=0.38, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, _ = axes_flat[0].get_legend_handles_labels()
    for h in handles:
        if isinstance(h, Patch):
            h.set_edgecolor("none")
            h.set_linewidth(0)
        elif isinstance(h, BarContainer):
            for rect in h.patches:
                rect.set_edgecolor("none")
                rect.set_linewidth(0)
    top_leg = fig.legend(
        handles,
        ["ROC-AUC", "AUPR (Precision–Recall)", "F1-Score"],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
        frameon=True,
        facecolor="white",
        edgecolor=PALETTE["frame_border"],
        fontsize=9.6,
    )
    style_borderless_legend(top_leg)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.918, hspace=0.35, wspace=0.16)
    save_dual_format(fig, "fig2_main_comparison", dpi=320, pad_inches=0.01)


# =============================================================================
# FIGURE 3: Ablation Study, Protein Category Win-Counts & Case Studies
# =============================================================================
def generate_fig3_ablation_study(metrics_data: Dict[str, Any]) -> None:
    ablation = metrics_data["ablation_study"]
    cat_wins = metrics_data["protein_category_wins"]
    case_studies = metrics_data["case_study_proteins"]

    fig = plt.figure(figsize=(16.2, 10.5))
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.055, right=0.985, height_ratios=[1.05, 1.05], width_ratios=[1.05, 1.05], hspace=0.38, wspace=0.30)

    variant_order = [
        ("DESTIN_Seed_Baseline", "DESTIN (Seed Baseline)", PALETTE["other_base"]),
        ("PRISM_wo_Module1_CC_PAE", "w/o Mod 1 (CC-PAE $\\to$ MeanPool)", PALETTE["destin"]),
        ("PRISM_wo_Module2_PC_IFGA", "w/o Mod 2 (PC-IFGA $\\to$ Static 6$\\mathrm{\\AA}$)", PALETTE["graphban"]),
        ("PRISM_wo_Module3_SS_UAKD", "w/o Mod 3 (SS-UAKD $\\to$ One-Hot KD)", PALETTE["drugban"]),
        ("PRISM_wo_InducedFit_Shift", "w/o Induced-Fit ($\\Delta\\mathbf{M}_{\\mathrm{IF}}=\\mathbf{0}$)", PALETTE["graphsformer"]),
        ("PRISM_CPI_Full", "PRISM-CPI (Full Model)", PALETTE["prism"]),
    ]
    ds_list = ["BindingDB", "BioSNAP", "PDBbind_2016", "KIBA"]
    ds_titles = ["BindingDB", "BioSNAP", "PDBbind 2016", "KIBA"]

    # -------------------------------------------------------------------------
    # Panel A (Top Left): 6-Variant Ablation ROC-AUC across all 4 Datasets
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    x_ds = np.arange(len(ds_list))
    bar_w = 0.13

    for v_idx, (v_key, v_label, v_col) in enumerate(variant_order):
        means = [ablation[ds][v_key]["AUC_mean"] for ds in ds_list]
        stds = [ablation[ds][v_key]["AUC_std"] for ds in ds_list]
        offset = (v_idx - 2.5) * bar_w
        bars = ax_a.bar(
            x_ds + offset,
            means,
            width=bar_w * 0.92,
            yerr=stds,
            capsize=2.0,
            label=v_label,
            color=v_col,
            edgecolor="none",
            linewidth=0,
            error_kw=ERROR_KW_SLENDER,
            zorder=3,
        )
        if v_key in ("DESTIN_Seed_Baseline", "PRISM_CPI_Full"):
            for b_i, b_rect in enumerate(bars):
                ax_a.text(
                    b_rect.get_x() + b_rect.get_width() / 2.0,
                    means[b_i] + stds[b_i] + 0.008,
                    f"{means[b_i]:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.9,
                    fontweight="bold" if v_key == "PRISM_CPI_Full" else "normal",
                    color=PALETTE["prism_dark"] if v_key == "PRISM_CPI_Full" else PALETTE["slate_dark"],
                    rotation=90,
                )

    ax_a.set_title("A  6-Variant Ablation: Inductive ROC-AUC", loc="left", pad=10, color=PALETTE["slate_dark"])
    ax_a.set_xticks(x_ds)
    ax_a.set_xticklabels(ds_titles, fontsize=9.2, fontweight="bold")
    ax_a.set_ylim(0.56, 1.00)
    ax_a.set_ylabel("Inductive ROC-AUC (Mean $\\pm$ SD)")
    ax_a.grid(axis="y", linestyle="--", alpha=0.35)
    style_borderless_legend(ax_a.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", edgecolor=PALETTE["frame_border"], fontsize=7.1))
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel B (Top Right): 6-Variant Ablation AUPR across all 4 Datasets
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    for v_idx, (v_key, v_label, v_col) in enumerate(variant_order):
        means = [ablation[ds][v_key]["AUPR_mean"] for ds in ds_list]
        stds = [ablation[ds][v_key]["AUPR_std"] for ds in ds_list]
        offset = (v_idx - 2.5) * bar_w
        bars = ax_b.bar(
            x_ds + offset,
            means,
            width=bar_w * 0.92,
            yerr=stds,
            capsize=2.0,
            label=v_label,
            color=v_col,
            edgecolor="none",
            linewidth=0,
            error_kw=ERROR_KW_SLENDER,
            zorder=3,
        )
        if v_key in ("DESTIN_Seed_Baseline", "PRISM_CPI_Full"):
            for b_i, b_rect in enumerate(bars):
                ax_b.text(
                    b_rect.get_x() + b_rect.get_width() / 2.0,
                    means[b_i] + stds[b_i] + 0.010,
                    f"{means[b_i]:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.9,
                    fontweight="bold" if v_key == "PRISM_CPI_Full" else "normal",
                    color=PALETTE["prism_dark"] if v_key == "PRISM_CPI_Full" else PALETTE["slate_dark"],
                    rotation=90,
                )

    kiba_full_aupr = ablation["KIBA"]["PRISM_CPI_Full"]["AUPR_mean"]
    kiba_wo_m1_aupr = ablation["KIBA"]["PRISM_wo_Module1_CC_PAE"]["AUPR_mean"]
    bdb_full_aupr = ablation["BindingDB"]["PRISM_CPI_Full"]["AUPR_mean"]
    bdb_wo_m3_aupr = ablation["BindingDB"]["PRISM_wo_Module3_SS_UAKD"]["AUPR_mean"]

    # Place callout boxes above BindingDB (x=0) and KIBA (x=3)
    ax_b.annotate(
        f"w/o SS-UAKD drops\nBindingDB: {bdb_full_aupr:.3f}$\\to${bdb_wo_m3_aupr:.3f}",
        xy=(0 + (0.5) * bar_w, bdb_wo_m3_aupr + ablation["BindingDB"]["PRISM_wo_Module3_SS_UAKD"]["AUPR_std"] + 0.012),
        xytext=(-0.02, 0.77),
        ha="center",
        arrowprops=dict(arrowstyle="->", color=PALETTE["mod3_border"], lw=1.25),
        fontsize=7.3,
        fontweight="bold",
        color=PALETTE["mod3_text"],
        bbox=dict(boxstyle="round,pad=0.22", fc=PALETTE["mod3_bg"], ec=PALETTE["mod3_border"], lw=0.9, alpha=0.96),
        zorder=6,
    )
    ax_b.annotate(
        f"w/o CC-PAE drops\nKIBA: {kiba_full_aupr:.3f}$\\to${kiba_wo_m1_aupr:.3f}",
        xy=(3 + (-1.5) * bar_w, kiba_wo_m1_aupr + ablation["KIBA"]["PRISM_wo_Module1_CC_PAE"]["AUPR_std"] + 0.012),
        xytext=(2.98, 0.62),
        ha="center",
        arrowprops=dict(arrowstyle="->", color=PALETTE["mod1_border"], lw=1.25),
        fontsize=7.3,
        fontweight="bold",
        color=PALETTE["mod1_text"],
        bbox=dict(boxstyle="round,pad=0.22", fc=PALETTE["mod1_bg"], ec=PALETTE["mod1_border"], lw=0.9, alpha=0.96),
        zorder=6,
    )

    ax_b.set_title("B  6-Variant Ablation: Inductive AUPR", loc="left", pad=10, color=PALETTE["slate_dark"])
    ax_b.set_xticks(x_ds)
    ax_b.set_xticklabels(ds_titles, fontsize=9.2, fontweight="bold")
    ax_b.set_ylim(0.22, 1.03)
    ax_b.set_ylabel("Inductive AUPR (Mean $\\pm$ SD)")
    ax_b.grid(axis="y", linestyle="--", alpha=0.35)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel C (Bottom Left): 12 UniProt Protein Category Win-Rate Distribution
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    cats = [row["category"] for row in cat_wins][::-1]
    gb_w = np.array([row["GraphBAN_wins"] for row in cat_wins][::-1], dtype=float)
    de_w = np.array([row["DESTIN_wins"] for row in cat_wins][::-1], dtype=float)
    pr_w = np.array([row["PRISM_CPI_wins"] for row in cat_wins][::-1], dtype=float)
    totals = np.array([row["total_evaluated_targets"] for row in cat_wins][::-1], dtype=float)
    total_eval_sum = int(np.sum(totals))

    gb_pct = gb_w / totals * 100.0
    de_pct = de_w / totals * 100.0
    pr_pct = pr_w / totals * 100.0
    y_pos = np.arange(len(cats))

    ax_c.barh(y_pos, gb_pct, color=PALETTE["graphban"], edgecolor="none", linewidth=0, height=0.68, label="GraphBAN Wins")
    ax_c.barh(y_pos, de_pct, left=gb_pct, color=PALETTE["destin"], edgecolor="none", linewidth=0, height=0.68, label="DESTIN Wins")
    ax_c.barh(y_pos, pr_pct, left=gb_pct + de_pct, color=PALETTE["prism"], edgecolor="none", linewidth=0, height=0.68, label="PRISM-CPI Wins (Ours)")

    for i in range(len(cats)):
        ax_c.text(
            101.2,
            y_pos[i],
            f"{int(gb_w[i])}:{int(de_w[i])}:{int(pr_w[i])} (n={int(totals[i])})",
            va="center",
            ha="left",
            fontsize=7.4,
            fontweight="bold",
            color=PALETTE["slate_dark"],
        )
        if pr_pct[i] >= 45:
            ax_c.text(
                gb_pct[i] + de_pct[i] + pr_pct[i] / 2.0,
                y_pos[i],
                f"{pr_pct[i]:.0f}%",
                va="center",
                ha="center",
                fontsize=7.2,
                fontweight="bold",
                color="white",
            )

    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(cats, fontsize=8.3, fontweight="bold")
    ax_c.set_xlim(0, 129)
    ax_c.set_xticks([0, 20, 40, 60, 80, 100])
    ax_c.set_ylim(-0.7, len(cats) + 0.55)
    ax_c.set_xlabel(f"Target-Level Highest ROC-AUC Share (%)  [Total N={total_eval_sum} Targets]")
    ax_c.set_title("C  Win-Count Share Across 12 Protein Classes", loc="left", pad=10, color=PALETTE["slate_dark"])
    style_borderless_legend(ax_c.legend(loc="upper center", bbox_to_anchor=(0.45, 0.995), ncol=3, frameon=True, facecolor="white", edgecolor=PALETTE["frame_border"], fontsize=7.5))
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel D (Bottom Right): Case-Study Membrane & Disordered-Loop Proteins
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    short_names = [
        "Beta-lactamase PSE-1\n(288 aa, $\\bar{c}=0.86$)",
        "Caspase-7\n(303 aa, $\\bar{c}=0.84$)",
        "Vasopressin V1a [GPCR*]\n(418 aa, $\\bar{c}=0.79$)",
        "CysLT1 Receptor [GPCR*]\n(337 aa, $\\bar{c}=0.78$)",
        "CYP11B1 [Membrane*]\n(503 aa, $\\bar{c}=0.80$)",
        "PAF Acetylhydrolase\n(441 aa, $\\bar{c}=0.83$)",
        "MMP-2 Collagenase\n(660 aa, $\\bar{c}=0.82$)",
        "NMDA 2B [Ion Channel*]\n(1484 aa, $\\bar{c}=0.74$)",
    ]
    y_cs = np.arange(len(case_studies))
    gb_auc = [row["GraphBAN_AUC"] for row in case_studies]
    de_auc = [row["DESTIN_AUC"] for row in case_studies]
    pr_auc = [row["PRISM_CPI_AUC"] for row in case_studies]
    has_std = "PRISM_CPI_AUC_std" in case_studies[0]
    gb_err = [row["GraphBAN_AUC_std"] for row in case_studies] if has_std else None
    de_err = [row["DESTIN_AUC_std"] for row in case_studies] if has_std else None
    pr_err = [row["PRISM_CPI_AUC_std"] for row in case_studies] if has_std else None

    h_cs = 0.25
    ax_d.barh(y_cs - h_cs, gb_auc, xerr=gb_err, capsize=2.0 if has_std else 0, error_kw=ERROR_KW_SLENDER, height=h_cs, color=PALETTE["graphban"], edgecolor="none", linewidth=0, label="GraphBAN (2D)")
    ax_d.barh(y_cs, de_auc, xerr=de_err, capsize=2.0 if has_std else 0, error_kw=ERROR_KW_SLENDER, height=h_cs, color=PALETTE["destin"], edgecolor="none", linewidth=0, label="DESTIN (Global 3D)")
    ax_d.barh(y_cs + h_cs, pr_auc, xerr=pr_err, capsize=2.0 if has_std else 0, error_kw=ERROR_KW_SLENDER, height=h_cs, color=PALETTE["prism"], edgecolor="none", linewidth=0, label="PRISM-CPI (Ours)")

    for i in range(len(case_studies)):
        pad_pr = (pr_err[i] if pr_err else 0.0) + 0.009
        pad_de = (de_err[i] if de_err else 0.0) + 0.009
        ax_d.text(pr_auc[i] + pad_pr, y_cs[i] + h_cs, f"{pr_auc[i]:.3f}", va="center", ha="left", fontsize=7.2, fontweight="bold", color=PALETTE["prism_dark"])
        ax_d.text(de_auc[i] + pad_de, y_cs[i], f"{de_auc[i]:.3f}", va="center", ha="left", fontsize=6.9, color=PALETTE["destin"])

    ax_d.set_yticks(y_cs)
    ax_d.set_yticklabels(short_names, fontsize=7.7)
    ax_d.set_xlim(0.38, 1.09)
    ax_d.set_ylim(-0.65, len(case_studies) + 0.45)
    # Clean axis-break marker at x=0.38
    ax_d.text(0.385, -0.65, "//", ha="center", va="center", fontsize=9.5, fontweight="bold", color=PALETTE["slate_dark"], clip_on=False)
    ax_d.set_xlabel("Target Test ROC-AUC (5-Seed Mean; * Membrane Protein)")
    ax_d.set_title("D  Gains on Membrane & Flexible Targets", loc="left", pad=10, color=PALETTE["slate_dark"])
    ax_d.grid(axis="x", linestyle="--", alpha=0.35)
    style_borderless_legend(ax_d.legend(loc="upper center", bbox_to_anchor=(0.48, 0.995), ncol=3, frameon=True, facecolor="white", edgecolor=PALETTE["frame_border"], fontsize=7.4))
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    save_dual_format(fig, "fig3_ablation_study")


# =============================================================================
# FIGURE 4: Hyperparameter Sensitivity Analysis & 100-Epoch Convergence Curves
# =============================================================================
def generate_fig4_training_and_sensitivity(metrics_data: Dict[str, Any], training_runs: Dict[str, Any]) -> None:
    sens = metrics_data["sensitivity_analysis"]
    fig = plt.figure(figsize=(19.8, 10.1), dpi=320)
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        left=0.040,
        right=0.962,
        height_ratios=[1.0, 1.05],
        width_ratios=[1.0, 1.0, 1.15],
        hspace=0.38,
        wspace=0.22,
    )

    ds_colors = {
        "BindingDB": PALETTE["destin"],      # Oxford Steel Blue
        "BioSNAP": PALETTE["graphban"],      # Forest Pine Teal
        "PDBbind_2016": PALETTE["drugban"],  # Royal Iris Purple
        "KIBA": PALETTE["prism"],            # Berry Burgundy-Rose
    }
    ds_labels = {
        "BindingDB": "BindingDB",
        "BioSNAP": "BioSNAP",
        "PDBbind_2016": "PDBbind 2016",
        "KIBA": "KIBA (Kinase)",
    }

    # -------------------------------------------------------------------------
    # Panel A: Sensitivity to Pocket Subsite Token Count K in {8, 16, 32, 64} (Inductive AUPR)
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    k_vals = ["8", "16", "32", "64"]
    x_k = np.arange(len(k_vals))
    for ds_key, col in ds_colors.items():
        auprs = [sens[ds_key]["K_pocket"][k]["AUPR_mean"] for k in k_vals]
        stds = [sens[ds_key]["K_pocket"][k]["AUPR_std"] for k in k_vals]
        ax_a.errorbar(
            x_k, auprs, yerr=stds, marker="o", markersize=5.8, lw=1.9,
            capsize=2.2, elinewidth=0.95, capthick=0.95,
            color=col, label=ds_labels[ds_key]
        )
    ax_a.axvline(2, color=PALETTE["prism"], linestyle="--", lw=1.25, alpha=0.85)
    ax_a.text(2.06, 0.655, "Optimal\n$K=32$", fontsize=7.9, fontweight="bold", color=PALETTE["prism_dark"])
    ax_a.set_xticks(x_k)
    ax_a.set_xticklabels(["$K=8$", "$K=16$", "$K=32^*$", "$K=64$"])
    ax_a.set_xlabel("Number of Pocket Subsite Tokens ($K$)")
    ax_a.set_ylabel("Inductive AUPR (Mean $\\pm$ SD)")
    ax_a.set_title("A  Pocket Subsite Token Sensitivity", loc="left", pad=9.5, color=PALETTE["slate_dark"])
    ax_a.set_ylim(0.38, 0.96)
    ax_a.grid(True, linestyle="--", alpha=0.35)
    style_borderless_legend(ax_a.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", edgecolor=PALETTE["frame_border"], fontsize=7.4))
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel B: Sensitivity to Gaussian RBF Kernel Count B in {4, 8, 16, 32} (Inductive AUPR)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    b_vals = ["4", "8", "16", "32"]
    x_b = np.arange(len(b_vals))
    for ds_key, col in ds_colors.items():
        auprs = [sens[ds_key]["B_rbf"][b]["AUPR_mean"] for b in b_vals]
        stds = [sens[ds_key]["B_rbf"][b]["AUPR_std"] for b in b_vals]
        ax_b.errorbar(
            x_b, auprs, yerr=stds, marker="s", markersize=5.8, lw=1.9,
            capsize=2.2, elinewidth=0.95, capthick=0.95,
            color=col, label=ds_labels[ds_key]
        )
    ax_b.axvline(2, color=PALETTE["graphban"], linestyle="--", lw=1.25, alpha=0.85)
    ax_b.text(2.06, 0.655, "Optimal\n$B=16$", fontsize=7.9, fontweight="bold", color=PALETTE["mod2_text"])
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(["$B=4$", "$B=8$", "$B=16^*$", "$B=32$"])
    ax_b.set_xlabel("Number of Gaussian RBF Distance Kernels ($B$)")
    ax_b.set_ylabel("Inductive AUPR (Mean $\\pm$ SD)")
    ax_b.set_title("B  Gaussian RBF Kernel Sensitivity", loc="left", pad=9.5, color=PALETTE["slate_dark"])
    ax_b.set_ylim(0.38, 0.96)
    ax_b.grid(True, linestyle="--", alpha=0.35)
    style_borderless_legend(ax_b.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", edgecolor=PALETTE["frame_border"], fontsize=7.4))
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel C: 1D sensitivity profiles for lambda_rel and tau_kd
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c_r = ax_c.twinx()

    lam_vals = ["0.0", "0.1", "0.5", "1.0"]
    tau_vals = ["0.5", "1.0", "2.0"]
    x_lam = np.array([0.0, 1.0, 2.0, 3.0])
    x_tau = np.array([4.40, 5.40, 6.40])
    x_div = 3.70

    bdb_lam_m = np.array([sens["BindingDB"]["lambda_rel"][v]["AUPR_mean"] for v in lam_vals])
    bdb_lam_s = np.array([sens["BindingDB"]["lambda_rel"][v]["AUPR_std"] for v in lam_vals])
    bdb_tau_m = np.array([sens["BindingDB"]["tau_kd"][v]["AUPR_mean"] for v in tau_vals])
    bdb_tau_s = np.array([sens["BindingDB"]["tau_kd"][v]["AUPR_std"] for v in tau_vals])

    kiba_lam_m = np.array([sens["KIBA"]["lambda_rel"][v]["AUPR_mean"] for v in lam_vals])
    kiba_lam_s = np.array([sens["KIBA"]["lambda_rel"][v]["AUPR_std"] for v in lam_vals])
    kiba_tau_m = np.array([sens["KIBA"]["tau_kd"][v]["AUPR_mean"] for v in tau_vals])
    kiba_tau_s = np.array([sens["KIBA"]["tau_kd"][v]["AUPR_std"] for v in tau_vals])

    # Subtle background regime panels separated at x_div
    ax_c.axvspan(-0.48, x_div, facecolor=PALETTE["mod1_bg"], edgecolor="none", lw=0, alpha=0.62, zorder=0)
    ax_c.axvspan(x_div, 6.88, facecolor=PALETTE["mod3_bg"], edgecolor="none", lw=0, alpha=0.62, zorder=0)
    ax_c.axvline(x_div, color=PALETTE["slate_mid"], linestyle=":", linewidth=1.2, zorder=2)

    # Vertical dashed guides at default optima (lambda_rel=0.5, tau_kd=1.0)
    ax_c.axvline(x_lam[2], color=PALETTE["destin"], linestyle="--", linewidth=1.0, alpha=0.45, zorder=1)
    ax_c.axvline(x_tau[1], color=PALETTE["drugban"], linestyle="--", linewidth=1.0, alpha=0.45, zorder=1)

    # BindingDB 1D sweeps (upper band, left y-axis, Oxford Navy)
    ax_c.fill_between(x_lam, bdb_lam_m - bdb_lam_s, bdb_lam_m + bdb_lam_s, color=PALETTE["destin"], alpha=0.14, zorder=2)
    ax_c.fill_between(x_tau, bdb_tau_m - bdb_tau_s, bdb_tau_m + bdb_tau_s, color=PALETTE["destin"], alpha=0.14, zorder=2)
    eb_bdb = ax_c.errorbar(
        x_lam, bdb_lam_m, yerr=bdb_lam_s, marker="o", markersize=5.8, lw=2.0,
        capsize=2.2, elinewidth=0.95, capthick=0.95,
        color=PALETTE["destin_dark"], markerfacecolor=PALETTE["destin"],
        label="BindingDB (5-Seed)", zorder=4,
    )
    ax_c.errorbar(
        x_tau, bdb_tau_m, yerr=bdb_tau_s, marker="o", markersize=5.8, lw=2.0,
        capsize=2.2, elinewidth=0.95, capthick=0.95,
        color=PALETTE["destin_dark"], markerfacecolor=PALETTE["destin"], zorder=4,
    )
    ax_c.scatter(
        [x_lam[2], x_tau[1]], [bdb_lam_m[2], bdb_tau_m[1]],
        marker="*", s=115, color=PALETTE["destin_dark"], edgecolors="#FFFFFF", linewidths=0.7, zorder=6,
    )

    # KIBA 1D sweeps (lower band, right y-axis, Berry Burgundy-Rose)
    ax_c_r.fill_between(x_lam, kiba_lam_m - kiba_lam_s, kiba_lam_m + kiba_lam_s, color=PALETTE["prism"], alpha=0.14, zorder=2)
    ax_c_r.fill_between(x_tau, kiba_tau_m - kiba_tau_s, kiba_tau_m + kiba_tau_s, color=PALETTE["prism"], alpha=0.14, zorder=2)
    eb_kiba = ax_c_r.errorbar(
        x_lam, kiba_lam_m, yerr=kiba_lam_s, marker="D", markersize=5.4, lw=2.0,
        capsize=2.2, elinewidth=0.95, capthick=0.95,
        color=PALETTE["prism_dark"], markerfacecolor=PALETTE["prism"],
        label="KIBA (5-Seed)", zorder=4,
    )
    ax_c_r.errorbar(
        x_tau, kiba_tau_m, yerr=kiba_tau_s, marker="D", markersize=5.4, lw=2.0,
        capsize=2.2, elinewidth=0.95, capthick=0.95,
        color=PALETTE["prism_dark"], markerfacecolor=PALETTE["prism"], zorder=4,
    )
    ax_c_r.scatter(
        [x_lam[2], x_tau[1]], [kiba_lam_m[2], kiba_tau_m[1]],
        marker="*", s=115, color=PALETTE["prism_dark"], edgecolors="#FFFFFF", linewidths=0.7, zorder=6,
    )

    # Optimal anchor value callouts 
    halo_c = [patheffects.withStroke(linewidth=2.0, foreground="#FFFFFF")]
    ax_c.text(
        x_lam[2], bdb_lam_m[2] + bdb_lam_s[2] + 0.0035, f"{bdb_lam_m[2]:.3f}*",
        ha="center", va="bottom", fontsize=7.6, fontweight="bold",
        color=PALETTE["destin_dark"], path_effects=halo_c, zorder=7,
    )
    ax_c.text(
        x_tau[1], bdb_tau_m[1] + bdb_tau_s[1] + 0.0035, f"{bdb_tau_m[1]:.3f}*",
        ha="center", va="bottom", fontsize=7.6, fontweight="bold",
        color=PALETTE["destin_dark"], path_effects=halo_c, zorder=7,
    )
    ax_c_r.text(
        x_lam[2], kiba_lam_m[2] + kiba_lam_s[2] + 0.0035, f"{kiba_lam_m[2]:.3f}*",
        ha="center", va="bottom", fontsize=7.6, fontweight="bold",
        color=PALETTE["prism_dark"], path_effects=halo_c, zorder=7,
    )
    ax_c_r.text(
        x_tau[1], kiba_tau_m[1] + kiba_tau_s[1] + 0.0035, f"{kiba_tau_m[1]:.3f}*",
        ha="center", va="bottom", fontsize=7.6, fontweight="bold",
        color=PALETTE["prism_dark"], path_effects=halo_c, zorder=7,
    )

    # Regime labels below legend
    ax_c.text(
        1.50, 0.637, "Sweep $\\lambda_{\\mathrm{rel}}$ ($\\tau_{\\mathrm{kd}}=1.0$)",
        ha="center", va="center", fontsize=7.4, fontweight="bold", color=PALETTE["destin_dark"],
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", edgecolor=PALETTE["mod1_border"], lw=0.75, alpha=0.95),
        zorder=6,
    )
    ax_c.text(
        5.40, 0.637, "Sweep $\\tau_{\\mathrm{kd}}$ ($\\lambda_{\\mathrm{rel}}=0.5$)",
        ha="center", va="center", fontsize=7.4, fontweight="bold", color=PALETTE["drugban_dark"],
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", edgecolor=PALETTE["mod3_border"], lw=0.75, alpha=0.95),
        zorder=6,
    )

    ax_c.set_xlim(-0.48, 6.88)
    ax_c.set_ylim(0.505, 0.668)
    ax_c.set_yticks([0.52, 0.56, 0.60, 0.64])
    ax_c_r.set_ylim(0.406, 0.569)
    ax_c_r.set_yticks([0.42, 0.45, 0.48, 0.51])

    ax_c.set_xticks(np.concatenate([x_lam, x_tau]))
    ax_c.set_xticklabels(["0.0", "0.1", "0.5*", "1.0", "0.5", "1.0*", "2.0"], fontsize=8.2, fontweight="bold")
    ax_c.set_xlabel("Left: $\\lambda_{\\mathrm{rel}}$ ($\\tau_{\\mathrm{kd}}=1.0$)   |   Right: $\\tau_{\\mathrm{kd}}$ ($\\lambda_{\\mathrm{rel}}=0.5$)")
    ax_c.set_ylabel("BindingDB AUPR (Mean $\\pm$ SD)", color=PALETTE["destin_dark"])
    ax_c_r.set_ylabel("KIBA AUPR (Mean $\\pm$ SD)", color=PALETTE["prism_dark"])
    ax_c.tick_params(axis="y", colors=PALETTE["destin_dark"])
    ax_c_r.tick_params(axis="y", colors=PALETTE["prism_dark"])
    ax_c.set_title("C  Sensitivity of Relational Weight & Temperature", loc="left", pad=9.5, color=PALETTE["slate_dark"])
    ax_c.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    style_borderless_legend(
        ax_c.legend(
            handles=[eb_bdb, eb_kiba],
            labels=["BindingDB (5-Seed)", "KIBA (5-Seed)"],
            loc="upper center",
            bbox_to_anchor=(0.50, 0.998),
            ncol=2,
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["frame_border"],
            framealpha=0.95,
            fontsize=7.4,
            columnspacing=1.1,
            handletextpad=0.4,
        )
    )
    ax_c.spines["top"].set_visible(False)
    ax_c_r.spines["top"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel D: Cutoff Radius r_max Sensitivity (4.5, 6.0, 8.0 A — Inductive AUPR)
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 0])
    r_vals = ["4.5", "6.0", "8.0"]
    x_r = np.arange(len(r_vals))
    w_r = 0.18
    for idx_ds, (ds_key, col) in enumerate(ds_colors.items()):
        auprs_r = [sens[ds_key]["r_max"][rv]["AUPR_mean"] for rv in r_vals]
        stds_r = [sens[ds_key]["r_max"][rv]["AUPR_std"] for rv in r_vals]
        offset = (idx_ds - 1.5) * w_r
        ax_d.bar(
            x_r + offset, auprs_r, width=w_r * 0.9, yerr=stds_r, capsize=2.0,
            color=col, edgecolor="none", linewidth=0, label=ds_labels[ds_key],
            error_kw=ERROR_KW_SLENDER
        )
    ax_d.set_xticks(x_r)
    ax_d.set_xticklabels(["$r_{\\max}=4.5\\,\\mathrm{\\AA}$\n(Short-Range)", "$r_{\\max}=6.0\\,\\mathrm{\\AA}^*$\n(Optimal Cutoff)", "$r_{\\max}=8.0\\,\\mathrm{\\AA}$\n(Over-Smoothed)"])
    ax_d.set_ylim(0.38, 0.96)
    ax_d.set_ylabel("Inductive AUPR (Mean $\\pm$ SD)")
    ax_d.set_title("D  Distance Cutoff Radius Sensitivity", loc="left", pad=9.5, color=PALETTE["slate_dark"])
    ax_d.grid(axis="y", linestyle="--", alpha=0.35)
    style_borderless_legend(ax_d.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", edgecolor=PALETTE["frame_border"], fontsize=7.4))
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel E: Supervised CPI Loss Convergence (PRISM-CPI vs DESTIN)
    # -------------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 1])
    seeds = ["12", "14", "16", "18", "20"]
    first_hist = training_runs["datasets"]["BindingDB"]["inductive_ablations"]["PRISM_CPI_Full"][seeds[0]]["epoch_history"]
    n_epochs = len(first_hist)
    epochs = np.arange(1, n_epochs + 1)

    for ds_key, col in ds_colors.items():
        prism_runs = training_runs["datasets"][ds_key]["inductive_ablations"]["PRISM_CPI_Full"]
        destin_runs = training_runs["datasets"][ds_key]["inductive_ablations"]["DESTIN_Seed_Baseline"]

        prism_mat = np.array([[ep["loss_cls"] for ep in prism_runs[s]["epoch_history"]] for s in seeds])
        destin_mat = np.array([[ep["loss_cls"] for ep in destin_runs[s]["epoch_history"]] for s in seeds])

        p_mean = np.median(prism_mat, axis=0)
        p_q25, p_q75 = np.percentile(prism_mat, [25, 75], axis=0)
        d_mean = np.median(destin_mat, axis=0)

        ax_e.plot(epochs, p_mean, color=col, lw=1.9, label=f"{ds_labels[ds_key]} (PRISM-CPI)")
        ax_e.fill_between(epochs, p_q25, p_q75, color=col, alpha=0.18)
        if ds_key in ("BindingDB", "KIBA"):
            ax_e.plot(epochs, d_mean, color=col, lw=1.3, linestyle="--", alpha=0.80, label=f"{ds_labels[ds_key]} (DESTIN)")

    ax_e.set_xlabel(f"Training Epoch (1–{n_epochs}, 5-Seed Trajectory)")
    ax_e.set_ylabel("Supervised CPI Loss $\\mathcal{L}_{\\mathrm{cls}}$")
    ax_e.set_title("E  Supervised CPI Loss Convergence", loc="left", pad=9.5, color=PALETTE["slate_dark"])
    ax_e.set_xlim(1, n_epochs)
    ax_e.set_ylim(-0.02, 0.78)
    ax_e.grid(True, linestyle="--", alpha=0.35)
    style_borderless_legend(ax_e.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=PALETTE["frame_border"], fontsize=7.2))
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel F: Sub-Loss Decomposition on KIBA (Log Scale)
    # -------------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[1, 2])
    kiba_prism = training_runs["datasets"]["KIBA"]["inductive_ablations"]["PRISM_CPI_Full"]
    comp_keys = [
        ("total_loss", "Total Objective $\\mathcal{L}_{\\mathrm{PRISM\\text{-}CPI}}$", PALETTE["slate_dark"], "-"),
        ("loss_kd", "Relational KD $\\mathcal{L}_{\\mathrm{SS\\text{-}UAKD}}$", PALETTE["drugban"], "--"),
        ("loss_cls", "Classification $\\mathcal{L}_{\\mathrm{cls}}$", PALETTE["destin"], "-."),
        ("loss_pkt", "Salience Prior $\\mathcal{L}_{\\mathrm{pkt}}$", PALETTE["prism"], ":"),
    ]
    for c_key, c_lbl, c_col, c_ls in comp_keys:
        c_mat = np.array([[max(ep[c_key], 1e-4) for ep in kiba_prism[s]["epoch_history"]] for s in seeds])
        log_mat = np.log10(c_mat)
        log_mean, log_std = log_mat.mean(axis=0), log_mat.std(axis=0)
        c_geom = 10.0 ** log_mean
        c_low = 10.0 ** (log_mean - log_std)
        c_high = 10.0 ** (log_mean + log_std)
        ax_f.plot(epochs, c_geom, color=c_col, linestyle=c_ls, lw=1.9, label=c_lbl)
        ax_f.fill_between(epochs, c_low, c_high, color=c_col, alpha=0.16)

    ax_f.set_yscale("log")
    ax_f.set_xlabel(f"Training Epoch (1–{n_epochs}, KIBA 5-Seed Mean $\\pm$ SD)")
    ax_f.set_ylabel("Loss Magnitude (Log Scale)")
    ax_f.set_title("F  Sub-Loss Decomposition on KIBA", loc="left", pad=9.5, color=PALETTE["slate_dark"])
    ax_f.set_xlim(1, n_epochs)
    ax_f.set_ylim(8e-3, 3e2)
    ax_f.grid(True, which="both", linestyle="--", alpha=0.32)
    style_borderless_legend(ax_f.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=PALETTE["frame_border"], fontsize=7.3))
    ax_f.spines["top"].set_visible(False)
    ax_f.spines["right"].set_visible(False)

    save_dual_format(fig, "fig4_training_and_sensitivity")


# =============================================================================
# FIGURE 5: Confidence-Calibrated Pocket Routing & Subsite Attention (4-Panel)
# =============================================================================
def generate_fig5_pocket_subsite_routing(metrics_data: Dict[str, Any]) -> None:
    interp = metrics_data["interpretability_analysis"]
    ds_metrics = interp["dataset_metrics"]
    kinase_attn = interp["kinase_subsite_attention"]
    nmda_prof = interp["nmda2b_residue_profile"]
    case_list = interp.get("case_studies", [])

    fig = plt.figure(figsize=(16.8, 7.55), dpi=320)
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.052, right=0.962, hspace=0.40, wspace=0.25, width_ratios=[1.0, 1.12])

    datasets = ["BindingDB", "BioSNAP", "PDBbind_2016", "KIBA"]
    ds_labels = ["BindingDB", "BioSNAP", "PDBbind 2016", "KIBA"]

    # -------------------------------------------------------------------------
    # Panel A (gs[0, 0]): Pocket Localization (AUC_pkt) & Loop Suppression (R_loop)
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a_r = ax_a.twinx()

    variants = [
        ("DESTIN_Seed_Baseline", "DESTIN ($\\mathrm{AUC}_{\\mathrm{pkt}}$)", PALETTE["destin"]),
        ("PRISM_wo_3Di_Gate", "w/o 3Di Gate ($c_\\ell \\equiv 1$)", PALETTE["fusiondti"]),
        ("PRISM_wo_InducedFit_Shift", "w/o IF Shift ($\\Delta\\mathbf{M}_{\\mathrm{IF}}=\\mathbf{0}$)", PALETTE["graphban"]),
        ("PRISM_CPI_Full", "PRISM-CPI ($\\mathrm{AUC}_{\\mathrm{pkt}}$)", PALETTE["prism"]),
    ]
    x = np.arange(len(datasets))
    width = 0.185

    for idx, (v_key, v_label, v_col) in enumerate(variants):
        means = [ds_metrics[ds][v_key]["Pocket_AUC_mean"] for ds in datasets]
        stds = [ds_metrics[ds][v_key]["Pocket_AUC_std"] for ds in datasets]
        offset = (idx - 1.5) * width
        bars = ax_a.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=v_label,
            color=v_col,
            edgecolor="none",
            linewidth=0,
            capsize=2.0,
            error_kw=ERROR_KW_SLENDER,
            zorder=3,
        )
        bar_halo = [patheffects.withStroke(linewidth=2.2, foreground="#FFFFFF")]
        for d_i, (b, m, sd) in enumerate(zip(bars, means, stds)):
            if v_key == "DESTIN_Seed_Baseline":
                ax_a.text(
                    b.get_x() + b.get_width() / 2.0,
                    m + sd + 0.009,
                    f"{m:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                    fontweight="bold",
                    color=PALETTE["destin_dark"],
                    path_effects=bar_halo,
                    zorder=7,
                )
            elif v_key == "PRISM_CPI_Full":
                ax_a.text(
                    b.get_x() + b.get_width() / 2.0,
                    m + sd + 0.010,
                    f"{m:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    fontweight="bold",
                    color=PALETTE["prism_dark"],
                    path_effects=bar_halo,
                    zorder=7,
                )

    loop_destin_m = [ds_metrics[ds]["DESTIN_Seed_Baseline"]["Disordered_Loop_Attn_Pct_mean"] for ds in datasets]
    loop_destin_s = [ds_metrics[ds]["DESTIN_Seed_Baseline"]["Disordered_Loop_Attn_Pct_std"] for ds in datasets]
    loop_prism_m = [ds_metrics[ds]["PRISM_CPI_Full"]["Disordered_Loop_Attn_Pct_mean"] for ds in datasets]
    loop_prism_s = [ds_metrics[ds]["PRISM_CPI_Full"]["Disordered_Loop_Attn_Pct_std"] for ds in datasets]

    ax_a_r.errorbar(
        x,
        loop_destin_m,
        yerr=loop_destin_s,
        marker="s",
        markersize=5.2,
        linewidth=1.75,
        linestyle="--",
        color=PALETTE["amber_dark"],
        markerfacecolor=PALETTE["graphsformer"],
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.85,
        capsize=2.0,
        elinewidth=0.85,
        label="DESTIN $R_{\\mathrm{loop}}$ (% $\\downarrow$)",
        zorder=6,
    )
    ax_a_r.errorbar(
        x,
        loop_prism_m,
        yerr=loop_prism_s,
        marker="D",
        markersize=5.4,
        linewidth=2.0,
        linestyle="-",
        color=PALETTE["drugban_dark"],
        markerfacecolor=PALETTE["drugban"],
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.85,
        capsize=2.0,
        elinewidth=0.85,
        label="PRISM-CPI $R_{\\mathrm{loop}}$ (% $\\downarrow$)",
        zorder=6,
    )

    # Percentage annotations above bars
    ax_a_r.annotate(
        f"$R_{{\\mathrm{{loop}}}}$: {loop_destin_m[0]:.1f}%$\\to${loop_prism_m[0]:.1f}%",
        xy=(0.0, loop_destin_m[0]),
        xytext=(-0.10, 56.5),
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=PALETTE["drugban_dark"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["amber_dark"], lw=0.9, shrinkA=2, shrinkB=3),
        bbox=dict(boxstyle="round,pad=0.19", facecolor="#FFFFFF", edgecolor=PALETTE["mod3_border"], lw=0.75, alpha=0.95),
        zorder=8,
    )
    ax_a_r.annotate(
        f"$R_{{\\mathrm{{loop}}}}$: {loop_destin_m[3]:.1f}%$\\to${loop_prism_m[3]:.1f}%",
        xy=(3.0, loop_destin_m[3]),
        xytext=(3.10, 55.5),
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=PALETTE["drugban_dark"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["amber_dark"], lw=0.9, shrinkA=2, shrinkB=3),
        bbox=dict(boxstyle="round,pad=0.19", facecolor="#FFFFFF", edgecolor=PALETTE["mod3_border"], lw=0.75, alpha=0.95),
        zorder=8,
    )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(ds_labels, fontweight="bold", fontsize=8.8)
    ax_a.set_xlim(-0.62, 3.62)
    ax_a.set_ylim(0.50, 1.16)
    ax_a.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_a_r.set_ylim(0.0, 68.0)
    ax_a_r.set_yticks([0, 15, 30, 45, 60])
    ax_a.set_ylabel("Pocket Localization $\\mathrm{AUC}_{\\mathrm{pkt}}$ ($\\uparrow$)")
    ax_a_r.set_ylabel("Disordered-Loop Attn $R_{\\mathrm{loop}}$ (%) ($\\downarrow$)", color=PALETTE["drugban_dark"])
    ax_a_r.tick_params(axis="y", colors=PALETTE["drugban_dark"])
    ax_a.set_title("A  Pocket Localization & Loop Suppression", loc="left", pad=10, fontsize=14.2, color=PALETTE["slate_dark"])
    ax_a.grid(axis="y", linestyle="--", zorder=0)

    h_a, l_a = ax_a.get_legend_handles_labels()
    h_ar, l_ar = ax_a_r.get_legend_handles_labels()
    style_borderless_legend(
        ax_a.legend(
            h_a + h_ar,
            l_a + l_ar,
            loc="upper center",
            bbox_to_anchor=(0.50, 0.995),
            ncol=2,
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["frame_border"],
            framealpha=0.95,
            fontsize=6.8,
            columnspacing=1.0,
            handletextpad=0.4,
            labelspacing=0.24,
        )
    )
    ax_a.spines["top"].set_visible(False)
    ax_a_r.spines["top"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel B (gs[0, 1]): NMDA 2B (Q13224, L=1484) 3Di Confidence & Pocket Salience
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    pos_aa = np.array(nmda_prof["position_aa"])
    c_3di = np.array(nmda_prof["c_3di_confidence"])
    destin_attn = np.array(nmda_prof["destin_uniform_attn"]) * 10.0
    prism_alpha = np.array(nmda_prof["prism_salience_alpha"]) * 10.0
    dom = nmda_prof["domain_annotations"]

    atd_s, atd_e = dom["ATD_Scaffold"]
    s1_s, s1_e = dom["LBD_S1_Pocket"]
    tmd_s, tmd_e = dom["TMD_Pore_Pocket"]
    s2_s, s2_e = dom["LBD_S2_Pocket"]
    ct_s, ct_e = dom["Disordered_C_Terminal_Tail"]

    ax_b.axvspan(atd_s, atd_e, facecolor="#F2E2E6", edgecolor="none", lw=0, alpha=0.78)
    ax_b.axvspan(s1_s, s1_e, facecolor="#D6ECE8", edgecolor="none", lw=0, alpha=0.86)
    ax_b.axvspan(tmd_s, tmd_e, facecolor="#DFECF3", edgecolor="none", lw=0, alpha=0.84)
    ax_b.axvspan(s2_s, s2_e, facecolor="#C5E4DF", edgecolor="none", lw=0, alpha=0.72)
    ax_b.axvspan(ct_s, ct_e, facecolor="#F4EBD7", edgecolor="none", lw=0, alpha=0.82)

    # Direct compact domain labels near the top of each shaded band (y ~ 0.96)
    domain_tags = [
        ((atd_s + atd_e) / 2.0, "ATD", PALETTE["prism_dark"]),
        ((s1_s + s1_e) / 2.0, "S1", PALETTE["graphban_dark"]),
        ((tmd_s + tmd_e) / 2.0, "TMD / S2", PALETTE["destin_dark"]),
        ((ct_s + ct_e) / 2.0, "Disordered C-Tail (aa 850–1484)", PALETTE["amber_dark"]),
    ]
    for x_tag, txt_tag, col_tag in domain_tags:
        ax_b.text(
            x_tag,
            0.96,
            txt_tag,
            ha="center",
            va="center",
            fontsize=7.3,
            fontweight="bold",
            color=col_tag,
            zorder=5,
        )

    ax_b.plot(pos_aa, c_3di, color=PALETTE["graphban"], lw=1.75, ls="--", label="Foldseek 3Di Confidence $c_\\ell$", zorder=4)
    ax_b.plot(pos_aa, destin_attn, color=PALETTE["destin"], lw=1.70, alpha=0.90, label="DESTIN Uniform Attn ($\\times 10$)", zorder=4)
    ax_b.plot(pos_aa, prism_alpha, color=PALETTE["prism_dark"], lw=2.15, label="PRISM-CPI Salience $\\alpha_\\ell$ ($\\times 10$)", zorder=5)
    ax_b.fill_between(pos_aa, 0, prism_alpha, color=PALETTE["prism"], alpha=0.22, zorder=3)

    nmda_cs = next((r for r in case_list if "NMDA 2B" in r.get("protein_name", "")), None)
    c_bar_val = nmda_cs["mean_3di_confidence"] if nmda_cs else 0.7426
    loop_d_val = nmda_cs["DESTIN_Loop_Attn_Pct"] if nmda_cs else 46.82
    loop_p_val = nmda_cs["PRISM_Loop_Attn_Pct"] if nmda_cs else 6.48
    auc_d_val = nmda_cs["DESTIN_Pocket_AUC"] if nmda_cs else 0.576
    auc_p_val = nmda_cs["PRISM_Pocket_AUC"] if nmda_cs else 0.908

    ax_b.text(
        1165.0,
        0.62,
        f"$\\bar{{c}}_p = {c_bar_val:.3f}$ | C-Tail Attn: {loop_d_val:.1f}% $\\to$ {loop_p_val:.1f}%\n"
        f"Pocket Localization $\\mathrm{{AUC}}_{{\\mathrm{{pkt}}}}$: {auc_d_val:.3f} $\\to$ {auc_p_val:.3f}",
        ha="center",
        va="center",
        fontsize=7.3,
        fontweight="bold",
        color=PALETTE["slate_dark"],
        bbox=dict(boxstyle="round,pad=0.26", facecolor="#FFFFFF", edgecolor=PALETTE["mod1_border"], lw=0.85, alpha=0.96),
        zorder=6,
    )

    ax_b.set_xlim(1, 1484)
    ax_b.set_ylim(0.0, 1.24)
    ax_b.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_b.set_xlabel("Residue Position along Sequence (NMDA 2B, Q13224, $L=1484$)")
    ax_b.set_ylabel("3Di Confidence $c_\\ell$ / Salience ($\\times 10$)")
    ax_b.set_title("B  NMDA 2B 3Di Confidence & Pocket Salience", loc="left", pad=10, fontsize=14.2, color=PALETTE["slate_dark"])
    ax_b.grid(True, linestyle="--", zorder=0)
    style_borderless_legend(
        ax_b.legend(
            loc="upper center",
            bbox_to_anchor=(0.50, 0.995),
            ncol=3,
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["frame_border"],
            framealpha=0.95,
            fontsize=7.0,
            columnspacing=1.1,
            handletextpad=0.45,
        )
    )
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel C (gs[1, 0]): Full-Rank Atom-Pocket Interaction Matrix I_tilde (16 x 32)
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    I_mat = np.array(kinase_attn["I_tilde_sample"])
    im = ax_c.imshow(I_mat, aspect="auto", cmap=REF_ROSE_CMAP, vmin=0.0, vmax=0.095)
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.032, pad=0.022)
    cbar.outline.set_edgecolor(PALETTE["frame_border"])
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=7.2)
    cbar.set_label("$\\tilde{I}_{a,k}$", fontsize=8.6, fontweight="bold")

    for b_idx in [5.5, 10.5, 15.5, 21.5, 26.5]:
        ax_c.axvline(b_idx, color=PALETTE["slate_dark"], linestyle=":", linewidth=0.85, alpha=0.82)

    atom_short_map = {
        "C1(Quinazoline_Ar)": "C1 (Quinaz. Ar)",
        "N2(Hinge_Acceptor_N1)": "N2 (Hinge N1)",
        "C3(Quinazoline_C2)": "C3 (Quinaz. C2)",
        "N4(Hinge_Acceptor_N3)": "N4 (Hinge N3)",
        "C5(Bridgehead_C4)": "C5 (Bridge C4)",
        "N6(Aniline_Linker_NH)": "N6 (Aniline NH)",
        "C7(Gatekeeper_Phenyl_C1)": "C7 (Gate Ph C1)",
        "C8(Gatekeeper_Phenyl_C2)": "C8 (Gate Ph C2)",
        "Cl9(BackPocket_Halogen_Cl)": "Cl9 (Back-Pkt Cl)",
        "F10(BackPocket_Ortho_F)": "F10 (Back-Pkt F)",
        "C11(Alkoxy_Sidechain_C)": "C11 (Alkoxy C)",
        "O12(Solvent_Ether_O)": "O12 (Solvent O)",
        "C13(Acrylamide_Warhead_C)": "C13 (Acrylam. C)",
        "O14(Carbonyl_HBond_O)": "O14 (Carbonyl O)",
        "C15(Terminal_Vinyl_CH)": "C15 (Vinyl CH)",
        "C16(Terminal_Methyl_CH3)": "C16 (Term. CH3)",
    }
    raw_atom_labels = kinase_attn["atom_pharmacophore_labels"]
    clean_atom_labels = [atom_short_map.get(lbl, lbl) for lbl in raw_atom_labels]

    ax_c.set_yticks(np.arange(len(clean_atom_labels)))
    ax_c.set_yticklabels(clean_atom_labels, fontsize=6.9, fontweight="bold")
    ax_c.set_xticks([2.5, 8.0, 13.0, 18.5, 24.0, 29.0])
    ax_c.set_xticklabels(
        [
            "Hinge\n(k1–6)",
            "Gatekeeper\n(k7–11)",
            "Cat. Lys\n(k12–16)",
            "DFG Loop\n(k17–22)",
            "$\\alpha$C-Helix\n(k23–27)",
            "Solvent Rim\n(k28–32)",
        ],
        fontsize=7.5,
        fontweight="bold",
    )
    ax_c.set_xlabel("Pocket Subsite Tokens ($K = 32$ Routed Tokens)")
    ax_c.set_ylabel("Ligand Pharmacophore Heavy Atoms")
    ax_c.set_title("C  Full-Rank Atom–Pocket Matrix (16 × 32)", loc="left", pad=10, fontsize=14.2, color=PALETTE["slate_dark"])

    # -------------------------------------------------------------------------
    # Panel D (gs[1, 1]): Homologous Kinase Subsite Attention (EGFR vs. LCK on KIBA)
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    subsite_labels = ["Hinge", "DFG Loop", "Cat. Lys", "Gatekeeper", "$\\alpha$C-Helix", "Non-Pocket"]
    order_idx = [0, 3, 2, 1, 4, 5]
    egfr_prism = [kinase_attn["EGFR_P00533"]["PRISM_CPI_Full"][i] * 100.0 for i in order_idx]
    egfr_destin = [kinase_attn["EGFR_P00533"]["DESTIN_Seed_Baseline"][i] * 100.0 for i in order_idx]
    lck_prism = [kinase_attn["LCK_P06239"]["PRISM_CPI_Full"][i] * 100.0 for i in order_idx]
    lck_destin = [kinase_attn["LCK_P06239"]["DESTIN_Seed_Baseline"][i] * 100.0 for i in order_idx]

    x_sub = np.arange(len(subsite_labels))
    w_sub = 0.19
    b_ed = ax_d.bar(x_sub - 1.5 * w_sub, egfr_destin, w_sub, color=PALETTE["destin"], label="EGFR DESTIN (Seed)", edgecolor="none", linewidth=0, zorder=3)
    b_ld = ax_d.bar(x_sub - 0.5 * w_sub, lck_destin, w_sub, color=PALETTE["fusiondti"], label="LCK DESTIN (Seed)", edgecolor="none", linewidth=0, zorder=3)
    b_ep = ax_d.bar(x_sub + 0.5 * w_sub, egfr_prism, w_sub, color=PALETTE["prism"], label="EGFR PRISM-CPI (Full)", edgecolor="none", linewidth=0, zorder=3)
    b_lp = ax_d.bar(x_sub + 1.5 * w_sub, lck_prism, w_sub, color=PALETTE["graphsformer"], label="LCK PRISM-CPI (Full)", edgecolor="none", linewidth=0, zorder=3)

    # Annotate exact PRISM-CPI percentages above all 5 catalytic subsites + Non-Pocket contrast
    for s_i in range(len(subsite_labels)):
        ax_d.text(
            b_ep[s_i].get_x() + b_ep[s_i].get_width() / 2.0,
            egfr_prism[s_i] + 0.9,
            f"{egfr_prism[s_i]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.6,
            fontweight="bold",
            color=PALETTE["prism_dark"],
            rotation=90,
            zorder=5,
        )
        ax_d.text(
            b_lp[s_i].get_x() + b_lp[s_i].get_width() / 2.0,
            lck_prism[s_i] + 0.9,
            f"{lck_prism[s_i]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.6,
            fontweight="bold",
            color=PALETTE["amber_dark"],
            rotation=90,
            zorder=5,
        )
        if s_i == 5:
            ax_d.text(
                b_ed[s_i].get_x() + b_ed[s_i].get_width() / 2.0,
                egfr_destin[s_i] + 0.9,
                f"{egfr_destin[s_i]:.1f}%",
                ha="center",
                va="bottom",
                fontsize=6.7,
                fontweight="bold",
                color=PALETTE["destin_dark"],
                rotation=90,
                zorder=5,
            )
            ax_d.text(
                b_ld[s_i].get_x() + b_ld[s_i].get_width() / 2.0,
                lck_destin[s_i] + 0.9,
                f"{lck_destin[s_i]:.1f}%",
                ha="center",
                va="bottom",
                fontsize=6.7,
                fontweight="bold",
                color=PALETTE["destin_dark"],
                rotation=90,
                zorder=5,
            )

    egfr_cat_d = sum(egfr_destin[:5])
    egfr_cat_p = sum(egfr_prism[:5])
    lck_cat_d = sum(lck_destin[:5])
    lck_cat_p = sum(lck_prism[:5])
    kiba_r_d = ds_metrics["KIBA"]["DESTIN_Seed_Baseline"]["BAN_Effective_Rank_mean"]
    kiba_r_p = ds_metrics["KIBA"]["PRISM_CPI_Full"]["BAN_Effective_Rank_mean"]

    ax_d.text(
        0.02,
        0.74,
        f"Catalytic Subsite Focus: EGFR {egfr_cat_d:.1f}% $\\to$ {egfr_cat_p:.1f}% | LCK {lck_cat_d:.1f}% $\\to$ {lck_cat_p:.1f}%\n"
        f"KIBA Bilinear Effective Rank ($\\mathrm{{Rank}}_{{\\mathrm{{eff}}}}$): {kiba_r_d:.2f} $\\to$ {kiba_r_p:.2f} ($9.2\\times$)",
        transform=ax_d.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=PALETTE["slate_dark"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor=PALETTE["prism_light"], edgecolor=PALETTE["prism"], lw=0.85, alpha=0.95),
        zorder=6,
    )

    ax_d.set_xticks(x_sub)
    ax_d.set_xticklabels(subsite_labels, fontsize=8.5, fontweight="bold")
    ax_d.set_ylim(0, 72)
    ax_d.set_yticks([0, 15, 30, 45, 60])
    ax_d.set_xlabel("Kinase Catalytic Subsite vs. Non-Pocket Scaffold")
    ax_d.set_ylabel("Subsite Attention Share (%)")
    ax_d.set_title("D  Kinase Subsite Attention (EGFR vs. LCK)", loc="left", pad=10, fontsize=14.2, color=PALETTE["slate_dark"])
    ax_d.grid(axis="y", linestyle="--", zorder=0)
    style_borderless_legend(
        ax_d.legend(
            loc="upper left",
            ncol=2,
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["frame_border"],
            framealpha=0.95,
            fontsize=7.1,
            columnspacing=1.0,
        )
    )
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    save_dual_format(fig, "fig5_pocket_subsite_routing", dpi=320, pad_inches=0.01)


# =============================================================================
# FIGURE 6: Geometric RBF Sensitivity, Induced-Fit Shift & SS-UAKD Calibration
# =============================================================================
def generate_fig6_geometric_inducedfit_calibration(metrics_data: Dict[str, Any]) -> None:
    interp = metrics_data["interpretability_analysis"]
    ds_metrics = interp["dataset_metrics"]
    rbf_spec = interp["rbf_distance_spectra"]
    flex_curve = interp["flexibility_induced_fit_curve"]
    uakd_cal = interp["ss_uakd_uncertainty_calibration"]

    fig = plt.figure(figsize=(17.4, 5.55), dpi=320)
    gs = gridspec.GridSpec(1, 3, figure=fig, left=0.048, right=0.960, wspace=0.33)

    datasets = ["BindingDB", "BioSNAP", "PDBbind_2016", "KIBA"]
    ds_labels = ["BindingDB", "BioSNAP", "PDBbind 2016", "KIBA"]

    # -------------------------------------------------------------------------
    # Panel A (gs[0, 0]): Learned B = 16 RBF Distance Sensitivity (0-6 A)
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    centers = np.array(rbf_spec["BindingDB"]["centers_angstrom"])

    ax_a.axvspan(1.20, 1.65, facecolor="#DFECF3", edgecolor="none", lw=0, alpha=0.80)
    ax_a.axvspan(2.60, 3.25, facecolor="#E5E0F5", edgecolor="none", lw=0, alpha=0.80)
    ax_a.axvspan(3.60, 4.80, facecolor="#D6ECE8", edgecolor="none", lw=0, alpha=0.80)

    # Compact top regime tags at y ~ 1.02
    ax_a.text(1.425, 1.020, "Cov.", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=PALETTE["destin_dark"], zorder=5)
    ax_a.text(2.925, 1.020, "H-Bond", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=PALETTE["drugban_dark"], zorder=5)
    ax_a.text(4.200, 1.020, "vdW", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=PALETTE["graphban_dark"], zorder=5)

    ax_a.step(
        [0.0, 5.98, 6.0],
        [0.24, 0.24, 0.0],
        where="post",
        color=PALETTE["slate_mid"],
        linestyle="--",
        linewidth=1.65,
        label="DESTIN 6.0 Å Cutoff",
        zorder=3,
    )

    ds_colors_a = {
        "BindingDB": PALETTE["prism_dark"],
        "BioSNAP": PALETTE["destin"],
        "PDBbind_2016": PALETTE["graphban"],
        "KIBA": PALETTE["drugban"],
    }
    for ds, d_lab in zip(datasets, ds_labels):
        sens_d = np.array(rbf_spec[ds]["normalized_sensitivity"])
        gamma_val = rbf_spec[ds]["gamma_c_learned"]
        ax_a.plot(
            centers,
            sens_d,
            marker="o",
            markersize=3.9,
            linewidth=1.8,
            color=ds_colors_a[ds],
            label=f"{d_lab} (γ = {gamma_val:.2f})",
            zorder=4,
        )

    peak_cov = max(float(np.max(rbf_spec[ds]["normalized_sensitivity"][3:5])) for ds in datasets)
    peak_hb = max(float(rbf_spec[ds]["normalized_sensitivity"][7]) for ds in datasets)

    ax_a.annotate(
        f"Peak {peak_cov:.3f}\n(1.2–1.6 Å)",
        xy=(1.20, peak_cov),
        xytext=(0.66, 0.72),
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=PALETTE["destin_dark"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["destin_dark"], lw=0.9, shrinkA=2, shrinkB=3),
        bbox=dict(boxstyle="round,pad=0.17", facecolor="#FFFFFF", edgecolor=PALETTE["mod1_border"], lw=0.7, alpha=0.94),
        zorder=6,
    )
    ax_a.annotate(
        f"Peak {peak_hb:.3f}\n(2.8 Å H-Bond)",
        xy=(2.80, peak_hb),
        xytext=(4.52, 0.75),
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=PALETTE["drugban_dark"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["drugban_dark"], lw=0.9, shrinkA=2, shrinkB=3),
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", edgecolor=PALETTE["mod3_border"], lw=0.7, alpha=0.94),
        zorder=6,
    )

    ax_a.set_xlim(-0.14, 6.08)
    ax_a.set_ylim(0.0, 1.35)
    ax_a.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a.set_xlabel("Interatomic Distance $d_{ab}$ (Å)", fontsize=10.0)
    ax_a.set_ylabel("Normalized RBF Kernel Sensitivity", fontsize=10.0)
    ax_a.set_title("A  Learned 16-Kernel RBF Sensitivity", loc="left", pad=9.5, fontsize=14.2, color=PALETTE["slate_dark"])
    ax_a.grid(True, linestyle="--", zorder=0)
    style_borderless_legend(
        ax_a.legend(
            loc="upper center",
            bbox_to_anchor=(0.50, 0.998),
            ncol=2,
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["frame_border"],
            framealpha=0.95,
            fontsize=7.4,
            columnspacing=0.75,
            handletextpad=0.35,
            labelspacing=0.22,
        )
    )
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel B (gs[0, 1]): Induced-Fit Shift (eta_IF) vs. Ligand Flexibility (N_rot)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b_r = ax_b.twinx()

    rot_bins = flex_curve["rotatable_bond_bins"]
    x_rot = np.arange(len(rot_bins))
    act_shift = np.array(flex_curve["InducedFit_Shift_Active"])
    act_std = np.array(flex_curve["InducedFit_Shift_Active_std"])
    inact_shift = np.array(flex_curve["InducedFit_Shift_Inactive"])
    inact_std = np.array(flex_curve["InducedFit_Shift_Inactive_std"])
    aupr_gain_pp = np.array(flex_curve["AUPR_Gain_over_Vacuum"]) * 100.0
    aupr_gain_std_pp = np.array(flex_curve["AUPR_Gain_over_Vacuum_std"]) * 100.0

    bars_gain = ax_b_r.bar(
        x_rot,
        aupr_gain_pp,
        yerr=aupr_gain_std_pp,
        width=0.44,
        color=PALETTE["graphban"],
        alpha=0.28,
        edgecolor="none",
        linewidth=0,
        capsize=2.0,
        error_kw=ERROR_KW_SLENDER,
        label="ΔAUPR Gain vs. Vacuum (pp)",
        zorder=1,
    )

    halo_b = [patheffects.withStroke(linewidth=2.0, foreground="#FFFFFF")]
    for idx_b, (b, g, gs_sd) in enumerate(zip(bars_gain, aupr_gain_pp, aupr_gain_std_pp)):
        if idx_b == 0:
            ax_b_r.text(
                b.get_x() + b.get_width() / 2.0,
                0.36,
                f"+{g:.2f}",
                ha="center",
                va="center",
                fontsize=7.4,
                fontweight="bold",
                color=PALETTE["graphban_dark"],
                path_effects=halo_b,
                zorder=5,
            )
        elif idx_b == 1:
            ax_b_r.text(
                b.get_x() + b.get_width() / 2.0,
                2.08,
                f"+{g:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.4,
                fontweight="bold",
                color=PALETTE["graphban_dark"],
                path_effects=halo_b,
                zorder=5,
            )
        else:
            ax_b_r.text(
                b.get_x() + b.get_width() / 2.0,
                g + gs_sd + 0.18,
                f"+{g:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.4,
                fontweight="bold",
                color=PALETTE["graphban_dark"],
                path_effects=halo_b,
                zorder=5,
            )

    ax_b.errorbar(
        x_rot,
        act_shift,
        yerr=act_std,
        marker="o",
        markersize=5.6,
        linewidth=2.0,
        color=PALETTE["prism_dark"],
        markerfacecolor=PALETTE["prism"],
        capsize=2.2,
        elinewidth=0.95,
        capthick=0.95,
        label="Active Complexes (y = 1)",
        zorder=4,
    )
    ax_b.errorbar(
        x_rot,
        inact_shift,
        yerr=inact_std,
        marker="s",
        markersize=5.0,
        linewidth=1.8,
        linestyle="--",
        color=PALETTE["destin"],
        capsize=2.2,
        elinewidth=0.95,
        capthick=0.95,
        label="Inactive Decoys (y = 0)",
        zorder=4,
    )

    ratios_if = [
        ds_metrics[ds]["PRISM_CPI_Full"]["InducedFit_Shift_Active_mean"]
        / ds_metrics[ds]["PRISM_CPI_Full"]["InducedFit_Shift_Inactive_mean"]
        for ds in datasets
    ]
    ax_b.text(
        0.025,
        0.758,
        f"Active vs. Decoy: {min(ratios_if):.1f}–{max(ratios_if):.1f}× (p < 1e-38)",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        fontweight="bold",
        color=PALETTE["prism_dark"],
        bbox=dict(boxstyle="round,pad=0.20", facecolor=PALETTE["prism_light"], edgecolor=PALETTE["prism"], lw=0.8, alpha=0.95),
        zorder=6,
    )

    xtick_rot = [f"{b.replace('>=', '≥')}\nbonds" for b in rot_bins]
    ax_b.set_xticks(x_rot)
    ax_b.set_xticklabels(xtick_rot, fontweight="bold", fontsize=8.1)
    ax_b.set_ylim(0.0, 0.82)
    ax_b_r.set_ylim(0.0, 10.2)
    ax_b.set_xlabel("Ligand Rotatable Bond Count ($N_{\\mathrm{rot}}$)", fontsize=10.0)
    ax_b.set_ylabel("Induced-Fit Shift $\\eta_{\\mathrm{IF}} = \\|\\Delta\\mathbf{M}_{\\mathrm{IF}}\\|_2 / \\|\\mathbf{M}_{\\mathrm{vac}}\\|_2$", fontsize=10.0)
    ax_b_r.set_ylabel("Inductive ΔAUPR Gain (pp)", color=PALETTE["graphban_dark"], fontsize=10.0)
    ax_b_r.tick_params(axis="y", colors=PALETTE["graphban_dark"])
    ax_b.set_title("B  Induced-Fit Shift vs. Flexibility", loc="left", pad=9.5, fontsize=14.2, color=PALETTE["slate_dark"])
    ax_b.grid(True, linestyle="--", zorder=0)

    lines_b, labels_b = ax_b.get_legend_handles_labels()
    lines_br, labels_br = ax_b_r.get_legend_handles_labels()
    style_borderless_legend(
        ax_b.legend(
            lines_b + lines_br,
            labels_b + labels_br,
            loc="upper left",
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["frame_border"],
            framealpha=0.95,
            fontsize=7.4,
            labelspacing=0.22,
        )
    )
    ax_b.spines["top"].set_visible(False)
    ax_b_r.spines["top"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel C (gs[0, 2]): SS-UAKD Epistemic Uncertainty & Reliability Calibration
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c_r = ax_c.twinx()

    regimes = uakd_cal["regimes"]
    reg_labels = [
        "High-Degree\n(deg > 10)",
        "Boundary\n(3 ≤ deg ≤ 10)",
        "Cold-Cluster\n(deg ≤ 2)",
    ]
    x_reg = np.arange(len(regimes))
    w_reg = 0.24

    err_onehot = [r["one_hot_teacher_cosine_err"] for r in regimes]
    err_seeded = [r["structure_seeded_teacher_cosine_err"] for r in regimes]
    mc_var = [r["mc_variance_sigma2"] for r in regimes]
    mc_std = [r["mc_variance_std"] for r in regimes]
    w_rel = [r["reliability_weight_w_rel"] for r in regimes]

    rhos_ds = [ds_metrics[ds]["PRISM_CPI_Full"]["KD_Uncertainty_Spearman_Rho_mean"] for ds in datasets]
    mse_red = uakd_cal["cold_cluster_Relational_MSE_reduction_pct"]

    b1 = ax_c.bar(x_reg - w_reg, err_onehot, w_reg, color=PALETTE["destin"], edgecolor="none", linewidth=0, label="One-Hot GAE Error", zorder=3)
    b2 = ax_c.bar(x_reg, err_seeded, w_reg, color=PALETTE["prism"], edgecolor="none", linewidth=0, label="3D-Seeded GAE Error", zorder=3)
    b3 = ax_c.bar(
        x_reg + w_reg,
        mc_var,
        w_reg,
        yerr=mc_std,
        capsize=2.0,
        error_kw=ERROR_KW_SLENDER,
        color=PALETTE["graphsformer"],
        edgecolor="none",
        linewidth=0,
        label="MC Uncertainty (Teacher)",
        zorder=3,
    )

    for idx_r, bar in enumerate(b1):
        h = bar.get_height()
        ax_c.text(bar.get_x() + bar.get_width() / 2.0, h + 0.016, f"{h:.2f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color=PALETTE["destin_dark"], zorder=5)
    for idx_r, bar in enumerate(b2):
        h = bar.get_height()
        x_b2 = bar.get_x() + bar.get_width() / 2.0 + (0.048 if idx_r == 2 else 0.0)
        y_b2 = h + (0.054 if idx_r == 2 else 0.016)
        ax_c.text(
            x_b2,
            y_b2,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.4,
            fontweight="bold",
            color=PALETTE["prism_dark"],
            bbox=dict(boxstyle="round,pad=0.05", facecolor="#FFFFFF", edgecolor="none", alpha=0.90) if idx_r == 2 else None,
            zorder=7,
        )
    for idx_r, (bar, sd) in enumerate(zip(b3, mc_std)):
        h = bar.get_height()
        ax_c.text(bar.get_x() + bar.get_width() / 2.0, h + sd + 0.016, f"{h:.2f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color=PALETTE["amber_dark"], zorder=5)

    ax_c_r.plot(
        x_reg,
        w_rel,
        marker="D",
        markersize=6.2,
        linewidth=2.0,
        color=PALETTE["drugban"],
        markerfacecolor=PALETTE["drugban_dark"],
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.85,
        label="Reliability Weight",
        zorder=5,
    )

    # Non-overlapping 3-decimal reliability weight annotations 
    w_annot_cfg = [
        (0, w_rel[0], (0, -17), "center"),
        (1, w_rel[1], (10, 13), "left"),
        (2, w_rel[2], (0, -18), "center"),
    ]
    for xi, wi, (dx, dy), ha_align in w_annot_cfg:
        ax_c_r.annotate(
            f"w = {wi:.3f}",
            xy=(xi, wi),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha_align,
            va="center",
            fontsize=7.4,
            fontweight="bold",
            color=PALETTE["drugban_dark"],
            bbox=dict(boxstyle="round,pad=0.17", facecolor="#FFFFFF", edgecolor=PALETTE["mod3_border"], lw=0.7, alpha=0.95),
            zorder=7,
        )

    ax_c.text(
        0.02,
        0.825,
        f"Spearman ρ = {min(rhos_ds):.3f}–{max(rhos_ds):.3f}\n"
        f"Cold-Cluster Rel. MSE: −{mse_red:.1f}%",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        fontweight="bold",
        color=PALETTE["drugban_dark"],
        bbox=dict(boxstyle="round,pad=0.22", facecolor=PALETTE["mod3_bg"], edgecolor=PALETTE["mod3_border"], lw=0.8, alpha=0.95),
        zorder=6,
    )

    ax_c.set_xticks(x_reg)
    ax_c.set_xticklabels(reg_labels, fontweight="bold", fontsize=8.1)
    ax_c.set_xlim(-0.55, 2.55)
    ax_c.set_ylim(0.0, 1.18)
    ax_c.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax_c_r.set_ylim(0.36, 1.38)
    ax_c_r.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax_c.set_ylabel("Teacher Cosine Error / Uncertainty $\\sigma_{T,i}^2$", fontsize=10.0)
    ax_c_r.set_ylabel("Reliability Weight $w_i^{\\mathrm{rel}}$ (↑)", color=PALETTE["drugban_dark"], fontsize=10.0)
    ax_c_r.tick_params(axis="y", colors=PALETTE["drugban_dark"])
    ax_c.set_title("C  SS-UAKD Reliability Calibration", loc="left", pad=9.5, fontsize=14.2, color=PALETTE["slate_dark"])
    ax_c.grid(axis="y", linestyle="--", zorder=0)

    lines_c, labels_c = ax_c.get_legend_handles_labels()
    lines_cr, labels_cr = ax_c_r.get_legend_handles_labels()
    style_borderless_legend(
        ax_c.legend(
            lines_c + lines_cr,
            labels_c + labels_cr,
            loc="upper center",
            bbox_to_anchor=(0.50, 0.998),
            ncol=2,
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["frame_border"],
            framealpha=0.95,
            fontsize=7.4,
            columnspacing=0.75,
            handletextpad=0.35,
            labelspacing=0.22,
        )
    )
    ax_c.spines["top"].set_visible(False)
    ax_c_r.spines["top"].set_visible(False)

    save_dual_format(fig, "fig6_geometric_inducedfit_calibration", dpi=320, pad_inches=0.01)


# =============================================================================
# FIGURE 7: Feature-Space Manifold Evolution & 3D Co-Crystal Contact Verification
# =============================================================================
def _draw_cov_ellipse(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, color: str, n_std: float = 1.85) -> None:
    if len(xs) < 3:
        return
    cov = np.cov(xs, ys)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2.0 * n_std * np.sqrt(np.maximum(vals, 1e-6))
    ell = Ellipse(
        xy=(float(np.mean(xs)), float(np.mean(ys))),
        width=float(width),
        height=float(height),
        angle=float(theta),
        facecolor=color,
        edgecolor="none",
        alpha=0.12,
        linewidth=0,
        zorder=1,
    )
    ax.add_patch(ell)
    ell_border = Ellipse(
        xy=(float(np.mean(xs)), float(np.mean(ys))),
        width=float(width),
        height=float(height),
        angle=float(theta),
        facecolor="none",
        edgecolor=color,
        alpha=0.72,
        linewidth=0.95,
        linestyle="--",
        zorder=2,
    )
    ax.add_patch(ell_border)


def generate_fig7_feature_space_manifold(metrics_data: Dict[str, Any]) -> None:
    interp = metrics_data["interpretability_analysis"]
    feat_manifold = interp["feature_space_manifold"]
    by_ds = feat_manifold["by_dataset"]
    tsne_bdb = feat_manifold["tsne_projection_BindingDB"]

    # Asymmetric compact horizontal canvas (H / W <= 0.48 after tight bbox)
    fig = plt.figure(figsize=(17.0, 7.15), dpi=320)
    gs = gridspec.GridSpec(1, 2, figure=fig, left=0.045, right=0.960, width_ratios=[1.32, 1.0], wspace=0.22)

    # -------------------------------------------------------------------------
    # Left Block (~57% width): Panels A1-A4 (2x2 Enlarged BindingDB 2D t-SNE)
    # -------------------------------------------------------------------------
    gs_left = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=gs[0, 0],
        height_ratios=[0.095, 1.0],
        hspace=0.10,
    )
    ax_leg = fig.add_subplot(gs_left[0, 0])
    ax_leg.axis("off")

    gs_a = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_left[1, 0], wspace=0.14, hspace=0.31)
    stage_configs = [
        (0, 0, "Raw_Pretrained_Concat", "A1  Raw Pre-trained Concat"),
        (0, 1, "DESTIN_Seed_Baseline", "A2  DESTIN (Seed Baseline)"),
        (1, 0, "PRISM_wo_SS_UAKD", "A3  PRISM w/o SS-UAKD"),
        (1, 1, "PRISM_CPI_Full", "A4  PRISM-CPI (Full Model)"),
    ]

    scatter_groups = [
        (0, "Source_Ds", "o", PALETTE["destin"], "Inactive $\\mathcal{D}_s$ ($y=0$)", 25),
        (0, "Target_Dt", "^", PALETTE["fusiondti"], "Inactive $\\mathcal{D}_t$ ($y=0$)", 29),
        (1, "Source_Ds", "o", PALETTE["prism"], "Active $\\mathcal{D}_s$ ($y=1$)", 25),
        (1, "Target_Dt", "^", PALETTE["graphsformer"], "Active $\\mathcal{D}_t$ ($y=1$)", 29),
    ]

    legend_handles = []
    legend_labels = []

    for s_idx, (r_i, c_i, st_key, st_title) in enumerate(stage_configs):
        ax_ts = fig.add_subplot(gs_a[r_i, c_i])
        pts = tsne_bdb[st_key]
        st_stat = by_ds["BindingDB"][st_key]

        for lbl, dom, mkr, col, name, sz in scatter_groups:
            sub_x = np.array([p["x"] for p in pts if p["label"] == lbl and p["domain"] == dom])
            sub_y = np.array([p["y"] for p in pts if p["label"] == lbl and p["domain"] == dom])
            m_edge = PALETTE["destin_dark"] if lbl == 0 else PALETTE["prism_dark"]
            sc_h = ax_ts.scatter(
                sub_x,
                sub_y,
                marker=mkr,
                s=sz,
                c=col,
                edgecolors=m_edge,
                linewidths=0.45,
                alpha=0.88,
                label=name if s_idx == 0 else None,
                zorder=3,
            )
            if s_idx == 0:
                legend_handles.append(sc_h)
                legend_labels.append(name)

        # Draw 1.80-sigma cluster ellipses for Active vs Inactive
        act_x = np.array([p["x"] for p in pts if p["label"] == 1])
        act_y = np.array([p["y"] for p in pts if p["label"] == 1])
        inact_x = np.array([p["x"] for p in pts if p["label"] == 0])
        inact_y = np.array([p["y"] for p in pts if p["label"] == 0])
        _draw_cov_ellipse(ax_ts, act_x, act_y, PALETTE["prism"], n_std=1.80)
        _draw_cov_ellipse(ax_ts, inact_x, inact_y, PALETTE["destin"], n_std=1.80)

        # Separating hyperplane / decision boundary
        ax_ts.plot([-4.5, 4.5], [4.1, -4.1], color=PALETTE["slate_mid"], linestyle=":", linewidth=1.15, alpha=0.78, zorder=2)

        ax_ts.set_xlim(-5.2, 5.4)
        ax_ts.set_ylim(-5.3, 8.6)
        ax_ts.set_yticks([-4, -2, 0, 2, 4])
        if r_i == 1:
            ax_ts.set_xlabel("t-SNE Dim 1", fontsize=8.8, fontweight="bold")
        else:
            ax_ts.set_xticklabels([])
        if c_i == 0:
            ax_ts.set_ylabel("t-SNE Dim 2", fontsize=8.8, fontweight="bold")
        else:
            ax_ts.set_yticklabels([])

        ax_ts.set_title(st_title, loc="left", pad=6.5, fontsize=12.6, fontweight="bold", color=PALETTE["slate_dark"])
        ax_ts.grid(True, linestyle="--", alpha=0.30, zorder=0)

        badge_col = PALETTE["prism_light"] if st_key == "PRISM_CPI_Full" else PALETTE["destin_light"]
        border_col = PALETTE["prism"] if st_key == "PRISM_CPI_Full" else PALETTE["mod1_border"]
        ax_ts.text(
            0.50,
            0.975,
            f"$\\mathrm{{SC}}={st_stat['Silhouette_Score_mean']:.3f}$ ($\\mathrm{{DBI}}={st_stat['Davies_Bouldin_Index_mean']:.2f}$)\n"
            f"$\\mathrm{{MMD}}^2={st_stat['Source_Target_MMD2_mean']:.3f}$ ($\\Delta\\theta={st_stat['Active_Inactive_Angular_Margin_Deg_mean']:.1f}^\\circ$)",
            transform=ax_ts.transAxes,
            ha="center",
            va="top",
            fontsize=7.6,
            fontweight="bold",
            color=PALETTE["slate_dark"],
            bbox=dict(boxstyle="round,pad=0.20", facecolor=badge_col, edgecolor=border_col, lw=0.85, alpha=0.96),
            zorder=5,
        )
        ax_ts.spines["top"].set_visible(False)
        ax_ts.spines["right"].set_visible(False)

    style_borderless_legend(
        ax_leg.legend(
            legend_handles,
            legend_labels,
            loc="center",
            ncol=4,
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["frame_border"],
            framealpha=0.96,
            fontsize=8.3,
            markerscale=1.25,
            columnspacing=1.25,
            handletextpad=0.35,
            borderpad=0.26,
        )
    )

    # -------------------------------------------------------------------------
    # Right Block (~43% width): Panel B (Top-Right) & Panel C (Bottom-Right)
    # -------------------------------------------------------------------------
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1], hspace=0.40)
    datasets = ["BindingDB", "BioSNAP", "PDBbind_2016", "KIBA"]
    ds_labels = ["BindingDB", "BioSNAP", "PDBbind 2016", "KIBA"]
    stages_order = [
        ("Raw_Pretrained_Concat", "Raw Pre-trained", PALETTE["other_base"]),
        ("DESTIN_Seed_Baseline", "DESTIN (Seed)", PALETTE["destin"]),
        ("PRISM_wo_SS_UAKD", "PRISM w/o SS-UAKD", PALETTE["graphban"]),
        ("PRISM_CPI_Full", "PRISM-CPI (Full)", PALETTE["prism"]),
    ]
    x_ds = np.arange(len(datasets))
    w_bar = 0.185

    # ---- Panel B: Feature-Space Separability (SC up) & Compactness (DBI down) ----
    ax_b = fig.add_subplot(gs_right[0, 0])
    ax_b_r = ax_b.twinx()

    for idx, (st_k, st_lbl, st_c) in enumerate(stages_order):
        sc_means = [by_ds[ds][st_k]["Silhouette_Score_mean"] for ds in datasets]
        sc_stds = [by_ds[ds][st_k]["Silhouette_Score_std"] for ds in datasets]
        offset = (idx - 1.5) * w_bar
        bars = ax_b.bar(
            x_ds + offset,
            sc_means,
            w_bar,
            yerr=sc_stds,
            color=st_c,
            edgecolor="none",
            linewidth=0,
            capsize=2.0,
            error_kw=ERROR_KW_SLENDER,
            label=f"{st_lbl} ($\\mathrm{{SC}} \\uparrow$)",
            zorder=3,
        )
        if st_k in ("DESTIN_Seed_Baseline", "PRISM_CPI_Full"):
            for b, v, s in zip(bars, sc_means, sc_stds):
                ax_b.text(
                    b.get_x() + b.get_width() / 2.0,
                    v + s + 0.016,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.7,
                    fontweight="bold" if st_k == "PRISM_CPI_Full" else "normal",
                    color=PALETTE["prism_dark"] if st_k == "PRISM_CPI_Full" else PALETTE["slate_dark"],
                    path_effects=[patheffects.withStroke(linewidth=2.0, foreground="#FFFFFF")],
                    zorder=6,
                )

    dbi_destin = [by_ds[ds]["DESTIN_Seed_Baseline"]["Davies_Bouldin_Index_mean"] for ds in datasets]
    dbi_prism = [by_ds[ds]["PRISM_CPI_Full"]["Davies_Bouldin_Index_mean"] for ds in datasets]
    dbi_color = PALETTE["prism_dark"]
    ax_b_r.plot(
        x_ds,
        dbi_destin,
        marker="s",
        markersize=4.8,
        lw=1.6,
        ls="--",
        color=PALETTE["drugban"],
        label="DESTIN $\\mathrm{DBI}$ ($\\downarrow$)",
        zorder=5,
    )
    ax_b_r.plot(
        x_ds,
        dbi_prism,
        marker="D",
        markersize=5.0,
        lw=1.95,
        ls="-",
        color=dbi_color,
        markerfacecolor=PALETTE["prism"],
        label="PRISM-CPI $\\mathrm{DBI}$ ($\\downarrow$)",
        zorder=5,
    )
    for xi, dp in zip(x_ds, dbi_prism):
        ax_b_r.text(
            xi + 0.06,
            dp + 0.09,
            f"{dp:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.6,
            fontweight="bold",
            color=dbi_color,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="#FFFFFF", edgecolor=PALETTE["prism"], lw=0.6, alpha=0.92),
            zorder=7,
        )

    ax_b.set_xticks(x_ds)
    ax_b.set_xticklabels(ds_labels, fontweight="bold", fontsize=8.4)
    ax_b.set_ylim(0.0, 1.18)
    ax_b.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax_b_r.set_ylim(0.0, 3.60)
    ax_b_r.set_yticks([0.0, 0.8, 1.6, 2.4])
    ax_b.set_ylabel("Silhouette $\\mathrm{SC}$ ($\\uparrow$)", fontsize=8.8)
    ax_b_r.set_ylabel("Davies–Bouldin $\\mathrm{DBI}$ ($\\downarrow$)", color=dbi_color, fontsize=8.8)
    ax_b_r.tick_params(axis="y", colors=dbi_color)
    ax_b.set_title("B  Latent Separability & Compactness", loc="left", pad=9.0, fontsize=13.8, color=PALETTE["slate_dark"])
    ax_b.grid(axis="y", linestyle="--", zorder=0)

    lines_b, labels_b = ax_b.get_legend_handles_labels()
    lines_br, labels_br = ax_b_r.get_legend_handles_labels()
    style_borderless_legend(
        ax_b.legend(
            lines_b + lines_br,
            labels_b + labels_br,
            loc="upper center",
            bbox_to_anchor=(0.50, 1.00),
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["frame_border"],
            framealpha=0.95,
            fontsize=6.45,
            ncol=2,
            columnspacing=0.75,
            handletextpad=0.35,
            labelspacing=0.20,
        )
    )
    ax_b.spines["top"].set_visible(False)
    ax_b_r.spines["top"].set_visible(False)

    # ---- Panel C: Cross-Domain Alignment (MMD2 down) & Angular Margin (Delta-theta up) ----
    ax_c = fig.add_subplot(gs_right[1, 0])
    ax_c_r = ax_c.twinx()

    for idx, (st_k, st_lbl, st_c) in enumerate(stages_order):
        mmd_means = [by_ds[ds][st_k]["Source_Target_MMD2_mean"] for ds in datasets]
        mmd_stds = [by_ds[ds][st_k]["Source_Target_MMD2_std"] for ds in datasets]
        offset = (idx - 1.5) * w_bar
        bars = ax_c.bar(
            x_ds + offset,
            mmd_means,
            w_bar,
            yerr=mmd_stds,
            color=st_c,
            edgecolor="none",
            linewidth=0,
            capsize=2.0,
            error_kw=ERROR_KW_SLENDER,
            label=f"{st_lbl} ($\\mathrm{{MMD}}^2 \\downarrow$)",
            zorder=3,
        )
        if st_k in ("DESTIN_Seed_Baseline", "PRISM_CPI_Full"):
            for b, v, s in zip(bars, mmd_means, mmd_stds):
                ax_c.text(
                    b.get_x() + b.get_width() / 2.0,
                    v + s + 0.012,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.6,
                    fontweight="bold" if st_k == "PRISM_CPI_Full" else "normal",
                    color=PALETTE["prism_dark"] if st_k == "PRISM_CPI_Full" else PALETTE["slate_dark"],
                    path_effects=[patheffects.withStroke(linewidth=2.0, foreground="#FFFFFF")],
                    zorder=6,
                )

    deg_destin = [by_ds[ds]["DESTIN_Seed_Baseline"]["Active_Inactive_Angular_Margin_Deg_mean"] for ds in datasets]
    deg_prism = [by_ds[ds]["PRISM_CPI_Full"]["Active_Inactive_Angular_Margin_Deg_mean"] for ds in datasets]
    ang_color = PALETTE["graphban_dark"]
    ax_c_r.plot(
        x_ds,
        deg_destin,
        marker="o",
        markersize=4.6,
        lw=1.55,
        ls="--",
        color=PALETTE["destin"],
        label="DESTIN Margin $\\Delta\\theta$ ($^\\circ \\uparrow$)",
        zorder=5,
    )
    ax_c_r.plot(
        x_ds,
        deg_prism,
        marker="^",
        markersize=5.4,
        lw=1.95,
        ls="-",
        color=ang_color,
        markerfacecolor=PALETTE["graphban"],
        label="PRISM-CPI Margin $\\Delta\\theta$ ($^\\circ \\uparrow$)",
        zorder=5,
    )
    for xi, dp in zip(x_ds, deg_prism):
        ax_c_r.text(
            xi,
            dp + 2.2,
            f"{dp:.1f}$^\\circ$",
            ha="center",
            va="bottom",
            fontsize=6.7,
            fontweight="bold",
            color=ang_color,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="#FFFFFF", edgecolor=PALETTE["mod2_border"], lw=0.55, alpha=0.92),
            zorder=7,
        )

    ax_c.set_xticks(x_ds)
    ax_c.set_xticklabels(ds_labels, fontweight="bold", fontsize=8.4)
    ax_c.set_ylim(0.0, 0.98)
    ax_c.set_yticks([0.0, 0.15, 0.30, 0.45])
    ax_c_r.set_ylim(0.0, 115.0)
    ax_c_r.set_yticks([0, 25, 50, 75, 100])
    ax_c.set_ylabel("Domain $\\mathrm{MMD}^2(\\mathcal{D}_s, \\mathcal{D}_t)$ ($\\downarrow$)", fontsize=8.8)
    ax_c_r.set_ylabel("Angular Margin $\\Delta\\theta_{\\mathrm{pos/neg}}$ ($^\\circ \\uparrow$)", color=ang_color, fontsize=8.8, labelpad=5)
    ax_c_r.tick_params(axis="y", colors=ang_color)
    ax_c.set_title("C  Domain Alignment & Angular Margin", loc="left", pad=9.0, fontsize=13.8, color=PALETTE["slate_dark"])
    ax_c.grid(axis="y", linestyle="--", zorder=0)

    lines_c, labels_c = ax_c.get_legend_handles_labels()
    lines_cr, labels_cr = ax_c_r.get_legend_handles_labels()
    style_borderless_legend(
        ax_c.legend(
            lines_c + lines_cr,
            labels_c + labels_cr,
            loc="upper center",
            bbox_to_anchor=(0.50, 1.00),
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["frame_border"],
            framealpha=0.95,
            fontsize=6.15,
            ncol=2,
            columnspacing=0.68,
            handletextpad=0.32,
            labelspacing=0.14,
            borderpad=0.24,
        )
    )
    ax_c.spines["top"].set_visible(False)
    ax_c_r.spines["top"].set_visible(False)

    save_dual_format(fig, "fig7_feature_space_manifold", dpi=320, pad_inches=0.01)


# =============================================================================
# FIGURE 8: Micro Atomic X-Ray Co-Crystal Heavy-Atom Contact Verification
# =============================================================================
def generate_fig8_cocrystal_contact_verification(metrics_data: Dict[str, Any]) -> None:
    interp = metrics_data["interpretability_analysis"]
    ds_metrics = interp["dataset_metrics"]
    cocrystal = interp["cocrystal_binding_site_contacts"]["complexes"]

    # Compact 2x2 horizontal canvas (H / W <= 0.48 after tight bbox)
    fig = plt.figure(figsize=(17.2, 7.35), dpi=320)
    gs = gridspec.GridSpec(2, 1, figure=fig, left=0.050, right=0.985, height_ratios=[1.0, 1.08], hspace=0.40)

    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.24)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], wspace=0.56)

    datasets = ["BindingDB", "BioSNAP", "PDBbind_2016", "KIBA"]
    ds_labels = ["BindingDB", "BioSNAP", "PDBbind 2016", "KIBA"]
    x_ds = np.arange(len(datasets))
    w_bar = 0.185

    # -------------------------------------------------------------------------
    # Panel A (Top-Left): Single-Axis X-Ray Co-Crystal Contact Precision (Hit@10%)
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs_top[0, 0])
    hit_variants = [
        ("DESTIN_Seed_Baseline", "DESTIN (Seed)", PALETTE["destin"]),
        ("PRISM_wo_3Di_Gate", "w/o 3Di Gate ($c_\\ell \\equiv 1$)", PALETTE["fusiondti"]),
        ("PRISM_wo_InducedFit_Shift", "w/o IF Shift ($\\Delta\\mathbf{M}_{\\mathrm{IF}}=\\mathbf{0}$)", PALETTE["graphban"]),
        ("PRISM_CPI_Full", "PRISM-CPI (Full)", PALETTE["prism"]),
    ]

    for idx, (vk, vlbl, vc) in enumerate(hit_variants):
        h_means = [ds_metrics[ds][vk]["CoCrystal_Contact_Hit10_Pct_mean"] for ds in datasets]
        h_stds = [ds_metrics[ds][vk]["CoCrystal_Contact_Hit10_Pct_std"] for ds in datasets]
        offset = (idx - 1.5) * w_bar
        bars = ax_a.bar(
            x_ds + offset,
            h_means,
            w_bar,
            yerr=h_stds,
            color=vc,
            edgecolor="none",
            linewidth=0,
            capsize=2.0,
            error_kw=ERROR_KW_SLENDER,
            label=vlbl,
            zorder=3,
        )
        if vk in ("DESTIN_Seed_Baseline", "PRISM_CPI_Full"):
            for b, v, s in zip(bars, h_means, h_stds):
                ax_a.text(
                    b.get_x() + b.get_width() / 2.0,
                    v + s + 1.4,
                    f"{v:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    fontweight="bold" if vk == "PRISM_CPI_Full" else "normal",
                    color=PALETTE["prism_dark"] if vk == "PRISM_CPI_Full" else PALETTE["slate_dark"],
                    zorder=5,
                )

    gains_pp = [
        ds_metrics[ds]["PRISM_CPI_Full"]["CoCrystal_Contact_Hit10_Pct_mean"]
        - ds_metrics[ds]["DESTIN_Seed_Baseline"]["CoCrystal_Contact_Hit10_Pct_mean"]
        for ds in datasets
    ]
    ax_a.text(
        0.985,
        0.975,
        f"+{min(gains_pp):.1f} to +{max(gains_pp):.1f} pp\nvs. DESTIN ($p < 10^{{-4}}$)",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=6.9,
        fontweight="bold",
        color=PALETTE["prism_dark"],
        bbox=dict(boxstyle="round,pad=0.22", facecolor=PALETTE["prism_light"], edgecolor=PALETTE["prism"], lw=0.85, alpha=0.96),
        zorder=6,
    )

    ax_a.set_xticks(x_ds)
    ax_a.set_xticklabels(ds_labels, fontweight="bold", fontsize=8.6)
    ax_a.set_ylim(0.0, 122.0)
    ax_a.set_yticks([0, 20, 40, 60, 80, 100])
    ax_a.set_ylabel("Co-Crystal Contact Hit@10 (%) ($<4.0\\,\\mathrm{\\AA} \\uparrow$)", fontsize=8.9)
    ax_a.set_title("A  X-Ray Co-Crystal Contact Hit@10 (<4 Å)", loc="left", pad=9.5, fontsize=14.2, color=PALETTE["slate_dark"])
    ax_a.grid(axis="y", linestyle="--", zorder=0)
    style_borderless_legend(
        ax_a.legend(
            loc="upper left",
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["frame_border"],
            framealpha=0.95,
            fontsize=6.8,
            ncol=2,
            columnspacing=0.85,
            handletextpad=0.35,
            labelspacing=0.22,
        )
    )
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel B (Top-Right): Interatomic Crystal Distance vs. Bilinear Attention Decay
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs_top[0, 1])
    c_1m17 = cocrystal["EGFR_Erlotinib_1M17"]
    c_2pl0 = cocrystal["LCK_Imatinib_2PL0"]

    d_1m17 = np.array([c["crystal_distance_angstrom"] for c in c_1m17["contacts"]], dtype=float)
    w_1m17 = np.array([c["PRISM_Attention_Weight"] for c in c_1m17["contacts"]], dtype=float)
    d_2pl0 = np.array([c["crystal_distance_angstrom"] for c in c_2pl0["contacts"]], dtype=float)
    w_2pl0 = np.array([c["PRISM_Attention_Weight"] for c in c_2pl0["contacts"]], dtype=float)

    d_destin = np.concatenate([d_1m17, d_2pl0])
    w_destin = np.array(
        [c["DESTIN_Attention_Weight"] for c in c_1m17["contacts"]]
        + [c["DESTIN_Attention_Weight"] for c in c_2pl0["contacts"]],
        dtype=float,
    )

    # Subtle vertical physical-interaction regime tinting
    ax_b.axvspan(2.78, 3.26, facecolor="#FAF3F5", edgecolor="none", lw=0, alpha=0.72, zorder=0)
    ax_b.axvspan(3.26, 3.74, facecolor="#F2FAF8", edgecolor="none", lw=0, alpha=0.72, zorder=0)
    ax_b.axvline(3.26, color=PALETTE["frame_border"], linestyle=":", linewidth=1.0, zorder=1)
    ax_b.text(3.02, 0.187, "H-Bond / Salt-Bridge Regime", ha="center", va="bottom", fontsize=6.7, fontweight="bold", color=PALETTE["prism_dark"], zorder=4)
    ax_b.text(3.50, 0.187, "Hydrophobic / Cleft Packing", ha="center", va="bottom", fontsize=6.7, fontweight="bold", color=PALETTE["graphban_dark"], zorder=4)

    def _plot_linreg_with_ci(
        ax: plt.Axes,
        xs: np.ndarray,
        ys: np.ndarray,
        line_col: str,
        band_col: str,
        ls: str = "-",
        lw: float = 1.85,
        show_band: bool = True,
    ) -> None:
        res = stats.linregress(xs, ys)
        xg = np.linspace(float(xs.min()) - 0.02, float(xs.max()) + 0.02, 80)
        yh = res.intercept + res.slope * xg
        ax.plot(xg, yh, color=line_col, linestyle=ls, linewidth=lw, zorder=3)
        if show_band and len(xs) >= 4:
            dof = len(xs) - 2
            resid = ys - (res.intercept + res.slope * xs)
            s_err = float(np.sqrt(np.sum(resid ** 2) / dof))
            se_fit = s_err * np.sqrt(1.0 / len(xs) + ((xg - float(xs.mean())) ** 2) / float(np.sum((xs - xs.mean()) ** 2)))
            band_lo = np.maximum(yh - 1.65 * se_fit, 0.0)
            band_hi = np.minimum(yh + 1.65 * se_fit, 0.175)
            ax.fill_between(xg, band_lo, band_hi, facecolor=band_col, edgecolor="none", lw=0, alpha=0.14, zorder=2)

    _plot_linreg_with_ci(ax_b, d_1m17, w_1m17, PALETTE["prism_dark"], PALETTE["prism"], ls="-", lw=1.9, show_band=True)
    _plot_linreg_with_ci(ax_b, d_2pl0, w_2pl0, PALETTE["graphban_dark"], PALETTE["graphban"], ls="-", lw=1.9, show_band=True)
    _plot_linreg_with_ci(ax_b, d_destin, w_destin, PALETTE["destin_dark"], PALETTE["destin"], ls="--", lw=1.5, show_band=False)

    ax_b.scatter(
        d_1m17,
        w_1m17,
        marker="o",
        s=46,
        color=PALETTE["prism"],
        edgecolors=PALETTE["prism_dark"],
        linewidths=0.95,
        label="EGFR–Erlotinib (1M17, Type-I)",
        zorder=5,
    )
    ax_b.scatter(
        d_2pl0,
        w_2pl0,
        marker="D",
        s=42,
        color=PALETTE["graphban"],
        edgecolors=PALETTE["graphban_dark"],
        linewidths=0.95,
        label="LCK–Imatinib (2PL0, Type-II DFG-Out)",
        zorder=5,
    )
    ax_b.scatter(
        d_destin,
        w_destin,
        marker="s",
        s=28,
        color=PALETTE["destin"],
        edgecolors=PALETTE["destin_dark"],
        linewidths=0.75,
        alpha=0.88,
        label="DESTIN Seed Baseline (1M17 & 2PL0)",
        zorder=4,
    )

    # Non-overlapping pointer callouts for top crystallographic contacts
    callouts_b = [
        (2.88, 0.148, (2.792, 0.169), "Met793 ($2.88\\,\\mathrm{\\AA}$, #1)", PALETTE["prism_dark"], "arc3,rad=0.0"),
        (2.91, 0.152, (3.045, 0.163), "Met319 ($2.91\\,\\mathrm{\\AA}$, #1)", PALETTE["graphban_dark"], "arc3,rad=0.0"),
        (2.84, 0.138, (2.805, 0.068), "Thr316/Glu288 ($2.84\\,\\mathrm{\\AA}$)", PALETTE["graphban_dark"], "arc3,rad=0.0"),
        (2.96, 0.126, (2.970, 0.048), "Asp382 DFG ($2.96\\,\\mathrm{\\AA}$)", PALETTE["graphban_dark"], "arc3,rad=0.0"),
    ]
    for cx, cy, (tx, ty), c_txt, c_col, conn_sty in callouts_b:
        ax_b.annotate(
            c_txt,
            xy=(cx, cy),
            xytext=(tx, ty),
            ha="left",
            va="center",
            fontsize=6.25,
            fontweight="bold",
            color=c_col,
            arrowprops=dict(arrowstyle="->", color=c_col, lw=0.85, shrinkA=2, shrinkB=3, connectionstyle=conn_sty),
            bbox=dict(boxstyle="round,pad=0.13", facecolor="#FFFFFF", edgecolor=PALETTE["frame_border"], lw=0.6, alpha=0.94),
            zorder=6,
        )

    ax_b.text(
        0.985,
        0.725,
        f"1M17: $\\rho_{{\\mathrm{{dist}}}} = {c_1m17['distance_attn_spearman_rho']:.3f}$ ($p = {c_1m17['distance_attn_pvalue']:.1e}$)\n"
        f"2PL0: $\\rho_{{\\mathrm{{dist}}}} = {c_2pl0['distance_attn_spearman_rho']:.3f}$ ($p = {c_2pl0['distance_attn_pvalue']:.1e}$)",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=6.6,
        fontweight="bold",
        color=PALETTE["slate_dark"],
        bbox=dict(boxstyle="round,pad=0.19", facecolor=PALETTE["prism_light"], edgecolor=PALETTE["prism"], lw=0.80, alpha=0.96),
        zorder=6,
    )

    ax_b.set_xlim(2.78, 3.74)
    ax_b.set_ylim(0.0, 0.244)
    ax_b.set_yticks([0.00, 0.04, 0.08, 0.12, 0.16])
    ax_b.set_xlabel("X-Ray Heavy-Atom Contact Distance $d_{ab}$ ($\\mathrm{\\AA}$)", fontsize=8.9)
    ax_b.set_ylabel("Projected Contact Weight $(\\tilde{\\mathbf{I}}\\mathbf{A}_{\\mathrm{pkt}})_{a,\\ell}$ ($\\uparrow$)", fontsize=8.9)
    ax_b.set_title("B  Crystal Distance vs. Contact Attention", loc="left", pad=9.5, fontsize=14.2, color=PALETTE["slate_dark"])
    ax_b.grid(True, linestyle="--", zorder=0)
    style_borderless_legend(
        ax_b.legend(
            loc="upper center",
            bbox_to_anchor=(0.50, 1.00),
            ncol=2,
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["frame_border"],
            framealpha=0.95,
            fontsize=6.4,
            columnspacing=0.75,
            handletextpad=0.32,
            labelspacing=0.18,
        )
    )
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panels C & D: Co-crystal heavy-atom contact profiles
    # -------------------------------------------------------------------------
    complex_configs = [
        ("EGFR_Erlotinib_1M17", "C  Type-I Contacts: EGFR–Erlotinib (1M17)"),
        ("LCK_Imatinib_2PL0", "D  Type-II Contacts: LCK–Imatinib (2PL0)"),
    ]

    full_label_map = {
        ("EGFR_Erlotinib_1M17", 1): "Quinazoline N1 $\\leftrightarrow$ Met793 (Hinge)\n[Canonical H-Bond, $d = 2.88\\,\\mathrm{\\AA}$]",
        ("EGFR_Erlotinib_1M17", 2): "Quinazoline N3 $\\leftrightarrow$ Thr790/Gln791\n[Water Bridge, $d = 3.12\\,\\mathrm{\\AA}$]",
        ("EGFR_Erlotinib_1M17", 3): "Ethynylphenyl C$\\equiv$CH $\\leftrightarrow$ Thr790/Lys745\n[Back-Pocket Cleft, $d = 3.46\\,\\mathrm{\\AA}$]",
        ("EGFR_Erlotinib_1M17", 4): "Aniline NH $\\leftrightarrow$ Leu788/Glu762\n[$\\alpha$C-Helix Cleft, $d = 3.68\\,\\mathrm{\\AA}$]",
        ("EGFR_Erlotinib_1M17", 5): "Quinazoline C2-H $\\leftrightarrow$ Gln791 (Hinge)\n[Weak CH$\\cdots$O H-Bond, $d = 3.24\\,\\mathrm{\\AA}$]",
        ("EGFR_Erlotinib_1M17", 6): "Methoxyethoxy O12 $\\leftrightarrow$ Cys797/Asp800\n[Solvent-Rim Polar, $d = 3.54\\,\\mathrm{\\AA}$]",
        ("LCK_Imatinib_2PL0", 1): "Pyridine–Pyrimidine N1 $\\leftrightarrow$ Met319 (Hinge)\n[Canonical H-Bond, $d = 2.91\\,\\mathrm{\\AA}$]",
        ("LCK_Imatinib_2PL0", 2): "Benzamide NH $\\leftrightarrow$ Thr316/Glu288\n[Gatekeeper/$\\alpha$C H-Bond, $d = 2.84\\,\\mathrm{\\AA}$]",
        ("LCK_Imatinib_2PL0", 3): "Benzamide C=O $\\leftrightarrow$ Asp382 (DFG)\n[DFG-Out H-Bond, $d = 2.96\\,\\mathrm{\\AA}$]",
        ("LCK_Imatinib_2PL0", 4): "Tolyl Phenyl $\\leftrightarrow$ Ile314/Lys273\n[Selectivity Pocket, $d = 3.52\\,\\mathrm{\\AA}$]",
        ("LCK_Imatinib_2PL0", 5): "Piperazine $\\mathrm{N4^+}$ $\\leftrightarrow$ Ile361/His362\n[HRD Loop Salt-Bridge, $d = 3.08\\,\\mathrm{\\AA}$]",
        ("LCK_Imatinib_2PL0", 6): "Pyrimidine NH $\\leftrightarrow$ Thr316 ($\\mathrm{O\\gamma}$)\n[Gatekeeper H-Bond, $d = 3.18\\,\\mathrm{\\AA}$]",
    }

    for c_idx, (c_key, c_title) in enumerate(complex_configs):
        ax_cd = fig.add_subplot(gs_bot[0, c_idx])
        c_data = cocrystal[c_key]
        contacts = list(reversed(c_data["contacts"]))  # Rank #1 contact at top

        y_pos = np.arange(len(contacts))
        h_bar = 0.36
        prism_w = [c["PRISM_Attention_Weight"] for c in contacts]
        destin_w = [c["DESTIN_Attention_Weight"] for c in contacts]

        b_pr = ax_cd.barh(
            y_pos + h_bar / 2.0,
            prism_w,
            height=h_bar,
            color=PALETTE["prism"],
            edgecolor="none",
            linewidth=0,
            label=f"PRISM-CPI (Hit@10 = {c_data['prism_contact_hit_at_10_pct']:.0f}%)",
            zorder=3,
        )
        b_de = ax_cd.barh(
            y_pos - h_bar / 2.0,
            destin_w,
            height=h_bar,
            color=PALETTE["destin"],
            edgecolor="none",
            linewidth=0,
            label=f"DESTIN Seed (Hit@10 = {c_data['destin_contact_hit_at_10_pct']:.0f}%)",
            zorder=3,
        )

        y_labels = [full_label_map[(c_key, c["contact_rank"])] for c in contacts]

        for bar, c in zip(b_pr, contacts):
            w_val = bar.get_width()
            ax_cd.text(
                w_val + 0.003,
                bar.get_y() + bar.get_height() / 2.0,
                f"{w_val:.3f} (Rank #{c['PRISM_Attn_Rank']})",
                va="center",
                ha="left",
                fontsize=7.2,
                fontweight="bold",
                color=PALETTE["prism_dark"],
            )
        for bar, c in zip(b_de, contacts):
            w_val = bar.get_width()
            ax_cd.text(
                w_val + 0.003,
                bar.get_y() + bar.get_height() / 2.0,
                f"{w_val:.3f} (Rank #{c['DESTIN_Attn_Rank']})",
                va="center",
                ha="left",
                fontsize=7.0,
                color=PALETTE["destin_dark"],
            )

        ax_cd.set_yticks(y_pos)
        ax_cd.set_yticklabels(y_labels, fontsize=7.4, fontweight="bold")
        ax_cd.set_xlim(0.0, 0.245)
        ax_cd.set_xlabel("Projected Atom–Residue Contact Weight $(\\tilde{\\mathbf{I}}\\mathbf{A}_{\\mathrm{pkt}})_{a,\\ell}$ ($\\uparrow$)", fontsize=8.9)
        ax_cd.set_title(c_title, loc="left", pad=9.5, fontsize=14.2, color=PALETTE["slate_dark"])
        ax_cd.grid(axis="x", linestyle="--", zorder=0)
        style_borderless_legend(
            ax_cd.legend(
                loc="lower right",
                frameon=True,
                facecolor="#FFFFFF",
                edgecolor=PALETTE["frame_border"],
                framealpha=0.95,
                fontsize=7.2,
            )
        )
        ax_cd.spines["top"].set_visible(False)
        ax_cd.spines["right"].set_visible(False)

    save_dual_format(fig, "fig8_cocrystal_contact_verification", dpi=320, pad_inches=0.01)


# =============================================================================
# FIGURE 9: Exact Game-Theoretic SHAP Attribution (2^8 = 256 Coalitions)
# =============================================================================
def generate_fig9_shap_attribution(shap_data: Dict[str, Any]) -> None:
    channels = shap_data["channels"]
    macro_summary = shap_data["macro_channel_summary"]
    benchmarks = shap_data["benchmarks"]

    # Sort all 8 channels top-to-bottom (row 0 = highest macro_share_pct)
    sorted_idx = sorted(
        range(len(channels)),
        key=lambda idx: float(macro_summary[channels[idx]["id"]]["macro_share_pct"]),
        reverse=True,
    )
    sorted_ids = [channels[idx]["id"] for idx in sorted_idx]

    disp_label_full = {
        "SaProt_Pocket_Routing": "CC-PAE Pocket Routing",
        "Bilinear_Subsite_Map": "Full-Rank Subsite Map",
        "Global_Conformer_Skip": "Global Conformer Skip",
        "Induced_Fit_Shift": "Pocket Induced-Fit Shift",
        "SS_UAKD_Relational_Calib": "SS-UAKD 3D-kNN & $w_i^{\\mathrm{rel}}$",
        "ECDA_Domain_Alignment": "E-CDAN Alignment",
        "RBF_3D_Conformer_GAT": "Continuous $B=16$ RBF-GAT",
        "ProstT5_3Di_Gate": "ProstT5 3Di Gate",
    }
    disp_label_compact = {
        "SaProt_Pocket_Routing": "CC-PAE Routing",
        "Bilinear_Subsite_Map": "Subsite Map",
        "Global_Conformer_Skip": "Conformer Skip",
        "Induced_Fit_Shift": "Induced-Fit Shift",
        "SS_UAKD_Relational_Calib": "SS-UAKD $w_i^{\\mathrm{rel}}$",
        "ECDA_Domain_Alignment": "E-CDAN Align.",
        "RBF_3D_Conformer_GAT": "$B=16$ RBF-GAT",
        "ProstT5_3Di_Gate": "ProstT5 3Di Gate",
    }

    fig = plt.figure(figsize=(17.4, 5.55), dpi=320)
    gs = gridspec.GridSpec(1, 3, figure=fig, left=0.100, right=0.988, width_ratios=[1.05, 1.04, 1.02], wspace=0.32)

    y_rows = np.arange(len(sorted_ids), dtype=float)
    shared_ylim = (8.65, -1.05)  # Row 0 at top, with clean top legend band and bottom badge band

    # -------------------------------------------------------------------------
    # Panel A (gs[0, 0]): Cross-Benchmark Global Mean |SHAP| Share (%)
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ds_order = [
        ("BindingDB", "BindingDB", PALETTE["prism"]),
        ("BioSNAP", "BioSNAP", PALETTE["destin"]),
        ("PDBbind_2016", "PDBbind 2016", PALETTE["graphban"]),
        ("KIBA", "KIBA", PALETTE["drugban"]),
    ]

    per_ds_share: Dict[str, Dict[str, float]] = {}
    for ds_key, _, _ in ds_order:
        per_ds_share[ds_key] = {
            ch_entry["channel_id"]: float(ch_entry["importance_share_pct"])
            for ch_entry in benchmarks[ds_key]["channels"]
        }

    h_bar = 0.185
    for d_i, (ds_key, ds_lbl, ds_col) in enumerate(ds_order):
        offset = (d_i - 1.5) * h_bar
        vals = [per_ds_share[ds_key][cid] for cid in sorted_ids]
        ax_a.barh(
            y_rows + offset,
            vals,
            height=h_bar,
            color=ds_col,
            edgecolor="none",
            linewidth=0,
            label=ds_lbl,
            zorder=3,
        )

    for r_i, cid in enumerate(sorted_ids):
        max_ds_val = max(per_ds_share[ds_key][cid] for ds_key, _, _ in ds_order)
        macro_pct = float(macro_summary[cid]["macro_share_pct"])
        lbl_str = f"Macro {macro_pct:.1f}%" if r_i == 0 else f"{macro_pct:.1f}%"
        ax_a.text(
            max_ds_val + 0.75,
            y_rows[r_i],
            lbl_str,
            va="center",
            ha="left",
            fontsize=7.3,
            fontweight="bold",
            color=PALETTE["prism_dark"] if r_i < 2 else PALETTE["slate_dark"],
            path_effects=[patheffects.withStroke(linewidth=2.0, foreground="#FFFFFF")],
            zorder=5,
        )

    pc_ifga_tot = sum(
        float(macro_summary[cid]["macro_share_pct"])
        for cid in ("Bilinear_Subsite_Map", "Induced_Fit_Shift", "RBF_3D_Conformer_GAT")
    )
    cc_pae_tot = sum(
        float(macro_summary[cid]["macro_share_pct"])
        for cid in ("SaProt_Pocket_Routing", "ProstT5_3Di_Gate")
    )
    ss_uakd_tot = sum(
        float(macro_summary[cid]["macro_share_pct"])
        for cid in ("SS_UAKD_Relational_Calib", "ECDA_Domain_Alignment")
    )
    skip_tot = float(macro_summary["Global_Conformer_Skip"]["macro_share_pct"])

    ax_a.text(
        0.985,
        0.03,
        f"Module Share Totals:\n"
        f"PC-IFGA: {pc_ifga_tot:.1f}%  |  CC-PAE: {cc_pae_tot:.1f}%\n"
        f"SS-UAKD: {ss_uakd_tot:.1f}%  |  Global Skip: {skip_tot:.1f}%",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.85,
        fontweight="bold",
        color=PALETTE["slate_dark"],
        bbox=dict(
            boxstyle="round,pad=0.26",
            facecolor=PALETTE["mod1_bg"],
            edgecolor=PALETTE["mod1_border"],
            lw=0.85,
            alpha=0.96,
        ),
        zorder=6,
    )

    ax_a.set_yticks(y_rows)
    ax_a.set_yticklabels([disp_label_full[cid] for cid in sorted_ids], fontsize=7.9, fontweight="bold")
    ax_a.set_ylim(*shared_ylim)
    ax_a.set_xlim(0.0, 48.0)
    ax_a.set_xticks([0, 10, 20, 30, 40])
    ax_a.set_xlabel("Channel Mean $|\\mathrm{SHAP}|$ Share (%)")
    ax_a.set_title("A  Global Mean |SHAP| Share (%)", loc="left", pad=9.5, fontsize=14.2, color=PALETTE["slate_dark"])
    ax_a.grid(axis="x", linestyle="--", zorder=0)
    style_borderless_legend(
        ax_a.legend(
            loc="upper center",
            bbox_to_anchor=(0.50, 1.00),
            ncol=4,
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["frame_border"],
            framealpha=0.95,
            fontsize=6.7,
            columnspacing=0.75,
            handletextpad=0.35,
            handlelength=1.15,
        )
    )
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel B (gs[0, 1]): Inductive SHAP Beeswarm Summary (BioSNAP, N = 400)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    bee = shap_data["beeswarm_biosnap"]
    shap_mat = np.asarray(bee["shap_values"], dtype=float)
    quant_mat = np.asarray(bee["feature_quantiles"], dtype=float)
    biosnap_meta = benchmarks["BioSNAP"]
    rng = np.random.default_rng(42)

    ax_b.axvline(0.0, color=PALETTE["slate_mid"], linestyle="--", linewidth=1.15, zorder=2)
    for y_r in y_rows:
        ax_b.axhline(y_r, color="#E6ECF2", linestyle="-", linewidth=0.55, zorder=1)

    sc_handle = None
    for r_i, ch_i in enumerate(sorted_idx):
        sv_col = shap_mat[:, ch_i]
        fq_col = np.clip(quant_mat[:, ch_i], 0.0, 1.0)

        # Density-aware deterministic beeswarm vertical jitter within +/- 0.27
        counts, bin_edges = np.histogram(sv_col, bins=26)
        bin_ids = np.clip(np.digitize(sv_col, bin_edges[:-1]) - 1, 0, len(counts) - 1)
        rel_density = counts[bin_ids] / max(1.0, float(counts.max()))
        jitter = rng.uniform(-1.0, 1.0, size=len(sv_col)) * (0.045 + 0.225 * rel_density)

        # Sort points by distance from median
        draw_ord = np.argsort(np.abs(fq_col - 0.5))
        sc_handle = ax_b.scatter(
            sv_col[draw_ord],
            np.full_like(sv_col[draw_ord], y_rows[r_i]) + jitter[draw_ord],
            c=fq_col[draw_ord],
            cmap=REF_SHAP_CMAP,
            vmin=0.0,
            vmax=1.0,
            s=12,
            alpha=0.82,
            linewidths=0.0,
            edgecolors="none",
            zorder=4,
        )

    # Top header note inside top headroom band
    ax_b.text(
        0.50,
        0.975,
        f"BioSNAP Held-Out Target-Test ($N = {bee['n_samples']}$, 200 Pos / 200 Neg, $\\mathrm{{AUC}} = {biosnap_meta['out_of_sample_target_test_auc']:.4f}$)",
        transform=ax_b.transAxes,
        ha="center",
        va="top",
        fontsize=6.7,
        fontweight="bold",
        color=PALETTE["destin_dark"],
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#FFFFFF", edgecolor=PALETTE["frame_border"], lw=0.7, alpha=0.95),
        zorder=6,
    )

    # Horizontal feature-value colorbar
    cax = ax_b.inset_axes([0.04, 0.058, 0.41, 0.030])
    cax.imshow(np.linspace(0.0, 1.0, 256).reshape(1, -1), aspect="auto", cmap=REF_SHAP_CMAP, extent=[0.0, 1.0, 0.0, 1.0])
    cax.set_yticks([])
    cax.set_xticks([0.0, 0.5, 1.0])
    cax.set_xticklabels(["Low", "Mid", "High"], fontsize=7.0, fontweight="bold", color=PALETTE["slate_dark"])
    cax.tick_params(axis="x", length=2.0, pad=1.5, width=0.6, colors=PALETTE["slate_dark"])
    for sp_cb in cax.spines.values():
        sp_cb.set_edgecolor(PALETTE["frame_border"])
        sp_cb.set_linewidth(0.6)
    cax.set_title("Channel Activation (Low → High)", fontsize=7.0, fontweight="bold", pad=2.5, color=PALETTE["slate_dark"])

    # Exact coalitions & additivity badge in bottom-right footer band
    max_add_err = max(float(benchmarks[d]["max_additivity_error"]) for d in benchmarks)
    ax_b.text(
        0.985,
        0.025,
        f"Exact $2^8 = 256$ Coalitions\nMax Additivity Err $= {max_add_err:.2e}$",
        transform=ax_b.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.65,
        fontweight="bold",
        color=PALETTE["prism_dark"],
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor=PALETTE["prism_light"],
            edgecolor=PALETTE["prism"],
            lw=0.80,
            alpha=0.96,
        ),
        zorder=6,
    )

    ax_b.set_yticks(y_rows)
    ax_b.set_yticklabels([disp_label_compact[cid] for cid in sorted_ids], fontsize=7.6, fontweight="bold")
    ax_b.set_ylim(*shared_ylim)
    ax_b.set_xlim(-0.34, 0.25)
    ax_b.set_xticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2])
    ax_b.set_xlabel("Exact Shapley Contribution $\\phi_j(x)$ to Interaction Probability")
    ax_b.set_title("B  SHAP Beeswarm Summary (BioSNAP)", loc="left", pad=9.5, fontsize=14.2, color=PALETTE["slate_dark"])
    ax_b.grid(axis="x", linestyle="--", zorder=0)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel C (gs[0, 2]): Single-Complex Decision Waterfall (Hit vs. Decoy)
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    wf_bio = shap_data["waterfall_examples"]["BioSNAP"]
    hit_ex = wf_bio["active_hit"]
    dec_ex = wf_bio["inactive_decoy"]
    base_val = float(hit_ex["base_value"])

    hit_deltas = [float(hit_ex["shap_contributions"][cid]) for cid in sorted_ids]
    dec_deltas = [float(dec_ex["shap_contributions"][cid]) for cid in sorted_ids]
    hit_cum = np.cumsum([base_val] + hit_deltas)
    dec_cum = np.cumsum([base_val] + dec_deltas)

    ax_c.axvline(base_val, color=PALETTE["slate_mid"], linestyle="--", linewidth=1.25, zorder=2)
    for y_r in y_rows:
        ax_c.axhline(y_r, color="#E6ECF2", linestyle="-", linewidth=0.55, zorder=1)

    # Borderless horizontal waterfall delta bars for each channel step
    bar_h_wf = 0.44
    for r_i in range(len(sorted_ids)):
        # Active Hit step bar
        h_start, h_d = float(hit_cum[r_i]), float(hit_deltas[r_i])
        ax_c.barh(
            y_rows[r_i],
            h_d,
            left=h_start,
            height=bar_h_wf,
            color=PALETTE["prism"],
            alpha=0.88,
            edgecolor="none",
            linewidth=0,
            label="Active Hit $\\Delta\\phi_j$ ($y = 1$)" if r_i == 0 else None,
            zorder=3,
        )
        # Inactive Decoy step bar
        d_start, d_d = float(dec_cum[r_i]), float(dec_deltas[r_i])
        ax_c.barh(
            y_rows[r_i],
            d_d,
            left=d_start,
            height=bar_h_wf,
            color=PALETTE["destin"],
            alpha=0.88,
            edgecolor="none",
            linewidth=0,
            label="Inactive Decoy $\\Delta\\phi_j$ ($y = 0$)" if r_i == 0 else None,
            zorder=3,
        )

        # Annotate major channel contributions cleanly without clutter
        if abs(h_d) >= 0.025:
            x_txt_h = max(h_start, h_start + h_d) + 0.015
            ax_c.text(
                x_txt_h,
                y_rows[r_i],
                f"{h_d:+.3f}",
                va="center",
                ha="left",
                fontsize=6.6,
                fontweight="bold",
                color=PALETTE["prism_dark"],
                path_effects=[patheffects.withStroke(linewidth=1.8, foreground="#FFFFFF")],
                zorder=6,
            )
        if abs(d_d) >= 0.055:
            x_txt_d = min(d_start, d_start + d_d) - 0.015
            ax_c.text(
                x_txt_d,
                y_rows[r_i],
                f"{d_d:+.3f}",
                va="center",
                ha="right",
                fontsize=6.6,
                fontweight="bold",
                color=PALETTE["destin_dark"],
                path_effects=[patheffects.withStroke(linewidth=1.8, foreground="#FFFFFF")],
                zorder=6,
            )

    # Cumulative trajectory backbone curves connecting E[f(X)] through all 8 channel updates
    y_traj = np.concatenate([[-0.44], y_rows])
    ax_c.plot(
        hit_cum,
        y_traj,
        marker="o",
        markersize=3.8,
        linewidth=1.45,
        color=PALETTE["prism_dark"],
        markerfacecolor=PALETTE["prism_light"],
        markeredgewidth=1.0,
        zorder=5,
    )
    ax_c.plot(
        dec_cum,
        y_traj,
        marker="s",
        markersize=3.6,
        linewidth=1.45,
        color=PALETTE["destin_dark"],
        markerfacecolor=PALETTE["destin_light"],
        markeredgewidth=1.0,
        zorder=5,
    )

    # Prior baseline pill above Row 0 (shifted left so legend sits cleanly at upper-right)
    ax_c.text(
        base_val - 0.02,
        -0.56,
        f"$\\mathbb{{E}}[f(X)] = {base_val:.3f}$",
        ha="right",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=PALETTE["slate_dark"],
        bbox=dict(boxstyle="round,pad=0.18", facecolor=PALETTE["slate_light"], edgecolor=PALETTE["frame_border"], lw=0.7),
        zorder=6,
    )

    # Endpoint prediction labels
    hit_final = float(hit_ex["final_prediction"])
    dec_final = float(dec_ex["final_prediction"])
    ax_c.annotate(
        f"$f(x_{{\\mathrm{{decoy}}}}) = {dec_final:.3f}$\n($L = {dec_ex['seq_length']}$, $y = 0$)",
        xy=(dec_final, 7.22),
        xytext=(0.22, 8.05),
        ha="center",
        va="center",
        fontsize=6.65,
        fontweight="bold",
        color=PALETTE["destin_dark"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["destin_dark"], lw=0.95, shrinkA=2, shrinkB=2),
        bbox=dict(boxstyle="round,pad=0.22", facecolor=PALETTE["destin_light"], edgecolor=PALETTE["destin"], lw=0.85, alpha=0.96),
        zorder=6,
    )
    ax_c.annotate(
        f"$f(x_{{\\mathrm{{hit}}}}) = {hit_final:.3f}$\n($L = {hit_ex['seq_length']}$, $y = 1$)",
        xy=(hit_final, 7.22),
        xytext=(0.79, 8.05),
        ha="center",
        va="center",
        fontsize=6.65,
        fontweight="bold",
        color=PALETTE["prism_dark"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["prism_dark"], lw=0.95, shrinkA=2, shrinkB=2),
        bbox=dict(boxstyle="round,pad=0.22", facecolor=PALETTE["prism_light"], edgecolor=PALETTE["prism"], lw=0.85, alpha=0.96),
        zorder=6,
    )

    ax_c.set_yticks(y_rows)
    ax_c.set_yticklabels([disp_label_compact[cid] for cid in sorted_ids], fontsize=7.6, fontweight="bold")
    ax_c.set_ylim(*shared_ylim)
    ax_c.set_xlim(-0.11, 1.09)
    ax_c.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax_c.set_xlabel("Cumulative Predicted Interaction Probability $f(x)$")
    ax_c.set_title("C  Single-Complex SHAP Waterfall", loc="left", pad=9.5, fontsize=14.2, color=PALETTE["slate_dark"])
    ax_c.grid(axis="x", linestyle="--", zorder=0)
    style_borderless_legend(
        ax_c.legend(
            loc="upper right",
            bbox_to_anchor=(0.995, 1.00),
            ncol=1,
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["frame_border"],
            framealpha=0.95,
            fontsize=6.45,
            labelspacing=0.18,
            handlelength=1.15,
        )
    )
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    save_dual_format(fig, "fig9_shap_attribution", dpi=320, pad_inches=0.01)


# =============================================================================
# FIGURE 10: Cold-Start Bipartite Network, Vina Validation & 3D/2D Pocket Renders
# =============================================================================
def generate_fig10_network_and_3d_docking(
    dock_data: Dict[str, Any],
    docking_30_rows: Optional[List[Dict[str, Any]]] = None,
) -> None:
    fig = plt.figure(figsize=(17.6, 10.2), dpi=320)
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.024, right=0.986, hspace=0.27, wspace=0.16)

    # -------------------------------------------------------------------------
    # Panel A (gs[0, 0]): Cold-Start CPI Prediction Network & Vina Validation
    # -------------------------------------------------------------------------
    ax_a_frame = fig.add_subplot(gs[0, 0])
    ax_a_frame.axis("off")
    ax_a_frame.set_title(
        "A  Cold-Start CPI Network & Vina Validation",
        loc="left",
        pad=10.2,
        fontsize=14.8,
        color=PALETTE["slate_dark"],
    )

    gs_a = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=gs[0, 0],
        width_ratios=[1.45, 1.0],
        wspace=0.20,
    )

    # ---- Left Sub-Axis: Layered Bipartite CPI & Polypharmacology Network ----
    ax_net = fig.add_subplot(gs_a[0, 0])
    ax_net.set_xlim(0.0, 1.0)
    ax_net.set_ylim(0.0, 1.0)
    ax_net.axis("off")

    net = dock_data["bipartite_cpi_network"]
    edges = net["edges"]

    # Soft color palette bands (EGFR, NMDA 2B, MMP-2)
    family_bands = [
        (0.625, 0.905, PALETTE["terra_bg"], PALETTE["terra_border"]),
        (0.320, 0.600, PALETTE["mod3_bg"], PALETTE["mod3_border"]),
        (0.015, 0.295, PALETTE["mod2_bg"], PALETTE["mod2_border"]),
    ]
    for yb0, yb1, b_fc, b_ec in family_bands:
        ax_net.add_patch(
            FancyBboxPatch(
                (0.008, yb0),
                0.984,
                yb1 - yb0,
                boxstyle="round,pad=0.004,rounding_size=0.018",
                facecolor=b_fc,
                edgecolor=b_ec,
                linewidth=0.75,
                alpha=0.88,
                zorder=0,
            )
        )

    # Top edge-type legend banner 
    from matplotlib.lines import Line2D
    edge_leg_handles = [
        Line2D([0], [0], color=PALETTE["prism_dark"], linestyle="-", linewidth=1.85, label="Primary (p ≥ 0.96)"),
        Line2D([0], [0], color=PALETTE["amber_dark"], linestyle="--", linewidth=1.35, label="Polypharm."),
        Line2D([0], [0], color=PALETTE["slate_mid"], linestyle=":", linewidth=1.05, label="Decoy"),
    ]
    style_borderless_legend(
        ax_net.legend(
            handles=edge_leg_handles,
            loc="upper center",
            bbox_to_anchor=(0.50, 1.008),
            ncol=3,
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["frame_border"],
            framealpha=0.95,
            fontsize=7.1,
            columnspacing=0.55,
            handlelength=1.15,
            handletextpad=0.28,
            borderpad=0.24,
        )
    )

    # Node layout coordinates
    x_hit = 0.270
    x_hub = 0.562
    x_dec = 0.770
    hub_half_w = 0.098

    egfr_hits = ["Afatinib", "Dacomitinib", "Osimertinib", "Lapatinib", "Neratinib"]
    nmda_hits = ["Ifenprodil", "Traxoprodil", "Ro 25-6981", "Eliprodil", "Rislenemdaz"]
    mmp2_hits = ["Prinomastat", "Ilomastat", "Batimastat", "Marimastat", "Tanomastat"]

    node_pos: Dict[str, Tuple[float, float]] = {}
    hit_family: Dict[str, str] = {}
    for fam_key, h_list, y_top, y_bot in [
        ("EGFR", egfr_hits, 0.876, 0.654),
        ("NMDA_2B", nmda_hits, 0.571, 0.349),
        ("MMP2", mmp2_hits, 0.266, 0.044),
    ]:
        for h_name, y_val in zip(h_list, np.linspace(y_top, y_bot, 5)):
            node_pos[h_name] = (x_hit, float(y_val))
            hit_family[h_name] = fam_key

    # 6 Target Hubs (Primary Unseen Hub on upper row of band, Homolog Off-Target Hub on lower row)
    target_hub_specs = {
        "EGFR": (x_hub, 0.825, "EGFR\n(4G5J)", PALETTE["prism_dark"], "#FFFFFF", PALETTE["prism_dark"], True),
        "ERBB2": (x_hub, 0.692, "HER2\n(ERBB2)", PALETTE["ochre_sub"], PALETTE["amber_dark"], PALETTE["amber_dark"], False),
        "NMDA_2B": (x_hub, 0.520, "NMDA 2B\n(5EWJ)", PALETTE["drugban_dark"], "#FFFFFF", PALETTE["drugban_dark"], True),
        "GRIN2A": (x_hub, 0.387, "GluN2A\n(GRIN2A)", PALETTE["slate_light"], PALETTE["slate_dark"], PALETTE["slate_mid"], False),
        "MMP2": (x_hub, 0.215, "MMP-2\n(1HOV)", PALETTE["graphban_dark"], "#FFFFFF", PALETTE["graphban_dark"], True),
        "MMP9": (x_hub, 0.082, "MMP-9\n(Homolog)", PALETTE["ochre_sub"], PALETTE["amber_dark"], PALETTE["amber_dark"], False),
    }
    for t_id, (tx, ty, _, _, _, _, _) in target_hub_specs.items():
        node_pos[t_id] = (tx, ty)

    decoy_specs = [
        ("EGFR_Decoy1", x_dec, 0.858, "Decoy 1"),
        ("EGFR_Decoy2", x_dec, 0.795, "Decoy 2"),
        ("NMDA_2B_Decoy1", x_dec, 0.553, "Decoy 1"),
        ("NMDA_2B_Decoy2", x_dec, 0.490, "Decoy 2"),
        ("MMP2_Decoy1", x_dec, 0.248, "Decoy 1"),
        ("MMP2_Decoy2", x_dec, 0.185, "Decoy 2"),
    ]
    for d_id, dx, dy, _ in decoy_specs:
        node_pos[d_id] = (dx, dy)

    primary_edge_colors = {
        "EGFR": PALETTE["prism_dark"],
        "NMDA_2B": PALETTE["drugban_dark"],
        "MMP2": PALETTE["graphban_dark"],
    }

    # Draw all 36 bipartite edges cleanly to hub borders
    for e in edges:
        src = e["source"]
        tgt = e["target"]
        etype = e["edge_type"]
        sx, sy = node_pos[src]
        tx, ty = node_pos[tgt]
        if sx < tx:
            x0, x1 = sx + 0.012, tx - hub_half_w
        else:
            x0, x1 = sx - 0.012, tx + hub_half_w

        if etype == "verified_primary_hit":
            ecol = primary_edge_colors.get(tgt, PALETTE["prism_dark"])
            ax_net.plot([x0, x1], [sy, ty], color=ecol, linestyle="-", linewidth=1.8, alpha=0.88, zorder=3)
        elif etype == "verified_polypharmacology":
            ax_net.plot([x0, x1], [sy, ty], color=PALETTE["graphsformer"], linestyle="--", linewidth=1.3, alpha=0.86, zorder=2)
        else:
            ax_net.plot([x0, x1], [sy, ty], color=PALETTE["slate_mid"], linestyle=":", linewidth=0.95, alpha=0.72, zorder=1)

    # Draw 15 Clinical Hit ligand nodes + right-aligned labels in clean left margin
    fam_dot_color = {
        "EGFR": PALETTE["prism"],
        "NMDA_2B": PALETTE["drugban"],
        "MMP2": PALETTE["graphban"],
    }
    fam_edge_color = {
        "EGFR": PALETTE["prism_dark"],
        "NMDA_2B": PALETTE["drugban_dark"],
        "MMP2": PALETTE["graphban_dark"],
    }
    for h_name, fam_key in hit_family.items():
        hx, hy = node_pos[h_name]
        ax_net.scatter(
            [hx],
            [hy],
            s=36,
            color=fam_dot_color[fam_key],
            edgecolors=fam_edge_color[fam_key],
            linewidths=0.9,
            zorder=5,
        )
        ax_net.text(
            hx - 0.020,
            hy,
            h_name,
            ha="right",
            va="center",
            fontsize=7.1,
            fontweight="bold",
            color=PALETTE["slate_dark"],
            zorder=6,
        )

    # Draw 6 Cold-Start Decoy nodes + right-side labels
    for d_id, dx, dy, d_lbl in decoy_specs:
        ax_net.scatter(
            [dx],
            [dy],
            s=26,
            marker="s",
            color=PALETTE["other_base"],
            edgecolors=PALETTE["slate_mid"],
            linewidths=0.75,
            zorder=5,
        )
        ax_net.text(
            dx + 0.020,
            dy,
            d_lbl,
            ha="left",
            va="center",
            fontsize=7.1,
            color=PALETTE["slate_mid"],
            fontweight="bold",
            zorder=6,
        )

    # Draw 6 Target Hub badges + Homolog selectivity annotations on right
    for t_id, (tx, ty, t_lbl, t_fc, t_tc, t_ec, is_prim) in target_hub_specs.items():
        hw = hub_half_w
        hh = 0.046 if is_prim else 0.044
        ax_net.add_patch(
            FancyBboxPatch(
                (tx - hw, ty - hh),
                2.0 * hw,
                2.0 * hh,
                boxstyle="round,pad=0.004,rounding_size=0.016",
                facecolor=t_fc,
                edgecolor=t_ec,
                linewidth=1.15 if is_prim else 0.95,
                zorder=6,
            )
        )
        ax_net.text(
            tx,
            ty,
            t_lbl,
            ha="center",
            va="center",
            fontsize=7.2 if is_prim else 7.1,
            fontweight="bold",
            color=t_tc,
            linespacing=1.04,
            zorder=7,
        )

    # Right-column homolog selectivity notes 
    homolog_notes = [
        (0.678, 0.692, "Pan-HER (4/5)\nOsim. p = 0.41", PALETTE["amber_dark"]),
        (0.678, 0.387, "5/5 Spared\n(p ≤ 0.165)", PALETTE["drugban_dark"]),
        (0.678, 0.082, "Dual Zn(II)\nInhib. (5/5)", PALETTE["graphban_dark"]),
    ]
    for nx_pos, ny_pos, n_txt, n_col in homolog_notes:
        ax_net.text(
            nx_pos,
            ny_pos,
            n_txt,
            ha="left",
            va="center",
            fontsize=7.1,
            fontweight="bold",
            color=n_col,
            linespacing=1.06,
            zorder=6,
        )

    # ---- Right Sub-Axis: AutoDock Vina Delta-G Hit vs. Decoy Comparison (N = 30) ----
    ax_ar = fig.add_subplot(gs_a[0, 1])
    per_t = dock_data["per_target_statistics"]
    pooled = dock_data["pooled_statistics"]
    t_keys = ["EGFR", "NMDA_2B", "MMP2"]
    t_disp = ["EGFR\n(4G5J)", "NMDA 2B\n(5EWJ)", "MMP-2\n(1HOV)"]
    x_t = np.arange(len(t_keys), dtype=float)
    w_v = 0.32

    hit_means_pos = [-float(per_t[tk]["hits_mean_dg_kcal_mol"]) for tk in t_keys]
    hit_sds = [float(per_t[tk]["hits_sd_dg_kcal_mol"]) for tk in t_keys]
    dec_means_pos = [-float(per_t[tk]["decoys_mean_dg_kcal_mol"]) for tk in t_keys]
    dec_sds = [float(per_t[tk]["decoys_sd_dg_kcal_mol"]) for tk in t_keys]

    bars_hit = ax_ar.bar(
        x_t - w_v / 2.0,
        hit_means_pos,
        w_v,
        yerr=hit_sds,
        color=PALETTE["prism"],
        edgecolor="none",
        linewidth=0,
        capsize=2.2,
        error_kw=ERROR_KW_SLENDER,
        label=f"Clinical Hits (n = {pooled['n_clinical_hits']})",
        zorder=3,
    )
    bars_dec = ax_ar.bar(
        x_t + w_v / 2.0,
        dec_means_pos,
        w_v,
        yerr=dec_sds,
        color=PALETTE["destin"],
        alpha=0.82,
        edgecolor="none",
        linewidth=0,
        capsize=2.2,
        error_kw=ERROR_KW_SLENDER,
        label=f"Cold-Start Decoys (n = {pooled['n_cold_start_decoys']})",
        zorder=3,
    )

    # Gather exact individual complex Vina scores (15 hits + 15 decoys)
    indiv_hits: Dict[str, List[float]] = {
        tk: [-float(h["docking_score_kcal_mol"]) for h in per_t[tk]["hits_table"]]
        for tk in t_keys
    }
    indiv_decs: Dict[str, List[float]] = {tk: [] for tk in t_keys}
    if docking_30_rows:
        for r in docking_30_rows:
            if r.get("role") == "cold_start_decoy" and r.get("target") in indiv_decs:
                indiv_decs[r["target"]].append(-float(r["docking_score_kcal_mol"]))
    else:
        for e in edges:
            if e["edge_type"] == "rejected_decoy" and e["vina_dg"] is not None:
                indiv_decs[e["target"]].append(-float(e["vina_dg"]))

    jit_offsets = np.linspace(-0.085, 0.085, 5)
    for idx_t, tk in enumerate(t_keys):
        h_pts = sorted(indiv_hits[tk])
        ax_ar.scatter(
            np.full(len(h_pts), x_t[idx_t] - w_v / 2.0) + jit_offsets[: len(h_pts)],
            h_pts,
            s=17,
            color=PALETTE["prism_dark"],
            alpha=0.88,
            edgecolors="#FFFFFF",
            linewidths=0.45,
            zorder=4,
        )
        d_pts = sorted(indiv_decs[tk])
        if d_pts:
            ax_ar.scatter(
                np.full(len(d_pts), x_t[idx_t] + w_v / 2.0) + jit_offsets[: len(d_pts)],
                d_pts,
                s=16,
                color=PALETTE["destin_dark"],
                alpha=0.85,
                edgecolors="#FFFFFF",
                linewidths=0.45,
                zorder=4,
            )

        # Signed kcal/mol value labels cleanly above both errorbar caps & highest complex dot
        top_h = max([hit_means_pos[idx_t] + hit_sds[idx_t]] + h_pts)
        ax_ar.text(
            bars_hit[idx_t].get_x() + bars_hit[idx_t].get_width() / 2.0,
            top_h + 0.26,
            f"−{hit_means_pos[idx_t]:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color=PALETTE["prism_dark"],
            path_effects=[patheffects.withStroke(linewidth=2.0, foreground="#FFFFFF")],
            zorder=6,
        )
        top_d = max([dec_means_pos[idx_t] + dec_sds[idx_t]] + d_pts)
        ax_ar.text(
            bars_dec[idx_t].get_x() + bars_dec[idx_t].get_width() / 2.0,
            top_d + 0.26,
            f"−{dec_means_pos[idx_t]:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color=PALETTE["destin_dark"],
            path_effects=[patheffects.withStroke(linewidth=2.0, foreground="#FFFFFF")],
            zorder=6,
        )

    ax_ar.text(
        0.50,
        0.840,
        f"Pooled: −{abs(pooled['hits_mean_dg_kcal_mol']):.2f} vs. −{abs(pooled['decoys_mean_dg_kcal_mol']):.2f} kcal/mol\n"
        f"ΔΔG = −{abs(pooled['mean_ddg_advantage_kcal_mol']):.2f} kcal/mol (MW p = 5.25e-4)\n"
        f"Spearman ρ = {pooled['spearman_rho_neg_dg_vs_prism']:.3f} (N = 30)",
        transform=ax_ar.transAxes,
        ha="center",
        va="top",
        fontsize=7.1,
        fontweight="bold",
        color=PALETTE["slate_dark"],
        linespacing=1.15,
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor=PALETTE["prism_light"],
            edgecolor=PALETTE["prism"],
            lw=0.80,
            alpha=0.96,
        ),
        zorder=6,
    )

    ax_ar.set_xticks(x_t)
    ax_ar.set_xticklabels(t_disp, fontsize=8.2, fontweight="bold")
    ax_ar.set_xlim(-0.54, 2.54)
    ax_ar.set_ylim(0.0, 19.8)
    ax_ar.set_yticks([0, 3, 6, 9, 12])
    ax_ar.set_ylabel("$-\\Delta G_{\\mathrm{Vina}}$ (kcal/mol, $\\uparrow$ stronger)", fontsize=10.3)
    ax_ar.grid(axis="y", linestyle="--", zorder=0)
    style_borderless_legend(
        ax_ar.legend(
            loc="upper center",
            bbox_to_anchor=(0.50, 1.002),
            ncol=1,
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["frame_border"],
            framealpha=0.95,
            fontsize=7.2,
            labelspacing=0.16,
            handlelength=1.15,
        )
    )
    ax_ar.spines["top"].set_visible(False)
    ax_ar.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panels B, C, D (gs[0, 1], gs[1, 0], gs[1, 1]): 3D PyMOL Pocket & 2D Atom Weights
    # -------------------------------------------------------------------------
    flagship = dock_data["flagship_3d_complexes"]
    egfr_f = flagship["EGFR"]
    nmda_f = flagship["NMDA_2B"]
    mmp2_f = flagship["MMP2"]

    panel_configs = [
        (
            gs[0, 1],
            "EGFR",
            "B  EGFR (4G5J): Afatinib 3D & 2D Attribution",
            PALETTE["terra_bg"],
            PALETTE["terra_border"],
            PALETTE["prism_dark"],
            (
                f"Vina ΔG = −{abs(egfr_f['vina_dg_kcal_mol']):.2f} kcal/mol | "
                f"Shift = {egfr_f['pocket_centroid_shift_A']:.2f} Å | "
                f"Pro794 (2.78 Å), Met766 (2.83 Å), Lys745 (3.51 Å), Met793/Cys797"
            ),
            "Quinazoline 0.92 · Phenolic 0.93 · Acrylamide 0.88",
        ),
        (
            gs[1, 0],
            "NMDA_2B",
            "C  NMDA 2B (5EWJ): ATD Cleft & 2D Attribution",
            PALETTE["mod3_bg"],
            PALETTE["mod3_border"],
            PALETTE["drugban_dark"],
            (
                f"Vina ΔG = −{abs(nmda_f['vina_dg_kcal_mol']):.2f} kcal/mol | "
                f"RMSD = {nmda_f['cocrystal_heavy_rmsd_A']:.2f} Å "
                f"(Shift {nmda_f['pocket_centroid_shift_A']:.2f} Å) | "
                f"Gln110 (1.98 Å), Ser132, Phe176, Glu236"
            ),
            "Phenolic OH 0.93 · Piperidine 0.82 · Benzyl 0.72",
        ),
        (
            gs[1, 1],
            "MMP2",
            "D  MMP-2 (1HOV): Catalytic Pocket & 2D Weights",
            PALETTE["mod2_bg"],
            PALETTE["mod2_border"],
            PALETTE["graphban_dark"],
            (
                f"Vina ΔG = −{abs(mmp2_f['vina_dg_kcal_mol']):.2f} kcal/mol | "
                f"Strain = {mmp2_f['strain_kcal_mol']:.2f} kcal/mol | "
                f"Zn166 (2.32 Å), Glu121 (2.78 Å), His120 (3.08 Å), His124/Tyr142"
            ),
            "Hydroxamate 0.96 · Sulfonamide 0.84 · Pyridyl 0.74",
        ),
    ]

    for sub_spec, f_key, p_title, strip_fc, card_ec, accent_tc, strip_txt, pharm_txt in panel_configs:
        f_info = flagship[f_key]

        # Outer rounded color palette card frame + panel title
        ax_card = fig.add_subplot(sub_spec)
        ax_card.set_xlim(0.0, 1.0)
        ax_card.set_ylim(0.0, 1.0)
        ax_card.axis("off")
        ax_card.set_title(p_title, loc="left", pad=10.2, fontsize=14.8, color=PALETTE["slate_dark"])
        ax_card.add_patch(
            FancyBboxPatch(
                (0.004, 0.004),
                0.992,
                0.992,
                boxstyle="round,pad=0.003,rounding_size=0.018",
                facecolor="#FFFFFF",
                edgecolor=card_ec,
                linewidth=1.0,
                zorder=0,
            )
        )
        # Subtle vertical divider between 3D PyMOL pocket and 2D RDKit atom weights
        ax_card.plot([0.568, 0.568], [0.145, 0.965], color="#E2E8F0", linewidth=0.85, linestyle="-", zorder=2)

        gs_inner = gridspec.GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec=sub_spec,
            width_ratios=[1.32, 1.0],
            height_ratios=[1.0, 0.138],
            wspace=0.04,
            hspace=0.02,
        )

        # Left sub-axis: Auto-cropped, HSV-clamped PyMOL 3D pocket render
        ax_3d = fig.add_subplot(gs_inner[0, 0])
        pymol_path = Path(f_info["pymol_png"])
        if not pymol_path.exists():
            pymol_path = ROOT_DIR / "docking_3d" / "pymol_sessions" / pymol_path.name
        img_3d = load_trimmed_image(pymol_path, pad=14, top_pad=105, bottom_pad=12, min_aspect_wh=0.0, max_sat=0.64)
        ax_3d.imshow(img_3d, aspect="equal")
        ax_3d.axis("off")
        ax_3d.text(
            0.025,
            0.975,
            "PyMOL 3D Docked Pose vs. X-Ray Ref",
            transform=ax_3d.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            fontweight="bold",
            color=accent_tc,
            bbox=dict(boxstyle="round,pad=0.20", facecolor=strip_fc, edgecolor=card_ec, lw=0.75, alpha=0.95),
            zorder=5,
        )

        # Right sub-axis: Auto-cropped, HSV-clamped RDKit 2D bilinear atom-weight structure
        ax_2d = fig.add_subplot(gs_inner[0, 1])
        rdkit_path = Path(f_info["rdkit_2d_png"])
        if not rdkit_path.exists():
            rdkit_path = ROOT_DIR / "docking_3d" / "rdkit_2d_weights" / rdkit_path.name
        img_2d = load_trimmed_image(rdkit_path, pad=14, top_pad=84, bottom_pad=162, min_aspect_wh=0.92, max_sat=0.64)
        ax_2d.imshow(img_2d, aspect="equal")
        ax_2d.axis("off")
        ax_2d.text(
            0.03,
            0.975,
            f"PRISM-CPI 2D Atom Weights ({f_info['drug_name']})",
            transform=ax_2d.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            fontweight="bold",
            color=accent_tc,
            bbox=dict(boxstyle="round,pad=0.20", facecolor=strip_fc, edgecolor=card_ec, lw=0.75, alpha=0.95),
            zorder=5,
        )

        # Compact horizontal thermal attribution colorbar matching exact RDKit ramp [0.45, 0.95]
        drug_cmap_stops = {
            "Afatinib": [(0.0, "#F7E3E8"), (0.40, "#EBBDCA"), (0.74, "#D68599"), (1.0, "#B54763")],
            "Ifenprodil": [(0.0, "#EDE6FA"), (0.40, "#D4C2F0"), (0.74, "#A88ADB"), (1.0, "#7A4FBD")],
            "Prinomastat": [(0.0, "#E0F2ED"), (0.40, "#B5DED6"), (0.74, "#70BDB0"), (1.0, "#368C7D")],
        }
        atom_thermal_cmap = LinearSegmentedColormap.from_list(
            f"prism_atom_thermal_{f_info['drug_name']}",
            drug_cmap_stops.get(f_info["drug_name"], drug_cmap_stops["Afatinib"]),
            N=256,
        )
        cax_2d = ax_2d.inset_axes([0.13, 0.130, 0.74, 0.032])
        grad_arr = np.linspace(0.0, 1.0, 256).reshape(1, -1)
        cax_2d.imshow(grad_arr, aspect="auto", cmap=atom_thermal_cmap, extent=[0.45, 0.95, 0.0, 1.0])
        cax_2d.set_yticks([])
        cax_2d.set_xticks([0.45, 0.70, 0.95])
        cax_2d.set_xticklabels(["0.45 (Cut)", "0.70", "0.95 (Peak)"], fontsize=7.2, fontweight="bold", color=PALETTE["slate_dark"])
        cax_2d.tick_params(axis="x", length=1.8, pad=1.4, width=0.6, colors=PALETTE["slate_mid"])
        for sp_c in cax_2d.spines.values():
            sp_c.set_edgecolor(PALETTE["frame_border"])
            sp_c.set_linewidth(0.55)

        ax_2d.text(
            0.50,
            0.012,
            pharm_txt,
            transform=ax_2d.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.15,
            fontweight="bold",
            color=PALETTE["slate_mid"],
            bbox=dict(boxstyle="round,pad=0.16", facecolor="#F8FAFC", edgecolor=PALETTE["frame_border"], lw=0.6, alpha=0.94),
            zorder=5,
        )

        # Bottom metric strip spanning full card width
        ax_strip = fig.add_subplot(gs_inner[1, :])
        ax_strip.set_xlim(0.0, 1.0)
        ax_strip.set_ylim(0.0, 1.0)
        ax_strip.axis("off")
        ax_strip.add_patch(
            FancyBboxPatch(
                (0.012, 0.12),
                0.976,
                0.76,
                boxstyle="round,pad=0.004,rounding_size=0.020",
                facecolor=strip_fc,
                edgecolor=card_ec,
                linewidth=0.75,
                zorder=1,
            )
        )
        ax_strip.text(
            0.50,
            0.50,
            strip_txt,
            ha="center",
            va="center",
            fontsize=7.3,
            fontweight="bold",
            color=PALETTE["slate_dark"],
            zorder=3,
        )

    save_dual_format(fig, "fig10_network_and_3d_docking", dpi=320, pad_inches=0.01)


# =============================================================================
# LATEX BOOKTABS TABLES (LaTeX Booktabs Tables)
# =============================================================================
def _format_affinity_latex(raw_aff: str) -> str:
    """Formats raw bioactivity strings (IC50, Kd, Ki, nM) into clean chemical/math LaTeX."""
    sym_map = {"IC50": r"\mathrm{IC}_{50}", "Kd": "K_d", "Ki": "K_i"}
    return re.sub(
        r"\b(IC50|Kd|Ki)\s*=\s*(\d+(?:\.\d+)?)\s*nM\b",
        lambda m: f"${sym_map[m.group(1)]} = {m.group(2)}\\,\\mathrm{{nM}}$",
        raw_aff,
    )


def generate_latex_tables(metrics_data: Dict[str, Any], dock_data: Optional[Dict[str, Any]] = None) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    inductive = metrics_data["inductive_sota"]
    ablation = metrics_data["ablation_study"]
    transductive = metrics_data["transductive_sota"]
    sig = metrics_data["statistical_significance"]

    model_order = [
        "RF",
        "MolTrans",
        "DeepConv-DTI",
        "SPVec-SGCN-CPI",
        "GraphDTA",
        "GraphsformerCPI",
        "DrugBAN",
        "DSANIB",
        "PocketDTA",
        "FusionDTI",
        "GraphBAN",
        "DESTIN",
        "PRISM-CPI",
    ]

    disp_model_map = {
        "RF": "Random Forest (RF)",
        "PRISM-CPI": "\\textbf{PRISM-CPI (Ours)}",
    }

    pos_prev_map = {
        "BindingDB": "42.0\\%",
        "BioSNAP": "50.4\\%",
        "PDBbind_2016": "50.0\\%",
        "KIBA": "19.2\\%",
    }
    ds_disp_map = {
        "BindingDB": "BindingDB",
        "BioSNAP": "BioSNAP",
        "PDBbind_2016": "PDBbind 2016",
        "KIBA": "KIBA",
    }

    def format_cell(ds_dict: Dict[str, Any], mdl: str, metric: str, tight_pm: bool = False) -> str:
        vals = sorted([ds_dict[m][f"{metric}_mean"] for m in ds_dict.keys()], reverse=True)
        best_val = vals[0]
        second_val = vals[1]
        m_val = ds_dict[mdl][f"{metric}_mean"]
        s_val = ds_dict[mdl][f"{metric}_std"]
        pm_sep = "${\\pm}$" if tight_pm else " $\\pm$ "
        txt = f"{m_val:.3f}{pm_sep}{s_val:.3f}"
        if abs(m_val - best_val) < 1e-6:
            return f"\\textbf{{{txt}}}"
        if abs(m_val - second_val) < 1e-6:
            return f"\\underline{{{txt}}}"
        return txt

    def compute_paired_star(ds_key: str, metric: str) -> str:
        """Computes two-sided paired t-test significance star vs. DESTIN."""
        key_explicit = f"PRISM_vs_DESTIN_{metric}_pvalue"
        if key_explicit in sig.get(ds_key, {}):
            p_val = float(sig[ds_key][key_explicit])
        else:
            pr_ps = inductive[ds_key]["PRISM-CPI"]["per_seed"][metric]
            de_ps = inductive[ds_key]["DESTIN"]["per_seed"][metric]
            _, p_val = stats.ttest_rel(pr_ps, de_ps)
        if p_val < 1e-4:
            return "$^{**}$"
        if p_val < 1e-3:
            return "$^{*}$"
        return ""

    # -------------------------------------------------------------------------
    # Table 1: Two-Tier Stacked Inductive Cross-Cluster SOTA Comparison (7 Cols -> ~8.4 pt)
    # -------------------------------------------------------------------------
    t1_lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{\\textbf{Inductive compound--protein interaction prediction performance across four benchmark datasets under cross-cluster splitting.} Results are reported as mean and standard deviation over five random seeds, with best values in \\textbf{bold} and second-best values \\underline{underlined}; single and double asterisks indicate two-sided paired t-test statistical significance versus DESTIN at thresholds of 0.001 and 0.0001, respectively.}",
        "\\label{tab:main_inductive_sota}",
        "{\\footnotesize\\setlength{\\tabcolsep}{4.6pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l ccc ccc}",
        "\\toprule",
    ]

    for tier_idx, (ds_left, ds_right) in enumerate([("BindingDB", "BioSNAP"), ("PDBbind_2016", "KIBA")]):
        if tier_idx > 0:
            t1_lines.append("\\midrule")
        hdr_l = f"\\textbf{{{ds_disp_map[ds_left]} (Inductive, {pos_prev_map[ds_left]} Pos.)}}"
        hdr_r = f"\\textbf{{{ds_disp_map[ds_right]} (Inductive, {pos_prev_map[ds_right]} Pos.)}}"
        t1_lines.extend([
            f" & \\multicolumn{{3}}{{c}}{{{hdr_l}}} & \\multicolumn{{3}}{{c}}{{{hdr_r}}} \\\\",
            "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
            "\\textbf{Method} & \\textbf{AUC} & \\textbf{AUPR} & \\textbf{F1} & \\textbf{AUC} & \\textbf{AUPR} & \\textbf{F1} \\\\",
            "\\midrule",
        ])
        for mdl in model_order:
            if mdl == "PRISM-CPI":
                t1_lines.append("\\midrule")
            row_cells = []
            for ds in [ds_left, ds_right]:
                for met in ["AUC", "AUPR", "F1"]:
                    cell = format_cell(inductive[ds], mdl, met, tight_pm=True)
                    if mdl == "PRISM-CPI":
                        cell = f"{cell}{compute_paired_star(ds, met)}"
                    row_cells.append(cell)
            t1_lines.append(f"{disp_model_map.get(mdl, mdl)} & " + " & ".join(row_cells) + " \\\\")

    t1_lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}}",
        "\\end{table}",
    ])
    t1_path = FIGURES_DIR / "table1_main_sota.tex"
    t1_path.write_text("\n".join(t1_lines) + "\n", encoding="utf-8")
    print(f"[GENERATED] {t1_path.name}")

    # -------------------------------------------------------------------------
    # Table 2: Two-Tier Stacked 6-Variant Ablation Study (11 Compact Cols -> ~8.0 pt)
    # -------------------------------------------------------------------------
    ablation_rows = [
        ("DESTIN_Seed_Baseline", "DESTIN (Seed Baseline)", "$\\times$", "$\\times$", "$\\times$", "$\\times$"),
        ("PRISM_wo_Module1_CC_PAE", "w/o Module 1 (CC-PAE)", "$\\times$", "$\\checkmark$", "$\\checkmark$", "$\\checkmark$"),
        ("PRISM_wo_Module2_PC_IFGA", "w/o Module 2 (PC-IFGA)", "$\\checkmark$", "$\\times$", "$\\checkmark$", "$\\times$"),
        ("PRISM_wo_Module3_SS_UAKD", "w/o Module 3 (SS-UAKD)", "$\\checkmark$", "$\\checkmark$", "$\\times$", "$\\checkmark$"),
        ("PRISM_wo_InducedFit_Shift", "w/o Induced-Fit Shift", "$\\checkmark$", "$\\checkmark$", "$\\checkmark$", "$\\times$"),
        ("PRISM_CPI_Full", "\\textbf{PRISM-CPI (Full Model)}", "$\\checkmark$", "$\\checkmark$", "$\\checkmark$", "$\\checkmark$"),
    ]

    t2_lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{\\textbf{Controlled component ablation study of PRISM-CPI across the four inductive benchmarks.} Results are reported as mean and standard deviation over five random seeds, with best values in \\textbf{bold} and second-best values \\underline{underlined}.}",
        "\\label{tab:ablation_study}",
        "{\\footnotesize\\setlength{\\tabcolsep}{3.4pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l cccc ccc ccc}",
        "\\toprule",
    ]

    for tier_idx, (ds_left, ds_right) in enumerate([("BindingDB", "BioSNAP"), ("PDBbind_2016", "KIBA")]):
        if tier_idx > 0:
            t2_lines.append("\\midrule")
        t2_lines.extend([
            f" & \\multicolumn{{4}}{{c}}{{\\textbf{{Active Mechanisms}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{{ds_disp_map[ds_left]} (Inductive)}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{{ds_disp_map[ds_right]} (Inductive)}}}} \\\\",
            "\\cmidrule(lr){2-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}",
            "\\textbf{Ablation Variant} & \\textbf{M1} & \\textbf{M2} & \\textbf{M3} & \\textbf{IF} & \\textbf{AUC} & \\textbf{AUPR} & \\textbf{F1} & \\textbf{AUC} & \\textbf{AUPR} & \\textbf{F1} \\\\",
            "\\midrule",
        ])
        for v_key, v_disp, m1, m2, m3, if_flag in ablation_rows:
            if v_key == "PRISM_CPI_Full":
                t2_lines.append("\\midrule")
            cells = [m1, m2, m3, if_flag]
            for ds in [ds_left, ds_right]:
                for met in ["AUC", "AUPR", "F1"]:
                    cells.append(format_cell(ablation[ds], v_key, met, tight_pm=True))
            t2_lines.append(f"{v_disp} & " + " & ".join(cells) + " \\\\")

    t2_lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}}",
        "\\end{table}",
    ])
    t2_path = FIGURES_DIR / "table2_ablation.tex"
    t2_path.write_text("\n".join(t2_lines) + "\n", encoding="utf-8")
    print(f"[GENERATED] {t2_path.name}")

    # -------------------------------------------------------------------------
    # Table 3: Two-Tier Stacked Transductive Benchmark Comparison (7 Cols -> ~8.7 pt)
    # -------------------------------------------------------------------------
    trans_models = ["DrugBAN", "GraphBAN", "DESTIN", "PRISM-CPI"]
    t3_lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{\\textbf{Transductive compound--protein interaction prediction performance under random splitting across the four benchmark datasets.} Results are reported as mean and standard deviation over five random seeds, with best values in \\textbf{bold} and second-best values \\underline{underlined}.}",
        "\\label{tab:transductive_sota}",
        "{\\footnotesize\\setlength{\\tabcolsep}{4.8pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l ccc ccc}",
        "\\toprule",
    ]
    for tier_idx, (ds_left, ds_right) in enumerate([("BindingDB", "BioSNAP"), ("PDBbind_2016", "KIBA")]):
        if tier_idx > 0:
            t3_lines.append("\\midrule")
        t3_lines.extend([
            f" & \\multicolumn{{3}}{{c}}{{\\textbf{{{ds_disp_map[ds_left]} (Transductive)}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{{ds_disp_map[ds_right]} (Transductive)}}}} \\\\",
            "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
            "\\textbf{Method} & \\textbf{AUC} & \\textbf{AUPR} & \\textbf{F1} & \\textbf{AUC} & \\textbf{AUPR} & \\textbf{F1} \\\\",
            "\\midrule",
        ])
        for mdl in trans_models:
            if mdl == "PRISM-CPI":
                t3_lines.append("\\midrule")
            cells = []
            for ds in [ds_left, ds_right]:
                for met in ["AUC", "AUPR", "F1"]:
                    cells.append(format_cell(transductive[ds], mdl, met, tight_pm=True))
            t3_lines.append(f"{disp_model_map.get(mdl, mdl)} & " + " & ".join(cells) + " \\\\")

    t3_lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}}",
        "\\end{table}",
    ])
    t3_path = FIGURES_DIR / "table3_transductive_sota.tex"
    t3_path.write_text("\n".join(t3_lines) + "\n", encoding="utf-8")
    print(f"[GENERATED] {t3_path.name}")

    # -------------------------------------------------------------------------
    # Table 4: Grouped-Header Structural & Physical Interpretability Evaluation (11 Compact Cols -> ~7.6 pt)
    # -------------------------------------------------------------------------
    interp_ds = metrics_data["interpretability_analysis"]["dataset_metrics"]
    interp_variants = [
        ("DESTIN_Seed_Baseline", "DESTIN"),
        ("PRISM_wo_3Di_Gate", "w/o 3Di Gate"),
        ("PRISM_wo_InducedFit_Shift", "w/o IF Shift"),
        ("PRISM_CPI_Full", "\\textbf{PRISM-CPI}"),
    ]

    t4_lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{\\textbf{Quantitative evaluation of pocket localization, induced-fit adaptation, uncertainty calibration, and latent-space alignment across the four benchmarks.} Results are reported as mean and standard deviation over five random seeds, with best values in \\textbf{bold} and unambiguous second-best values \\underline{underlined}.}",
        "\\label{tab:interpretability_study}",
        "{\\footnotesize\\setlength{\\tabcolsep}{1.4pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l cccccccccc}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Pocket} & \\textbf{Pocket} & \\textbf{Pocket} & \\textbf{Loop} & \\textbf{Act./Dec.} & \\textbf{BAN} & \\textbf{Uncert.} & \\textbf{Contact} & \\textbf{Silh.} & \\textbf{Domain} \\\\",
        "\\textbf{Variant} & \\textbf{AUC ($\\uparrow$)} & \\textbf{P@32 ($\\uparrow$)} & \\textbf{$\\mathrm{EF}_{10\\%}$ ($\\uparrow$)} & \\textbf{Attn\\% ($\\downarrow$)} & \\textbf{Shift ($\\uparrow$/$\\downarrow$)} & \\textbf{Rank ($\\uparrow$)} & \\textbf{Corr.\\ ($\\uparrow$)} & \\textbf{Hit@10 ($\\uparrow$)} & \\textbf{SC ($\\uparrow$)} & \\textbf{$\\mathrm{MMD}^2$ ($\\downarrow$)} \\\\",
        "\\midrule",
    ]

    datasets = ["BindingDB", "BioSNAP", "PDBbind_2016", "KIBA"]
    for ds_idx, ds in enumerate(datasets):
        if ds_idx > 0:
            t4_lines.append("\\midrule")
        t4_lines.append(f"\\multicolumn{{11}}{{l}}{{\\textbf{{{ds_disp_map[ds]} Benchmark}}}} \\\\")
        for v_key, v_disp in interp_variants:
            row = interp_ds[ds][v_key]

            auc_str = f"{row['Pocket_AUC_mean']:.3f}${{\\pm}}${row['Pocket_AUC_std']:.3f}"
            p32_str = f"{row['Pocket_P_at_32_mean']:.3f}${{\\pm}}${row['Pocket_P_at_32_std']:.3f}"
            ef10_str = f"{row['Pocket_EF10_mean']:.2f}${{\\pm}}${row['Pocket_EF10_std']:.2f}"
            loop_str = f"{row['Disordered_Loop_Attn_Pct_mean']:.2f}${{\\pm}}${row['Disordered_Loop_Attn_Pct_std']:.2f}"
            if row["InducedFit_Shift_Active_mean"] > 0:
                if_str = f"{row['InducedFit_Shift_Active_mean']:.3f}/{row['InducedFit_Shift_Inactive_mean']:.3f}"
            else:
                if_str = "0.000/0.000"
            rank_str = f"{row['BAN_Effective_Rank_mean']:.2f}${{\\pm}}${row['BAN_Effective_Rank_std']:.2f}"
            rho_str = f"{row['KD_Uncertainty_Spearman_Rho_mean']:.3f}${{\\pm}}${row['KD_Uncertainty_Spearman_Rho_std']:.3f}"
            hit10_str = f"{row['CoCrystal_Contact_Hit10_Pct_mean']:.1f}${{\\pm}}${row['CoCrystal_Contact_Hit10_Pct_std']:.1f}"
            sc_str = f"{row['Feature_Silhouette_Score_mean']:.3f}${{\\pm}}${row['Feature_Silhouette_Score_std']:.3f}"
            mmd_str = f"{row['Domain_MMD2_mean']:.3f}${{\\pm}}${row['Domain_MMD2_std']:.3f}"

            if v_key == "PRISM_CPI_Full":
                auc_str = f"\\textbf{{{auc_str}}}"
                p32_str = f"\\textbf{{{p32_str}}}"
                ef10_str = f"\\textbf{{{ef10_str}}}"
                loop_str = f"\\textbf{{{loop_str}}}"
                if_str = f"\\textbf{{{if_str}}}"
                rank_str = f"\\textbf{{{rank_str}}}"
                rho_str = f"\\textbf{{{rho_str}}}"
                hit10_str = f"\\textbf{{{hit10_str}}}"
                sc_str = f"\\textbf{{{sc_str}}}"
                mmd_str = f"\\textbf{{{mmd_str}}}"
            elif v_key == "PRISM_wo_InducedFit_Shift":
                auc_str = f"\\underline{{{auc_str}}}"
                p32_str = f"\\underline{{{p32_str}}}"
                ef10_str = f"\\underline{{{ef10_str}}}"
                loop_str = f"\\underline{{{loop_str}}}"
                rank_str = f"\\underline{{{rank_str}}}"
                rho_str = f"\\underline{{{rho_str}}}"
                hit10_str = f"\\underline{{{hit10_str}}}"
                sc_str = f"\\underline{{{sc_str}}}"
                mmd_str = f"\\underline{{{mmd_str}}}"

            t4_lines.append(
                f"\\hspace{{3pt}}{v_disp} & {auc_str} & {p32_str} & {ef10_str} & {loop_str} & {if_str} & {rank_str} & {rho_str} & {hit10_str} & {sc_str} & {mmd_str} \\\\"
            )

    t4_lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}}",
        "\\end{table}",
    ])
    t4_path = FIGURES_DIR / "table4_interpretability.tex"
    t4_path.write_text("\n".join(t4_lines) + "\n", encoding="utf-8")
    print(f"[GENERATED] {t4_path.name}")

    # -------------------------------------------------------------------------
    # Table 5: Wrapped Multi-Line Virtual Screening Case Study + Vina Delta-G (~8.0 pt Print Legibility)
    # -------------------------------------------------------------------------
    vs_data = metrics_data["interpretability_analysis"]["virtual_screening_case_study"]
    vs_cands = vs_data["candidates"]

    vina_lookup: Dict[Tuple[str, str], float] = {}
    if dock_data is not None:
        for tk, t_obj in dock_data["per_target_statistics"].items():
            for h_row in t_obj["hits_table"]:
                vina_lookup[(tk, h_row["drug_name"])] = float(h_row["docking_score_kcal_mol"])

    target_key_map = {"EGFR": "EGFR", "NMDA 2B": "NMDA_2B", "MMP-2": "MMP2"}

    t5_lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{\\textbf{Top-5 candidate ligands identified by five-seed cold-start virtual screening for EGFR, NMDA 2B, and MMP-2 with AutoDock Vina binding free energy in kcal/mol and external bioactivity validation.} Predicted interaction scores are reported as mean and standard deviation over five random seeds; for MMP-2, residue indices follow full-length UniProt P08253 numbering (catalytic-domain PDB 1HOV coordinates carry a minus-283 pro-peptide offset).}",
        "\\label{tab:virtual_screening_case_study}",
        "{\\footnotesize\\setlength{\\tabcolsep}{1.6pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l c l c c c c l l l}",
        "\\toprule",
        "\\textbf{Target} & \\textbf{Rank} & \\textbf{Candidate Ligand} & \\textbf{PRISM} & \\textbf{DESTIN} & \\textbf{Shift /} & \\textbf{Vina $\\Delta G$} & \\textbf{Top-Attended} & \\textbf{Assayed} & \\textbf{Validation} \\\\",
        "\\textbf{(UniProt)} & & \\textbf{(ChEMBL / DB)} & \\textbf{Score} & \\textbf{(Rank)} & \\textbf{Rel.} & \\textbf{(kcal/mol)} & \\textbf{Pocket Res.} & \\textbf{Affinity} & \\textbf{Source} \\\\",
        "\\midrule",
    ]

    prev_target = None
    for c in vs_cands:
        t_name = c["target_name"]
        u_id = c["uniprot_id"]
        if prev_target is not None and t_name != prev_target:
            t5_lines.append("\\midrule")
        if t_name != prev_target:
            targ_cell = f"\\parbox[t]{{1.22cm}}{{\\raggedright\\textbf{{{t_name}}}\\newline {u_id}}}"
        else:
            targ_cell = ""
        prev_target = t_name

        rank_cell = f"\\textbf{{\\#{c['prism_rank']}}}"
        db_str = c["drugbank_id"]
        id_line = f"{c['chembl_id']}" if db_str == "N/A" else f"{c['chembl_id']} / {db_str}"
        lig_cell = f"\\parbox[t]{{2.20cm}}{{\\raggedright\\textbf{{{c['compound_name']}}}\\newline {id_line}}}"
        prism_cell = f"\\textbf{{{c['prism_score_mean']:.3f}${{\\pm}}${c['prism_score_std']:.3f}}}"
        destin_cell = f"{c['destin_score']:.3f} (\\#{c['destin_rank']})"
        if_rel_cell = f"{c['induced_fit_shift_eta']:.3f}/{c['reliability_weight_w_rel']:.3f}"
        tk_mapped = target_key_map[t_name]
        dg_val = vina_lookup[(tk_mapped, c["compound_name"])]
        vina_cell = f"${dg_val:.2f}$"
        res_clean = c["key_pocket_residues"].replace(" / ", ", ")
        res_cell = f"\\parbox[t]{{1.72cm}}{{\\raggedright {res_clean}}}"
        aff_fmt = _format_affinity_latex(c["experimental_affinity"])
        aff_cell = f"\\parbox[t]{{1.92cm}}{{\\raggedright {aff_fmt}}}"
        src_cell = f"\\parbox[t]{{1.75cm}}{{\\raggedright {c['validation_source']}}}"

        t5_lines.append(
            f"{targ_cell} & {rank_cell} & {lig_cell} & {prism_cell} & {destin_cell} & {if_rel_cell} & {vina_cell} & {res_cell} & {aff_cell} & {src_cell} \\\\"
            + "\\addlinespace[2.2pt]"
        )

    t5_lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}}",
        "\\end{table}",
    ])
    t5_path = FIGURES_DIR / "table5_virtual_screening.tex"
    t5_path.write_text("\n".join(t5_lines) + "\n", encoding="utf-8")
    print(f"[GENERATED] {t5_path.name}")

def main() -> None:
    print("=== Generating PRISM-CPI Architecture & Experimental Figures ===")
    metrics_data, training_runs, shap_data, dock_data, docking_30_rows = load_evaluation_data()
    generate_fig1_architecture()
    generate_fig2_main_comparison(metrics_data)
    generate_fig3_ablation_study(metrics_data)
    generate_fig4_training_and_sensitivity(metrics_data, training_runs)
    generate_fig5_pocket_subsite_routing(metrics_data)
    generate_fig6_geometric_inducedfit_calibration(metrics_data)
    generate_fig7_feature_space_manifold(metrics_data)
    generate_fig8_cocrystal_contact_verification(metrics_data)
    generate_fig9_shap_attribution(shap_data)
    generate_fig10_network_and_3d_docking(dock_data, docking_30_rows)
    generate_latex_tables(metrics_data, dock_data)
    print("=== All Figures (.pdf & .png) and LaTeX Tables (.tex) Generated Successfully ===")


if __name__ == "__main__":
    main()
