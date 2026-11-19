#!/usr/bin/env python3
"""
Renders the PRISM-CPI model architecture diagram (figures/fig1_model_architecture.pdf/.png)
and graphical abstract (figures/graphical_abstract.pdf/.png).
"""

import math
import os
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_prism_cpi_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb, rgb_to_hsv, to_rgba
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Polygon,
    Rectangle,
)
from matplotlib.path import Path as MplPath
import numpy as np

REF_CMAPS = {
    "ref_contact": LinearSegmentedColormap.from_list(
        "ref_contact", ["#F5F9EE", "#DBE9D8", "#8BBCC3", "#5186A8", "#2E4D72"], N=256
    ),
    "ref_rose": LinearSegmentedColormap.from_list(
        "ref_rose", ["#FDF7F8", "#F2DFE4", "#DDAAB7", "#BC6E80", "#914153"], N=256
    ),
    "ref_purple": LinearSegmentedColormap.from_list(
        "ref_purple", ["#F7F5FC", "#E3DDF3", "#B1A3D4", "#7862A8", "#5E468A"], N=256
    ),
    "ref_blue": LinearSegmentedColormap.from_list(
        "ref_blue", ["#F4F8FC", "#DCE8F4", "#9BB8D4", "#5B7C9B", "#2E4D72"], N=256
    ),
    "ref_teal": LinearSegmentedColormap.from_list(
        "ref_teal", ["#F2FAF8", "#D6ECE8", "#9ECAC3", "#43877E", "#2D6B64"], N=256
    ),
    "ref_amber": LinearSegmentedColormap.from_list(
        "ref_amber", ["#FAF5EA", "#F4EBD7", "#E4CFA6", "#C28B4E", "#9D6833"], N=256
    ),
}

REF_RBF_ARCH_CMAP = LinearSegmentedColormap.from_list(
    "ref_rbf_arch", ["#4E769E", "#5186A8", "#43877E", "#C28B4E", "#BC6E80"], N=256
)


def _clamp_neon_only(color, max_sat: float = 0.70):
    """Caps HSV color saturation at max_sat."""
    if color is None or (isinstance(color, str) and color.lower() == "none"):
        return color
    try:
        r, g, b, a = to_rgba(color)
    except Exception:
        return color
    if a == 0.0:
        return "none"
    hsv = rgb_to_hsv(np.array([r, g, b], dtype=float))
    h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])
    if s <= max_sat:
        return color
    rgb_new = hsv_to_rgb(np.array([h, max_sat, v]))
    return (float(rgb_new[0]), float(rgb_new[1]), float(rgb_new[2]), float(a))


def enforce_reference_palette(ax: plt.Axes, max_sat: float = 0.70) -> None:
    """Applies color saturation bounds across axes artists."""
    for p in ax.patches:
        fc = p.get_facecolor()
        if fc is not None and len(fc) == 4 and fc[3] > 0.0:
            p.set_facecolor(_clamp_neon_only(fc, max_sat=max_sat))
        ec = p.get_edgecolor()
        if ec is not None and len(ec) == 4 and ec[3] > 0.0 and p.get_linewidth() > 0.0:
            p.set_edgecolor(_clamp_neon_only(ec, max_sat=max_sat))

    for ln in ax.lines:
        ln.set_color(_clamp_neon_only(ln.get_color(), max_sat=max_sat))
        mfc = ln.get_markerfacecolor()
        if mfc is not None and not (isinstance(mfc, str) and mfc.lower() == "none"):
            ln.set_markerfacecolor(_clamp_neon_only(mfc, max_sat=max_sat))
        mec = ln.get_markeredgecolor()
        if mec is not None and not (isinstance(mec, str) and mec.lower() == "none"):
            ln.set_markeredgecolor(_clamp_neon_only(mec, max_sat=max_sat))

    for col in ax.collections:
        fcs = col.get_facecolor()
        if fcs is not None and len(fcs) > 0:
            col.set_facecolor([_clamp_neon_only(c, max_sat=max_sat) for c in fcs])
        ecs = col.get_edgecolor()
        if ecs is not None and len(ecs) > 0:
            col.set_edgecolor([_clamp_neon_only(c, max_sat=max_sat) for c in ecs])

    for txt in ax.texts:
        txt.set_color(_clamp_neon_only(txt.get_color(), max_sat=max_sat))
        bb = txt.get_bbox_patch()
        if bb is not None:
            bfc = bb.get_facecolor()
            if bfc is not None and len(bfc) == 4 and bfc[3] > 0.0:
                bb.set_facecolor(_clamp_neon_only(bfc, max_sat=max_sat))
            bec = bb.get_edgecolor()
            if bec is not None and len(bec) == 4 and bec[3] > 0.0 and bb.get_linewidth() > 0.0:
                bb.set_edgecolor(_clamp_neon_only(bec, max_sat=max_sat))
        ap = getattr(txt, "arrow_patch", None)
        if ap is not None:
            afc = ap.get_facecolor()
            if afc is not None and len(afc) == 4 and afc[3] > 0.0:
                ap.set_facecolor(_clamp_neon_only(afc, max_sat=max_sat))
            aec = ap.get_edgecolor()
            if aec is not None and len(aec) == 4 and aec[3] > 0.0:
                ap.set_edgecolor(_clamp_neon_only(aec, max_sat=max_sat))


# =============================================================================
# Vector Primitive & Orthogonal Routing Helpers (1.0 unit == 1.0 physical inch)
# =============================================================================

def add_panel_container(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    badge: str,
    title: str,
    bg_fc: str,
    border_ec: str,
    header_fc: str,
    header_h: float = 0.46,
    title_fontsize: float = 13.8,
    zorder: int = 1,
) -> None:
    """Draws a module container with a header bar."""
    shadow = FancyBboxPatch(
        (x + 0.032, y - 0.032),
        w,
        h,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        facecolor="#0F172A",
        edgecolor="none",
        alpha=0.08,
        zorder=zorder,
    )
    ax.add_patch(shadow)

    body = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        facecolor=bg_fc,
        edgecolor=border_ec,
        linewidth=1.75,
        zorder=zorder + 1,
    )
    ax.add_patch(body)

    hy = y + h - header_h
    header = FancyBboxPatch(
        (x, hy),
        w,
        header_h,
        boxstyle="round,pad=0.0,rounding_size=0.09",
        facecolor=header_fc,
        edgecolor=border_ec,
        linewidth=1.3,
        zorder=zorder + 2,
    )
    ax.add_patch(header)
    header_sq = Rectangle(
        (x + 0.01, hy),
        w - 0.02,
        header_h * 0.48,
        facecolor=header_fc,
        edgecolor="none",
        zorder=zorder + 2,
    )
    ax.add_patch(header_sq)

    bw = 0.37
    bh = 0.35
    bx = x + 0.12
    by = hy + (header_h - bh) / 2.0
    badge_patch = FancyBboxPatch(
        (bx, by),
        bw,
        bh,
        boxstyle="round,pad=0.0,rounding_size=0.07",
        facecolor="#FFFFFF",
        edgecolor="none",
        zorder=zorder + 3,
    )
    ax.add_patch(badge_patch)
    ax.text(
        bx + bw / 2.0,
        hy + header_h / 2.0,
        badge,
        ha="center",
        va="center",
        fontsize=14.4,
        fontweight="bold",
        color=header_fc,
        zorder=zorder + 4,
    )

    ax.text(
        bx + bw + 0.13,
        hy + header_h / 2.0,
        title,
        ha="left",
        va="center",
        fontsize=title_fontsize,
        fontweight="bold",
        color="#FFFFFF",
        linespacing=1.18,
        zorder=zorder + 4,
    )


def add_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    fc: str = "#FFFFFF",
    ec: str = "#94A3B8",
    header_fc: str = "#EFF6FF",
    title_color: str = "#0F172A",
    title_fontsize: float = 11.5,
    lw: float = 1.25,
    header_h: float = 0.34,
    zorder: int = 5,
) -> Tuple[float, float, float, float]:
    """Draws a sub-module card with a header bar."""
    sh = FancyBboxPatch(
        (x + 0.020, y - 0.020),
        w,
        h,
        boxstyle="round,pad=0.0,rounding_size=0.07",
        facecolor="#0F172A",
        edgecolor="none",
        alpha=0.055,
        zorder=zorder,
    )
    ax.add_patch(sh)

    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.0,rounding_size=0.07",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder + 1,
    )
    ax.add_patch(card)

    hy = y + h - header_h
    hdr = FancyBboxPatch(
        (x, hy),
        w,
        header_h,
        boxstyle="round,pad=0.0,rounding_size=0.065",
        facecolor=header_fc,
        edgecolor="none",
        zorder=zorder + 2,
    )
    ax.add_patch(hdr)
    hdr_sq = Rectangle(
        (x, hy),
        w,
        header_h * 0.46,
        facecolor=header_fc,
        edgecolor="none",
        zorder=zorder + 2,
    )
    ax.add_patch(hdr_sq)
    ax.plot([x, x + w], [hy, hy], color=ec, lw=0.85, alpha=0.70, zorder=zorder + 3)

    border_top = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.0,rounding_size=0.07",
        facecolor="none",
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder + 4,
    )
    ax.add_patch(border_top)

    ax.text(
        x + w / 2.0,
        hy + header_h / 2.0,
        title,
        ha="center",
        va="center",
        fontsize=title_fontsize,
        fontweight="bold",
        color=title_color,
        zorder=zorder + 5,
    )

    return (x + 0.08, y + 0.06, w - 0.16, h - header_h - 0.08)


def add_subpanel(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    fc: str = "#F1F5F9",
    ec: str = "#CBD5E1",
    lw: float = 1.0,
    zorder: int = 7,
) -> None:
    """Draws a clean rounded sub-panel inside a card."""
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(panel)


def draw_op_circle(
    ax: plt.Axes,
    cx: float,
    cy: float,
    symbol: str,
    radius: float = 0.12,
    fc: str = "#EFF6FF",
    ec: str = "#1D4ED8",
    tc: str = "#1E3A8A",
    fontsize: float = 11.0,
    zorder: int = 12,
) -> None:
    """Draws a circular operator node."""
    circ = Circle(
        (cx, cy),
        radius=radius,
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.35,
        zorder=zorder,
    )
    ax.add_patch(circ)
    ax.text(
        cx,
        cy,
        symbol,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=tc,
        zorder=zorder + 1,
    )


def draw_3d_atom(
    ax: plt.Axes,
    cx: float,
    cy: float,
    r: float,
    base_color: str,
    zorder: int = 12,
) -> None:
    """Draws a shaded 3D atom sphere."""
    shadow = Circle(
        (cx + r * 0.15, cy - r * 0.15),
        r,
        facecolor="#0F172A",
        edgecolor="none",
        alpha=0.18,
        zorder=zorder,
    )
    ax.add_patch(shadow)
    sphere = Circle(
        (cx, cy),
        r,
        facecolor=base_color,
        edgecolor="#0F172A",
        linewidth=0.75,
        zorder=zorder + 1,
    )
    ax.add_patch(sphere)
    spec = Circle(
        (cx - r * 0.30, cy + r * 0.30),
        r * 0.33,
        facecolor="#FFFFFF",
        edgecolor="none",
        alpha=0.70,
        zorder=zorder + 2,
    )
    ax.add_patch(spec)


def draw_double_bond_inset(
    ax: plt.Axes,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    center: Tuple[float, float],
    color: str = "#475569",
    lw: float = 1.15,
    zorder: int = 7,
) -> None:
    """Draws a clean inner aromatic double-bond segment parallel to p1-p2."""
    q1 = (0.78 * p1[0] + 0.22 * center[0], 0.78 * p1[1] + 0.22 * center[1])
    q2 = (0.78 * p2[0] + 0.22 * center[0], 0.78 * p2[1] + 0.22 * center[1])
    ms1 = (0.82 * q1[0] + 0.18 * q2[0], 0.82 * q1[1] + 0.18 * q2[1])
    ms2 = (0.18 * q1[0] + 0.82 * q2[0], 0.18 * q1[1] + 0.82 * q2[1])
    ax.plot([ms1[0], ms2[0]], [ms1[1], ms2[1]], color=color, lw=lw, solid_capstyle="round", zorder=zorder)


def draw_manhattan_arrow(
    ax: plt.Axes,
    pts: Sequence[Tuple[float, float]],
    color: str = "#1E3A8A",
    lw: float = 1.75,
    linestyle: str = "-",
    arrow: bool = True,
    corner_r: float = 0.06,
    halo: bool = True,
    label: str = "",
    label_xy: Optional[Tuple[float, float]] = None,
    label_fontsize: float = 10.5,
    label_pad: float = 0.10,
    zorder: int = 18,
) -> None:
    """Draws an orthogonal connector arrow with rounded corners."""
    if len(pts) < 2:
        return

    verts: List[Tuple[float, float]] = [pts[0]]
    codes: List[int] = [MplPath.MOVETO]

    for i in range(1, len(pts) - 1):
        p_prev = np.array(pts[i - 1], dtype=float)
        p_curr = np.array(pts[i], dtype=float)
        p_next = np.array(pts[i + 1], dtype=float)

        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        len1 = float(np.linalg.norm(v1))
        len2 = float(np.linalg.norm(v2))
        r = min(corner_r, len1 * 0.45, len2 * 0.45)

        if r > 1e-4:
            u1 = v1 / len1
            u2 = v2 / len2
            p_in = p_curr - u1 * r
            p_out = p_curr + u2 * r
            verts.append((float(p_in[0]), float(p_in[1])))
            codes.append(MplPath.LINETO)
            verts.append((float(p_curr[0]), float(p_curr[1])))
            codes.append(MplPath.CURVE3)
            verts.append((float(p_out[0]), float(p_out[1])))
            codes.append(MplPath.CURVE3)
        else:
            verts.append((float(p_curr[0]), float(p_curr[1])))
            codes.append(MplPath.LINETO)

    p_end = np.array(pts[-1], dtype=float)
    p_pen = np.array(pts[-2], dtype=float)
    seg_vec = p_end - p_pen
    seg_len = float(np.linalg.norm(seg_vec))
    trim = 0.055 if (arrow and seg_len > 0.11) else 0.0
    p_line_end = p_end - (seg_vec / max(seg_len, 1e-6)) * trim

    verts.append((float(p_line_end[0]), float(p_line_end[1])))
    codes.append(MplPath.LINETO)
    mpl_path = MplPath(verts, codes)

    if halo:
        halo_patch = PathPatch(
            mpl_path,
            facecolor="none",
            edgecolor="#FFFFFF",
            linewidth=lw + 2.3,
            capstyle="round",
            joinstyle="round",
            alpha=0.92,
            zorder=zorder - 1,
        )
        ax.add_patch(halo_patch)

    stroke_patch = PathPatch(
        mpl_path,
        facecolor="none",
        edgecolor=color,
        linewidth=lw,
        linestyle=linestyle,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(stroke_patch)

    if arrow and seg_len > 1e-5:
        u_end = seg_vec / seg_len
        p_tail = p_end - u_end * min(0.14, seg_len * 0.85)
        head = FancyArrowPatch(
            (float(p_tail[0]), float(p_tail[1])),
            (float(p_end[0]), float(p_end[1])),
            arrowstyle="-|>",
            mutation_scale=13.5,
            linewidth=lw,
            color=color,
            zorder=zorder + 1,
        )
        head.set_arrowstyle("-|>", head_length=0.44, head_width=0.28)
        head.set_mutation_scale(13.5)
        ax.add_patch(head)

    if label and label_xy is not None:
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            ha="center",
            va="center",
            fontsize=label_fontsize,
            fontweight="bold",
            color=color,
            bbox=dict(
                boxstyle=f"round,pad={label_pad:.2f},rounding_size=0.06",
                fc="#FFFFFF",
                ec=color,
                lw=0.95,
                alpha=0.99,
            ),
            zorder=zorder + 4,
        )


def draw_mini_matrix(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    data: np.ndarray,
    cmap_name: str = "ref_blue",
    ec: str = "#2E4D72",
    cell_border: str = "#FFFFFF",
    lw_cell: float = 0.42,
    desat_blend: float = 0.0,
    zorder: int = 8,
) -> None:
    """Draws a 2D heatmap grid."""
    nrows, ncols = data.shape
    cw = w / ncols
    ch = h / nrows
    cmap = REF_CMAPS.get(cmap_name, REF_CMAPS["ref_blue"])
    dmin, dmax = float(np.min(data)), float(np.max(data))
    denom = max(dmax - dmin, 1e-8)

    for r in range(nrows):
        for c in range(ncols):
            val = (float(data[r, c]) - dmin) / denom
            raw_rgba = cmap(0.06 + 0.88 * val)
            r_c = (1.0 - desat_blend) * raw_rgba[0] + desat_blend * 0.96
            g_c = (1.0 - desat_blend) * raw_rgba[1] + desat_blend * 0.96
            b_c = (1.0 - desat_blend) * raw_rgba[2] + desat_blend * 0.97
            rect = Rectangle(
                (x + c * cw, y + (nrows - 1 - r) * ch),
                cw,
                ch,
                facecolor=(r_c, g_c, b_c, 1.0),
                edgecolor=cell_border,
                linewidth=lw_cell,
                zorder=zorder,
            )
            ax.add_patch(rect)

    frame = Rectangle((x, y), w, h, facecolor="none", edgecolor=ec, linewidth=0.95, zorder=zorder + 1)
    ax.add_patch(frame)


# =============================================================================
# Architecture diagram renderer
# =============================================================================

def render_nc_fig1_architecture(figures_dir: Path, ms_dir: Optional[Path] = None) -> None:
    """Renders fig1_model_architecture (.pdf & .png) and graphical_abstract (.pdf & .png)."""
    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(13.58, 8.88), dpi=300, facecolor="#FFFFFF")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="#FFFFFF")
    ax.set_xlim(0.11, 13.69)
    ax.set_ylim(0.05, 8.93)
    ax.set_aspect("equal")
    ax.axis("off")

    # =========================================================================
    # 1. Three Main Module Containers (A, B, C) 
    # =========================================================================
    add_panel_container(
        ax,
        x=0.12,
        y=4.89,
        w=8.00,
        h=4.03,
        badge="A",
        title="Module 1: Pocket-Aware 3D Protein Encoder (CC-PAE)",
        bg_fc="#F4F8FC",
        border_ec="#88A6C4",
        header_fc="#2E4D72",
        header_h=0.46,
        title_fontsize=13.8,
    )

    add_panel_container(
        ax,
        x=0.12,
        y=0.53,
        w=8.00,
        h=3.90,
        badge="B",
        title="Module 2: Induced-Fit Geometric Cross-Attention (PC-IFGA)",
        bg_fc="#F2FAF8",
        border_ec="#84B5AE",
        header_fc="#2D6B64",
        header_h=0.46,
        title_fontsize=13.5,
    )

    add_panel_container(
        ax,
        x=9.14,
        y=0.53,
        w=4.54,
        h=8.39,
        badge="C",
        title="Module 3: Uncertainty-Aware\nKD & Alignment (SS-UAKD)",
        bg_fc="#F6F4FC",
        border_ec="#A599C8",
        header_fc="#5E468A",
        header_h=0.62,
        title_fontsize=13.5,
    )

    # =========================================================================
    # 2. PANEL A CARDS & VISUALS (Module 1: CC-PAE 
    # =========================================================================

    # -------------------------------------------------------------------------
    # Card A1: 3D Protein & SaProt PLM (y: 6.46 .. 8.37)
    # -------------------------------------------------------------------------
    add_card(
        ax,
        x=0.24,
        y=6.46,
        w=2.42,
        h=1.91,
        title="3D Protein & SaProt PLM",
        fc="#FFFFFF",
        ec="#88A6C4",
        header_fc="#DFECF3",
        title_color="#2E4D72",
        title_fontsize=11.3,
    )

    # 3D Protein Fold Cartoon (Pocket Glow + Alpha-Helix + Beta-Sheet + Loop)
    for rad_w, rad_h, alp in [(0.56, 0.40, 0.16), (0.38, 0.28, 0.26), (0.22, 0.16, 0.36)]:
        ax.add_patch(
            Ellipse((1.31, 7.59), rad_w, rad_h, facecolor="#43877E", edgecolor="none", alpha=alp, zorder=6)
        )

    hx_l = np.linspace(0.36, 1.00, 140)
    hy_l = 7.60 + 0.14 * np.sin((hx_l - 0.36) * 2.0 * np.pi / 0.215)
    ax.plot(hx_l, hy_l + 0.025, color="#9BB8D4", lw=4.2, solid_capstyle="round", zorder=7)
    ax.plot(hx_l, hy_l, color="#2E4D72", lw=2.8, solid_capstyle="round", zorder=8)
    ax.text(0.71, 7.89, r"$\alpha$-Helix", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#2E4D72", zorder=10)

    # Pocket cleft + pharmacophore spheres
    theta_pkt = np.linspace(1.12 * np.pi, 1.88 * np.pi, 50)
    ax.plot(1.31 + 0.20 * np.cos(theta_pkt), 7.65 + 0.16 * np.sin(theta_pkt), color="#2D6B64", lw=1.7, ls="--", zorder=9)
    lig_sil = [(1.23, 7.60), (1.31, 7.55), (1.39, 7.61)]
    ax.plot([p[0] for p in lig_sil], [p[1] for p in lig_sil], color="#9D6833", lw=1.7, zorder=10)
    for lx_s, ly_s in lig_sil:
        ax.scatter([lx_s], [ly_s], s=26, color="#C28B4E", edgecolor="#9D6833", linewidth=0.8, zorder=11)
    ax.text(1.24, 7.28, "Pocket", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#2D6B64", zorder=10)

    # Cerulean-Teal Beta-Sheet strand
    beta_top = Polygon(
        [(1.56, 7.69), (1.92, 7.69), (1.92, 7.75), (2.06, 7.62), (1.92, 7.49), (1.92, 7.55), (1.56, 7.55)],
        closed=True,
        facecolor="#5186A8",
        edgecolor="#2E4D72",
        linewidth=1.0,
        zorder=8,
    )
    ax.add_patch(beta_top)
    ax.text(1.84, 7.89, r"$\beta$-Sheet", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#2E4D72", zorder=10)

    # Flexible Loop
    lx_r = np.linspace(2.10, 2.55, 60)
    ly_r = 7.60 + 0.10 * np.sin((lx_r - 2.10) * 12.0) + 0.05 * np.cos((lx_r - 2.10) * 20.0)
    ax.plot(lx_r, ly_r, color="#BC6E80", lw=2.1, linestyle="--", zorder=8)
    ax.text(2.33, 7.28, "Loop", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#914153", zorder=10)

    # Bold 1D AA & 3Di Color Strips
    ax.text(0.34, 7.05, "1D AA", ha="left", va="center", fontsize=10.2, fontweight="bold", color="#2B3542", zorder=9)
    aa_mat = np.array([[0.22, 0.35, 0.28, 0.88, 0.94, 0.86, 0.32, 0.20, 0.25, 0.18]])
    draw_mini_matrix(ax, x=1.02, y=6.98, w=1.52, h=0.14, data=aa_mat, cmap_name="ref_teal", ec="#2D6B64", lw_cell=0.40)

    ax.text(0.34, 6.85, "3Di", ha="left", va="center", fontsize=10.2, fontweight="bold", color="#2E4D72", zorder=9)
    p3di_mat = np.array([
        [0.85, 0.80, 0.18, 0.92, 0.95, 0.90, 0.78, 0.22, 0.15, 0.12],
        [0.10, 0.15, 0.75, 0.06, 0.04, 0.08, 0.16, 0.58, 0.62, 0.68],
    ])
    draw_mini_matrix(ax, x=1.02, y=6.77, w=1.52, h=0.15, data=p3di_mat, cmap_name="ref_blue", ec="#2E4D72", lw_cell=0.40)

    ax.text(
        1.45,
        6.59,
        r"SaProt Embeddings $\mathbf{H}_{\mathrm{raw}}$",
        ha="center",
        va="center",
        fontsize=10.6,
        fontweight="bold",
        color="#2E4D72",
        zorder=10,
    )

    # -------------------------------------------------------------------------
    # Card A2: 3Di Entropy Confidence (y: 5.00 .. 6.26) — Borderless #BC6E80 & #9AA6B5 Bars
    # -------------------------------------------------------------------------
    add_card(
        ax,
        x=0.24,
        y=5.00,
        w=2.42,
        h=1.26,
        title="3Di Entropy Confidence",
        fc="#FFFFFF",
        ec="#88A6C4",
        header_fc="#DFECF3",
        title_color="#2E4D72",
        title_fontsize=11.4,
    )

    px0, py0, pw0, ph0 = 0.35, 5.34, 2.20, 0.51
    ax.add_patch(Rectangle((px0, py0), pw0, ph0, facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=0.85, zorder=6))
    conf_vals = np.array([0.24, 0.82, 0.90, 0.94, 0.96, 0.92, 0.88, 0.91, 0.85, 0.32, 0.20, 0.15, 0.14])
    nb = len(conf_vals)
    bw_bar = (pw0 - 0.12) / nb
    bx_centers, by_tops = [], []
    for i_b, cv in enumerate(conf_vals):
        bx_i = px0 + 0.06 + i_b * bw_bar
        bh_i = cv * (ph0 - 0.19)
        bcol = "#BC6E80" if cv >= 0.55 else "#9AA6B5"
        ax.add_patch(
            Rectangle((bx_i, py0 + 0.02), bw_bar * 0.78, bh_i, facecolor=bcol, edgecolor="none", linewidth=0, alpha=0.95, zorder=7)
        )
        bx_centers.append(bx_i + bw_bar * 0.39)
        by_tops.append(py0 + 0.02 + bh_i)
    ax.plot(bx_centers, by_tops, color="#2B3542", lw=1.10, alpha=0.78, zorder=8)
    ax.text(
        px0 + 0.73,
        py0 + ph0 - 0.11,
        r"Core $c_\ell\!\uparrow$",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color="#914153",
        bbox=dict(boxstyle="round,pad=0.05,rounding_size=0.04", fc="#FFFFFF", ec="none", alpha=0.86),
        zorder=10,
    )
    ax.text(
        px0 + 1.74,
        py0 + ph0 - 0.14,
        r"Loop $c_\ell\!\downarrow$",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color="#526071",
        zorder=10,
    )

    ax.text(
        1.45,
        5.15,
        r"Confidence Score $c_\ell$",
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color="#2E4D72",
        zorder=10,
    )

    # -------------------------------------------------------------------------
    # Card A3: Confidence-Gated CNN (y: 6.76 .. 8.37)
    # -------------------------------------------------------------------------
    add_card(
        ax,
        x=2.87,
        y=6.76,
        w=2.42,
        h=1.61,
        title="Confidence-Gated CNN",
        fc="#FFFFFF",
        ec="#88A6C4",
        header_fc="#DFECF3",
        title_color="#2E4D72",
        title_fontsize=11.5,
    )

    seq_x0 = 3.00
    for r_i in range(6):
        rx = seq_x0 + r_i * 0.20
        in_win = 1 <= r_i <= 4
        ax.add_patch(
            Rectangle(
                (rx, 7.34),
                0.16,
                0.36,
                facecolor="#DCE8F4" if in_win else "#F1F5F9",
                edgecolor="#4E769E" if in_win else "#9AA6B5",
                linewidth=1.0 if in_win else 0.7,
                zorder=7,
            )
        )
    win_box = FancyBboxPatch(
        (seq_x0 + 1 * 0.20 - 0.025, 7.30),
        4 * 0.20 + 0.01,
        0.44,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor="none",
        edgecolor="#914153",
        linewidth=1.5,
        linestyle="--",
        zorder=9,
    )
    ax.add_patch(win_box)
    ax.text(seq_x0 + 2.9 * 0.20, 7.87, "Conv1D", ha="center", va="center", fontsize=10.3, fontweight="bold", color="#914153", zorder=10)

    ax.plot([seq_x0 + 5 * 0.20 + 0.17, 4.38], [7.52, 7.52], color="#2E4D72", lw=1.5, zorder=8)
    draw_op_circle(ax, cx=4.51, cy=7.52, symbol="⊕", radius=0.125, fc="#DFECF3", ec="#4E769E", tc="#2E4D72", fontsize=11.2)
    ax.plot([4.64, 4.86], [7.52, 7.52], color="#2B3542", lw=1.5, zorder=8)
    draw_op_circle(ax, cx=4.99, cy=7.52, symbol="⊙", radius=0.125, fc="#F4EBD7", ec="#C28B4E", tc="#9D6833", fontsize=11.2)
    ax.plot([4.99, 4.99], [7.39, 7.20], color="#2E4D72", lw=1.5, zorder=8)

    add_subpanel(ax, x=3.02, y=6.85, w=2.12, h=0.35, fc="#DFECF3", ec="#4E769E", lw=1.25, zorder=8)
    ax.text(
        4.08,
        7.025,
        r"Gated Features $\tilde{\mathbf{H}}$",
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color="#2E4D72",
        zorder=10,
    )

    # -------------------------------------------------------------------------
    # Card A4: Ligand-Guided Salience (y: 5.00 .. 6.54)
    # -------------------------------------------------------------------------
    add_card(
        ax,
        x=2.87,
        y=5.00,
        w=2.42,
        h=1.54,
        title="Ligand-Guided Salience",
        fc="#FFFFFF",
        ec="#88A6C4",
        header_fc="#DFECF3",
        title_color="#2E4D72",
        title_fontsize=11.4,
    )

    sx0, sy0, sw0, sh0 = 2.98, 5.40, 2.20, 0.72
    ax.add_patch(Rectangle((sx0, sy0), sw0, sh0, facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=0.85, zorder=6))
    t_sal = np.linspace(0.0, 1.0, 180)
    sal_curve = (
        0.10
        + 0.76 * np.exp(-((t_sal - 0.30) / 0.060) ** 2)
        + 0.84 * np.exp(-((t_sal - 0.70) / 0.055) ** 2)
        + 0.22 * np.exp(-((t_sal - 0.50) / 0.050) ** 2)
    )
    sal_curve = np.clip(sal_curve, 0.0, 0.95)
    xs_sal = sx0 + 0.06 + t_sal * (sw0 - 0.12)
    ys_sal = sy0 + 0.04 + sal_curve * (sh0 - 0.26)
    ax.fill_between(xs_sal, sy0 + 0.04, ys_sal, color="#BC6E80", alpha=0.26, zorder=7)
    ax.plot(xs_sal, ys_sal, color="#914153", lw=1.85, zorder=8)
    ax.plot([sx0 + 0.05, sx0 + sw0 - 0.05], [sy0 + 0.11, sy0 + 0.11], color="#9AA6B5", lw=1.0, ls="--", zorder=7)
    p1x = sx0 + 0.06 + 0.30 * (sw0 - 0.12)
    p2x = sx0 + 0.06 + 0.70 * (sw0 - 0.12)
    ax.scatter(
        [p1x, p2x],
        [sy0 + 0.04 + 0.86 * (sh0 - 0.26), sy0 + 0.04 + 0.94 * (sh0 - 0.26)],
        s=32,
        color="#C28B4E",
        edgecolor="#914153",
        lw=1.0,
        zorder=9,
    )
    ax.text(
        4.08,
        sy0 + sh0 - 0.13,
        r"Pocket Peaks $\alpha_\ell\!\uparrow$",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color="#914153",
        zorder=10,
    )

    ax.text(
        4.08,
        5.18,
        r"Salience $\alpha_\ell$ & $\mathcal{L}_{\mathrm{pkt}}$",
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color="#2E4D72",
        zorder=10,
    )

    # -------------------------------------------------------------------------
    # Card A5: Pocket Token Routing (y: 5.00 .. 8.37) — Warm Amber Pocket Tokens P
    # -------------------------------------------------------------------------
    add_card(
        ax,
        x=5.50,
        y=5.00,
        w=2.50,
        h=3.37,
        title="Pocket Token Routing",
        fc="#FFFFFF",
        ec="#88A6C4",
        header_fc="#DFECF3",
        title_color="#2E4D72",
        title_fontsize=11.6,
    )

    ax.text(
        6.75,
        7.84,
        "Multi-Query Routing",
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color="#2E4D72",
        zorder=10,
    )
    route_mat = np.full((5, 10), 0.08)
    for rk, ck in [(0, 2), (1, 3), (2, 4), (3, 6), (4, 7)]:
        route_mat[rk, ck] = 0.94
        if ck + 1 < 10:
            route_mat[rk, ck + 1] = 0.50
    draw_mini_matrix(ax, x=5.66, y=7.16, w=1.42, h=0.52, data=route_mat, cmap_name="ref_blue", ec="#2E4D72")
    ax.text(
        7.52,
        7.42,
        r"$\mathbf{A}_{\mathrm{pkt}}$",
        ha="center",
        va="center",
        fontsize=11.8,
        fontweight="bold",
        color="#2E4D72",
        zorder=10,
    )

    draw_manhattan_arrow(ax, [(6.75, 7.12), (6.75, 6.94)], color="#9D6833", lw=1.65, halo=False)

    # Middle Warm Amber Bronze Highlight Sub-Card: Pocket Tokens P
    add_subpanel(ax, x=5.62, y=5.96, w=2.26, h=0.95, fc="#FAF5EA", ec="#9D6833", lw=1.75, zorder=7)
    ax.add_patch(
        FancyBboxPatch(
            (5.63, 6.56), 2.24, 0.34,
            boxstyle="round,pad=0.0,rounding_size=0.05",
            facecolor="#F4EBD7", edgecolor="none", zorder=7.5,
        )
    )
    ax.text(
        6.75,
        6.73,
        r"Pocket Tokens $\mathbf{P}$",
        ha="center",
        va="center",
        fontsize=11.4,
        fontweight="bold",
        color="#9D6833",
        zorder=10,
    )
    p_grid = rng.uniform(0.32, 0.98, size=(4, 9))
    draw_mini_matrix(ax, x=5.76, y=6.05, w=1.98, h=0.44, data=p_grid, cmap_name="ref_amber", ec="#9D6833", zorder=8)

    # Bottom Oxford Blue Sub-Card: Global Fold p_glob
    add_subpanel(ax, x=5.62, y=5.09, w=2.26, h=0.75, fc="#F4F8FC", ec="#4E769E", lw=1.45, zorder=7)
    ax.text(
        6.75,
        5.63,
        r"Global Fold $\mathbf{p}_{\mathrm{glob}}$",
        ha="center",
        va="center",
        fontsize=11.2,
        fontweight="bold",
        color="#2E4D72",
        zorder=10,
    )
    pglob_vec = np.linspace(0.22, 0.96, 10).reshape(1, 10)
    draw_mini_matrix(ax, x=5.76, y=5.19, w=1.98, h=0.23, data=pglob_vec, cmap_name="ref_blue", ec="#2E4D72", zorder=8)

    # =========================================================================
    # 3. PANEL B CARDS & VISUALS (Module 2: PC-IFGA 
    # =========================================================================

    # -------------------------------------------------------------------------
    # Card B1: 3D Conformer & Uni-Mol (y: 2.05 .. 3.77)
    # -------------------------------------------------------------------------
    add_card(
        ax,
        x=0.24,
        y=2.05,
        w=2.42,
        h=1.72,
        title="3D Conformer & Uni-Mol",
        fc="#FFFFFF",
        ec="#84B5AE",
        header_fc="#D6ECE8",
        title_color="#2D6B64",
        title_fontsize=11.4,
    )

    b_len = 0.132
    c1x, c1y = 0.64, 2.91
    c2x, c2y = c1x + math.sqrt(3.0) * b_len, c1y
    ring_bz = [
        (c1x + b_len * math.cos(math.radians(90 - 60 * k)), c1y + b_len * math.sin(math.radians(90 - 60 * k)))
        for k in range(6)
    ]
    ring_py = [
        (c2x + b_len * math.cos(math.radians(90 - 60 * k)), c2y + b_len * math.sin(math.radians(90 - 60 * k)))
        for k in range(6)
    ]
    o_me = (ring_bz[5][0] + 0.126 * math.cos(math.radians(150)), ring_bz[5][1] + 0.126 * math.sin(math.radians(150)))
    c_me = (o_me[0] + 0.112 * math.cos(math.radians(215)), o_me[1] + 0.112 * math.sin(math.radians(215)))
    n_nh = (ring_py[0][0] + 0.128 * math.cos(math.radians(55)), ring_py[0][1] + 0.128 * math.sin(math.radians(55)))
    b_ph = 0.118
    phi_ph = -10.0
    c_ph = (
        n_nh[0] + (0.126 + b_ph) * math.cos(math.radians(phi_ph)),
        n_nh[1] + (0.126 + b_ph) * math.sin(math.radians(phi_ph)),
    )
    ring_ph = [
        (c_ph[0] + b_ph * math.cos(math.radians(phi_ph - 60 * k)), c_ph[1] + b_ph * math.sin(math.radians(phi_ph - 60 * k)))
        for k in range(6)
    ]
    f_sub = (
        ring_ph[5][0] + 0.115 * math.cos(math.radians(phi_ph + 60)),
        ring_ph[5][1] + 0.115 * math.sin(math.radians(phi_ph + 60)),
    )

    mol_bonds = [
        (ring_bz[0], ring_bz[1]), (ring_bz[1], ring_bz[2]), (ring_bz[2], ring_bz[3]),
        (ring_bz[3], ring_bz[4]), (ring_bz[4], ring_bz[5]), (ring_bz[5], ring_bz[0]),
        (ring_py[0], ring_py[1]), (ring_py[1], ring_py[2]), (ring_py[2], ring_py[3]),
        (ring_py[3], ring_py[4]), (ring_py[5], ring_py[0]),
        (ring_bz[5], o_me), (o_me, c_me),
        (ring_py[0], n_nh), (n_nh, ring_ph[3]),
        (ring_ph[0], ring_ph[1]), (ring_ph[1], ring_ph[2]), (ring_ph[2], ring_ph[3]),
        (ring_ph[3], ring_ph[4]), (ring_ph[4], ring_ph[5]), (ring_ph[5], ring_ph[0]),
        (ring_ph[5], f_sub),
    ]
    for p_a, p_b in mol_bonds:
        ax.plot([p_a[0], p_b[0]], [p_a[1], p_b[1]], color="#2B3542", lw=1.6, solid_capstyle="round", zorder=7)

    draw_double_bond_inset(ax, ring_bz[0], ring_bz[5], (c1x, c1y))
    draw_double_bond_inset(ax, ring_bz[1], ring_bz[2], (c1x, c1y))
    draw_double_bond_inset(ax, ring_bz[3], ring_bz[4], (c1x, c1y))
    draw_double_bond_inset(ax, ring_py[0], ring_py[1], (c2x, c2y))
    draw_double_bond_inset(ax, ring_py[2], ring_py[3], (c2x, c2y))
    draw_double_bond_inset(ax, ring_ph[0], ring_ph[1], c_ph)
    draw_double_bond_inset(ax, ring_ph[2], ring_ph[3], c_ph)
    draw_double_bond_inset(ax, ring_ph[4], ring_ph[5], c_ph)

    arc_top_d = FancyArrowPatch(
        o_me, n_nh, connectionstyle="arc3,rad=-0.34", arrowstyle="-",
        linestyle="--", linewidth=1.15, color="#43877E", zorder=6,
    )
    ax.add_patch(arc_top_d)

    cpk_atoms = [
        (ring_bz[0], "#2B3542", 0.036), (ring_bz[1], "#2B3542", 0.036), (ring_bz[2], "#2B3542", 0.036),
        (ring_bz[3], "#2B3542", 0.036), (ring_bz[4], "#2B3542", 0.036), (ring_bz[5], "#2B3542", 0.036),
        (ring_py[0], "#2B3542", 0.036), (ring_py[1], "#4E769E", 0.046), (ring_py[2], "#2B3542", 0.036),
        (ring_py[3], "#4E769E", 0.046),
        (o_me, "#BC6E80", 0.046), (c_me, "#2B3542", 0.036),
        (n_nh, "#4E769E", 0.046),
        (ring_ph[0], "#2B3542", 0.036), (ring_ph[1], "#2B3542", 0.036), (ring_ph[2], "#2B3542", 0.036),
        (ring_ph[3], "#2B3542", 0.036), (ring_ph[4], "#2B3542", 0.036), (ring_ph[5], "#2B3542", 0.036),
        (f_sub, "#43877E", 0.046),
    ]
    for (ax_i, ay_i), acol, arad in cpk_atoms:
        draw_3d_atom(ax, ax_i, ay_i, r=arad, base_color=acol, zorder=9)

    add_subpanel(ax, x=1.48, y=2.70, w=1.08, h=0.63, fc="#F2FAF8", ec="#43877E", lw=1.1, zorder=7)
    ax.text(2.02, 3.19, "Uni-Mol 3D", ha="center", va="center", fontsize=10.3, fontweight="bold", color="#2D6B64", zorder=10)
    u_vec = np.linspace(0.25, 0.95, 6).reshape(1, 6)
    draw_mini_matrix(ax, x=1.58, y=2.98, w=0.88, h=0.11, data=u_vec, cmap_name="ref_teal", ec="#2D6B64", zorder=8)
    u_mat = rng.uniform(0.22, 0.92, size=(2, 6))
    draw_mini_matrix(ax, x=1.58, y=2.77, w=0.88, h=0.16, data=u_mat, cmap_name="ref_contact", ec="#2D6B64", zorder=8)

    ax.text(
        1.45,
        2.49,
        r"Conformer $\mathbf{u}'$",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="bold",
        color="#2D6B64",
        zorder=10,
    )
    ax.text(
        1.45,
        2.22,
        r"Atoms $\mathbf{U}_{\mathrm{atom}}$",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="bold",
        color="#2B3542",
        zorder=10,
    )

    # -------------------------------------------------------------------------
    # Card B2: Continuous RBF Basis (y: 0.74 .. 1.88)
    # -------------------------------------------------------------------------
    add_card(
        ax,
        x=0.24,
        y=0.74,
        w=2.42,
        h=1.14,
        title="Continuous RBF Basis",
        fc="#FFFFFF",
        ec="#84B5AE",
        header_fc="#D6ECE8",
        title_color="#2D6B64",
        title_fontsize=11.5,
    )

    rx0, ry0, rw0, rh0 = 0.35, 1.09, 2.20, 0.40
    ax.add_patch(Rectangle((rx0, ry0), rw0, rh0, facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=0.85, zorder=6))
    d_vals = np.linspace(0.0, 6.6, 220)
    r_max = 6.0
    env_vals = 0.5 * (np.cos(np.pi * np.clip(d_vals, 0.0, r_max) / r_max) + 1.0) * (d_vals <= r_max)
    mus = np.linspace(0.3, 5.7, 14)
    gam = 1.0 / (0.44 ** 2)
    rbf_cols = REF_RBF_ARCH_CMAP(np.linspace(0.05, 0.95, 14))
    xd_plot = rx0 + 0.06 + (d_vals / 6.6) * (rw0 - 0.12)
    for m_i, mu_v in enumerate(mus):
        rbf_y = env_vals * np.exp(-gam * (d_vals - mu_v) ** 2)
        yd_plot = ry0 + 0.04 + rbf_y * (rh0 - 0.10)
        ax.plot(xd_plot, yd_plot, color=rbf_cols[m_i], lw=0.95, alpha=0.88, zorder=7)
    ax.plot(xd_plot, ry0 + 0.04 + env_vals * (rh0 - 0.10), color="#914153", lw=1.85, zorder=8)
    x_6a = rx0 + 0.06 + (6.0 / 6.6) * (rw0 - 0.12)
    ax.plot([x_6a, x_6a], [ry0 + 0.03, ry0 + rh0 - 0.03], color="#9D6833", lw=1.05, ls=":", zorder=8)

    ax.text(
        1.45,
        0.90,
        r"SE(3) RBF Basis $\boldsymbol{\rho}_{ab}$ ($\phi_{\mathrm{env}}$)",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color="#2D6B64",
        zorder=10,
    )

    # -------------------------------------------------------------------------
    # Card B3: 3D Distance-Gated GAT (y: 2.05 .. 3.77) 
    # -------------------------------------------------------------------------
    add_card(
        ax,
        x=2.87,
        y=2.05,
        w=2.42,
        h=1.72,
        title="3D Distance-Gated GAT",
        fc="#FFFFFF",
        ec="#84B5AE",
        header_fc="#D6ECE8",
        title_color="#2D6B64",
        title_fontsize=11.5,
    )

    gcx, gcy = 3.46, 2.96
    shell = Circle((gcx, gcy), 0.37, facecolor="#F2FAF8", edgecolor="#43877E", linewidth=1.05, linestyle="--", zorder=6)
    ax.add_patch(shell)
    nbrs = [
        (gcx - 0.24, gcy + 0.20, "#4E769E"),
        (gcx + 0.25, gcy + 0.17, "#C28B4E"),
        (gcx - 0.21, gcy - 0.21, "#7862A8"),
        (gcx + 0.23, gcy - 0.19, "#BC6E80"),
    ]
    for nx_i, ny_i, hcol in nbrs:
        ax.plot([nx_i, gcx], [ny_i, gcy], color=hcol, lw=1.75, zorder=7)
        draw_3d_atom(ax, nx_i, ny_i, r=0.052, base_color=hcol, zorder=8)
    draw_3d_atom(ax, gcx, gcy, r=0.074, base_color="#2D6B64", zorder=9)

    ax.plot([3.85, 3.95], [2.96, 2.96], color="#2D6B64", lw=1.5, zorder=8)
    draw_op_circle(ax, cx=4.07, cy=2.96, symbol="⊕", radius=0.115, fc="#D6ECE8", ec="#43877E", tc="#2D6B64", fontsize=11.0)
    ax.plot([4.19, 4.29], [2.96, 2.96], color="#2D6B64", lw=1.5, zorder=8)

    mvac_grid = rng.uniform(0.15, 0.96, size=(5, 6))
    draw_mini_matrix(ax, x=4.31, y=2.64, w=0.86, h=0.64, data=mvac_grid, cmap_name="ref_contact", ec="#2D6B64", zorder=8)

    add_subpanel(ax, x=3.00, y=2.15, w=2.16, h=0.36, fc="#D6ECE8", ec="#43877E", lw=1.2, zorder=8)
    ax.text(
        4.08,
        2.33,
        r"Vacuum Conformer $\mathbf{M}_{\mathrm{vac}}$",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="bold",
        color="#2D6B64",
        zorder=10,
    )

    # -------------------------------------------------------------------------
    # Card B4: Global 3D Skip Fusion (y: 0.74 .. 1.88)
    # -------------------------------------------------------------------------
    add_card(
        ax,
        x=2.87,
        y=0.74,
        w=2.42,
        h=1.14,
        title="Global 3D Skip Fusion",
        fc="#FFFFFF",
        ec="#84B5AE",
        header_fc="#D6ECE8",
        title_color="#2D6B64",
        title_fontsize=11.5,
    )

    add_subpanel(ax, x=3.00, y=1.18, w=0.80, h=0.30, fc="#D6ECE8", ec="#43877E", lw=1.1, zorder=8)
    ax.text(3.40, 1.33, r"$\mathbf{u}'$", ha="center", va="center", fontsize=11.2, fontweight="bold", color="#2D6B64", zorder=10)

    draw_op_circle(ax, cx=4.01, cy=1.33, symbol="∥", radius=0.11, fc="#F2FAF8", ec="#43877E", tc="#2D6B64", fontsize=10.4)

    add_subpanel(ax, x=4.22, y=1.18, w=0.95, h=0.30, fc="#DFECF3", ec="#4E769E", lw=1.1, zorder=8)
    ax.text(4.695, 1.33, r"$\mathbf{p}_{\mathrm{glob}}$", ha="center", va="center", fontsize=11.0, fontweight="bold", color="#2E4D72", zorder=10)

    ax.text(
        4.08,
        0.92,
        r"Skip Vector $\mathbf{g}_{\mathrm{skip}}$",
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color="#9D6833",
        zorder=10,
    )

    # -------------------------------------------------------------------------
    # Card B5: Induced-Fit Adaptation (y: 2.35 .. 3.77) 
    # -------------------------------------------------------------------------
    add_card(
        ax,
        x=5.50,
        y=2.35,
        w=2.50,
        h=1.42,
        title="Induced-Fit Adaptation",
        fc="#FAF5EA",
        ec="#CFAF82",
        header_fc="#F4EBD7",
        title_color="#9D6833",
        title_fontsize=11.5,
    )

    add_subpanel(ax, x=5.64, y=2.94, w=0.70, h=0.38, fc="#D6ECE8", ec="#43877E", lw=1.15, zorder=8)
    ax.text(5.99, 3.13, r"$\mathbf{M}_{\mathrm{vac}}$", ha="center", va="center", fontsize=11.0, fontweight="bold", color="#2D6B64", zorder=10)

    add_subpanel(ax, x=7.20, y=2.94, w=0.66, h=0.38, fc="#EAD6B6", ec="#9D6833", lw=1.15, zorder=8)
    ax.text(7.53, 3.13, r"$\mathbf{P}$", ha="center", va="center", fontsize=11.4, fontweight="bold", color="#9D6833", zorder=10)

    arc_top = FancyArrowPatch((7.16, 3.24), (6.38, 3.24), connectionstyle="arc3,rad=0.18", arrowstyle="-|>", mutation_scale=11.5, lw=1.45, color="#9D6833", zorder=8)
    arc_bot = FancyArrowPatch((6.38, 3.02), (7.16, 3.02), connectionstyle="arc3,rad=0.18", arrowstyle="-|>", mutation_scale=11.5, lw=1.45, color="#2D6B64", zorder=8)
    ax.add_patch(arc_top)
    ax.add_patch(arc_bot)
    draw_op_circle(ax, cx=6.77, cy=3.13, symbol="⊙", radius=0.12, fc="#F4EBD7", ec="#C28B4E", tc="#9D6833", fontsize=11.2)

    add_subpanel(ax, x=5.64, y=2.45, w=2.22, h=0.36, fc="#EAD6B6", ec="#9D6833", lw=1.15, zorder=8)
    ax.text(
        6.75,
        2.63,
        r"Holo States ($\mathbf{M}_{\mathrm{holo}}, \mathbf{P}_{\mathrm{holo}}$)",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#9D6833",
        zorder=10,
    )

    # -------------------------------------------------------------------------
    # Card B6: Pocket-Resolved BAN (y: 0.74 .. 2.18) 
    # -------------------------------------------------------------------------
    add_card(
        ax,
        x=5.50,
        y=0.74,
        w=2.50,
        h=1.44,
        title="Pocket-Resolved BAN",
        fc="#FAF3F5",
        ec="#C99AA5",
        header_fc="#F2E2E6",
        title_color="#914153",
        title_fontsize=11.6,
        lw=1.75,
    )

    ax.text(5.73, 1.53, r"$\tilde{\mathbf{I}}$", ha="center", va="center", fontsize=11.6, fontweight="bold", color="#914153", zorder=10)
    ban_map = np.full((6, 9), 0.08) + rng.uniform(0.0, 0.12, size=(6, 9))
    for ar, pk, val in [(0, 2, 0.95), (1, 2, 0.80), (1, 3, 0.90), (3, 5, 0.98), (4, 5, 0.84), (4, 6, 0.88), (5, 7, 0.92)]:
        ban_map[ar, pk] = val
    draw_mini_matrix(ax, x=5.92, y=1.29, w=0.78, h=0.47, data=ban_map, cmap_name="ref_rose", ec="#914153", lw_cell=0.40)

    ax.plot([6.70, 6.84], [1.53, 1.53], color="#914153", lw=1.45, zorder=8)
    add_subpanel(ax, x=6.84, y=1.35, w=0.60, h=0.36, fc="#F2E2E6", ec="#BC6E80", lw=1.15, zorder=8)
    ax.text(7.14, 1.53, "Pool", ha="center", va="center", fontsize=10.6, fontweight="bold", color="#914153", zorder=10)

    ax.plot([7.44, 7.58], [1.53, 1.53], color="#914153", lw=1.45, zorder=8)
    draw_op_circle(ax, cx=7.72, cy=1.53, symbol="⊕", radius=0.12, fc="#DFECF3", ec="#4E769E", tc="#2E4D72", fontsize=11.0)
    ax.plot([7.72, 7.72], [1.41, 1.21], color="#914153", lw=1.55, zorder=8)

    out_b6 = FancyBboxPatch(
        (5.68, 0.83),
        2.16,
        0.38,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        facecolor="#BC6E80",
        edgecolor="#914153",
        linewidth=1.3,
        zorder=9,
    )
    ax.add_patch(out_b6)
    ax.text(
        6.76,
        1.02,
        r"Joint Repr. $\mathbf{b}'_i$",
        ha="center",
        va="center",
        fontsize=11.4,
        fontweight="bold",
        color="#FFFFFF",
        zorder=10,
    )

    # =========================================================================
    # 4. PANEL C CARDS & VISUALS (Module 3: SS-UAKD 
    # =========================================================================

    # -------------------------------------------------------------------------
    # Card C1: 3D-Seeded GAE Teacher (y: 6.26 .. 8.20)
    # -------------------------------------------------------------------------
    add_card(
        ax,
        x=9.26,
        y=6.26,
        w=4.30,
        h=1.94,
        title="3D-Seeded GAE Teacher",
        fc="#FFFFFF",
        ec="#A599C8",
        header_fc="#E5E0F5",
        title_color="#5E468A",
        title_fontsize=11.8,
    )

    add_subpanel(ax, x=9.36, y=6.36, w=1.56, h=1.40, fc="#FAF8FE", ec="#D5CEEB", zorder=6)
    vc_pts = [(9.68, 7.45), (9.68, 7.20), (9.68, 6.95)]
    vp_pts = [(10.60, 7.45), (10.60, 7.20), (10.60, 6.95)]

    ax.text(9.68, 7.64, "C", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#2D6B64", zorder=10)
    ax.text(10.60, 7.64, "P", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#2E4D72", zorder=10)

    for ic, ip in [(0, 0), (0, 1), (1, 1), (2, 0), (2, 2)]:
        ax.plot([vc_pts[ic][0], vp_pts[ip][0]], [vc_pts[ic][1], vp_pts[ip][1]], color="#5E468A", lw=1.35, alpha=0.85, zorder=7)

    for (i1, i2, rad_c) in [(0, 1, 0.45), (1, 2, 0.45), (0, 2, 0.55)]:
        ax.add_patch(FancyArrowPatch(vc_pts[i1], vc_pts[i2], connectionstyle=f"arc3,rad={rad_c}", arrowstyle="-", linestyle="--", lw=1.2, color="#43877E", zorder=7))
    for (j1, j2, rad_p) in [(0, 1, -0.45), (1, 2, -0.45), (0, 2, -0.55)]:
        ax.add_patch(FancyArrowPatch(vp_pts[j1], vp_pts[j2], connectionstyle=f"arc3,rad={rad_p}", arrowstyle="-", linestyle="--", lw=1.2, color="#4E769E", zorder=7))

    for cx_n, cy_n in vc_pts:
        draw_3d_atom(ax, cx_n, cy_n, r=0.068, base_color="#43877E", zorder=9)
    for px_n, py_n in vp_pts:
        hex_verts = [(px_n + 0.080 * math.cos(math.radians(60 * k)), py_n + 0.080 * math.sin(math.radians(60 * k))) for k in range(6)]
        ax.add_patch(Polygon(hex_verts, closed=True, facecolor="#4E769E", edgecolor="#2B3542", lw=0.8, zorder=9))

    gx = np.linspace(9.44, 10.84, 120)
    for mu_g, sig_g, col_g in [(10.06, 0.17, "#7862A8"), (10.14, 0.19, "#5E468A"), (10.21, 0.16, "#9D6833"), (10.17, 0.21, "#BC6E80")]:
        gy = 6.41 + 0.39 * np.exp(-0.5 * ((gx - mu_g) / sig_g) ** 2)
        ax.plot(gx, gy, color=col_g, lw=1.2, alpha=0.88, zorder=8)
    ax.fill_between(gx, 6.41, 6.41 + 0.38 * np.exp(-0.5 * ((gx - 10.14) / 0.19) ** 2), color="#7862A8", alpha=0.20, zorder=7)

    add_subpanel(ax, x=11.00, y=6.36, w=2.46, h=1.40, fc="#FAF8FE", ec="#D5CEEB", zorder=7)
    ax.text(12.23, 7.47, r"3D Node Seeding ($\mathbf{u}'_c, \bar{\mathbf{p}}_{\mathrm{node},p}$)", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#2E4D72", zorder=10)
    ax.text(12.23, 7.05, "GraphSAGE + MC-Dropout", ha="center", va="center", fontsize=11.2, fontweight="bold", color="#5E468A", zorder=10)
    ax.text(12.23, 6.62, r"Teacher $\mathbf{t}_i$ & Variance $\sigma^2_{T,i}$", ha="center", va="center", fontsize=11.0, fontweight="bold", color="#914153", zorder=10)

    # -------------------------------------------------------------------------
    # Card C2: Uncertainty-Gated Relational KD (y: 4.22 .. 6.08) 
    # -------------------------------------------------------------------------
    add_card(
        ax,
        x=9.26,
        y=4.22,
        w=4.30,
        h=1.86,
        title="Uncertainty-Gated Relational KD",
        fc="#FFFFFF",
        ec="#A599C8",
        header_fc="#E5E0F5",
        title_color="#5E468A",
        title_fontsize=11.8,
    )

    add_subpanel(ax, x=9.36, y=4.32, w=1.56, h=1.32, fc="#FAF8FE", ec="#D5CEEB", zorder=6)
    base_sim = np.array([
        [1.00, 0.82, 0.78, 0.18, 0.12],
        [0.82, 1.00, 0.80, 0.20, 0.15],
        [0.78, 0.80, 1.00, 0.22, 0.18],
        [0.18, 0.20, 0.22, 1.00, 0.85],
        [0.12, 0.15, 0.18, 0.85, 1.00],
    ])
    stud_sim = np.clip(base_sim + rng.normal(0.0, 0.03, size=(5, 5)), 0.0, 1.0)
    draw_mini_matrix(ax, x=9.44, y=4.48, w=0.59, h=0.59, data=stud_sim, cmap_name="ref_blue", ec="#2E4D72", lw_cell=0.42)
    ax.text(9.735, 5.34, r"$\hat{\mathbf{S}}^S$", ha="center", va="center", fontsize=11.4, fontweight="bold", color="#2E4D72", zorder=10)

    ax.annotate("", xy=(10.24, 4.78), xytext=(10.04, 4.78), arrowprops=dict(arrowstyle="<->", color="#5E468A", lw=1.35), zorder=10)

    draw_mini_matrix(ax, x=10.25, y=4.48, w=0.59, h=0.59, data=base_sim, cmap_name="ref_purple", ec="#5E468A", lw_cell=0.42)
    ax.text(10.545, 5.34, r"$\hat{\mathbf{S}}^T$", ha="center", va="center", fontsize=11.4, fontweight="bold", color="#5E468A", zorder=10)

    add_subpanel(ax, x=11.00, y=4.32, w=2.46, h=1.32, fc="#FAF8FE", ec="#D5CEEB", zorder=7)
    ax.text(12.23, 5.36, r"Reliability Weight $w_i^{\mathrm{rel}}$", ha="center", va="center", fontsize=11.2, fontweight="bold", color="#5E468A", zorder=10)
    ax.text(12.23, 4.97, "Pointwise + Relational KD", ha="center", va="center", fontsize=11.0, fontweight="bold", color="#2B3542", zorder=10)
    ax.text(12.23, 4.56, r"Distillation Loss $\mathcal{L}_{\mathrm{SS\text{-}UAKD}}$", ha="center", va="center", fontsize=11.2, fontweight="bold", color="#914153", zorder=10)

    # -------------------------------------------------------------------------
    # Card C3: Entropy-Gated Domain Alignment (y: 2.18 .. 4.04)
    # -------------------------------------------------------------------------
    add_card(
        ax,
        x=9.26,
        y=2.18,
        w=4.30,
        h=1.86,
        title="Entropy-Gated Domain Alignment",
        fc="#FFFFFF",
        ec="#A599C8",
        header_fc="#E5E0F5",
        title_color="#5E468A",
        title_fontsize=11.8,
    )

    add_subpanel(ax, x=9.36, y=2.28, w=1.56, h=1.32, fc="#FAF8FE", ec="#D5CEEB", zorder=6)
    ax.text(10.14, 3.41, r"$\mathcal{D}_s$ (●) / $\mathcal{D}_t$ (▲)", ha="center", va="center", fontsize=10.3, fontweight="bold", color="#2B3542", zorder=10)
    ax.plot([9.46, 10.82], [2.44, 3.18], color="#526071", lw=1.3, ls="--", zorder=7)

    act_src = [(9.60, 3.06), (9.78, 3.16), (9.72, 2.92)]
    act_tgt = [(9.68, 2.98), (9.88, 3.10), (9.86, 2.86)]
    ina_src = [(10.32, 2.66), (10.50, 2.52), (10.64, 2.74)]
    ina_tgt = [(10.42, 2.58), (10.58, 2.66), (10.72, 2.50)]

    ax.scatter([p[0] for p in act_src], [p[1] for p in act_src], s=38, marker="o", color="#BC6E80", edgecolor="#2B3542", lw=0.55, zorder=9)
    ax.scatter([p[0] for p in act_tgt], [p[1] for p in act_tgt], s=44, marker="^", color="#C28B4E", edgecolor="#2B3542", lw=0.55, zorder=9)
    ax.scatter([p[0] for p in ina_src], [p[1] for p in ina_src], s=38, marker="o", color="#4E769E", edgecolor="#2B3542", lw=0.55, zorder=9)
    ax.scatter([p[0] for p in ina_tgt], [p[1] for p in ina_tgt], s=44, marker="^", color="#5186A8", edgecolor="#2B3542", lw=0.55, zorder=9)

    add_subpanel(ax, x=11.00, y=2.28, w=2.46, h=1.32, fc="#FAF8FE", ec="#D5CEEB", zorder=7)
    ax.text(12.23, 3.32, r"Certainty Gate $e_i$", ha="center", va="center", fontsize=11.2, fontweight="bold", color="#5E468A", zorder=10)
    ax.text(11.38, 2.93, "GRL", ha="center", va="center", fontsize=10.8, fontweight="bold", color="#2B3542", zorder=10)
    draw_op_circle(ax, cx=11.83, cy=2.93, symbol="⊗", radius=0.105, fc="#E5E0F5", ec="#7862A8", tc="#5E468A", fontsize=10.4)
    ax.text(12.68, 2.93, r"Classifier $\hat{\mathbf{y}}_i$", ha="center", va="center", fontsize=10.8, fontweight="bold", color="#2B3542", zorder=10)
    ax.text(12.23, 2.54, r"Adversarial Loss $\mathcal{L}_{\mathrm{adv}}$", ha="center", va="center", fontsize=11.2, fontweight="bold", color="#5E468A", zorder=10)

    # -------------------------------------------------------------------------
    # Card C4: Unified Objective & CPI Prediction (y: 0.64 .. 2.00) 
    # -------------------------------------------------------------------------
    add_card(
        ax,
        x=9.26,
        y=0.64,
        w=4.30,
        h=1.36,
        title="Unified Objective & CPI Prediction",
        fc="#FAF3F5",
        ec="#914153",
        header_fc="#F2E2E6",
        title_color="#914153",
        title_fontsize=11.8,
        lw=1.80,
    )

    ax.text(
        11.41,
        1.43,
        r"$\mathcal{L}_{\mathrm{cls}} + \lambda_{\mathrm{kd}}\mathcal{L}_{\mathrm{SS\text{-}UAKD}} + \lambda_{\mathrm{pkt}}\mathcal{L}_{\mathrm{pkt}} - \beta\mathcal{L}_{\mathrm{adv}}$",
        ha="center",
        va="center",
        fontsize=11.4,
        fontweight="bold",
        color="#2B3542",
        zorder=10,
    )

    out_pill = FancyBboxPatch(
        (9.35, 0.74),
        4.12,
        0.44,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        facecolor="#BC6E80",
        edgecolor="#914153",
        linewidth=1.4,
        zorder=9,
    )
    ax.add_patch(out_pill)
    ax.text(
        11.41,
        0.96,
        r"Binding Probability  $\hat{y}_i = P(y_i = 1 \mid C_i, P_i)$",
        ha="center",
        va="center",
        fontsize=11.7,
        fontweight="bold",
        color="#FFFFFF",
        zorder=10,
    )

    # =========================================================================
    # 5. Inter-module connector arrows
    # =========================================================================

    # --- Panel A Internal Orthogonal Arrows ---
    draw_manhattan_arrow(ax, [(1.45, 6.46), (1.45, 6.26)], color="#4E769E", lw=1.65)
    draw_manhattan_arrow(ax, [(2.66, 7.45), (2.87, 7.45)], color="#2E4D72", lw=1.70)
    draw_manhattan_arrow(ax, [(2.66, 5.75), (2.765, 5.75), (2.765, 7.05), (2.87, 7.05)], color="#4E769E", lw=1.55)
    draw_manhattan_arrow(ax, [(2.66, 5.42), (2.87, 5.42)], color="#4E769E", lw=1.55)
    draw_manhattan_arrow(ax, [(4.08, 6.76), (4.08, 6.54)], color="#2E4D72", lw=1.65)
    draw_manhattan_arrow(ax, [(5.29, 7.45), (5.50, 7.45)], color="#2E4D72", lw=1.70)
    draw_manhattan_arrow(ax, [(5.29, 5.75), (5.395, 5.75), (5.395, 7.18), (5.50, 7.18)], color="#914153", lw=1.60)
    draw_manhattan_arrow(ax, [(5.29, 5.42), (5.50, 5.42)], color="#914153", lw=1.55)

    # --- Panel B Internal Orthogonal Arrows ---
    draw_manhattan_arrow(ax, [(1.45, 2.05), (1.45, 1.88)], color="#43877E", lw=1.65)
    draw_manhattan_arrow(ax, [(2.66, 2.94), (2.87, 2.94)], color="#2D6B64", lw=1.70)
    draw_manhattan_arrow(ax, [(2.66, 1.32), (2.765, 1.32), (2.765, 2.50), (2.87, 2.50)], color="#43877E", lw=1.55)
    draw_manhattan_arrow(ax, [(3.50, 2.05), (3.50, 1.88)], color="#43877E", lw=1.55)
    draw_manhattan_arrow(ax, [(5.29, 2.98), (5.50, 2.98)], color="#9D6833", lw=1.70)
    draw_manhattan_arrow(ax, [(6.75, 2.35), (6.75, 2.18)], color="#914153", lw=1.80)
    draw_manhattan_arrow(ax, [(5.29, 1.15), (5.50, 1.15)], color="#43877E", lw=1.60)

    # --- Cross-Panel Orthogonal Buses in Wide Corridors ---

    # Bus 1 (x = 8.29, Solid Warm Amber): Pocket Tokens P -> Card B5
    draw_manhattan_arrow(
        ax,
        [(8.00, 6.44), (8.29, 6.44), (8.29, 3.13), (8.00, 3.13)],
        color="#9D6833",
        lw=2.15,
        linestyle="-",
        label=r"$\mathbf{P}$",
        label_xy=(8.29, 5.95),
        label_fontsize=10.5,
        label_pad=0.10,
    )

    # Bus 2 (x = 8.58, Dashed Oxford Blue): Node seed up to C1, Global Fold p_glob down to B4 & C2
    draw_manhattan_arrow(
        ax,
        [(8.00, 5.47), (9.26, 5.47)],
        color="#4E769E",
        lw=1.60,
        linestyle="--",
    )
    draw_manhattan_arrow(
        ax,
        [(8.58, 5.47), (8.58, 7.22), (9.26, 7.22)],
        color="#4E769E",
        lw=1.60,
        linestyle="--",
        label=r"$\bar{\mathbf{p}}_{\mathrm{node},p}$",
        label_xy=(8.58, 6.84),
        label_fontsize=10.2,
        label_pad=0.08,
    )
    draw_manhattan_arrow(
        ax,
        [(8.58, 5.47), (8.58, 0.62), (4.695, 0.62), (4.695, 0.74)],
        color="#4E769E",
        lw=1.50,
        linestyle="--",
        label=r"$\mathbf{p}_{\mathrm{glob}}$",
        label_xy=(8.58, 2.38),
        label_fontsize=10.2,
        label_pad=0.08,
    )

    # Bus 3 (x = 8.85, Solid Berry Burgundy): Student Joint Repr. b'_i -> C2 KD & C3 Alignment
    draw_manhattan_arrow(
        ax,
        [(8.00, 1.02), (8.85, 1.02), (8.85, 4.97), (9.26, 4.97)],
        color="#914153",
        lw=2.15,
        linestyle="-",
        label=r"$\mathbf{b}'_i$",
        label_xy=(8.85, 1.55),
        label_fontsize=10.5,
        label_pad=0.09,
    )
    draw_manhattan_arrow(
        ax,
        [(8.85, 2.93), (9.26, 2.93)],
        color="#914153",
        lw=2.05,
        linestyle="-",
    )

    # Bus 4 (x = 9.03, Dashed Forest Teal): Conformer Prior u' -> Horiz. Corridor -> A4 & C1
    draw_manhattan_arrow(
        ax,
        [(2.02, 3.77), (2.02, 3.87), (9.03, 3.87), (9.03, 4.66), (4.08, 4.66), (4.08, 5.00)],
        color="#2D6B64",
        lw=1.55,
        linestyle="--",
        label=r"$\mathbf{u}'$",
        label_xy=(6.05, 4.66),
        label_fontsize=10.5,
        label_pad=0.10,
    )
    draw_manhattan_arrow(
        ax,
        [(9.03, 4.66), (9.03, 7.62), (9.26, 7.62)],
        color="#2D6B64",
        lw=1.55,
        linestyle="--",
    )

    # --- Panel C Internal Vertical Step-to-Step Arrows ---
    draw_manhattan_arrow(ax, [(11.41, 6.26), (11.41, 6.08)], color="#5E468A", lw=1.80)
    draw_manhattan_arrow(ax, [(11.41, 4.22), (11.41, 4.04)], color="#5E468A", lw=1.80)
    draw_manhattan_arrow(ax, [(11.41, 2.18), (11.41, 2.00)], color="#914153", lw=1.90)

    # =========================================================================
    # 6. Bottom Legend Bar — 6 Concise Items at 10.6 pt Bold (y: 0.06 .. 0.44)
    # =========================================================================
    leg_bar = FancyBboxPatch(
        (0.12, 0.06),
        13.56,
        0.38,
        boxstyle="round,pad=0.0,rounding_size=0.07",
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        linewidth=1.2,
        zorder=4,
    )
    ax.add_patch(leg_bar)

    ym = 0.25
    draw_op_circle(ax, cx=0.56, cy=ym, symbol="⊕", radius=0.12, fc="#DFECF3", ec="#4E769E", tc="#2E4D72", fontsize=11.0)
    ax.text(0.75, ym, "Residual+LN", ha="left", va="center", fontsize=10.6, fontweight="bold", color="#2B3542", zorder=10)

    draw_op_circle(ax, cx=2.56, cy=ym, symbol="⊙", radius=0.12, fc="#F4EBD7", ec="#C28B4E", tc="#9D6833", fontsize=11.0)
    ax.text(2.75, ym, "Gate", ha="left", va="center", fontsize=10.6, fontweight="bold", color="#2B3542", zorder=10)

    draw_op_circle(ax, cx=3.92, cy=ym, symbol="⊗", radius=0.12, fc="#E5E0F5", ec="#7862A8", tc="#5E468A", fontsize=11.0)
    ax.text(4.11, ym, "Conditioning", ha="left", va="center", fontsize=10.6, fontweight="bold", color="#2B3542", zorder=10)

    draw_op_circle(ax, cx=6.04, cy=ym, symbol="∥", radius=0.12, fc="#D6ECE8", ec="#43877E", tc="#2D6B64", fontsize=10.6)
    ax.text(6.23, ym, "Concat", ha="left", va="center", fontsize=10.6, fontweight="bold", color="#2B3542", zorder=10)

    draw_manhattan_arrow(ax, [(7.56, ym), (8.18, ym)], color="#2E4D72", lw=1.95, halo=False)
    ax.text(8.28, ym, "Solid: Forward", ha="left", va="center", fontsize=10.6, fontweight="bold", color="#2B3542", zorder=10)

    draw_manhattan_arrow(ax, [(10.38, ym), (11.00, ym)], color="#2D6B64", lw=1.75, linestyle="--", halo=False)
    ax.text(11.10, ym, "Dashed: StopGrad/Cond.", ha="left", va="center", fontsize=10.6, fontweight="bold", color="#2B3542", zorder=10)

    # Apply color saturation bounds
    enforce_reference_palette(ax, max_sat=0.68)

    # =========================================================================
    # 7. Save Dual Vector PDF + 300-DPI PNG & Render Dedicated Graphical Abstract
    # =========================================================================
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figures_dir / "fig1_model_architecture.pdf"
    png_path = figures_dir / "fig1_model_architecture.png"

    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.01, facecolor="#FFFFFF")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.01, facecolor="#FFFFFF")
    plt.close(fig)

    if ms_dir is not None and ms_dir.exists():
        ms_fig_dir = ms_dir / "figures"
        ms_fig_dir.mkdir(parents=True, exist_ok=True)
        for src in [pdf_path, png_path]:
            shutil.copy2(src, ms_fig_dir / src.name)

    print(
        f"[GENERATED] {pdf_path.name} ({pdf_path.stat().st_size // 1024} KB) "
        f"& {png_path.name} ({png_path.stat().st_size // 1024} KB)"
    )
    render_graphical_abstract(figures_dir, ms_dir)


def render_graphical_abstract(
    figures_dir: Path,
    ms_dir: Optional[Path] = None,
    metrics_data: Optional[dict] = None,
) -> None:
    """Renders an ACS JCIM Table of Contents (TOC) Graphic ('For Table of Contents Only')
    strictly within 3.25 in x 1.75 in (8.25 cm x 4.45 cm, 600 DPI -> 1950 x 1050 px,
    234.0 x 126.0 pt PDF) with zero manuscript title banner, 6.3-7.6 pt bold actual-size
    typography, 100% borderless bars, and the 5-Family Reference Academic Palette."""
    import json
    from matplotlib.transforms import Bbox
    from PIL import Image

    if metrics_data is None:
        lf_path = figures_dir.parent / "locked_facts.json"
        if lf_path.exists():
            with open(lf_path, "r", encoding="utf-8") as f:
                lf = json.load(f)
            metrics_data = lf["metrics_data"]["results/metrics_summary.json"]

    if metrics_data is not None:
        ind = metrics_data["inductive_sota"]
        kiba_pr = ind["KIBA"]["PRISM-CPI"]["AUPR_mean"]
        kiba_de = ind["KIBA"]["DESTIN"]["AUPR_mean"]
        kiba_gain = (kiba_pr - kiba_de) / kiba_de * 100.0

        bdb_pr = ind["BindingDB"]["PRISM-CPI"]["AUPR_mean"]
        bdb_de = ind["BindingDB"]["DESTIN"]["AUPR_mean"]
        bdb_gain = (bdb_pr - bdb_de) / bdb_de * 100.0

        vs_sum = metrics_data["interpretability_analysis"]["virtual_screening_case_study"]["target_screening_summary"]
        hit_rate = int(round(np.mean([v["PRISM_Hit_at_10_Pct"] for v in vs_sum.values()])))
    else:
        kiba_pr, kiba_de, kiba_gain = 0.4473, 0.2805, 59.5
        bdb_pr, bdb_de, bdb_gain = 0.6046, 0.5162, 17.1
        hit_rate = 90

    W_IN, H_IN, TOC_DPI = 3.25, 1.75, 600
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=TOC_DPI, facecolor="#FFFFFF")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="#FFFFFF")
    ax.set_xlim(0.0, W_IN)
    ax.set_ylim(0.0, H_IN)
    ax.axis("off")

    # Pure white background canvas (no outer title banner per ACS JCIM TOC spec)
    ax.add_patch(Rectangle((0.0, 0.0), W_IN, H_IN, facecolor="#FFFFFF", edgecolor="none", linewidth=0, zorder=0))

    def toc_column(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        bg_fc: str,
        border_ec: str,
        header_fc: str,
        header_h: float = 0.210,
        title_fs: float = 7.2,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.0,rounding_size=0.045",
                facecolor=bg_fc,
                edgecolor=border_ec,
                linewidth=0.85,
                zorder=1,
            )
        )
        hy = y + h - header_h
        ax.add_patch(
            FancyBboxPatch(
                (x, hy),
                w,
                header_h,
                boxstyle="round,pad=0.0,rounding_size=0.042",
                facecolor=header_fc,
                edgecolor=border_ec,
                linewidth=0.75,
                zorder=2,
            )
        )
        ax.add_patch(
            Rectangle(
                (x + 0.008, hy),
                w - 0.016,
                header_h * 0.48,
                facecolor=header_fc,
                edgecolor="none",
                linewidth=0,
                zorder=2,
            )
        )
        ax.text(
            x + w / 2.0,
            hy + header_h / 2.0,
            title,
            ha="center",
            va="center",
            fontsize=title_fs,
            fontweight="bold",
            color="#FFFFFF",
            zorder=4,
        )

    def toc_card(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        fc: str,
        ec: str,
        header_fc: str,
        title_col: str,
        header_h: float = 0.155,
        title_fs: float = 6.4,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.0,rounding_size=0.036",
                facecolor=fc,
                edgecolor=ec,
                linewidth=0.75,
                zorder=4,
            )
        )
        if header_h > 0.0 and title:
            hy = y + h - header_h
            ax.add_patch(
                FancyBboxPatch(
                    (x, hy),
                    w,
                    header_h,
                    boxstyle="round,pad=0.0,rounding_size=0.034",
                    facecolor=header_fc,
                    edgecolor="none",
                    linewidth=0,
                    zorder=5,
                )
            )
            ax.add_patch(
                Rectangle(
                    (x + 0.006, hy),
                    w - 0.012,
                    header_h * 0.48,
                    facecolor=header_fc,
                    edgecolor="none",
                    linewidth=0,
                    zorder=5,
                )
            )
            ax.text(
                x + w / 2.0,
                hy + header_h / 2.0,
                title,
                ha="center",
                va="center",
                fontsize=title_fs,
                fontweight="bold",
                color=title_col,
                zorder=7,
            )

    def toc_arrow(
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        color: str,
        lw: float = 1.25,
        mut: float = 7.5,
        style: str = "-|>",
    ) -> None:
        arr = FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=mut,
            linewidth=lw,
            color=color,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=9,
        )
        ax.add_patch(arr)

    def toc_atom(cx: float, cy: float, r: float, col: str, ec: str = "#FFFFFF", lw: float = 0.45) -> None:
        ax.add_patch(Circle((cx, cy), radius=r, facecolor=col, edgecolor=ec, linewidth=lw, zorder=9))

    def toc_matrix(x: float, y: float, w: float, h: float, data: np.ndarray, cmap_name: str, ec: str) -> None:
        nr, nc = data.shape
        cw, ch = w / nc, h / nr
        cmap = REF_CMAPS[cmap_name]
        for ri in range(nr):
            for ci in range(nc):
                val = float(np.clip(data[ri, ci], 0.0, 1.0))
                ax.add_patch(
                    Rectangle(
                        (x + ci * cw, y + (nr - 1 - ri) * ch),
                        cw,
                        ch,
                        facecolor=cmap(val),
                        edgecolor="#FFFFFF",
                        linewidth=0.35,
                        zorder=7,
                    )
                )
        ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=ec, linewidth=0.65, zorder=8))

    # =========================================================================
    # COLUMN 1: 3D Inputs (x: 0.025 .. 0.765, w = 0.740)
    # =========================================================================
    toc_column(
        x=0.025,
        y=0.025,
        w=0.740,
        h=1.700,
        title="3D Inputs",
        bg_fc="#F4F8FC",
        border_ec="#88A6C4",
        header_fc="#2E4D72",
        header_h=0.210,
        title_fs=7.2,
    )

    # Upper Input Card: Seq + 3Di (Folded Pocket + Muted Loop)
    toc_card(
        x=0.050,
        y=0.805,
        w=0.690,
        h=0.675,
        title="Seq + 3Di",
        fc="#FFFFFF",
        ec="#88A6C4",
        header_fc="#DFECF3",
        title_col="#2E4D72",
        header_h=0.165,
        title_fs=6.4,
    )
    ax.add_patch(
        Ellipse(
            (0.330, 1.135),
            0.32,
            0.21,
            facecolor="#FAF5EA",
            edgecolor="#C28B4E",
            linewidth=0.75,
            linestyle="--",
            zorder=6,
        )
    )
    t_fold = np.linspace(0.0, 1.0, 90)
    xf = 0.105 + 0.425 * t_fold
    yf = 1.135 + 0.070 * np.sin(4.5 * math.pi * t_fold)
    ax.plot(xf, yf, color="#2E4D72", lw=1.70, solid_capstyle="round", zorder=7)
    t_loop = np.linspace(0.0, 1.0, 45)
    xl = 0.530 + 0.165 * t_loop
    yl = yf[-1] - 0.044 * np.sin(1.8 * math.pi * t_loop)
    ax.plot(xl, yl, color="#9AA6B5", lw=1.20, linestyle="--", solid_capstyle="round", zorder=7)
    for px, py in [(0.255, 1.173), (0.335, 1.090), (0.415, 1.168)]:
        toc_atom(px, py, r=0.023, col="#BC6E80", ec="#914153", lw=0.55)
    ax.text(0.395, 0.885, "Pocket / Loop", ha="center", va="center", fontsize=6.3, fontweight="bold", color="#2B3542", zorder=8)

    # Lower Input Card: 3D Conformer (Uni-Mol Molecular Ring)
    toc_card(
        x=0.050,
        y=0.060,
        w=0.690,
        h=0.700,
        title="3D Ligand",
        fc="#FFFFFF",
        ec="#84B5AE",
        header_fc="#D6ECE8",
        title_col="#2D6B64",
        header_h=0.165,
        title_fs=6.4,
    )
    ax.add_patch(
        Circle(
            (0.395, 0.405),
            0.150,
            facecolor="#F2FAF8",
            edgecolor="#43877E",
            linewidth=0.70,
            linestyle="--",
            zorder=6,
        )
    )
    cx_m, cy_m, r_m = 0.370, 0.405, 0.084
    ring_xy = [(cx_m + r_m * math.cos(math.radians(60 * k)), cy_m + r_m * math.sin(math.radians(60 * k))) for k in range(6)]
    subs_xy = [(0.530, 0.458), (0.600, 0.390), (0.195, 0.452)]
    for k in range(6):
        p1, p2 = ring_xy[k], ring_xy[(k + 1) % 6]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#2B3542", lw=1.15, zorder=7)
    ax.plot([ring_xy[0][0], subs_xy[0][0]], [ring_xy[0][1], subs_xy[0][1]], color="#2B3542", lw=1.15, zorder=7)
    ax.plot([subs_xy[0][0], subs_xy[1][0]], [subs_xy[0][1], subs_xy[1][1]], color="#2B3542", lw=1.15, zorder=7)
    ax.plot([ring_xy[3][0], subs_xy[2][0]], [ring_xy[3][1], subs_xy[2][1]], color="#2B3542", lw=1.15, zorder=7)
    for k, (ax_x, ax_y) in enumerate(ring_xy):
        toc_atom(ax_x, ax_y, r=0.020, col="#4E769E" if k in (1, 4) else "#2B3542")
    toc_atom(subs_xy[0][0], subs_xy[0][1], r=0.022, col="#43877E")
    toc_atom(subs_xy[1][0], subs_xy[1][1], r=0.023, col="#BC6E80")
    toc_atom(subs_xy[2][0], subs_xy[2][1], r=0.022, col="#C28B4E")
    ax.text(0.395, 0.140, "Uni-Mol 3D", ha="center", va="center", fontsize=6.3, fontweight="bold", color="#2D6B64", zorder=8)

    # =========================================================================
    # COLUMN 2: PRISM-CPI (x: 0.845 .. 2.115, w = 1.270)
    # =========================================================================
    toc_column(
        x=0.845,
        y=0.025,
        w=1.270,
        h=1.700,
        title="PRISM-CPI",
        bg_fc="#F8FAFC",
        border_ec="#88A6C4",
        header_fc="#2B3542",
        header_h=0.210,
        title_fs=7.5,
    )

    # Module 1: 1. 3Di Pocket Routing + K=32 — Warm Amber-Bronze Header + Navy/Rose Borderless Bars
    toc_card(
        x=0.875,
        y=1.050,
        w=1.210,
        h=0.430,
        title="1. 3Di Pocket Routing",
        fc="#FAF5EA",
        ec="#CFAF82",
        header_fc="#9D6833",
        title_col="#FFFFFF",
        header_h=0.152,
        title_fs=6.3,
    )
    m1_hs = [0.065, 0.195, 0.215, 0.075, 0.205, 0.185, 0.060]
    m1_cs = ["#4E769E", "#BC6E80", "#BC6E80", "#4E769E", "#BC6E80", "#BC6E80", "#4E769E"]
    for bi, (bh, bc) in enumerate(zip(m1_hs, m1_cs)):
        ax.add_patch(
            Rectangle(
                (0.905 + bi * 0.061, 1.082),
                0.044,
                bh,
                facecolor=bc,
                edgecolor="none",
                linewidth=0,
                zorder=7,
            )
        )
    ax.text(1.455, 1.248, "K=32", ha="center", va="center", fontsize=6.2, fontweight="bold", color="#9D6833", zorder=8)
    toc_arrow((1.355, 1.150), (1.555, 1.150), color="#9D6833", lw=1.15, mut=6.5)
    pkt_mat = np.array(
        [
            [0.86, 0.42, 0.92, 0.56, 0.88],
            [0.48, 0.90, 0.38, 0.95, 0.52],
            [0.89, 0.54, 0.85, 0.44, 0.91],
        ]
    )
    toc_matrix(x=1.595, y=1.088, w=0.445, h=0.195, data=pkt_mat, cmap_name="ref_rose", ec="#914153")

    # Module 2: 2. Induced-Fit 3D GAT — Forest Pine-Teal Header + RBF Waves + Contact Grid
    toc_card(
        x=0.875,
        y=0.555,
        w=1.210,
        h=0.430,
        title="2. Induced-Fit 3D GAT",
        fc="#F2FAF8",
        ec="#84B5AE",
        header_fc="#2D6B64",
        title_col="#FFFFFF",
        header_h=0.152,
        title_fs=6.4,
    )
    xr = np.linspace(0.920, 1.420, 80)
    ax.plot([0.920, 1.420], [0.592, 0.592], color="#84B5AE", lw=0.65, zorder=6)
    for mu_r, col_r in [(1.02, "#4E769E"), (1.12, "#43877E"), (1.22, "#C28B4E"), (1.32, "#BC6E80")]:
        yr = 0.592 + 0.185 * np.exp(-((xr - mu_r) / 0.062) ** 2)
        ax.plot(xr, yr, color=col_r, lw=1.15, zorder=7)
    toc_arrow((1.440, 0.690), (1.615, 0.690), color="#2D6B64", lw=1.15, mut=6.2, style="<->")
    if_grid = np.array(
        [
            [0.16, 0.92, 0.24, 0.18],
            [0.88, 0.28, 0.20, 0.85],
            [0.20, 0.24, 0.95, 0.28],
            [0.18, 0.82, 0.26, 0.90],
        ]
    )
    toc_matrix(x=1.635, y=0.590, w=0.405, h=0.200, data=if_grid, cmap_name="ref_contact", ec="#2D6B64")

    # Module 3: 3. Uncertainty GAE KD — Royal Iris-Purple Header + Bipartite k-NN + Manifold
    toc_card(
        x=0.875,
        y=0.060,
        w=1.210,
        h=0.430,
        title="3. Uncertainty GAE KD",
        fc="#F6F4FC",
        ec="#A599C8",
        header_fc="#5E468A",
        title_col="#FFFFFF",
        header_h=0.152,
        title_fs=6.3,
    )
    c_nds = [(0.975, 0.275), (0.975, 0.195), (0.975, 0.115)]
    p_nds = [(1.325, 0.275), (1.325, 0.195), (1.325, 0.115)]
    for ic, ip in [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]:
        ax.plot([c_nds[ic][0], p_nds[ip][0]], [c_nds[ic][1], p_nds[ip][1]], color="#5E468A", lw=0.95, zorder=6)
    for nx_c, ny_c in c_nds:
        toc_atom(nx_c, ny_c, r=0.024, col="#43877E")
    for nx_p, ny_p in p_nds:
        toc_atom(nx_p, ny_p, r=0.024, col="#4E769E")
    toc_arrow((1.395, 0.195), (1.565, 0.195), color="#5E468A", lw=1.15, mut=6.5)
    ax.add_patch(
        FancyBboxPatch(
            (1.585, 0.092),
            0.455,
            0.206,
            boxstyle="round,pad=0.0,rounding_size=0.028",
            facecolor="#FFFFFF",
            edgecolor="#A599C8",
            linewidth=0.65,
            zorder=6,
        )
    )
    ax.plot([1.615, 2.005], [0.112, 0.278], color="#526071", lw=0.75, linestyle="--", zorder=7)
    ax.scatter([1.665, 1.725, 1.690], [0.238, 0.262, 0.205], s=11, marker="o", color="#BC6E80", edgecolors="none", zorder=8)
    ax.scatter([1.755, 1.645], [0.225, 0.258], s=13, marker="^", color="#C28B4E", edgecolors="none", zorder=8)
    ax.scatter([1.885, 1.955, 1.915], [0.145, 0.125, 0.172], s=11, marker="o", color="#4E769E", edgecolors="none", zorder=8)
    ax.scatter([1.855, 1.975], [0.132, 0.158], s=13, marker="^", color="#5186A8", edgecolors="none", zorder=8)

    # Inter-module & inter-column flow arrows
    toc_arrow((0.768, 1.255), (0.845, 1.255), color="#2E4D72", lw=1.35, mut=7.0)
    toc_arrow((0.768, 0.720), (0.845, 0.720), color="#2D6B64", lw=1.35, mut=7.0)
    toc_arrow((1.480, 1.050), (1.480, 0.988), color="#9D6833", lw=1.20, mut=6.2)
    toc_arrow((1.480, 0.555), (1.480, 0.493), color="#5E468A", lw=1.20, mut=6.2)
    toc_arrow((2.118, 0.770), (2.195, 0.770), color="#914153", lw=1.45, mut=7.4)

    # =========================================================================
    # COLUMN 3: Cold-Start Gains (x: 2.195 .. 3.225, w = 1.030)
    # =========================================================================
    toc_column(
        x=2.195,
        y=0.025,
        w=1.030,
        h=1.700,
        title="Cold-Start Gains",
        bg_fc="#FAF3F5",
        border_ec="#C99AA5",
        header_fc="#914153",
        header_h=0.210,
        title_fs=6.9,
    )

    # KPI 1: KIBA AUPR +59.5%
    toc_card(
        x=2.222,
        y=1.050,
        w=0.976,
        h=0.430,
        title="",
        fc="#FFFFFF",
        ec="#C99AA5",
        header_fc="none",
        title_col="#914153",
        header_h=0.0,
    )
    ax.text(2.710, 1.392, f"KIBA AUPR +{kiba_gain:.1f}%", ha="center", va="center", fontsize=6.25, fontweight="bold", color="#914153", zorder=8)
    ax.add_patch(Rectangle((2.262, 1.215), 0.360, 0.086, facecolor="#4E769E", edgecolor="none", linewidth=0, zorder=7))
    ax.text(2.645, 1.258, f"{kiba_de:.2f}", ha="left", va="center", fontsize=6.3, fontweight="bold", color="#2E4D72", zorder=8)
    ax.add_patch(Rectangle((2.262, 1.092), 0.585, 0.086, facecolor="#BC6E80", edgecolor="none", linewidth=0, zorder=7))
    ax.text(2.870, 1.135, f"{kiba_pr:.2f}", ha="left", va="center", fontsize=6.3, fontweight="bold", color="#914153", zorder=8)

    # KPI 2: BindingDB +17.1%
    toc_card(
        x=2.222,
        y=0.555,
        w=0.976,
        h=0.430,
        title="",
        fc="#FFFFFF",
        ec="#88A6C4",
        header_fc="none",
        title_col="#2E4D72",
        header_h=0.0,
    )
    ax.text(2.710, 0.897, f"BindingDB +{bdb_gain:.1f}%", ha="center", va="center", fontsize=6.25, fontweight="bold", color="#2E4D72", zorder=8)
    ax.add_patch(Rectangle((2.262, 0.720), 0.495, 0.086, facecolor="#4E769E", edgecolor="none", linewidth=0, zorder=7))
    ax.text(2.780, 0.763, f"{bdb_de:.2f}", ha="left", va="center", fontsize=6.3, fontweight="bold", color="#2E4D72", zorder=8)
    ax.add_patch(Rectangle((2.262, 0.597), 0.585, 0.086, facecolor="#BC6E80", edgecolor="none", linewidth=0, zorder=7))
    ax.text(2.870, 0.640, f"{bdb_pr:.2f}", ha="left", va="center", fontsize=6.3, fontweight="bold", color="#914153", zorder=8)

    # KPI 3: 90% Top-10 Hits (Sub-μM Verified)
    toc_card(
        x=2.222,
        y=0.060,
        w=0.976,
        h=0.430,
        title="",
        fc="#FFFFFF",
        ec="#84B5AE",
        header_fc="none",
        title_col="#2D6B64",
        header_h=0.0,
    )
    ax.text(2.710, 0.405, f"{hit_rate}% Top-10 Hits", ha="center", va="center", fontsize=6.35, fontweight="bold", color="#2D6B64", zorder=8)
    ax.add_patch(Rectangle((2.262, 0.236), 0.615, 0.088, facecolor="#D6ECE8", edgecolor="none", linewidth=0, zorder=6))
    ax.add_patch(Rectangle((2.262, 0.236), 0.615 * (hit_rate / 100.0), 0.088, facecolor="#43877E", edgecolor="none", linewidth=0, zorder=7))
    ax.text(2.900, 0.280, "9/10", ha="left", va="center", fontsize=6.3, fontweight="bold", color="#2D6B64", zorder=8)
    ax.text(2.710, 0.132, "Sub-μM Verified", ha="center", va="center", fontsize=6.3, fontweight="bold", color="#2B3542", zorder=8)

    enforce_reference_palette(ax, max_sat=0.68)

    ga_pdf = figures_dir / "graphical_abstract.pdf"
    ga_png = figures_dir / "graphical_abstract.png"
    exact_bbox = Bbox([[0.0, 0.0], [W_IN, H_IN]])
    fig.savefig(ga_pdf, format="pdf", dpi=TOC_DPI, bbox_inches=exact_bbox, pad_inches=0.0, facecolor="#FFFFFF")
    fig.savefig(ga_png, format="png", dpi=TOC_DPI, bbox_inches=exact_bbox, pad_inches=0.0, facecolor="#FFFFFF")
    plt.close(fig)

    if ms_dir is not None and ms_dir.exists():
        ms_fig_dir = ms_dir / "figures"
        ms_fig_dir.mkdir(parents=True, exist_ok=True)
        for src in [ga_pdf, ga_png]:
            shutil.copy2(src, ms_fig_dir / src.name)

    # Export standalone ACS JCIM portal upload copies (600 DPI RGB TIFF + vector PDF)
    final_review_dir = figures_dir.parent.parent / "06_final_review"
    if final_review_dir.exists():
        toc_pdf_out = final_review_dir / "toc_graphic_jcim.pdf"
        toc_tif_out = final_review_dir / "toc_graphic_jcim.tif"
        shutil.copy2(ga_pdf, toc_pdf_out)
        with Image.open(ga_png) as im_png:
            im_rgb = im_png.convert("RGB")
            im_rgb.save(toc_tif_out, format="TIFF", dpi=(TOC_DPI, TOC_DPI), compression="tiff_lzw")

    print(
        f"[GENERATED ACS JCIM 3.25x1.75 TOC GRAPHIC] {ga_pdf.name} ({ga_pdf.stat().st_size // 1024} KB) "
        f"& {ga_png.name} ({ga_png.stat().st_size // 1024} KB)"
    )


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    render_nc_fig1_architecture(repo_root / "figures", None)
