"""README header figure built from the authors' own recordings (BadmintonGRF, held-out fold sub_001).

Usage:
    python3 scripts/readme_figures/hero.py --data-root /path/to/BadmintonGRF/data --run-dir outputs/runs/corrected_q75

Left: six video frames around one landing with the 2D pose used by the model, and the force-plate curve (used
only to build labels). Right: the CN-HiLDNet ranking of the same trial and the shortlist the coach reviews.
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "assets"
sys.path.insert(0, str(ROOT))
from badminton_impact_ai.io import read_csv  # noqa: E402

INK, MUTED, LINE, BLUE, ORANGE = "#1f2933", "#6b7785", "#d5dbe1", "#0072B2", "#E69F00"
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
plt.rcParams.update({"font.family": "DejaVu Sans"})


def lighten(color, amount):
    c = np.array(matplotlib.colors.to_rgb(color))
    return tuple(c + (1 - c) * amount)


def ranking(run: Path, fold: str, trial: str):
    views = defaultdict(list)
    for r in read_csv(Path(run / "predictions" / f"{fold}__cn_hildnet__cls_peak.csv")):
        if r["split"] == "test" and r["trial_id"] == trial:
            views[r["unique_impact_key_candidate"]].append((float(r["y_prob"]), float(r["y_true_cls"])))
    out = sorted(((np.mean([p for p, _ in v]), v[0][1], key) for key, v in views.items()), reverse=True)
    return out


def best_segment(data: Path, subject: str, trial: str, impact: str):
    n = impact.split("_")[-1]
    best = None
    for f in glob.glob(str(data / subject / "segments" / f"{trial}_cam*_impact_{n}.npz")):
        d = np.load(f, allow_pickle=True)
        quality = float(np.nanmean(d["scores"][:, 11:17]))
        if best is None or quality > best[0]:
            best = (quality, f)
    return best[1]


def read_frame(video: Path, index: int):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def crop_box(kp, scores, shape, size_scale=1.9, aspect=0.82):
    ok = scores > 0.3
    pts = kp[ok] if ok.sum() >= 6 else kp
    cx, cy = (pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2
    h = max(pts[:, 1].max() - pts[:, 1].min(), 200) * size_scale
    w = h * aspect
    h_img, w_img = shape[:2]
    x0, y0 = int(np.clip(cx - w / 2, 0, w_img - w)), int(np.clip(cy - h / 2, 0, h_img - h))
    return x0, y0, int(w), int(h)


def draw_pose(ax, kp, scores, x0, y0, lw=2.4, ms=4.5):
    for a, b in EDGES:
        if scores[a] > 0.3 and scores[b] > 0.3:
            ax.plot(
                [kp[a, 0] - x0, kp[b, 0] - x0],
                [kp[a, 1] - y0, kp[b, 1] - y0],
                color="white",
                lw=lw + 1.8,
                solid_capstyle="round",
            )
            ax.plot(
                [kp[a, 0] - x0, kp[b, 0] - x0],
                [kp[a, 1] - y0, kp[b, 1] - y0],
                color=BLUE,
                lw=lw,
                solid_capstyle="round",
            )
    m = scores > 0.3
    ax.plot(kp[m, 0] - x0, kp[m, 1] - y0, "o", ms=ms + 2, color="white", mec="none")
    ax.plot(kp[m, 0] - x0, kp[m, 1] - y0, "o", ms=ms, color=ORANGE, mec="none")


def frame_at(data, seg_path, offset, fixed_box=None):
    d = np.load(seg_path, allow_pickle=True)
    ev = int(d["ev_idx"])
    i = int(np.clip(ev + offset, 0, len(d["frame_indices"]) - 1))
    trial, cam = str(d["trial"]), int(d["camera"])
    video = data / str(d["subject"]) / "video" / f"cam{cam}" / f"{trial}_cam{cam}.mp4"
    img = read_frame(video, int(d["frame_indices"][i]))
    kp, sc = d["keypoints_px"][i], d["scores"][i]
    return img, kp, sc, d, i, ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ASSETS / "hero.png")
    a = ap.parse_args()
    fold, trial = "sub_001", "fatigue_stage3_01"
    rank = ranking(a.run_dir, fold, f"{trial}")
    show = [0, 1, 2, len(rank) - 1]
    segs = {r: best_segment(a.data_root, fold, f"{fold}_{trial}", rank[r][2]) for r in show}

    fig = plt.figure(figsize=(14, 6.6), facecolor="white")
    fig.text(
        0.035,
        0.94,
        "Which landings should a coach look at first?",
        fontsize=22,
        fontweight="bold",
        color=INK,
        va="center",
    )
    fig.text(
        0.035,
        0.885,
        "BadmintonImpact turns multi-camera video into a ranked shortlist: each landing is scored from 2D pose, "
        "relative to its own drill and fatigue state.",
        fontsize=10.5,
        color=MUTED,
        va="center",
    )

    # ---- left: filmstrip of the top-ranked landing
    offsets = [-42, -28, -14, 0, 14, 28]
    seg0 = segs[0]
    _, kp0, sc0, d0, i0, ev0 = frame_at(a.data_root, seg0, 0)
    img0 = frame_at(a.data_root, seg0, 0)[0]
    x0, y0, w, h = crop_box(kp0, sc0, img0.shape, size_scale=1.85, aspect=0.78)
    fig.text(
        0.035,
        0.822,
        "1",
        color="white",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        bbox={"boxstyle": "circle,pad=0.35", "fc": BLUE, "ec": "none"},
    )
    fig.text(0.052, 0.826, "Video  →  2D pose", fontsize=13, fontweight="bold", color=INK, va="center")
    fig.text(
        0.052,
        0.793,
        f"one landing, six frames ({offsets[0] / 119.88:+.2f} s … {offsets[-1] / 119.88:+.2f} s)",
        fontsize=9.5,
        color=MUTED,
        va="center",
    )
    left, width, gap = 0.035, 0.087, 0.006
    for j, off in enumerate(offsets):
        img, kp, sc, d, i, ev = frame_at(a.data_root, seg0, off)
        ax = fig.add_axes([left + j * (width + gap), 0.50, width, 0.235])
        ax.imshow(img[y0 : y0 + h, x0 : x0 + w])
        draw_pose(ax, kp, sc, x0, y0, lw=1.8, ms=3)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xticks([]), ax.set_yticks([])
        impact = off == 0
        for s in ax.spines.values():
            s.set_edgecolor(ORANGE if impact else LINE)
            s.set_linewidth(3 if impact else 1)
        ax.set_title(
            "impact" if impact else f"{off / 119.88:+.2f} s",
            fontsize=8.5,
            color=ORANGE if impact else MUTED,
            fontweight="bold" if impact else "normal",
            pad=3,
        )

    # force curve for the same landing
    d = np.load(seg0, allow_pickle=True)
    t = d["timestamps_grf"] - float(d["grf_peak_sec"])
    fz = d["grf_1200hz"][:, 2] / float(d["body_weight_N"])
    if abs(fz.min()) > abs(fz.max()):
        fz = -fz
    axf = fig.add_axes([left + 0.02, 0.17, 6 * (width + gap) - 0.05, 0.25])
    axf.fill_between(t, 0, fz, color=lighten(BLUE, 0.7), lw=0)
    axf.plot(t, fz, color=BLUE, lw=1.6)
    pk = int(np.argmax(fz))
    axf.plot([t[pk]], [fz[pk]], "o", color=ORANGE, ms=7, zorder=5)
    axf.annotate(
        f"peak {fz[pk]:.1f} BW",
        (t[pk], fz[pk]),
        xytext=(t[pk] + 0.07, fz[pk] * 0.93),
        fontsize=9,
        fontweight="bold",
        color=INK,
        va="center",
    )
    axf.set_xlim(offsets[0] / 119.88 - 0.05, offsets[-1] / 119.88 + 0.05)
    axf.set_ylim(min(0, fz.min()), fz.max() * 1.12)
    axf.set_ylabel("force (BW)", fontsize=8.5, color=MUTED)
    axf.tick_params(labelsize=8, colors=MUTED, length=2)
    for s in ("top", "right"):
        axf.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axf.spines[s].set_color(LINE)
    fig.text(
        left + 0.02,
        0.075,
        "Force plate: used only to build training labels, never an input of the model at inference.",
        fontsize=9,
        color=MUTED,
        va="center",
    )

    # ---- right: ranked review list
    fig.add_artist(plt.Line2D([0.60, 0.60], [0.06, 0.85], color=LINE, lw=1, transform=fig.transFigure))
    fig.text(
        0.625,
        0.822,
        "2",
        color="white",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        bbox={"boxstyle": "circle,pad=0.35", "fc": BLUE, "ec": "none"},
    )
    fig.text(0.642, 0.826, "Score  →  ranked review list", fontsize=13, fontweight="bold", color=INK, va="center")
    fig.text(
        0.642,
        0.793,
        f"same trial, {len(rank)} landings; the coach reviews the top 20 % ({max(1, int(np.ceil(0.2 * len(rank))))})",
        fontsize=9.5,
        color=MUTED,
        va="center",
    )
    card_h, top = 0.152, 0.752
    for n, r in enumerate(show):
        score, label, key = rank[r]
        y = top - (n + (0.22 if n == 3 else 0)) * (card_h + 0.012)
        inreview = r < 3
        fig.add_artist(
            FancyBboxPatch(
                (0.615, y - card_h),
                0.36,
                card_h,
                boxstyle="round,pad=0,rounding_size=0.01",
                fc=lighten(BLUE, 0.92) if inreview else "#f5f6f8",
                ec=BLUE if inreview else LINE,
                lw=1.3,
                transform=fig.transFigure,
                zorder=0,
            )
        )
        img, kp, sc, dd, i, ev = frame_at(a.data_root, segs[r], 0)
        bx0, by0, bw, bh = crop_box(kp, sc, img.shape, size_scale=1.7, aspect=0.8)
        axt = fig.add_axes([0.621, y - card_h + 0.01, 0.082, card_h - 0.02], zorder=3)
        axt.imshow(img[by0 : by0 + bh, bx0 : bx0 + bw])
        draw_pose(axt, kp, sc, bx0, by0, lw=1.6, ms=2.5)
        axt.set_xlim(0, bw), axt.set_ylim(bh, 0), axt.axis("off")
        fig.text(
            0.715,
            y - 0.036,
            f"#{r + 1}",
            fontsize=17,
            fontweight="bold",
            color=BLUE if inreview else MUTED,
            va="center",
        )
        fig.text(0.715, y - 0.08, f"landing {int(key.split('_')[-1])}", fontsize=8.5, color=MUTED, va="center")
        axb = fig.add_axes([0.785, y - 0.086, 0.12, 0.02], zorder=3)
        axb.barh(0, 1, color=lighten("#8a96a3", 0.7), height=1)
        axb.barh(0, score, color=BLUE if inreview else lighten("#8a96a3", 0.2), height=1)
        axb.set_xlim(0, 1), axb.axis("off")
        fig.text(
            0.915,
            y - 0.076,
            f"{score:.2f}",
            fontsize=10.5,
            fontweight="bold",
            color=INK if inreview else MUTED,
            va="center",
        )
        if label == 1:
            fig.text(0.715, y - 0.123, "● relatively high in its context", fontsize=8.2, color=BLUE, va="center")
        elif not inreview:
            fig.text(0.715, y - 0.123, "low in its context", fontsize=8.2, color=MUTED, va="center")
        if n == 2:
            fig.text(0.795, y - card_h - 0.012, "⋮", fontsize=14, color=MUTED, ha="center", va="center")
    fig.text(
        0.795,
        0.035,
        "Held-out athlete (sub_001); every threshold and model was fit on the other nine.",
        fontsize=8.6,
        color=MUTED,
        ha="center",
        va="center",
    )
    fig.savefig(a.out, dpi=200)


if __name__ == "__main__":
    main()
