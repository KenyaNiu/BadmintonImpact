"""Generate the README hero figure from prepared labels and a completed run.

Usage: python3 docs/assets/make_idea_figure.py --prepared outputs/prepared --run-dir outputs/runs/corrected_q75
Left/middle: real peak-force distributions of unique fresh landings per drill (global vs per-context cut-off).
Right: the CN-HiLDNet ranking of one held-out trial (median precision@20% among trials of 12-22 landings).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde

INK, MUTED, LINE = "#1f2933", "#6b7785", "#d5dbe1"
CTX = [("stage2", "Smash", "#D55E00"), ("stage1", "Net-front", "#009E73"), ("stage3", "Reaction", "#CC79A7"), ("rally", "Rally", "#0072B2")]
BLUE = "#0072B2"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def lighten(color: str, amount: float) -> tuple[float, float, float]:
    c = np.array(matplotlib.colors.to_rgb(color))
    return tuple(c + (1 - c) * amount)


def load_peaks(prepared: Path) -> dict[str, np.ndarray]:
    seen: dict[str, dict[str, float]] = defaultdict(dict)
    for row in csv.DictReader(open(prepared / "context_labels_event_q75.csv")):
        if row["fatigue_state"] == "fresh":
            seen[row["stage"]][row["unique_impact_key_candidate"]] = float(row["peak_fz"])
    return {k: np.array(list(v.values())) for k, v in seen.items()}


def example_trial(run: Path):
    rows = list(csv.DictReader(open(run / "analysis" / "cn_hildnet__cls_peak__top_fraction.csv")))
    cand = [r for r in rows if 12 <= int(r["n_impacts"]) <= 22 and 3 <= int(r["n_positive"]) <= 7]
    cand.sort(key=lambda r: (float(r["precision_at_fraction"]), r["fold"], r["trial_id"]))
    pick = cand[len(cand) // 2]
    views = defaultdict(list)
    path = run / "predictions" / f"{pick['fold']}__cn_hildnet__cls_peak.csv"
    for r in csv.DictReader(open(path)):
        if r["split"] == "test" and r["trial_id"] == pick["trial_id"]:
            views[r["unique_impact_key_candidate"]].append((float(r["y_prob"]), float(r["y_true_cls"])))
    items = sorted(((np.mean([p for p, _ in v]), v[0][1]) for v in views.values()), reverse=True)
    return items, int(pick["k"]), pick


def distribution_panel(ax, peaks, global_cut, mode):
    xs = np.linspace(0, 9.4, 400)
    for i, (key, label, color) in enumerate(CTX):
        v = peaks[key]
        cut = global_cut if mode == "global" else float(np.quantile(v, 0.75))
        y0 = len(CTX) - 1 - i
        d = gaussian_kde(v, bw_method=0.28)(xs)
        d = 0.82 * d / max(d.max(), 1e-9)
        below, above = xs < cut, xs >= cut
        ax.fill_between(xs, y0, y0 + d, where=below, color=lighten(color, 0.72), lw=0)
        ax.fill_between(xs, y0, y0 + d, where=above, color=color, lw=0, alpha=0.95)
        ax.plot(xs, y0 + d, color=color, lw=1.2)
        ax.plot([0, 9.4], [y0, y0], color=LINE, lw=0.8, zorder=0)
        share = float((v >= cut).mean())
        ax.text(9.95, y0 + 0.3, f"{share:.0%}", ha="left", va="center", fontsize=15, fontweight="bold", color=color if share > 0.12 else MUTED)
        ax.text(9.95, y0 + 0.02, "labelled high", ha="left", va="center", fontsize=7.5, color=MUTED)
        if mode == "context":
            ax.plot([cut, cut], [y0 - 0.05, y0 + 0.95], color=INK, lw=1.6, zorder=3)
            ax.text(cut + 0.14, y0 + 0.62, f"τ = {cut:.1f}", ha="left", va="center", fontsize=8, color=INK, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9), zorder=5)
    if mode == "global":
        ax.plot([global_cut, global_cut], [-0.15, len(CTX) - 0.05], color=INK, lw=2, ls=(0, (4, 2)), zorder=3)
        ax.text(global_cut + 0.15, len(CTX) - 0.08, f"one cut-off {global_cut:.1f} BW", fontsize=8.5, color=INK, fontweight="bold", va="bottom")
    ax.set_yticks([len(CTX) - 1 - i + 0.2 for i in range(len(CTX))], [c[1] for c in CTX], fontsize=10.5, fontweight="bold")
    for t, (_, _, color) in zip(ax.get_yticklabels(), CTX):
        t.set_color(color)
    ax.set_xlim(0, 12.6)
    ax.set_ylim(-0.25, len(CTX) + 0.35)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED, length=2)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlabel("peak vertical force, body weights (BW)", fontsize=8.5, color=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LINE)


def ranking_panel(ax, items, k):
    n = len(items)
    ax.set_xlim(-1.2, 1.55)
    ax.set_ylim(n + 0.3, -1.95)
    ax.add_patch(FancyBboxPatch((-1.15, -0.48), 2.65, k, boxstyle="round,pad=0.02,rounding_size=0.12", fc=lighten(BLUE, 0.9), ec=BLUE, lw=1.2))
    for rank, (score, label) in enumerate(items):
        y = rank
        top = rank < k
        ax.text(-1.05, y, f"{rank + 1}", ha="center", va="center", fontsize=9, color=INK if top else MUTED, fontweight="bold" if top else "normal")
        ax.barh(y, score, left=0, height=0.62, color=BLUE if label == 1 else lighten("#8a96a3", 0.45), zorder=3)
        ax.text(score + 0.04, y, f"{score:.2f}", va="center", fontsize=8, color=INK if top else MUTED)
        if label == 1:
            ax.text(1.42, y, "●", ha="center", va="center", fontsize=9, color=BLUE)
    ax.text(-1.15, -0.86, f"review top 20 %  ({k} of {n} landings)", ha="left", va="center", fontsize=9, color=BLUE, fontweight="bold")
    ax.plot([-1.15, 1.5], [k - 0.5, k - 0.5], color=BLUE, lw=1.2, ls=(0, (3, 2)))
    ax.text(-1.05, -1.5, "rank", fontsize=8, color=MUTED, ha="center")
    ax.text(0.0, -1.5, "model score", fontsize=8, color=MUTED, ha="left")
    ax.text(1.42, -1.5, "true high", fontsize=7.5, color=MUTED, ha="center", va="center", linespacing=0.9)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "idea.png")
    args = parser.parse_args()
    peaks = load_peaks(args.prepared)
    global_cut = float(np.quantile(np.concatenate(list(peaks.values())), 0.75))
    items, k, pick = example_trial(args.run_dir)

    fig = plt.figure(figsize=(13, 6.5), facecolor="white")
    fig.text(0.04, 0.945, "Which landings should a coach look at first?", fontsize=21, fontweight="bold", color=INK, va="center")
    fig.text(0.04, 0.885, "BadmintonImpact scores each landing by how demanding it is for its own drill and fatigue state, then reviews only the top of the list.",
             fontsize=11, color=MUTED, va="center")
    heads = [(0.04, "1", "A single force cut-off", "the label just reveals the drill"),
             (0.395, "2", "A cut-off per context (ours)", "the label means relatively severe"),
             (0.705, "3", "A ranked review list", "the coach starts at the top")]
    for x, num, title, sub in heads:
        fig.patches.append(plt.Circle((x + 0.012, 0.795), 0.0135, transform=fig.transFigure, color=BLUE if num in "23" else "#8a96a3", figure=fig))
        fig.text(x + 0.012, 0.795, num, color="white", fontsize=11, fontweight="bold", ha="center", va="center")
        fig.text(x + 0.034, 0.805, title, fontsize=12.5, fontweight="bold", color=INK, va="center")
        fig.text(x + 0.034, 0.772, sub, fontsize=9.5, color=MUTED, va="center")

    ax1 = fig.add_axes([0.075, 0.28, 0.255, 0.46])
    ax2 = fig.add_axes([0.43, 0.28, 0.235, 0.46])
    ax3 = fig.add_axes([0.705, 0.255, 0.265, 0.49])
    distribution_panel(ax1, peaks, global_cut, "global")
    distribution_panel(ax2, peaks, global_cut, "context")
    ranking_panel(ax3, items, k)
    for xa in (0.338, 0.682):
        fig.add_artist(plt.Line2D([xa, xa], [0.2, 0.74], color=LINE, lw=1, transform=fig.transFigure))
    cap = dict(fontsize=8.8, color=MUTED, va="top", linespacing=1.35)
    fig.text(0.075, 0.185, "Smash always looks \"high\" and rally never does:\na model can win by recognising the drill.", **cap)
    fig.text(0.43, 0.185, "Each context uses its own train-subject-only 75th\npercentile, so every drill has high and low landings.", **cap)
    fig.text(0.705, 0.185, f"Example held-out trial ({pick['fold']}, {pick['trial_id']}):\nCN-HiLDNet scores; ● = relatively high in its context.", **cap)

    fig.add_artist(FancyBboxPatch((0.04, 0.02), 0.93, 0.095, boxstyle="round,pad=0,rounding_size=0.012", fc=lighten(BLUE, 0.92), ec="none", transform=fig.transFigure))
    stats = [("0.942", "unique-impact AUROC (HGB 0.861)"), ("0.536 / 0.635", "precision / recall at top 20 %"), ("10", "leave-one-subject-out folds"), ("2,242", "physical landings")]
    for x, (big, small) in zip((0.16, 0.38, 0.60, 0.82), stats):
        fig.text(x, 0.087, big, fontsize=15, fontweight="bold", color=BLUE, va="center", ha="center")
        fig.text(x, 0.045, small, fontsize=9, color=INK, va="center", ha="center")
    fig.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
