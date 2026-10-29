#!/usr/bin/env python3
"""E4 figures, built from the CSVs written by analyze_e4.py.

  fig_passk_per_op.png          strict pass@1 / pass@64 vs op, lines per reward
  fig_passk_curves.png          strict pass@k vs k, per band, lines per reward
  fig_hacking_rate.png          reward-hacking rate per reward (overall + per op)
  fig7_bars_<ctx>.png           paper-Fig.7-style bars: strict pass@1 & pass@64
  fig_error_breakdown_<ctx>.png paper-Fig.8-style error-category bars per band
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e4_registry import BANDS, BY_LABEL, LABELS  # noqa: E402

CTX_TITLE = {"zoo": "crazy_zootopia (context A)", "teachers": "teachers_in_school (context B)"}
RL_LABELS = [l for l in LABELS]  # base included everywhere as the reference


def style(label):
    c = BY_LABEL[label]
    return {"color": c.color, "label": c.display,
            "linestyle": "--" if label == "base" else "-"}


def present(df, col="label"):
    return [l for l in LABELS if l in set(df[col])]


def available_ks(per_op: pd.DataFrame) -> list[int]:
    import re
    ks = sorted({int(m.group(1)) for c in per_op.columns
                 if (m := re.fullmatch(r"strict_pass@(\d+)", c))})
    return [ks[0]] if len(ks) == 1 else [ks[0], ks[-1]]


def fig_passk_per_op(per_op: pd.DataFrame, out: Path) -> None:
    ctxs = [c for c in ("zoo", "teachers") if c in set(per_op.ctx)]
    ks_pair = available_ks(per_op)
    fig, axes = plt.subplots(len(ctxs), len(ks_pair),
                             figsize=(5.5 * len(ks_pair), 4 * len(ctxs)), squeeze=False)
    for i, ctx in enumerate(ctxs):
        sub_ctx = per_op[per_op.ctx == ctx]
        for j, k in enumerate(ks_pair):
            ax = axes[i][j]
            for label in present(sub_ctx):
                s = sub_ctx[sub_ctx.label == label].sort_values("op")
                val = 100 * s[f"strict_pass@{k}"]
                lo = 100 * s[f"strict_pass@{k}_lo"]
                hi = 100 * s[f"strict_pass@{k}_hi"]
                ax.plot(s.op, val, marker="o", **style(label))
                ax.fill_between(s.op, lo, hi, color=BY_LABEL[label].color, alpha=0.15)
            ax.set_title(f"{CTX_TITLE[ctx]} — strict pass@{k}")
            ax.set_xlabel("op count")
            ax.set_ylabel("strict pass@k (%)")
            ax.set_xticks(sorted(sub_ctx.op.unique()))
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_passk_per_op.png", dpi=200)
    plt.close(fig)


def fig_passk_curves(curves: pd.DataFrame, out: Path) -> None:
    scopes = list(BANDS.keys()) + ["overall"]
    ctxs = [c for c in ("zoo", "teachers") if c in set(curves.ctx)]
    fig, axes = plt.subplots(len(ctxs), len(scopes),
                             figsize=(4.2 * len(scopes), 3.8 * len(ctxs)), squeeze=False)
    for i, ctx in enumerate(ctxs):
        for j, scope in enumerate(scopes):
            ax = axes[i][j]
            sub = curves[(curves.ctx == ctx) & (curves.scope == scope)]
            for label in present(sub):
                s = sub[sub.label == label].sort_values("k")
                ax.plot(s.k, 100 * s.strict, marker="o", markersize=3, **style(label))
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(sub.k.unique()))
            ax.set_xticklabels([str(k) for k in sorted(sub.k.unique())])
            ax.set_title(f"{ctx} — {scope}", fontsize=10)
            ax.set_xlabel("k")
            ax.set_ylabel("strict pass@k (%)")
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_passk_curves.png", dpi=200)
    plt.close(fig)


def fig_hacking(hacking: pd.DataFrame, out: Path) -> None:
    ctxs = [c for c in ("zoo", "teachers") if c in set(hacking.ctx)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    # Left: overall bars grouped by context
    ax = axes[0]
    labels_all = present(hacking)
    width = 0.8 / max(1, len(labels_all))
    xs = np.arange(len(ctxs))
    for li, label in enumerate(labels_all):
        vals, los, his = [], [], []
        for ctx in ctxs:
            m = hacking[(hacking.ctx == ctx) & (hacking.label == label)
                        & (hacking.scope == "overall")]
            vals.append(100 * m.hacking_rate.iloc[0] if len(m) else np.nan)
            los.append(100 * m.hacking_rate_lo.iloc[0] if len(m) else np.nan)
            his.append(100 * m.hacking_rate_hi.iloc[0] if len(m) else np.nan)
        pos = xs + (li - (len(labels_all) - 1) / 2) * width
        err = [np.array(vals) - np.array(los), np.array(his) - np.array(vals)]
        ax.bar(pos, vals, width=width * 0.95, color=BY_LABEL[label].color,
               label=BY_LABEL[label].display,
               yerr=err, capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(xs)
    ax.set_xticklabels([CTX_TITLE[c] for c in ctxs], fontsize=9)
    ax.set_ylabel("reward-hacking rate (%)")
    ax.set_title("Right answer via wrong steps (overall)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    # Right: per-op lines (first context with data, usually both plotted)
    ax = axes[1]
    for ctx, ls in zip(ctxs, ("-", "--")):
        for label in labels_all:
            m = hacking[(hacking.ctx == ctx) & (hacking.label == label)
                        & (hacking.scope.str.startswith("op"))].copy()
            if not len(m):
                continue
            m["op"] = m.scope.str[2:].astype(int)
            m = m.sort_values("op")
            ax.plot(m.op, 100 * m.hacking_rate, marker="o", markersize=3,
                    color=BY_LABEL[label].color, linestyle=ls,
                    label=f"{BY_LABEL[label].display} ({ctx})")
    ax.set_xlabel("op count")
    ax.set_ylabel("reward-hacking rate (%)")
    ax.set_title("Per op (solid=zoo, dashed=teachers)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "fig_hacking_rate.png", dpi=200)
    plt.close(fig)


def fig7_bars(curves: pd.DataFrame, ctx: str, out: Path) -> None:
    """Paper Fig. 7 analogue: strict pass@1 and pass@64, edge vs hard bands."""
    sub = curves[curves.ctx == ctx]
    if not len(sub):
        return
    bands = list(BANDS.keys())
    kmax = int(sub.k.max())
    panels = [(k, band) for k in dict.fromkeys((1, kmax)) for band in bands]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.75 * len(panels), 3.6),
                             squeeze=False)
    axes = axes[0]
    labels_all = present(sub)
    for ax, (k, band) in zip(axes, panels):
        vals = []
        for label in labels_all:
            m = sub[(sub.label == label) & (sub.scope == band) & (sub.k == k)]
            vals.append(100 * m.strict.iloc[0] if len(m) else np.nan)
        xs = np.arange(len(labels_all))
        ax.bar(xs, vals, color=[BY_LABEL[l].color for l in labels_all])
        for x, v in zip(xs, vals):
            if np.isfinite(v):
                ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        base_v = vals[labels_all.index("base")] if "base" in labels_all else None
        if base_v is not None and np.isfinite(base_v):
            ax.axhline(base_v, color="#555555", lw=0.8, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels([BY_LABEL[l].display for l in labels_all],
                           rotation=30, ha="right", fontsize=6.5)
        ax.set_title(f"pass@{k} — {band}", fontsize=9)
        ax.set_ylabel("strict pass@k (%)")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Strict pass@k by reward composition — {CTX_TITLE[ctx]}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"fig7_bars_{ctx}.png", dpi=200)
    plt.close(fig)


def fig_error_breakdown(errors: pd.DataFrame, ctx: str, out: Path) -> None:
    """Paper Fig. 8 analogue: shares of samples per error category and band."""
    sub = errors[errors.ctx == ctx]
    if not len(sub):
        return
    cats = ["correct", "dependency_mismatch", "missing_nodes", "unparseable"]
    bands = list(BANDS.keys())
    labels_all = present(sub)
    fig, axes = plt.subplots(len(bands), len(cats),
                             figsize=(3.3 * len(cats), 3.2 * len(bands)), squeeze=False)
    for i, band in enumerate(bands):
        for j, cat in enumerate(cats):
            ax = axes[i][j]
            vals = []
            for label in labels_all:
                m = sub[(sub.label == label) & (sub.band == band) & (sub.category == cat)]
                vals.append(100 * m.share.iloc[0] if len(m) else np.nan)
            xs = np.arange(len(labels_all))
            ax.bar(xs, vals, color=[BY_LABEL[l].color for l in labels_all])
            for x, v in zip(xs, vals):
                if np.isfinite(v):
                    ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(xs)
            ax.set_xticklabels([BY_LABEL[l].display for l in labels_all],
                               rotation=30, ha="right", fontsize=6)
            if j == 0:
                ax.set_ylabel(f"{band}\nshare of samples (%)", fontsize=8)
            if i == 0:
                ax.set_title(cat.replace("_", " "), fontsize=10)
            ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Error breakdown by reward — {CTX_TITLE[ctx]}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"fig_error_breakdown_{ctx}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--analysis-dir", type=Path, default=root / "analysis")
    args = ap.parse_args()
    d = args.analysis_dir

    per_op = pd.read_csv(d / "per_op_passk.csv")
    curves = pd.read_csv(d / "passk_curves.csv")
    hacking = pd.read_csv(d / "hacking_rate.csv")
    errors = pd.read_csv(d / "error_breakdown.csv")

    fig_passk_per_op(per_op, d)
    fig_passk_curves(curves, d)
    fig_hacking(hacking, d)
    for ctx in ("zoo", "teachers"):
        fig7_bars(curves, ctx, d)
        fig_error_breakdown(errors, ctx, d)
    print(f"[plots] wrote figures to {d}")


if __name__ == "__main__":
    main()
