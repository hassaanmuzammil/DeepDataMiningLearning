"""Metric vs. training context length, from a sweep whose runs are named <model>_ctx<L>.

    python -m nanolm.sweep --sweep nanolm/configs/sweeps/context_length.yaml
    python -m nanolm.context_curve outputs/nanolm/sweeps/context_length
    python -m nanolm.context_curve outputs/nanolm/sweeps/context_length --out figs/context.png --metric val_loss

Prints mean +- std over seeds per (model, context) with the gain from each doubling, and writes
context_curve.png (default: into the sweep directory). A gain below ~2x the seed std is noise.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUN_RE = re.compile(r"(?P<model>.+)_ctx(?P<ctx>\d+)_s\d+")


def load(sweep_dir: Path, metric: str) -> dict[tuple[str, int], list[float]]:
    res = defaultdict(list)
    for mf in sorted(sweep_dir.glob("*/metrics.json")):
        m = RUN_RE.fullmatch(mf.parent.name)
        if m:
            res[(m["model"], int(m["ctx"]))].append(json.loads(mf.read_text())[metric])
    if not res:
        raise SystemExit(f"no <model>_ctx<L>_s<seed>/metrics.json runs under {sweep_dir}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sweep_dir")
    ap.add_argument("--metric", default="val_bpb")
    ap.add_argument("--out", default=None, help="output png (default: <sweep_dir>/context_curve.png)")
    args = ap.parse_args(argv)
    sweep = Path(args.sweep_dir)
    res = load(sweep, args.metric)

    from .viz import SERIES, style
    style()
    print(f"{'model':<14}{'ctx':>6}{'n':>3}{'mean':>10}{'std':>8}{'gain vs prev':>14}")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, model in enumerate(sorted({m for m, _ in res})):
        ctxs = sorted(c for m, c in res if m == model)
        mean = np.array([np.mean(res[(model, c)]) for c in ctxs])
        std = np.array([np.std(res[(model, c)]) for c in ctxs])
        for j, c in enumerate(ctxs):
            gain = f"{mean[j - 1] - mean[j]:+.4f}" if j else ""
            print(f"{model:<14}{c:>6}{len(res[(model, c)]):>3}{mean[j]:>10.4f}{std[j]:>8.4f}{gain:>14}")
        ax.errorbar(ctxs, mean, yerr=std, marker="o", capsize=3, color=SERIES[i % len(SERIES)], label=model)
    all_ctx = sorted({c for _, c in res})
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_ctx, [str(c) for c in all_ctx])
    ax.set_xlabel("training / evaluation context length (tokens)")
    ax.set_ylabel(f"{args.metric} (lower is better)")
    n_seeds = max(len(v) for v in res.values())
    ax.set_title(f"{args.metric} vs. context length (mean of {n_seeds} seeds, error bar = std)", loc="left", fontsize=10)
    ax.legend()
    out = Path(args.out) if args.out else sweep / "context_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
