"""Paired differences between runs of one or more sweeps, an optional 2x2 factorial, and curves.

    python -m nanolm.paired_diffs outputs/nanolm/sweeps/laptop_recipe_quick \
        --pairs rmsnorm_rope-textbook wsd_adamw-rmsnorm_rope muon_wsd-wsd_adamw muon_cosine-rmsnorm_rope \
        --factorial rmsnorm_rope,wsd_adamw,muon_cosine,muon_wsd \
        --order textbook rmsnorm_rope wsd_adamw muon_wsd muon_cosine --figdir docs/.../figures --prefix recipe_quick

`--pairs B-A` reports B − A (negative = B is better), per seed and as a mean, with a paired t-test over
the seeds both runs share. With 2 seeds the t-test has 1 degree of freedom, so p is only small when the
two per-seed differences nearly agree; "resolved" therefore also requires |mean diff| > 2x the noise
floor (median within-run seed std over all runs given).

`--factorial a,b,c,d` = the cells (A0B0, A1B0, A0B1, A1B1) of a 2x2 design, e.g.
(adamw+cosine, adamw+wsd, muon+cosine, muon+wsd). It prints the main effect of each factor (averaged
over the other) and the interaction, all computed per seed then averaged.

Writes <prefix>_curves.png (val bpb vs. tokens, mean of seeds with a min/max band), <prefix>_bars.png
(final metric, seed error bars) and <prefix>_diffs.md.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def load(roots: list[str], metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, curves = [], []
    for root in roots:
        for mf in sorted(Path(root).glob("*/metrics.json")):
            m = json.loads(mf.read_text())
            rows.append({"name": m["name"], "seed": m["seed"], "value": m[metric], "params": m["params"],
                         "params_non_embedding": m["params_non_embedding"], "tokens_seen": m["tokens_seen"],
                         "train_min": m["train_time_sec"] / 60, "tok_per_sec": m["tok_per_sec"],
                         "peak_mem_mb": m.get("peak_mem_mb")})
            ev = mf.parent / "eval_log.jsonl"
            if ev.exists():
                c = pd.read_json(ev, lines=True)
                curves.append(c.assign(name=m["name"], seed=m["seed"]))
    if not rows:
        raise SystemExit(f"no runs under {roots}")
    return pd.DataFrame(rows), (pd.concat(curves, ignore_index=True) if curves else pd.DataFrame())


def pivot(df):
    return df.pivot_table(index="seed", columns="name", values="value")


def ttest(d: np.ndarray):
    if len(d) < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0
    t = d.mean() / (sd / math.sqrt(len(d)))
    from scipy import stats
    return float(2 * stats.t.sf(abs(t), df=len(d) - 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description="paired differences / 2x2 factorial / curves")
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--metric", default="val_bpb")
    ap.add_argument("--pairs", nargs="*", default=[], help="B-A: report B minus A")
    ap.add_argument("--factorial", default="", help="a,b,c,d = A0B0,A1B0,A0B1,A1B1")
    ap.add_argument("--factor-names", default="A,B")
    ap.add_argument("--order", nargs="*", default=None, help="run order for the summary and figures")
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--prefix", default="compare")
    ap.add_argument("--logx", action="store_true")
    args = ap.parse_args(argv)

    df, curves = load(args.roots, args.metric)
    P = pivot(df)
    order = args.order or list(dict.fromkeys(df.name))
    noise = float(np.nanmedian(P.std(ddof=1).values))
    out = [f"metric `{args.metric}` (lower is better); seeds {list(P.index)}; "
           f"noise floor (median within-run seed std) = {noise:.4f}\n",
           "| run | mean ± std | per seed | non-emb params | tokens | train time | tok/s | peak MB |",
           "|---|---:|---|---:|---:|---:|---:|---:|"]
    g = df.groupby("name")
    for n in order:
        if n not in P:
            continue
        v = P[n].dropna()
        r = g.get_group(n)
        out.append(f"| `{n}` | {v.mean():.4f} ± {v.std(ddof=1):.4f} | {' / '.join(f'{x:.4f}' for x in v)} "
                   f"| {r.params_non_embedding.iloc[0] / 1e6:.2f}M | {r.tokens_seen.iloc[0] / 1e6:.1f}M "
                   f"| {r.train_min.mean():.2f} min | {r.tok_per_sec.mean() / 1e6:.2f}M | {r.peak_mem_mb.max():.0f} |")
    if args.pairs:
        out += ["", "| comparison (B − A) | mean diff | per seed | paired p | resolved? |", "|---|---:|---|---:|---|"]
        for pr in args.pairs:
            b, a = pr.split("-", 1) if "--" not in pr else pr.split("--", 1)
            common = P[[a, b]].dropna()
            d = (common[b] - common[a]).values
            p = ttest(d)
            same_sign = len(d) > 0 and (np.all(d < 0) or np.all(d > 0))
            res = "yes" if abs(d.mean()) > 2 * noise and same_sign else "not resolved"
            out.append(f"| `{b}` − `{a}` | {d.mean():+.4f} | {' / '.join(f'{x:+.4f}' for x in d)} | {p:.3f} | {res} |")
    if args.factorial:
        a0b0, a1b0, a0b1, a1b1 = args.factorial.split(",")
        fa, fb = args.factor_names.split(",")
        c = P[[a0b0, a1b0, a0b1, a1b1]].dropna()
        mainA = ((c[a1b0] - c[a0b0]) + (c[a1b1] - c[a0b1])) / 2
        mainB = ((c[a0b1] - c[a0b0]) + (c[a1b1] - c[a1b0])) / 2
        inter = (c[a1b1] - c[a0b1]) - (c[a1b0] - c[a0b0])
        out += ["", f"2x2 factorial: cells {a0b0} | {a1b0} | {a0b1} | {a1b1}", "",
                "| effect | mean | per seed | paired p |", "|---|---:|---|---:|"]
        for lab, s in ((f"main effect of {fa}", mainA), (f"main effect of {fb}", mainB),
                       (f"interaction ({fa} effect with {fb} − without)", inter)):
            out.append(f"| {lab} | {s.mean():+.4f} | {' / '.join(f'{x:+.4f}' for x in s)} | {ttest(s.values):.3f} |")
    md = "\n".join(out) + "\n"
    print(md)
    figdir = Path(args.figdir or args.roots[0])
    figdir.mkdir(parents=True, exist_ok=True)
    (Path(args.roots[0]) / f"{args.prefix}_diffs.md").write_text(md)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(curves):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for n in order:
            c = curves[curves.name == n]
            if not len(c):
                continue
            s = c.groupby("train_tokens")["bpb"].agg(["mean", "min", "max"]).reset_index()
            (line,) = ax.plot(s.train_tokens / 1e6, s["mean"], marker=".", label=n)
            ax.fill_between(s.train_tokens / 1e6, s["min"], s["max"], color=line.get_color(), alpha=0.18)
        shown = curves[curves.name.isin(order)]
        # y range: every run's last 80% of training, so short runs are not clipped next to long ones
        tail = shown[shown.train_tokens >= shown.groupby("name").train_tokens.transform("max") * 0.2]
        ax.set_ylim(float(shown.bpb.min()) - 0.02, float(tail.bpb.max()) + 0.02)
        if args.logx:
            ax.set_xscale("log")
        ax.set_xlabel("training tokens seen (M)")
        ax.set_ylabel("val bits-per-byte (periodic estimate)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title("mean of seeds, band = min/max over seeds")
        fig.tight_layout()
        fig.savefig(figdir / f"{args.prefix}_curves.png", dpi=130)
        plt.close(fig)

    names = [n for n in order if n in P]
    mean, std = P[names].mean(), P[names].std(ddof=1).fillna(0)
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(names)), 4.5))
    ax.bar(range(len(names)), mean, yerr=std, capsize=4, color="#4C78A8")
    for i, n in enumerate(names):
        ax.scatter([i] * len(P[n].dropna()), P[n].dropna(), color="k", s=10, zorder=3)
        ax.annotate(f"{mean[n]:.3f}", (i, mean[n]), ha="center", va="bottom", fontsize=8,
                    xytext=(0, 6), textcoords="offset points")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    pad = max(float(std.max()), 2e-3) * 3 + 0.15 * float(mean.max() - mean.min())
    ax.set_ylim(float(mean.min()) - pad, float(mean.max()) + pad)
    ax.set_ylabel(f"final {args.metric} (bars: mean ± std, dots: seeds)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / f"{args.prefix}_bars.png", dpi=130)
    plt.close(fig)
    print(f"wrote {args.roots[0]}/{args.prefix}_diffs.md and {figdir}/{args.prefix}_{{curves,bars}}.png")


if __name__ == "__main__":
    main()
