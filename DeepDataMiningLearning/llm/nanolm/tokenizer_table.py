"""Tokenizer-study table: final AND best validation bpb, the step of the best, and passes over the data.

    python -m nanolm.tokenizer_table outputs/nanolm/sweeps/laptop_tokenizers
    python -m nanolm.tokenizer_table outputs/nanolm/sweeps/laptop_tokenizers --figdir docs/lm_from_scratch_part2/figures

Per run (one run dir = one tokenizer x one seed):
    final bpb    metrics.json -> val_bpb (full validation split, last checkpoint)
    best bpb     min bpb in eval_log.jsonl (periodic estimate) and the step where it occurs
    train tokens the token count of the train split, as train.py logs it in stdout.log
                 ("[data] ... | train N tok"); bytes/token = val_bytes / val_tokens
    passes       tokens_seen / train_tokens

Writes tokenizer_table.csv / .md into the sweep dir and three figures (into --figdir or the sweep dir):
    tokenizer_curves.png   val bpb vs. tokens seen, one line per tokenizer (mean of seeds, min/max band)
    tokenizer_bars.png     final and best val bpb per tokenizer, seed error bars
    tokenizer_overfit.png  passes over the data vs. (final - best) bpb
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_RE = re.compile(r"\|\s*train ([\d,]+) tok")


def load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, curves = [], []
    for mf in sorted(root.glob("*/metrics.json")):
        d = mf.parent
        m = json.loads(mf.read_text())
        ev = pd.read_json(d / "eval_log.jsonl", lines=True)
        log = (d / "stdout.log").read_text(errors="replace") if (d / "stdout.log").exists() else ""
        hit = TRAIN_RE.search(log)
        train_tokens = int(hit.group(1).replace(",", "")) if hit else np.nan
        best = ev.loc[ev["bpb"].idxmin()]
        rows.append({
            "name": m["name"], "seed": m["seed"], "tokenizer": m["tokenizer"],
            "vocab": None, "params": m["params"], "params_non_embedding": m["params_non_embedding"],
            "bytes_per_token": m["val_bytes_per_token"], "train_tokens": train_tokens,
            "tokens_seen": m["tokens_seen"], "passes": m["tokens_seen"] / train_tokens,
            "val_loss": m["val_loss"], "val_ppl": m["val_ppl"], "final_bpb": m["val_bpb"],
            "best_bpb": float(best["bpb"]), "best_step": int(best["step"]),
            "train_min": m["train_time_sec"] / 60, "peak_mem_mb": m.get("peak_mem_mb"),
        })
        vocab = re.search(r"vocab=(\d+)", log)
        rows[-1]["vocab"] = int(vocab.group(1)) if vocab else None
        ev = ev.assign(name=m["name"], seed=m["seed"])
        curves.append(ev)
    if not rows:
        raise SystemExit(f"no runs under {root}")
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("name", sort=False)
    out = pd.DataFrame({
        "vocab": g["vocab"].first(),
        "params_M": g["params"].first() / 1e6,
        "non_emb_M": g["params_non_embedding"].first() / 1e6,
        "bytes_per_token": g["bytes_per_token"].mean(),
        "train_tokens_M": g["train_tokens"].first() / 1e6,
        "passes": g["passes"].mean(),
        "val_loss": g["val_loss"].mean(),
        "val_ppl": g["val_ppl"].mean(),
        "final_bpb": g["final_bpb"].mean(),
        "final_std": g["final_bpb"].std(ddof=1),
        "best_bpb": g["best_bpb"].mean(),
        "best_std": g["best_bpb"].std(ddof=1),
        "best_steps": g["best_step"].apply(lambda s: "/".join(str(int(v)) for v in s)),
        "final_minus_best": (g["final_bpb"].mean() - g["best_bpb"].mean()),
        "n_seeds": g.size(),
    })
    return out.sort_values("vocab", kind="stable")


def to_markdown(t: pd.DataFrame) -> str:
    head = ("| tokenizer | vocab | params (non-emb) | bytes/token | passes | val loss (nats/token) | val ppl "
            "| final val bpb | best val bpb (step, per seed) | final − best |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    lines = []
    for name, r in t.iterrows():
        lines.append(f"| `{name}` | {int(r.vocab):,} | {r.params_M:.2f}M ({r.non_emb_M:.2f}M) | {r.bytes_per_token:.2f} "
                     f"| {r.passes:.1f} | {r.val_loss:.3f} | {r.val_ppl:,.1f} "
                     f"| {r.final_bpb:.4f} ± {r.final_std:.4f} | {r.best_bpb:.4f} ± {r.best_std:.4f} ({r.best_steps}) "
                     f"| {r.final_minus_best:+.3f} |")
    return head + "\n".join(lines) + "\n"


def plot(t: pd.DataFrame, curves: pd.DataFrame, figdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(t.index)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name in order:
        c = curves[curves.name == name].groupby("train_tokens")["bpb"].agg(["mean", "min", "max"]).reset_index()
        (line,) = ax.plot(c.train_tokens / 1e6, c["mean"], marker=".", label=f"{name} (V={int(t.loc[name, 'vocab'])})")
        ax.fill_between(c.train_tokens / 1e6, c["min"], c["max"], color=line.get_color(), alpha=0.15)
    ax.set_xlabel("training tokens seen (M)")
    ax.set_ylabel("val bits-per-byte (periodic estimate)")
    ax.set_ylim(2.0, min(3.2, float(curves.bpb.max()) + 0.02))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Tokenizer study: val bpb vs. tokens (mean of seeds, min/max band)")
    fig.tight_layout()
    fig.savefig(figdir / "tokenizer_curves.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(order))
    ax.bar(x - 0.2, t.final_bpb, 0.4, yerr=t.final_std.fillna(0), capsize=3, label="final")
    ax.bar(x + 0.2, t.best_bpb, 0.4, yerr=t.best_std.fillna(0), capsize=3, label="best checkpoint")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(1.95, float(t.final_bpb.max()) + 0.1)
    ax.set_ylabel("val bits-per-byte")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "tokenizer_bars.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(t.passes, t.final_minus_best, s=40)
    placed = []  # points already labelled: put a label below its point if another point is close by
    for name, r in t.sort_values("passes").iterrows():
        close = any(abs(r.passes - x) < 2 and abs(r.final_minus_best - y) < 0.03 for x, y in placed)
        ax.annotate(name, (r.passes, r.final_minus_best), fontsize=8, xytext=(4, -11 if close else 3),
                    textcoords="offset points")
        placed.append((r.passes, r.final_minus_best))
    ax.set_xlabel("passes over the training data")
    ax.set_ylabel("final − best val bpb (overfitting)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "tokenizer_overfit.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="tokenizer study table (final/best bpb, passes)")
    ap.add_argument("root")
    ap.add_argument("--figdir", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root)
    df, curves = load(root)
    t = aggregate(df)
    df.to_csv(root / "tokenizer_runs.csv", index=False)
    t.to_csv(root / "tokenizer_table.csv")
    md = to_markdown(t)
    (root / "tokenizer_table.md").write_text(md)
    print(md)
    figdir = Path(args.figdir or root)
    figdir.mkdir(parents=True, exist_ok=True)
    plot(t, curves, figdir)
    print(f"wrote {root}/tokenizer_table.{{csv,md}}, tokenizer_runs.csv and figures in {figdir}")


if __name__ == "__main__":
    main()
