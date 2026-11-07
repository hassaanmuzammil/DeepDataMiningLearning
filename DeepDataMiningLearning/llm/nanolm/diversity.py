"""Sample diversity vs. temperature and top-p for ONE trained model (so val bpb is fixed).

    python -m nanolm.diversity --run outputs/nanolm/sweeps/char_models/gpt_modern_s1337
    python -m nanolm.diversity --run <run> --temperatures 0.5 1.0 --top-ps 0.9 1.0 --samples 32 --out d.png

Only the decoding rule changes between settings; the model, and hence its validation bpb, does not.
For every setting it draws --samples continuations of --length tokens (batched, one prompt per
sample = a random --prompt-len window of the validation text) with the same sample_next() that
nanolm.generate uses, and measures on the generated part only:

    distinct-4  fraction of the character 4-grams in a sample that are distinct (mean over samples;
                per-sample at a fixed length, because the corpus-level ratio falls with length)
    word-valid  fraction of generated words ([A-Za-z']+, lower-cased) that occur in the TRAINING
                text -- a cheap quality proxy for a character model

Reference: the same metrics on --samples real validation windows of the same length.
Writes <run>/diversity.{json,png} unless --out is given.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

WORD_RE = re.compile(r"[A-Za-z']+")


def distinct_ngrams(text: str, n: int = 4) -> float:
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / len(grams) if grams else float("nan")


def word_valid(text: str, vocab: set[str]) -> float:
    words = [w.lower() for w in WORD_RE.findall(text)]
    return sum(w in vocab for w in words) / len(words) if words else float("nan")


@torch.no_grad()
def sample_batch(model, prompts: torch.Tensor, n_new: int, ctx_len: int, dinfo,
                 temperature: float, top_p: float, seed: int) -> torch.Tensor:
    """Continue every row of prompts [S, P] by n_new tokens; returns only the new tokens [S, n_new]."""
    from .device import autocast_ctx
    from .evaluate import sample_next
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = prompts.to(dinfo.device)
    for _ in range(n_new):
        with autocast_ctx(dinfo):
            logits = model(idx[:, -ctx_len:])[:, -1, :]
        idx = torch.cat([idx, sample_next(logits, temperature, 0, top_p, g)], dim=1)
    return idx[:, prompts.size(1):].cpu()


def measure(texts: list[str], vocab: set[str]) -> dict:
    d4 = np.array([distinct_ngrams(t) for t in texts])
    wv = np.array([word_valid(t, vocab) for t in texts])
    return {"distinct4": float(d4.mean()), "distinct4_std": float(d4.std()),
            "word_valid": float(np.nanmean(wv)), "word_valid_std": float(np.nanstd(wv))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--temperatures", type=float, nargs="*", default=[0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5],
                    help="temperature sweep, at top-p 1.0 (0 = greedy)")
    ap.add_argument("--top-ps", type=float, nargs="*", default=[0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
                    help="top-p sweep, at --p-temperature")
    ap.add_argument("--p-temperature", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--length", type=int, default=1000, help="generated tokens per sample")
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output png (json is written next to it)")
    args = ap.parse_args(argv)

    from .data import read_shard
    from .generate import load_run
    cfg, model, tok, dinfo, data_meta = load_run(args.run)
    ctx_len = cfg.train.seq_len
    metrics = json.loads((Path(args.run) / "metrics.json").read_text())
    train_text = tok.decode(np.asarray(read_shard(data_meta["train_files"][0])).tolist())
    val = np.asarray(read_shard(data_meta["val_files"][0]))
    vocab = {w.lower() for w in WORD_RE.findall(train_text)}

    rng = np.random.default_rng(args.seed)
    starts = rng.integers(0, len(val) - args.prompt_len - args.length, size=2 * args.samples)
    prompts = torch.tensor(np.stack([val[s:s + args.prompt_len] for s in starts[:args.samples]]).astype(np.int64))
    reference = measure([tok.decode(val[s:s + args.length].tolist()) for s in starts[args.samples:]], vocab)

    print(f"run: {args.run}  val_bpb {metrics['val_bpb']:.4f} (fixed)  ctx {ctx_len}  "
          f"{args.samples} samples x {args.length} tokens")
    print(f"{'setting':<18}{'distinct-4':>12}{'word-valid':>12}")
    print(f"{'real val text':<18}{reference['distinct4']:>12.3f}{reference['word_valid']:>12.3f}")
    results = {"run": args.run, "val_bpb": metrics["val_bpb"], "samples": args.samples, "length": args.length,
               "reference": reference, "temperature": [], "top_p": [], "examples": {}}
    settings = [("temperature", t, t, 1.0) for t in args.temperatures] + \
               [("top_p", p, args.p_temperature, p) for p in args.top_ps]       # (sweep, x value, T, p)
    for kind, value, temp, top_p in settings:
        new = sample_batch(model, prompts, args.length, ctx_len, dinfo, temp, top_p, args.seed)
        texts = [tok.decode(row.tolist()) for row in new]
        r = {"value": value, "temperature": temp, "top_p": top_p, **measure(texts, vocab)}
        results[kind].append(r)
        results["examples"][f"T={temp} p={top_p}"] = texts[0][:300]
        label = f"T={temp:g}" if kind == "temperature" else f"p={top_p:g} (T={temp:g})"
        print(f"{label:<18}{r['distinct4']:>12.3f}{r['word_valid']:>12.3f}")

    out = Path(args.out) if args.out else Path(args.run) / "diversity.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=1))
    plot(results, out, args.p_temperature)
    print(f"wrote {out} and {out.with_suffix('.json')}")


def plot(results: dict, out: Path, p_temperature: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .viz import INK2, SERIES, style
    style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ref = results["reference"]
    for ax, kind, xlabel in [(axes[0], "temperature", "temperature (top-p = 1)"),
                             (axes[1], "top_p", f"top-p (temperature = {p_temperature:g})")]:
        rows = results[kind]
        x = [r["value"] for r in rows]
        for i, (key, name) in enumerate([("distinct4", "distinct-4 (diversity)"),
                                         ("word_valid", "word-valid (quality proxy)")]):
            ax.errorbar(x, [r[key] for r in rows], yerr=[r[key + "_std"] for r in rows], marker="o",
                        capsize=3, color=SERIES[i], label=name)
            ax.axhline(ref[key], color=SERIES[i], ls="--", lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("fraction (mean over samples, bar = std)")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[1].text(0.02, 0.03, "dashed = real validation text", transform=axes[1].transAxes, fontsize=8, color=INK2)
    ax = axes[2]                                   # the trade-off: quality at equal diversity
    for i, (kind, name, fmt) in enumerate([("temperature", "temperature sweep", "T={:g}"),
                                           ("top_p", "top-p sweep", "p={:g}")]):
        rows = results[kind]
        ax.plot([r["distinct4"] for r in rows], [r["word_valid"] for r in rows], marker="o",
                color=SERIES[2 + i], label=name)
        for r in rows:
            ax.annotate(fmt.format(r["value"]), (r["distinct4"], r["word_valid"]), fontsize=6.5, color=INK2,
                        xytext=(3, 3 if kind == "temperature" else -9), textcoords="offset points")
    ax.plot([ref["distinct4"]], [ref["word_valid"]], marker="*", ms=12, ls="", color=INK2, label="real val text")
    ax.set_xlabel("distinct-4 (diversity)")
    ax.set_ylabel("word-valid (quality proxy)")
    ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Sample diversity vs. decoding at fixed val bpb {results['val_bpb']:.3f} "
                 f"({results['samples']} x {results['length']} chars per setting)", x=0.01, ha="left", fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
