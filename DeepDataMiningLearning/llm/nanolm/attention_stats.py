"""Classify the attention heads of a trained GPT by where they look, with numbers (tutorial §1.6.2).

    python -m nanolm.attention_stats --run outputs/nanolm/sweeps/char_models/gpt_modern_s1337
    python -m nanolm.attention_stats --run <run> --text "ROMEO:\\nBut soft!" --text "abc abc " --inspect L3H2

For every text: draws the per-layer/per-head attention maps (same figure as `nanolm.viz attention`,
written to --out-dir instead of over the tutorial figure) and prints, for every head, the attention
weight on one relevant key position averaged over query positions t >= 1:

    prev    t-1 (previous token)
    self    t
    first   position 0
    wstart  first letter of the current word (letters >= 2 into a word, so never the same key as prev)
    delim   most recent ':' newline . , ! ? at least 2 back
    induct  repeated text only (period P): t-P+1, the token after the previous occurrence -- an
            induction head's target
    dist    mean look-back distance sum_s A[t,s] (t-s)

plus the uniform-attention baseline for prev/wstart, and for repeated text the bits/char of each copy
(a model that copies in context predicts the second copy better). A literal "\\n" in --text is a newline.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_TEXTS = {
    "shakespeare": "ROMEO:\nBut soft! what light through yonder window breaks?",
    "modern": "The new phone has a battery that lasts all day, she said.",
    "repeated": "the cat sat on the mat. " * 2,
}
DELIMS = ":\n.!?,"


def period(text: str) -> int | None:
    """Smallest P < len(text) with text == text[:P] * k, or None."""
    for p in range(1, len(text) // 2 + 1):
        if len(text) % p == 0 and text[:p] * (len(text) // p) == text:
            return p
    return None


def word_start(text: str, t: int) -> int | None:
    if not text[t].isalpha():
        return None
    s = t
    while s > 0 and (text[s - 1].isalpha() or text[s - 1] == "'"):
        s -= 1
    return s


def head_scores(A: np.ndarray, text: str, P: int | None) -> dict[str, float]:
    """A: [T, T] causal attention of one head."""
    T = len(text)
    q = range(1, T)
    out = {
        "prev": np.mean([A[t, t - 1] for t in q]),
        "self": np.mean([A[t, t] for t in q]),
        "first": np.mean([A[t, 0] for t in q]),
    }
    ws = [(t, word_start(text, t)) for t in q]
    ws = [(t, s) for t, s in ws if s is not None and t - s >= 2]
    out["wstart"] = np.mean([A[t, s] for t, s in ws]) if ws else math.nan
    delim = [i for i, c in enumerate(text) if c in DELIMS]
    dl = [(t, max(d for d in delim if d <= t - 2)) for t in q if any(d <= t - 2 for d in delim)]
    out["delim"] = np.mean([A[t, d] for t, d in dl]) if dl else math.nan
    if P:
        out["induct"] = np.mean([A[t, t - P + 1] for t in range(P, T)])
    out["dist"] = np.mean([(A[t, :t + 1] * (t - np.arange(t + 1))).sum() for t in q])
    return out


def uniform_baseline(text: str) -> dict[str, float]:
    q = range(1, len(text))
    ws = [t for t in q if (s := word_start(text, t)) is not None and t - s >= 2]
    return {"prev": np.mean([1 / (t + 1) for t in q]),
            "wstart": np.mean([1 / (t + 1) for t in ws]) if ws else math.nan}


def analyze_text(model, tok, text: str, inspect: list[str]):
    from .viz import attention_maps
    idx = torch.tensor([tok.encode(text)])
    maps = attention_maps(model, idx)
    P = period(text)
    cols = ["prev", "self", "first", "wstart", "delim"] + (["induct"] if P else []) + ["dist"]
    print(f"{'head':<8}" + "".join(f"{c:>9}" for c in cols))
    for l, M in enumerate(maps):
        for h in range(M.shape[0]):
            s = head_scores(M[h], text, P)
            print(f"L{l}H{h:<5}" + "".join(f"{s[c]:>9.3f}" for c in cols))
    base = uniform_baseline(text)
    print(f"uniform-attention baseline: prev {base['prev']:.3f}, wstart {base['wstart']:.3f}"
          + (f", induct ~{np.mean([1 / (t + 1) for t in range(P, len(text))]):.3f}" if P else ""))
    for name in inspect:                         # e.g. "L3H2": the keys this head looks at most
        l, h = (int(v) for v in name[1:].split("H"))
        mean_w = maps[l][h][1:].mean(0)
        top = mean_w.argsort()[::-1][:4]
        print(f"{name} top keys (mean weight over queries): "
              + ", ".join(f"{k}:{text[k]!r} {mean_w[k]:.3f}" for k in top))
    if P:
        with torch.no_grad():
            logits = model(idx)[0].float()
        nll = F.cross_entropy(logits[:-1], idx[0, 1:], reduction="none") / math.log(2)
        copies = [nll[max(i * P - 1, 0):(i + 1) * P - 1].mean().item() for i in range(len(text) // P)]
        print("bits/char per copy: " + ", ".join(f"copy {i + 1} {b:.3f}" for i, b in enumerate(copies)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run dir of a trained arch=gpt model")
    ap.add_argument("--text", action="append", default=None,
                    help="text to analyze (repeatable; default: Shakespeare, modern prose, repeated text)")
    ap.add_argument("--name", action="append", default=None, help="file-name label per --text")
    ap.add_argument("--out-dir", default=None, help="where to write attention_<name>.png (default: <run>/attention)")
    ap.add_argument("--inspect", nargs="*", default=[], help="heads to list top keys for, e.g. L3H2")
    args = ap.parse_args(argv)

    if args.text:
        texts = [t.replace("\\n", "\n") for t in args.text]
        names = args.name or [f"text{i}" for i in range(len(texts))]
        if len(names) != len(texts):
            raise SystemExit("give one --name per --text")
        texts = dict(zip(names, texts))
    else:
        texts = DEFAULT_TEXTS
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.run) / "attention"

    from . import viz
    viz.style()
    cfg, model, tok, _, _ = viz.load_run(args.run)
    if cfg.model.arch != "gpt":
        raise SystemExit(f"attention analysis needs an arch=gpt run, got {cfg.model.arch!r}")
    model.eval()
    print(f"run: {args.run}  ({cfg.model.n_layers} layers x {cfg.model.n_heads} heads)")
    viz.FIG_DIR = out_dir
    for name, text in texts.items():
        print(f"\n=== {name}: {text!r}  (T={len(text)})")
        analyze_text(model, tok, text, args.inspect)
        viz.fig_attention(run_dir=args.run, text=text)
        (out_dir / "p1_attention_maps.png").replace(out_dir / f"attention_{name}.png")
        print(f"wrote {out_dir / f'attention_{name}.png'}")


if __name__ == "__main__":
    main()
