"""Scaling plot: val bpb vs. non-embedding parameters and vs. training FLOPs, one line per token budget.

    python -m nanolm.scaling_plot outputs/nanolm/sweeps/laptop_scaling outputs/nanolm/sweeps/laptop_scaling_2x
    python -m nanolm.scaling_plot <dirs...> --figdir docs/lm_from_scratch_part2/figures

Each argument is a sweep dir in which every run has the same token budget D (tokens_seen). Per run:
    N      params_non_embedding (metrics.json)
    D      tokens_seen
    FLOPs  6 N D                         (the usual "6ND" estimate: forward + backward over the weights)
    FLOPs+ 6 N D + 6 L T d D             (+ attention scores QK^T and AV, Kaplan et al. 2020, eq. 2.1)

Writes scaling_table.csv/.md into the first dir and two figures:
    scaling_params.png   bpb vs. N (log x), one line per D, mean of seeds with min/max bars,
                         and a fit  L(N) = E + A N^(-alpha)  per D
    scaling_flops.png    bpb vs. 6ND (log x), all runs, with the compute-efficient frontier
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load(dirs: list[str]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for mf in sorted(Path(d).glob("*/metrics.json")):
            m = json.loads(mf.read_text())
            cfg = yaml.safe_load((mf.parent / "config.yaml").read_text())
            L, dim, T = cfg["model"]["n_layers"], cfg["model"]["dim"], cfg["train"]["seq_len"]
            N, D = m["params_non_embedding"], m["tokens_seen"]
            rows.append({"sweep": Path(d).name, "name": m["name"], "seed": m["seed"], "dim": dim, "n_layers": L,
                         "N": N, "params": m["params"], "D": D, "flops": 6 * N * D,
                         "flops_attn": 6 * N * D + 6 * L * T * dim * D, "val_bpb": m["val_bpb"],
                         "train_min": m["train_time_sec"] / 60, "tok_per_sec": m["tok_per_sec"],
                         "peak_mem_mb": m.get("peak_mem_mb")})
    if not rows:
        raise SystemExit(f"no runs under {dirs}")
    return pd.DataFrame(rows)


def fit_power(N: np.ndarray, y: np.ndarray):
    """L(N) = E + A N^-alpha by least squares; None if it does not converge."""
    try:
        from scipy.optimize import curve_fit
        f = lambda n, E, A, a: E + A * np.power(n / 1e6, -a)
        p, _ = curve_fit(f, N, y, p0=(float(y.min()) - 0.1, 0.3, 0.3), bounds=([0, 0, 0], [5, 50, 3]),
                         maxfev=20000)
        return p, f
    except Exception:
        return None, None


def frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Compute-efficient frontier: the points with lower bpb than every cheaper point."""
    pts = df.sort_values("flops")
    keep, best = [], np.inf
    for i, r in pts.iterrows():
        if r.bpb < best:
            keep.append(i)
            best = r.bpb
    return pts.loc[keep]


def main(argv=None):
    ap = argparse.ArgumentParser(description="scaling plot (bpb vs. params and FLOPs)")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--figdir", default=None)
    args = ap.parse_args(argv)
    df = load(args.dirs)
    out = Path(args.dirs[0])
    df.to_csv(out / "scaling_runs.csv", index=False)

    g = df.groupby(["D", "name"], sort=True)
    t = g.agg(dim=("dim", "first"), n_layers=("n_layers", "first"), N=("N", "first"), params=("params", "first"),
              flops=("flops", "first"), flops_attn=("flops_attn", "first"), bpb=("val_bpb", "mean"),
              bpb_std=("val_bpb", "std"), lo=("val_bpb", "min"), hi=("val_bpb", "max"),
              n_seeds=("val_bpb", "size"), train_min=("train_min", "mean"), tok_per_sec=("tok_per_sec", "mean"),
              peak_mem_mb=("peak_mem_mb", "max")).reset_index().sort_values(["D", "N"])
    t.to_csv(out / "scaling_table.csv", index=False)

    lines = ["| D (tokens) | run | dim × layers | non-emb params | D / N | 6ND FLOPs | val bpb (mean ± std) "
             "| gain vs. next smaller | train time | tok/s |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    fits = {}
    for D, s in t.groupby("D"):
        prev = None
        for _, r in s.iterrows():
            gain = "—" if prev is None else f"{r.bpb - prev:+.4f}"
            lines.append(f"| {D / 1e6:.1f}M | `{r['name']}` | {r.dim} × {r.n_layers} | {r.N / 1e6:.2f}M | {D / r.N:.1f} "
                         f"| {r.flops:.2e} | {r.bpb:.4f} ± {r.bpb_std:.4f} | {gain} | {r.train_min:.2f} min "
                         f"| {r.tok_per_sec / 1e6:.2f}M |")
            prev = r.bpb
        p, f = fit_power(s.N.values.astype(float), s.bpb.values)
        if p is not None:
            fits[D] = (p, f)
    md = "\n".join(lines) + "\n"
    if fits:
        md += "\nFit L(N) = E + A · (N / 1M)^(−α), per budget:\n\n| D | E | A | α |\n|---:|---:|---:|---:|\n"
        for D, (p, _) in fits.items():
            md += f"| {D / 1e6:.1f}M | {p[0]:.3f} | {p[1]:.3f} | {p[2]:.3f} |\n"
    # "2x model or 2x tokens?": from (N, D), both cost 2x the FLOPs. 2x tokens is measured (the 2D sweep);
    # 2x model is read off the D fit at 2N (interpolated between measured sizes, extrapolated past the last).
    Ds = sorted(t.D.unique())
    for D in Ds:
        if 2 * D not in Ds or D not in fits:
            continue
        p, f = fits[D]
        small, big = t[t.D == D].set_index("name"), t[t.D == 2 * D].set_index("name")
        md += (f"\n2x model vs. 2x tokens, starting from D = {D / 1e6:.1f}M (same FLOPs: 12 N D):\n\n"
               "| from | N | bpb now | 2x tokens (measured) | 2x model (fit at 2N) | better choice | margin |\n"
               "|---|---:|---:|---:|---:|---|---:|\n")
        for name, r in small.iterrows():
            if name not in big.index:
                continue
            two_d = big.loc[name, "bpb"]
            two_n = float(f(2 * r.N, *p))
            extrap = " (extrap.)" if 2 * r.N > small.N.max() else ""
            md += (f"| `{name}` | {r.N / 1e6:.2f}M | {r.bpb:.4f} | {two_d:.4f} | {two_n:.4f}{extrap} "
                   f"| {'2x tokens' if two_d < two_n else '2x model'} | {abs(two_d - two_n):.4f} |\n")
    (out / "scaling_table.md").write_text(md)
    print(md)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = Path(args.figdir or out)
    figdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for D, s in t.groupby("D"):
        e = ax.errorbar(s.N, s.bpb, yerr=[s.bpb - s.lo, s.hi - s.bpb], marker="o", capsize=3,
                        label=f"D = {D / 1e6:.1f}M tokens")
        for _, r in s.iterrows():
            ax.annotate(r["name"], (r.N, r.bpb), fontsize=7, xytext=(4, 4), textcoords="offset points")
        if D in fits:
            p, f = fits[D]
            n = np.geomspace(s.N.min() * 0.8, s.N.max() * 1.5, 100)
            ax.plot(n, f(n, *p), "--", color=e[0].get_color(), alpha=0.6, lw=1,
                    label=f"fit: {p[0]:.2f} + {p[1]:.2f}·(N/1M)$^{{-{p[2]:.2f}}}$")
            ax.axhline(p[0], color=e[0].get_color(), ls=":", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("non-embedding parameters N")
    ax.set_ylabel("val bits-per-byte")
    ax.set_title("Model size at a fixed token budget")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "scaling_params.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for D, s in t.groupby("D"):
        ax.errorbar(s.flops, s.bpb, yerr=[s.bpb - s.lo, s.hi - s.bpb], marker="o", capsize=3, ls="-", alpha=0.85,
                    label=f"D = {D / 1e6:.1f}M tokens")
        for _, r in s.iterrows():
            ax.annotate(r["name"], (r.flops, r.bpb), fontsize=7, xytext=(4, 4), textcoords="offset points")
    fr = frontier(t)
    ax.plot(fr.flops, fr.bpb, color="k", lw=2, alpha=0.35, label="compute-efficient frontier")
    ax.set_xscale("log")
    ax.set_xlabel("training FLOPs (6 N D)")
    ax.set_ylabel("val bits-per-byte")
    ax.set_title("bpb vs. compute")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "scaling_flops.png", dpi=130)
    plt.close(fig)
    print(f"wrote {out}/scaling_table.{{csv,md}}, scaling_runs.csv, {figdir}/scaling_params.png, scaling_flops.png")


if __name__ == "__main__":
    main()
