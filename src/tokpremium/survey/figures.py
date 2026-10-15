"""Figures and summary tables, drawn from results/survey.parquet.

    python -m tokpremium.survey.figures

Reads the committed parquet, so anyone can redraw every figure without
re-running the survey or owning a GPU. Nothing here recomputes a metric; all
the arithmetic lives in aggregate.py where it is tested.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .aggregate import by_language, failure_mode_table, premium_matrix  # noqa: E402


def fig_premium_bars(df: pd.DataFrame, tokenizer_id: str, out: Path) -> None:
    """Token premium against English, one bar per language.

    The headline figure. Sorted so the shape of the disparity is the first
    thing a reader sees, and colored by failure mode.
    """
    fm = failure_mode_table(df)
    fm = fm[fm["tokenizer_id"] == tokenizer_id].sort_values("premium_median")
    colors = [
        "#c1440e" if m == "script-uncovered" else "#4a6fa5" for m in fm["failure_mode"]
    ]

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(fm) + 1.5))
    ax.barh(fm["lang"], fm["premium_median"], color=colors)
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("tokens relative to English, same sentence (median)")
    ax.set_title(f"Tokenization premium: {tokenizer_id}")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#c1440e"),
        plt.Rectangle((0, 0), 1, 1, color="#4a6fa5"),
    ]
    ax.legend(handles, ["script uncovered", "poor merges"], loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_premium_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Languages by tokenizers.

    If a language is expensive in every column, the problem is structural
    rather than one vendor's choice, which is the claim the report wants to
    make and this figure is the evidence for it.
    """
    wide = premium_matrix(df)
    fig, ax = plt.subplots(figsize=(1.6 * len(wide.columns) + 3, 0.3 * len(wide) + 1.5))
    im = ax.imshow(wide.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(wide.columns)), wide.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(wide.index)), wide.index)
    for i in range(wide.shape[0]):
        for j in range(wide.shape[1]):
            ax.text(j, i, f"{wide.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="median premium vs English")
    ax.set_title("Token premium across tokenizers")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_failure_modes(df: pd.DataFrame, tokenizer_id: str, out: Path) -> None:
    """Premium against sub-character rate. This is the target-selection plot.

    Two languages at the same premium sit in different places here when they
    fail for different reasons. Pick one target from each cluster rather than
    the two worst overall.
    """
    fm = failure_mode_table(df)
    fm = fm[fm["tokenizer_id"] == tokenizer_id]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(fm["subchar_rate"], fm["premium_median"], s=40, color="#4a6fa5")
    for _, r in fm.iterrows():
        ax.annotate(
            r["lang"],
            (r["subchar_rate"], r["premium_median"]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    ax.set_xlabel("fraction of tokens that are character fragments")
    ax.set_ylabel("median premium vs English")
    ax.set_title(f"Failure mode: {tokenizer_id}")
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--survey", default="results/survey.parquet")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.survey)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    by_language(df).to_csv(outdir / "by_language.csv", index=False)
    failure_mode_table(df).to_csv(outdir / "failure_modes.csv", index=False)

    for tid in sorted(df["tokenizer_id"].unique()):
        slug = tid.replace("/", "_")
        fig_premium_bars(df, tid, outdir / f"premium_{slug}.png")
        fig_failure_modes(df, tid, outdir / f"failure_{slug}.png")
    fig_premium_heatmap(df, outdir / "premium_heatmap.png")

    print(f"wrote figures and tables to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
