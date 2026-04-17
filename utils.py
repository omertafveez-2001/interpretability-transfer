import json
import math
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from pathlib import Path


_RESULTS_DIR = Path(__file__).parent / "steering-vectors" / "results"

_MODEL_DISPLAY = {
    "btlm":  "Hermes BTLM-3b",
    "llama": "Llama-2-7b",
    "mamba": "Hermes Mamba-2.8b",
    "rwkv":  "Hermes RWKV-v5-7b",
}

# ── Shared style ─────────────────────────────────────────────────────────────

PALETTE = {
    # steering behaviors
    "coordination":  "#4C72B0",
    "corrigibility": "#DD8452",
    "survival":      "#55A868",
    "sycophancy":    "#C44E52",
    "myopic":        "#8172B3",
    "hallucination": "#937860",
    "refusal":       "#DA8BC3",
    # logit-lens models — specific keys must come before generic "mamba"
    "pythia":        "#4C72B0",
    "mamba-790m":    "#f4a261",
    "mamba-1.4b":    "#DD8452",
    "mamba-2.8b":    "#a84300",
    "rwkv-v4-3b":    "#55A868",
    "btlm-3b":       "#8172B3",
}


def _resolve_color(label: str) -> str | None:
    lower = label.lower()
    for key, color in PALETTE.items():
        if key in lower:
            return color
    return None


def _apply_style(ax):
    """White background, light grid, clean spines — no font overrides."""
    ax.set_facecolor("white")
    ax.grid(True, color="#CCCCCC", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")


# ── Steering-vector layer helpers ─────────────────────────────────────────────

def _load_layer_data(model_name: str, layer: str) -> dict:
    """Read all behavior jsonl files for one model/layer into {behavior: DataFrame}."""
    layer_dir = _RESULTS_DIR / model_name / layer
    if not layer_dir.exists():
        raise FileNotFoundError(f"No results at {layer_dir}")
    data = {}
    for path in sorted(layer_dir.glob("*.jsonl")):
        rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        df   = pd.DataFrame(rows).sort_values("multiplier").reset_index(drop=True)
        data[df["behavior"].iloc[0]] = df
    return data


def plot_layer(model_name: str, layer: str, figsize=None):
    """
    Plot steering curves for every behavior at one layer of a model.

    Parameters
    ----------
    model_name : str
        One of "btlm", "llama", "mamba", "rwkv".
    layer : str
        Layer folder name, e.g. "layer10" or "layer22".
    figsize : tuple, optional
        Passed to plt.subplots. Uses matplotlib default if not given.
    """
    data        = _load_layer_data(model_name, layer)
    model_label = _MODEL_DISPLAY.get(model_name, model_name)
    layer_num   = layer.replace("layer", "")

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    _apply_style(ax)

    for behavior, df in data.items():
        ax.plot(
            df["multiplier"],
            df["mean_prob_diff"],
            marker="o",
            label=behavior.replace("_", " ").title(),
            color=_resolve_color(behavior),
            zorder=3,
        )

    ax.axhline(0, color="#888888", linewidth=1, linestyle="--", zorder=2)
    ax.set_xlabel("Multiplier")
    ax.set_ylabel("Difference in the probability of the behavior")
    ax.set_title(f"Steering Effect on {model_label} — Layer {layer_num}")
    ax.legend(frameon=True, edgecolor="#CCCCCC")

    fig.tight_layout()
    return ax


def plot_model_grid(model_name: str, ncols: int = 4, figsize=None):
    """
    Plot steering curves for all available layers of a model in a grid.

    Parameters
    ----------
    model_name : str
        One of "btlm", "llama", "mamba", "rwkv".
    ncols : int
        Columns in the grid (default 4).
    figsize : tuple, optional
        Total figure size. Uses matplotlib default scaled by grid size if not given.
    """
    model_dir = _RESULTS_DIR / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"No results for '{model_name}' at {model_dir}")

    layers = sorted(
        [d.name for d in model_dir.iterdir() if d.is_dir()],
        key=lambda x: int(x.replace("layer", "")),
    )

    all_data      = {layer: _load_layer_data(model_name, layer) for layer in layers}
    all_behaviors = sorted({b for d in all_data.values() for b in d})

    nrows = math.ceil(len(layers) / ncols)
    if figsize is None:
        figsize = (4.5 * ncols, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor="white")
    axes = axes.flatten()

    model_label = _MODEL_DISPLAY.get(model_name, model_name)

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        _apply_style(ax)

        for behavior, df in all_data[layer].items():
            ax.plot(
                df["multiplier"],
                df["mean_prob_diff"],
                marker="o",
                label=behavior.replace("_", " ").title(),
                color=_resolve_color(behavior),
                zorder=3,
            )

        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=2)
        ax.set_title(f"Layer {layer.replace('layer', '')}")
        ax.set_xlabel("Multiplier")
        ax.set_ylabel("ΔP(behavior)")

    for idx in range(len(layers), len(axes)):
        axes[idx].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=_resolve_color(b), marker="o",
                   label=b.replace("_", " ").title())
        for b in all_behaviors
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(all_behaviors),
               frameon=True, edgecolor="#CCCCCC", bbox_to_anchor=(0.5, 0))
    fig.suptitle(f"Steering Effect on {model_label} — All Layers", fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


# ── plot_steering_curves ──────────────────────────────────────────────────────

def plot_steering_curves(dfs, labels=None, x_col="multiplier", y_col="mean_prob_diff",
                         title="Steering Eval", figsize=None):
    """Plot steering-vector evaluation curves for one or more behaviours."""
    labels = labels or [f"series_{i}" for i in range(len(dfs))]

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    _apply_style(ax)

    for df, label in zip(dfs, labels):
        ax.plot(df[x_col], df[y_col], marker="o",
                label=label.replace("_", " ").title(),
                color=_resolve_color(label), zorder=3)

    ax.axhline(0, color="#888888", linewidth=1, linestyle="--", zorder=2)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(frameon=True, edgecolor="#CCCCCC")

    fig.tight_layout()
    return ax


# ── plot_layer_bits_multi_frac ────────────────────────────────────────────────

def plot_layer_bits_multi_frac(folder, title="Logit Lens — Bits-per-byte by Layer",
                               figsize=None):
    """Plot logit-lens bits-per-byte vs fractional depth for all models in a folder.

    Parameters
    ----------
    folder : str | Path
        Directory containing one JSON file per model (e.g. results/logit_lens/).
        Each file must be a flat dict of {layer_name: bpb, ..., "final": bpb}.
        The filename stem is used as the model label.
    title : str
        Plot title.
    figsize : tuple, optional
        Passed to plt.subplots.
    """
    folder = Path(folder)
    paths  = sorted(folder.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No .json files found in {folder}")

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    _apply_style(ax)

    for p in paths:
        label      = p.stem
        layer_dict = json.loads(p.read_text())

        rows = []
        for name, val in layer_dict.items():
            if name == "final":
                continue
            m = re.search(r"(\d+)$", name)
            if m:
                rows.append((int(m.group(1)), val))

        if not rows:
            continue

        df    = pd.DataFrame(rows, columns=["layer", "bits"]).sort_values("layer")
        depth = df["layer"].max()
        df["frac_depth"] = (df["layer"] + 1) / (depth + 1)

        ax.plot(df["frac_depth"], df["bits"], marker="o", markersize=3,
                label=label, color=_resolve_color(label), zorder=3)

        if "final" in layer_dict:
            ax.scatter([1.0], [layer_dict["final"]], marker="*", s=120,
                       color=_resolve_color(label), zorder=4)

    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Fraction of Depth")
    ax.set_ylabel("Bits-per-byte (bpb)")
    ax.set_title(title)
    ax.legend(frameon=True, edgecolor="#CCCCCC")

    fig.tight_layout()
    return ax
