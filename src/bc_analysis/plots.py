"""图表生成。图表文字使用英文，避免在没有中文字体的机器上出现乱码。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .data import NEGATIVE_LABEL, POSITIVE_LABEL, TARGET  # noqa: E402

# ---- 配色（经色盲安全校验的分类色 + 中性文字色） ----
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BENIGN = "#2a78d6"  # 蓝
MALIGNANT = "#eb6834"  # 橙
CLASS_COLORS = {NEGATIVE_LABEL: BENIGN, POSITIVE_LABEL: MALIGNANT}
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32a2a"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#f0f5fb", "#86b6ef", "#256abf", "#0d366b"])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "text.color": TEXT,
    }
)


def _pretty(name: str) -> str:
    return name.replace("_", " ")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_effect_sizes(tests: pd.DataFrame, path: Path) -> Path:
    """各特征区分恶性/良性的效应量（Cohen's d）。"""
    t = tests.sort_values("cohens_d")
    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    ax.barh(
        [_pretty(f) for f in t["feature"]], t["cohens_d"], color=MALIGNANT, height=0.68, zorder=2
    )
    ax.axvline(0, color=TEXT_2, linewidth=1, zorder=3)
    # 只画数据实际触及的效应量阈值线
    thresholds = [0.8] + ([-0.8] if t["cohens_d"].min() < -0.8 else [])
    for x in thresholds:
        ax.axvline(x, color=TEXT_2, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    ax.text(
        0.86, 1.0, "← large effect threshold (d = 0.8)", color=TEXT_2, fontsize=8, va="center",
    )
    ax.set_xlim(min(-0.4, t["cohens_d"].min() - 0.15), None)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Cohen's d  (malignant mean − benign mean, in pooled SDs)")
    ax.set_title("How strongly each feature separates malignant from benign")
    ax.margins(y=0.01)
    return _save(fig, path)


def plot_top_distributions(df: pd.DataFrame, features: list[str], path: Path) -> Path:
    """效应量最大的几个特征在两类中的分布对比。"""
    n = len(features)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, feat in zip(axes, features):
        bins = np.histogram_bin_edges(df[feat], bins=30)
        for label in (NEGATIVE_LABEL, POSITIVE_LABEL):
            ax.hist(
                df.loc[df[TARGET] == label, feat],
                bins=bins,
                color=CLASS_COLORS[label],
                alpha=0.55,
                edgecolor=SURFACE,
                linewidth=0.6,
                label=label,
                zorder=2,
            )
        ax.set_title(_pretty(feat), fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("samples")
    for ax in axes[n:]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(1, 1.02))
    fig.suptitle(
        "Distributions of the most discriminative features",
        x=0.01,
        y=1.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_correlation(df: pd.DataFrame, features: list[str], path: Path) -> Path:
    corr = df[features].corr()
    labels = [_pretty(f).replace("mean ", "") for f in features]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(features)):
        for j in range(len(features)):
            v = corr.iloc[i, j]
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color=SURFACE if abs(v) > 0.6 else TEXT,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.set_label("Pearson r", color=TEXT_2)
    ax.set_title("Correlation between the ten 'mean' features")
    return _save(fig, path)


def plot_roc(roc_curves: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 1], [0, 1], color=TEXT_2, linewidth=1, linestyle=(0, (3, 3)), label="chance")
    for color, (name, (fpr, tpr, auc)) in zip(MODEL_COLORS, roc_curves.items()):
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name}  (AUC {auc:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right")
    ax.set_title("ROC curves on the held-out test set")

    # 所有模型都贴近左上角，放大该区域便于比较
    inset = ax.inset_axes([0.46, 0.38, 0.42, 0.36], zorder=5)
    inset.set_facecolor(SURFACE)
    for color, (fpr, tpr, _) in zip(MODEL_COLORS, roc_curves.values()):
        inset.plot(fpr, tpr, color=color, linewidth=2)
    inset.set_xlim(-0.005, 0.2)
    inset.set_ylim(0.85, 1.005)
    inset.tick_params(labelsize=7)
    inset.set_title("zoom: top-left corner", fontsize=8, color=TEXT_2, loc="left")
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color(GRID)
    return _save(fig, path)


def plot_confusion(cm: np.ndarray, model_name: str, path: Path) -> Path:
    labels = [NEGATIVE_LABEL, POSITIVE_LABEL]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap=SEQUENTIAL)
    ax.set_xticks([0, 1], [f"pred. {l}" for l in labels])
    ax.set_yticks([0, 1], [f"actual {l}" for l in labels])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    vmax = cm.max()
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center", fontsize=16, fontweight="bold",
                color=SURFACE if cm[i, j] > vmax * 0.5 else TEXT,
            )
    ax.set_title(f"Confusion matrix · {model_name}")
    return _save(fig, path)


def plot_coefficients(coefs: pd.Series, path: Path, top: int = 12) -> Path:
    c = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(top).sort_values()
    colors = [MALIGNANT if v > 0 else BENIGN for v in c]
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.barh([_pretty(f) for f in c.index], c.values, color=colors, height=0.66, zorder=2)
    ax.axvline(0, color=TEXT_2, linewidth=1, zorder=3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Standardized coefficient (log-odds per 1 SD)")
    ax.set_title(f"Top {top} drivers in the logistic regression model")
    lo, hi = ax.get_xlim()
    ax.text(hi, -0.9, "→ pushes toward malignant", ha="right", color=TEXT_2, fontsize=8)
    ax.text(lo, -0.9, "← pushes toward benign", ha="left", color=TEXT_2, fontsize=8)
    ax.set_ylim(-1.3, len(c) - 0.4)
    return _save(fig, path)
