from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..data import LABELS

COLORS = ["#52677b", "#247aab", "#cf6733"]
KINDS = ["baseline", "bert", "finbert"]


def save(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plots(out, reports, aggregate, robustness, seeds):
    folder = Path(out) / "figures"
    folder.mkdir()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["macro_f1", "accuracy"]):
        values = [aggregate[k]["classification"][metric] for k in KINDS]
        ax.bar(KINDS, [v["mean"] for v in values], yerr=[v["std"] for v in values], color=COLORS, capsize=5)
        ax.set(title=metric.replace("_", " "), ylim=(0, 1), ylabel="Test score")
    fig.suptitle("Frozen test classification — mean ± sample SD, three seeds")
    fig.tight_layout()
    save(fig, folder / "classification_comparison.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, metric in zip(axes, ["precision", "recall", "f1"]):
        for j, kind in enumerate(KINDS):
            values = [aggregate[kind]["classification"]["per_class"][label][metric] for label in LABELS]
            ax.bar(np.arange(3)+(j-1)*.25, [v["mean"] for v in values], .25,
                   yerr=[v["std"] for v in values], label=kind, color=COLORS[j], capsize=3)
        ax.set(xticks=np.arange(3), xticklabels=LABELS, title=metric, ylim=(0, 1))
    axes[-1].legend(fontsize=8)
    fig.suptitle("Per-class test metrics — mean ± sample SD")
    fig.tight_layout()
    save(fig, folder / "per_class_comparison.png")
    for kind in ["bert", "finbert"]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, seed in zip(axes, seeds):
            matrix = np.asarray(reports[f"{kind}-{seed}"]["classification"]["confusion_matrix"])
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i,j] > matrix.max()/2 else "black")
            ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
                   xlabel="Predicted", ylabel="True", title=f"{kind}, seed {seed}")
        fig.tight_layout()
        save(fig, folder / f"{kind}_confusion_matrices.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, group in enumerate(["allagree", "non_allagree"]):
        values = [aggregate[k]["subgroups"][group]["macro_f1"] for k in KINDS]
        ax.bar(np.arange(3)+(j-.5)*.3, [v["mean"] for v in values], .3,
               yerr=[v["std"] for v in values], capsize=4, label=group)
    ax.set(xticks=range(3), xticklabels=KINDS, ylim=(0, 1), ylabel="Macro-F1", title="Agreement subgroups — mean ± sample SD")
    ax.legend()
    save(fig, folder / "agreement_subgroups.png")
    for bins in [10, 15]:
        fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=True, sharey=True)
        for col, kind in enumerate(["bert", "finbert"]):
            for row, seed in enumerate(seeds):
                ax = axes[row, col]
                c = reports[f"{kind}-{seed}"]["calibration"]
                table = [b for b in c["bins"][str(bins)] if b["count"]]
                ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
                ax.plot([b["mean_confidence"] for b in table], [b["accuracy"] for b in table], "o-", color=COLORS[col+1])
                ax.set(title=f"{kind}, seed {seed}, ECE={c[f'ece_{bins}']:.3f}", xlim=(0,1), ylim=(0,1),
                       xlabel="Mean top-label confidence", ylabel="Bin accuracy")
        fig.suptitle(f"Reliability diagrams — {bins} equal-width bins; empty bins omitted")
        fig.tight_layout()
        save(fig, folder / f"reliability_{bins}_bins.png")
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True)
    for col, kind in enumerate(["bert", "finbert"]):
        for row, seed in enumerate(seeds):
            table = reports[f"{kind}-{seed}"]["calibration"]["bins"]["10"]
            axes[row,col].bar([b["lower"]+.05 for b in table], [b["count"] for b in table], width=.09, color=COLORS[col+1])
            axes[row,col].set(title=f"{kind}, seed {seed}", xlabel="Top-label confidence", ylabel="Test samples", xlim=(0,1))
    fig.tight_layout()
    save(fig, folder / "confidence_histograms.png")
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, metric in zip(axes, ["ece_10", "ece_15", "brier", "nll"]):
        values = [aggregate[k]["calibration"][metric] for k in KINDS]
        ax.bar(KINDS, [v["mean"] for v in values], yerr=[v["std"] for v in values], capsize=4, color=COLORS)
        ax.set(title=metric, ylabel="Lower is better")
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Raw probability calibration — mean ± sample SD; no fitted scaling")
    fig.tight_layout()
    save(fig, folder / "calibration_comparison.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    for j, kind in enumerate(KINDS):
        values = [robustness[kind][t]["visible_flip_rate"] for t in ["company", "numerical", "negation"]]
        ax.bar(np.arange(3)+(j-1)*.25, [v["mean"] if v else 0 for v in values], .25,
               yerr=[v["std"] if v else 0 for v in values], color=COLORS[j], label=kind, capsize=4)
    ax.set(xticks=range(3), xticklabels=["Company identity", "Monetary scale", "Negation"],
           ylabel="Class-flip rate", title="Perturbation sensitivity — model-visible edits only (not error rates)", ylim=(0,1))
    ax.legend()
    save(fig, folder / "robustness_flip_rates.png")
    ref = reports["reference"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, bins in zip(axes, [10, 15]):
        table = [b for b in ref["calibration"]["bins"][str(bins)] if b["count"]]
        ax.plot([0,1], [0,1], "--", color="gray")
        ax.plot([b["mean_confidence"] for b in table], [b["accuracy"] for b in table], "o-", color="#79628e")
        ax.set(title=f"{bins} bins", xlim=(0,1), ylim=(0,1), xlabel="Confidence", ylabel="Accuracy")
    fig.suptitle("ProsusAI external reference — NON-INDEPENDENT")
    fig.tight_layout()
    save(fig, folder / "external_reference_reliability.png")
