import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


HISTORY_PATH      = "./checkpoints_cls_only/history.json"
TEST_RESULTS_PATH = "./checkpoints_cls_only/test_results.json"
SAVE_DIR          = "./checkpoints_cls_only"

CLASS_NAMES = [
    "Abyssinian", "Am. Bulldog", "Am. Pit Bull", "Basset Hound", "Beagle",
    "Bengal", "Birman", "Bombay", "Boxer", "Br. Shorthair", "Chihuahua",
    "Egyptian Mau", "Eng. Cocker", "Eng. Setter", "Ger. Shorthaired",
    "Gr. Pyrenees", "Havanese", "Japanese Chin", "Keeshond", "Leonberger",
    "Maine Coon", "Mini Pinscher", "Newfoundland", "Persian", "Pomeranian",
    "Pug", "Ragdoll", "Russian Blue", "Saint Bernard", "Samoyed",
    "Scottish Terr.", "Shiba Inu", "Siamese", "Sphynx",
    "Stafford. Bull", "Wheaten Terr.", "Yorkshire Terr.",
]


# ── Helper ────────────────────────────────────────────────────────────────────

def plot_single_curve(ax, epochs, train_vals, val_vals, test_val,
                      title, ylabel, lower_better=False):
    ax.plot(epochs, train_vals, label="Train", color="steelblue", linewidth=2)
    ax.plot(epochs, val_vals,   label="Val",   color="tomato",    linewidth=2)

    if test_val is not None:
        ax.axhline(y=test_val, color="green", linestyle="--",
                   linewidth=1.5, label=f"Test ({test_val:.4f})")

    best_idx   = int(np.argmin(val_vals) if lower_better else np.argmax(val_vals))
    best_val   = val_vals[best_idx]
    best_epoch = best_idx + 1
    ax.scatter(best_epoch, best_val, color="tomato", zorder=5,
               s=60, marker="o",
               label=f"Best ({best_val:.4f} @ ep{best_epoch})")

    ax.set_title(title,    fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel(ylabel,  fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))


# ── Figure 1: Loss Curves ─────────────────────────────────────────────────────

def plot_loss_curves(history, test, epochs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training & Validation — Loss Curves",
                 fontsize=14, fontweight="bold")

    plot_single_curve(axes[0], epochs,
                      history["train_loss"],     history["val_loss"],
                      test.get("test_loss"),
                      "Total Loss", "Loss", lower_better=True)

    plot_single_curve(axes[1], epochs,
                      history["train_cls_loss"], history["val_cls_loss"],
                      None,
                      "Classification Loss", "Loss", lower_better=True)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "plot_loss_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")



# ── Figure 2: Classification Metrics Curves ───────────────────────────────────

def plot_classification_metrics(history, test, epochs):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Training & Validation — Classification Metrics",
                 fontsize=14, fontweight="bold")

    plot_single_curve(axes[0, 0], epochs,
                      history["train_top1"], history["val_top1"],
                      test.get("test_top1"),
                      "Top-1 Accuracy", "Accuracy")

    plot_single_curve(axes[0, 1], epochs,
                      history["train_prec"], history["val_prec"],
                      test.get("test_precision"),
                      "Precision (Macro)", "Precision")

    plot_single_curve(axes[1, 0], epochs,
                      history["train_rec"], history["val_rec"],
                      test.get("test_recall"),
                      "Recall (Macro)", "Recall")

    plot_single_curve(axes[1, 1], epochs,
                      history["train_f1"], history["val_f1"],
                      test.get("test_f1"),
                      "F1 Score (Macro)", "F1")

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "plot_classification_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Figure 3: Per-Class Accuracy ──────────────────────────────────────────────

def plot_per_class_accuracy(cm_matrix):
    per_class_acc = cm_matrix.diagonal() / (cm_matrix.sum(axis=1) + 1e-8)

    colors = ["tomato" if a < 0.5 else
              "gold"   if a < 0.75 else
              "steelblue" for a in per_class_acc]

    fig, ax = plt.subplots(figsize=(18, 7))
    bars = ax.bar(CLASS_NAMES, per_class_acc, color=colors, edgecolor="white")

    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_title("Per-Class Accuracy — Test Set",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Breed",    fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.axhline(y=per_class_acc.mean(), color="black", linestyle="--",
               linewidth=1.5,
               label=f"Mean Acc ({per_class_acc.mean():.4f})")
    ax.grid(True, axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend = [
        Patch(color="steelblue", label=">=0.75 (Good)"),
        Patch(color="gold",      label="0.50–0.75 (Medium)"),
        Patch(color="tomato",    label="<0.50 (Poor)"),
    ]
    ax.legend(handles=legend, fontsize=9, loc="upper right")

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "plot_per_class_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Figure 4: Per-Class Precision, Recall, F1 ────────────────────────────────

def plot_per_class_prf(cm_matrix):
    cm = cm_matrix.astype(float)
    tp = cm.diagonal()
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    x     = np.arange(len(CLASS_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(22, 8))
    ax.bar(x - width, precision, width, label="Precision",
           color="steelblue", edgecolor="white")
    ax.bar(x,          recall,   width, label="Recall",
           color="tomato",    edgecolor="white")
    ax.bar(x + width,  f1,       width, label="F1 Score",
           color="seagreen",  edgecolor="white")

    ax.set_title("Per-Class Precision, Recall & F1 Score — Test Set",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Breed",  fontsize=11)
    ax.set_ylabel("Score",  fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.axhline(y=precision.mean(), color="steelblue", linestyle="--",
               linewidth=1, alpha=0.6,
               label=f"Mean Prec ({precision.mean():.4f})")
    ax.axhline(y=recall.mean(),    color="tomato",    linestyle="--",
               linewidth=1, alpha=0.6,
               label=f"Mean Rec ({recall.mean():.4f})")
    ax.axhline(y=f1.mean(),        color="seagreen",  linestyle="--",
               linewidth=1, alpha=0.6,
               label=f"Mean F1 ({f1.mean():.4f})")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "plot_per_class_prf.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Figure 5: Confusion Matrix (normalized only) ──────────────────────────────

def plot_confusion_matrix(cm_matrix):
    fig, ax = plt.subplots(figsize=(22, 20))
    cm_norm = cm_matrix.astype(float) / (cm_matrix.sum(axis=1, keepdims=True) + 1e-8)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        linewidths=0.3,
        linecolor="lightgray",
        annot_kws={"size": 6},
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title("Normalized Confusion Matrix — Test Set",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(CLASS_NAMES, rotation=0,  fontsize=7)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "plot_confusion_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")



# ── Figure 6: Classification Metrics Bar Chart ────────────────────────────────

def plot_classification_bar(test):
    metrics = {
        "Top-1 Accuracy": test["test_top1"],
        "Precision"     : test["test_precision"],
        "Recall"        : test["test_recall"],
        "F1 Score"      : test["test_f1"],
    }

    names  = list(metrics.keys())
    values = list(metrics.values())
    colors = ["steelblue", "tomato", "seagreen", "darkorange"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, values, color=colors, edgecolor="white", width=0.4)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center", va="bottom",
                fontsize=13, fontweight="bold")

    ax.set_title("Classification Metrics — Test Set",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "plot_cls_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Figure 7: Confusion Matrix With Numbers ──────────────────────────────────

def plot_confusion_matrix_with_numbers(cm_matrix):
    fig, ax = plt.subplots(figsize=(28, 24))

    cm_norm = cm_matrix.astype(float) / (cm_matrix.sum(axis=1, keepdims=True) + 1e-8)

    # Each cell shows: normalized value + raw count
    annot = np.array([
        [f"{cm_norm[i, j]:.2f}\n({int(cm_matrix[i, j])})"
         for j in range(cm_matrix.shape[1])]
        for i in range(cm_matrix.shape[0])]
    )

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        linewidths=0.4,
        linecolor="lightgray",
        annot_kws={"size": 5.5},
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title(
        "Confusion Matrix — Test Set\n(Normalized proportion + raw count)",
        fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(CLASS_NAMES, rotation=0,  fontsize=7)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "plot_confusion_matrix_numbers.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def plot():
    with open(HISTORY_PATH) as f:
        history = json.load(f)
    with open(TEST_RESULTS_PATH) as f:
        test = json.load(f)

    epochs = list(range(1, len(history["train_loss"]) + 1))
    cm_matrix = np.array(test["confusion_matrix"])

    print("Generating plots...")
    plot_loss_curves                  (history, test, epochs)
    plot_classification_metrics       (history, test, epochs)
    plot_classification_bar           (test)
    plot_per_class_accuracy           (cm_matrix)
    plot_per_class_prf                (cm_matrix)
    plot_confusion_matrix             (cm_matrix)
    plot_confusion_matrix_with_numbers(cm_matrix)

    print(f"\nAll plots saved to: {SAVE_DIR}")


if __name__ == "__main__":
    plot()