
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Model performance results from 5-fold cross-validation
models = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost"
]

accuracy = [76.37, 83.11, 85.66]
balanced_accuracy = [77.66, 74.67, 68.95]
f1_score = [57.21, 58.995, 52.01]

# Prepare graph
x = np.arange(len(models))
width = 0.24

fig, ax = plt.subplots(figsize=(12, 7))

bars1 = ax.bar(
    x - width,
    accuracy,
    width,
    label="Accuracy"
)

bars2 = ax.bar(
    x,
    balanced_accuracy,
    width,
    label="Balanced Accuracy"
)

bars3 = ax.bar(
    x + width,
    f1_score,
    width,
    label="F1-Score"
)

# Add percentage labels
for bars in [bars1, bars2, bars3]:
    ax.bar_label(
        bars,
        fmt="%.2f%%",
        padding=3,
        fontsize=9
    )

# Graph formatting
ax.set_title(
    "Comparative Performance of Machine Learning Models",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Machine Learning Models", fontsize=12)
ax.set_ylabel("Performance (%)", fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels(models)

ax.set_ylim(0, 105)
ax.legend(loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

# Save graph
output_path = Path("data/model_performance_comparison.png")
output_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output_path, dpi=300, bbox_inches="tight")

print("MODEL PERFORMANCE GRAPH GENERATED SUCCESSFULLY!")
print(f"Saved location: {output_path.resolve()}")

plt.close()
