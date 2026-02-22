"""
Explore age_gender.csv: show sample images, labels, and data statistics in a single figure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load data
df = pd.read_csv("data/MNIST/raw/age_gender.csv")

# Parse pixels and reshape to 48x48 images
def parse_pixels(pixel_str):
    return np.array(pixel_str.split(), dtype=np.uint8).reshape(48, 48)

# Ethnicity and gender labels for display
ETHNICITY_NAMES = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}
GENDER_NAMES = {0: "Male", 1: "Female"}

# Sample diverse examples: different ages, genders, ethnicities
np.random.seed(42)
sample_indices = []
for g in [0, 1]:
    for e in range(5):
        subset = df[(df["gender"] == g) & (df["ethnicity"] == e)]
        if len(subset) > 0:
            idx = subset.sample(1, random_state=42 + e + g * 5).index[0]
            sample_indices.append(idx)

# Ensure we have 10 samples (2 rows x 5 cols), fill with random if needed
while len(sample_indices) < 10:
    extra = df.sample(1, random_state=len(sample_indices) * 7).index[0]
    if extra not in sample_indices:
        sample_indices.append(extra)
sample_indices = sample_indices[:10]

fig = plt.figure(figsize=(16, 12), constrained_layout=True)
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.3)

# --- Top: Sample images (2 rows x 5 cols) ---
gs_top = gs[0, :].subgridspec(2, 5, hspace=0.15, wspace=0.1)
for i, idx in enumerate(sample_indices):
    row = df.loc[idx]
    img = parse_pixels(row["pixels"])
    ax = fig.add_subplot(gs_top[i // 5, i % 5])
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Age: {row['age']}, {GENDER_NAMES[row['gender']]}\n{ETHNICITY_NAMES[row['ethnicity']]}", fontsize=8)
    ax.axis("off")

# Add section title
fig.text(0.5, 0.92, "Sample Faces with Labels", ha="center", fontsize=14, fontweight="bold")

# --- Bottom left: Age distribution histogram ---
ax_age = fig.add_subplot(gs[1, 0])
ax_age.hist(df["age"], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
ax_age.axvline(df["age"].median(), color="red", linestyle="--", linewidth=2, label=f"Median: {df['age'].median():.0f}")
ax_age.axvline(df["age"].mean(), color="orange", linestyle=":", linewidth=2, label=f"Mean: {df['age'].mean():.1f}")
ax_age.set_xlabel("Age")
ax_age.set_ylabel("Count")
ax_age.set_title("Age Distribution")
ax_age.legend()

# --- Bottom right: Gender & Ethnicity bar charts ---
gs_cat = gs[1, 1].subgridspec(2, 1, hspace=0.4)
gender_counts = df["gender"].value_counts().sort_index()
eth_counts = df["ethnicity"].value_counts().sort_index()

ax1 = fig.add_subplot(gs_cat[0])
ax1.bar([0, 1], [gender_counts.get(0, 0), gender_counts.get(1, 0)],
        color=["#3498db", "#e74c3c"], width=0.6)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["Male", "Female"])
ax1.set_ylabel("Count")
ax1.set_title("Gender Distribution")

ax2 = fig.add_subplot(gs_cat[1])
ax2.bar(range(5), [eth_counts.get(i, 0) for i in range(5)],
        color=plt.cm.Set3(np.linspace(0, 1, 5)), width=0.7)
ax2.set_xticks(range(5))
ax2.set_xticklabels([ETHNICITY_NAMES[i] for i in range(5)], rotation=25, ha="right")
ax2.set_ylabel("Count")
ax2.set_title("Ethnicity Distribution")

# --- Bottom: Text statistics ---
stats_text = (
    f"Dataset: {len(df):,} face images (48×48 grayscale)\n"
    f"Age range: {df['age'].min()}–{df['age'].max()} years  |  "
    f"Mean: {df['age'].mean():.1f}  |  Median: {df['age'].median():.0f}\n"
    f"Gender: Male {gender_counts.get(0, 0):,} | Female {gender_counts.get(1, 0):,}\n"
    f"Ethnicity: White {eth_counts.get(0, 0):,} | Black {eth_counts.get(1, 0):,} | "
    f"Asian {eth_counts.get(2, 0):,} | Indian {eth_counts.get(3, 0):,} | Other {eth_counts.get(4, 0):,}"
)
ax_stats = fig.add_subplot(gs[2, :])
ax_stats.axis("off")
ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes, fontsize=11,
              verticalalignment="center", horizontalalignment="center",
              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), family="monospace")

plt.suptitle("Age & Gender Dataset Exploration", fontsize=16, fontweight="bold", y=0.98)
plt.savefig("figures_cDCGAN/age_gender_exploration.png", dpi=150, bbox_inches="tight")
print("Saved: figures_cDCGAN/age_gender_exploration.png")
plt.show()
