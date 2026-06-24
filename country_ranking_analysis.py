import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 15)


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

df = pd.read_csv("data/Country-data.csv")

print("=" * 60)
print("  COUNTRY SOCIOECONOMIC RANKING — PCA & FACTOR ANALYSIS")
print("=" * 60)
print(f"\nDataset: {df.shape[0]} countries × {df.shape[1]} variables")
print(f"Missing values: {df.isnull().sum().sum()}\n")
print(df.describe().round(2))


# ---------------------------------------------------------------------------
# 2. Standardization
# ---------------------------------------------------------------------------

features = df.drop("country", axis=1)
countries = df["country"].values

scaler = StandardScaler()
X_std = scaler.fit_transform(features)
df_std = pd.DataFrame(X_std, columns=features.columns, index=countries)

print("\nStandardized data — mean ≈ 0, std ≈ 1:")
print(pd.DataFrame({"Mean": df_std.mean().round(10), "Std Dev": df_std.std().round(4)}))


# ---------------------------------------------------------------------------
# 3. Correlation Matrix
# ---------------------------------------------------------------------------

rho = df_std.corr()

# Show only the lower triangle to avoid redundant cells
mask = np.triu(np.ones_like(rho, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(
    rho, annot=True, fmt=".2f", cmap="RdYlGn",
    center=0, vmin=-1, vmax=1, ax=ax,
    mask=mask, square=True, linewidths=0.6,
    cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    annot_kws={"size": 11},
)
ax.set_title("Correlation Matrix of Socioeconomic Indicators",
             fontsize=14, fontweight="bold", pad=15)
ax.tick_params(axis="both", labelsize=11)
plt.tight_layout()
plt.savefig("data/correlation_matrix.png", bbox_inches="tight")
plt.show()


# ---------------------------------------------------------------------------
# 4. Bartlett's Sphericity Test
# ---------------------------------------------------------------------------

def bartlett_sphericity_test(corr_matrix, n):
    p = corr_matrix.shape[0]
    det = np.linalg.det(corr_matrix)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(det)
    dof = int(p * (p - 1) / 2)
    p_value = stats.chi2.sf(chi2, dof)
    return chi2, dof, p_value


chi2_stat, dof, p_val = bartlett_sphericity_test(rho.values, len(df))

print("\n" + "=" * 52)
print("         Bartlett's Sphericity Test")
print("=" * 52)
print(f"  Chi-square statistic : {chi2_stat:>10.4f}")
print(f"  Degrees of freedom   : {dof:>10d}")
print(f"  p-value              : {p_val:>10.2e}")
print("-" * 52)
if p_val < 0.05:
    print("  RESULT: Reject H₀ (p < 0.05)")
    print("  ✓ PCA is appropriate — variables are significantly correlated.")
else:
    print("  RESULT: Fail to reject H₀")
    print("  ✗ Warning: PCA may not be appropriate.")
print("=" * 52)


# ---------------------------------------------------------------------------
# 5. Principal Component Analysis
# ---------------------------------------------------------------------------

pca = PCA()
pca.fit(X_std)

eigenvalues = pca.explained_variance_
var_ratio = pca.explained_variance_ratio_
cumulative_var = np.cumsum(var_ratio)
n_comp = len(eigenvalues)

# Kaiser criterion: retain components with eigenvalue > 1
k = int(np.sum(eigenvalues > 1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bar_colors = ["#1565C0" if ev > 1 else "#90CAF9" for ev in eigenvalues]
axes[0].bar(range(1, n_comp + 1), eigenvalues, color=bar_colors, edgecolor="black", alpha=0.9)
axes[0].axhline(y=1, color="crimson", linestyle="--", linewidth=1.8, label="Kaiser criterion (λ = 1)")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Eigenvalue")
axes[0].set_title("Scree Plot")
axes[0].set_xticks(range(1, n_comp + 1))
axes[0].set_xticklabels([f"PC{i}" for i in range(1, n_comp + 1)])
axes[0].legend()

axes[1].plot(range(1, n_comp + 1), cumulative_var * 100,
             marker="o", color="#1565C0", linewidth=2, markersize=8, zorder=3)
axes[1].fill_between(range(1, n_comp + 1), cumulative_var * 100, alpha=0.15, color="#1565C0")
axes[1].axhline(y=70, color="orange", linestyle="--", linewidth=1.5, label="70% threshold")
axes[1].axvline(x=k, color="crimson", linestyle="--", linewidth=1.5, label=f"{k} factors retained")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Explained Variance (%)")
axes[1].set_title("Cumulative Explained Variance")
axes[1].set_xticks(range(1, n_comp + 1))
axes[1].set_xticklabels([f"PC{i}" for i in range(1, n_comp + 1)])
axes[1].set_ylim(0, 108)
axes[1].legend()

plt.suptitle("PCA — Variance Explained", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("data/pca_variance.png", bbox_inches="tight")
plt.show()

print(f"\nKaiser criterion retains {k} factors")
print(f"These {k} factors explain {cumulative_var[k - 1] * 100:.2f}% of total variance\n")

summary_df = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(n_comp)],
    "Eigenvalue": eigenvalues.round(4),
    "Explained Variance (%)": (var_ratio * 100).round(2),
    "Cumulative Variance (%)": (cumulative_var * 100).round(2),
    "Retained": ["Yes" if ev > 1 else "No" for ev in eigenvalues],
})
print(summary_df.to_string(index=False))


# ---------------------------------------------------------------------------
# 6. Factor Loadings & Communalities
# ---------------------------------------------------------------------------

eigenvectors = pca.components_.T               # shape: (n_features, n_components)
factor_loadings = eigenvectors[:, :k] * np.sqrt(eigenvalues[:k])
communalities = np.sum(factor_loadings ** 2, axis=1)

fl_df = pd.DataFrame(
    factor_loadings,
    index=features.columns,
    columns=[f"F{i+1}" for i in range(k)],
)
fl_df["Communality"] = communalities

print("\nFactor Loadings and Communalities:")
print(fl_df.round(4).to_string())

var_names = features.columns.tolist()
var_display = {
    "child_mort": "Child Mortality",
    "exports":    "Exports",
    "health":     "Health Spending",
    "imports":    "Imports",
    "income":     "Income",
    "inflation":  "Inflation",
    "life_expec": "Life Expectancy",
    "total_fer":  "Fertility Rate",
    "gdpp":       "GDP per Capita",
}
factor_labels = [
    "Human Development",
    "Trade & Health Balance",
    "Inflation vs. Healthcare",
    "Economic Output",
]

n_cols = 2
n_rows = (k + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5.5 * n_rows))
axes_flat = np.array(axes).flatten()

for i in range(k):
    ax = axes_flat[i]
    values = factor_loadings[:, i]
    labels = [var_display.get(v, v) for v in var_names]

    # Sort bars by absolute loading so strongest drivers appear at the top
    order = np.argsort(np.abs(values))
    sorted_labels = [labels[j] for j in order]
    sorted_values = values[order]
    bar_colors = ["#C62828" if v < 0 else "#1565C0" for v in sorted_values]

    ax.barh(sorted_labels, sorted_values, color=bar_colors,
            edgecolor="white", linewidth=0.5, height=0.65)
    ax.axvline(x=0,    color="black", linewidth=1.0)
    ax.axvline(x= 0.5, color="gray",  linestyle="--", linewidth=0.8, alpha=0.4)
    ax.axvline(x=-0.5, color="gray",  linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_xlim(-1.45, 1.45)
    ax.set_xlabel("Loading", fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="x", alpha=0.25)

    label = factor_labels[i] if i < len(factor_labels) else f"Factor {i+1}"
    ax.set_title(
        f"Factor {i+1} — {label}  ({var_ratio[i]*100:.1f}% of variance)",
        fontweight="bold", fontsize=11, pad=10,
    )

    # Value labels positioned outside each bar tip — never occluded
    for val, ypos in zip(sorted_values, range(len(sorted_values))):
        if val >= 0:
            ax.text(val + 0.06, ypos, f"{val:.2f}",
                    va="center", ha="left", fontsize=9, fontweight="bold", color="#1565C0")
        else:
            ax.text(val - 0.06, ypos, f"{val:.2f}",
                    va="center", ha="right", fontsize=9, fontweight="bold", color="#C62828")

for j in range(k, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle("Factor Loadings — What Each Dimension Captures",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("data/factor_loadings.png", bbox_inches="tight")
plt.show()


# ---------------------------------------------------------------------------
# 7. Factor Scores & Final Ranking
# ---------------------------------------------------------------------------

pca_scores = pca.transform(X_std)   # shape: (n_countries, n_components)

# Determine the sign for each factor so that higher score = better conditions.
# PCA eigenvector signs are arbitrary; we orient each factor by checking whether
# the net loading of clearly beneficial variables (income, gdpp, life_expec) is
# positive. If it is not, we flip the sign.
beneficial = ["income", "gdpp", "life_expec", "health", "exports"]
detrimental = ["child_mort", "total_fer", "inflation"]
col_list = list(features.columns)

factor_score_cols = {}
print("\nFactor sign orientation:")
for i in range(k):
    pos_net = sum(factor_loadings[col_list.index(v), i] for v in beneficial if v in col_list)
    neg_net = sum(factor_loadings[col_list.index(v), i] for v in detrimental if v in col_list)
    sign = 1 if (pos_net - neg_net) > 0 else -1
    factor_score_cols[f"Factor{i+1}"] = sign * pca_scores[:, i] / np.sqrt(eigenvalues[i])
    print(f"  Factor {i+1}: sign = {'+1' if sign == 1 else '-1'}"
          f"  (beneficial net loading: {pos_net - neg_net:.3f})")

composite_score = sum(
    factor_score_cols[f"Factor{i+1}"] * var_ratio[i]
    for i in range(k)
)

df_ranked = df.copy()
for col, vals in factor_score_cols.items():
    df_ranked[col] = vals
df_ranked["Score"] = composite_score
df_ranked = df_ranked.sort_values("Score", ascending=False).reset_index(drop=True)
df_ranked.index = range(1, len(df_ranked) + 1)
df_ranked.index.name = "Rank"

key_cols = ["country", "child_mort", "health", "life_expec", "income", "gdpp", "Score"]

print("\n" + "=" * 60)
print("  TOP 20 — Countries with the Best Socioeconomic Conditions")
print("=" * 60)
print(df_ranked[key_cols].head(20).to_string())

print("\n" + "=" * 60)
print("  BOTTOM 20 — Countries Most in Need of Aid")
print("=" * 60)
print(df_ranked[key_cols].tail(20).sort_values("Score").to_string())


# ---------------------------------------------------------------------------
# 8. Visualization — Top / Bottom 20 & Score Distribution
# ---------------------------------------------------------------------------

top_20    = df_ranked.head(20)
bottom_20 = df_ranked.tail(20).sort_values("Score")

# Gradient colormaps so bar darkness encodes rank intensity
def make_colors(scores, cmap_name):
    norm = plt.Normalize(scores.min(), scores.max())
    cmap = plt.get_cmap(cmap_name)
    return [cmap(norm(s)) for s in scores]

bot_colors = make_colors(-bottom_20["Score"], "YlOrRd")   # worst = darkest red
top_colors = make_colors(top_20["Score"],  "YlGn")         # best = darkest green

fig, axes = plt.subplots(1, 2, figsize=(22, 11))

# --- Bottom 20 ---
axes[0].barh(bottom_20["country"], bottom_20["Score"],
             color=bot_colors, edgecolor="white", linewidth=0.4)
axes[0].axvline(x=0, color="black", linewidth=1.0)
axes[0].set_xlim(bottom_20["Score"].min() * 1.18, 0.12)
axes[0].set_title("20 Countries Most in Need of Aid\n(Lowest Composite Score)",
                  fontsize=13, fontweight="bold", pad=12)
axes[0].set_xlabel("Composite Score", fontsize=11)
axes[0].tick_params(axis="y", labelsize=10)
axes[0].grid(axis="x", alpha=0.3)

# Score labels centred inside each bar
for _, row in bottom_20.iterrows():
    axes[0].text(
        row["Score"] / 2, bottom_20["country"].tolist().index(row["country"]),
        f'{row["Score"]:.3f}',
        va="center", ha="center", fontsize=8.5, fontweight="bold", color="white",
    )

# --- Top 20 ---
axes[1].barh(top_20["country"], top_20["Score"],
             color=top_colors, edgecolor="white", linewidth=0.4)
axes[1].axvline(x=0, color="black", linewidth=1.0)
axes[1].set_xlim(-0.05, top_20["Score"].max() * 1.18)
axes[1].invert_yaxis()
axes[1].set_title("20 Countries with Best Conditions\n(Highest Composite Score)",
                  fontsize=13, fontweight="bold", pad=12)
axes[1].set_xlabel("Composite Score", fontsize=11)
axes[1].tick_params(axis="y", labelsize=10)
axes[1].grid(axis="x", alpha=0.3)

top_countries = top_20["country"].tolist()
for _, row in top_20.iterrows():
    axes[1].text(
        row["Score"] / 2, top_countries.index(row["country"]),
        f'{row["Score"]:.3f}',
        va="center", ha="center", fontsize=8.5, fontweight="bold", color="white",
    )

plt.suptitle("Country Socioeconomic Ranking — PCA Factor Analysis",
             fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("data/ranking.png", bbox_inches="tight")
plt.show()

# --- Score distribution with KDE and country annotations ---
fig, ax = plt.subplots(figsize=(14, 5))
sns.histplot(df_ranked["Score"], bins=30, kde=True, ax=ax,
             color="#1565C0", alpha=0.55, line_kws={"linewidth": 2.5, "color": "#0D47A1"})

ax.axvline(df_ranked["Score"].mean(),   color="orange", linestyle="--", linewidth=1.8,
           label=f"Mean:   {df_ranked['Score'].mean():.3f}")
ax.axvline(df_ranked["Score"].median(), color="crimson", linestyle="-.", linewidth=1.8,
           label=f"Median: {df_ranked['Score'].median():.3f}")

# Annotate a handful of notable outliers
top_y = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10
notable = {
    "Haiti":                    ("red",       -0.18),
    "Central African Rep.":     ("red",        0.08),
    "Qatar":                    ("darkgreen", -0.18),
    "Norway":                   ("darkgreen",  0.08),
    "Lesotho":                  ("darkorange", 0.0),
}
for country_key, (color, y_offset_frac) in notable.items():
    match = df_ranked[df_ranked["country"].str.startswith(country_key[:6])]
    if match.empty:
        continue
    score = match["Score"].values[0]
    ymax = ax.get_ylim()[1]
    ax.axvline(score, color=color, linestyle=":", linewidth=1.4, alpha=0.8)
    ax.text(score + y_offset_frac * 0.05, ymax * 0.88,
            country_key, rotation=90, va="top",
            ha="right" if y_offset_frac < 0 else "left",
            fontsize=8.5, color=color, fontweight="bold")

ax.set_xlabel("Composite Score", fontsize=11)
ax.set_ylabel("Number of Countries", fontsize=11)
ax.set_title("Score Distribution Across All 167 Countries", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("data/score_distribution.png", bbox_inches="tight")
plt.show()

print("\nDone. Charts saved to data/")
