import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import umap
from matplotlib.lines import Line2D
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

INPUT_FILE = "data/safespoon_annotations_top5.csv"
OUTPUT_HTML = "data/output/query_embeddings_umap.html"
OUTPUT_FLAGS_PNG = "data/output/query_embeddings_flags.png"
OUTPUT_FLAGS_HTML = "data/output/query_embeddings_flags.html"
QUERY_COL = "query"
CATEGORY_COL = "category"
CATEGORIES = ["Health", "Moral Values and Religion", "Economic and Financial"]
FLAG_COLS = ["TRASH", "DEMO", "NOT_IN_DOMAIN", "NOT_IN_TOPIC"]

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
UMAP_RANDOM_STATE = 42
UMAP_MIN_DIST = 0.1
UMAP_N_NEIGHBORS_CAP = 15
HOVER_TEXT_TRUNC = 160

CATEGORY_COLORS = {
    "Economic and Financial": "#2a78d6",
    "Health": "#008300",
    "Moral Values and Religion": "#e34948",
}

df = pd.read_csv(INPUT_FILE)
print(f"Raw rows: {len(df)}")

df = df[df[CATEGORY_COL].isin(CATEGORIES)].reset_index(drop=True)
df[FLAG_COLS] = df[FLAG_COLS].fillna(False).astype(bool)
print(f"Rows after dropping non-standard/corrupt categories: {len(df)}")
print(df[CATEGORY_COL].value_counts().to_string())

print(f"Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(df[QUERY_COL].tolist(), show_progress_bar=True)
print(f"Embeddings shape: {embeddings.shape}")

pca = PCA(n_components=2)
pca.fit_transform(embeddings)
print(f"PCA explained variance ratio (PC1, PC2): {pca.explained_variance_ratio_}")

n_samples = len(df)
n_neighbors = min(UMAP_N_NEIGHBORS_CAP, n_samples - 1)
print(f"Running UMAP (n_neighbors={n_neighbors}, n_samples={n_samples})")

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=n_neighbors,
    min_dist=UMAP_MIN_DIST,
    metric="cosine",
    random_state=UMAP_RANDOM_STATE,
)
coords_2d = reducer.fit_transform(embeddings)
df["x"] = coords_2d[:, 0]
df["y"] = coords_2d[:, 1]

def hover_text_for(sub, category):
    return [
        f"<b>{category}</b><br>"
        f"query: {q[:HOVER_TEXT_TRUNC]}{'...' if len(q) > HOVER_TEXT_TRUNC else ''}<br>"
        f"TRASH: {t} | DEMO: {d} | NOT_IN_DOMAIN: {nid} | NOT_IN_TOPIC: {nit}"
        for q, t, d, nid, nit in zip(
            sub[QUERY_COL], sub["TRASH"], sub["DEMO"], sub["NOT_IN_DOMAIN"], sub["NOT_IN_TOPIC"]
        )
    ]


fig = go.Figure()

for category in CATEGORIES:
    sub = df[df[CATEGORY_COL] == category]
    hover_text = hover_text_for(sub, category)
    fig.add_trace(go.Scatter(
        x=sub["x"], y=sub["y"],
        mode="markers",
        name=f"{category} (n={len(sub)})",
        marker=dict(size=10, color=CATEGORY_COLORS[category], line=dict(width=1, color="white")),
        text=hover_text,
        hoverinfo="text",
    ))

fig.update_layout(
    title=f"UMAP projection of query embeddings ({EMBEDDING_MODEL}) by category",
    xaxis_title="UMAP-1",
    yaxis_title="UMAP-2",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    template="plotly_white",
)

fig.write_html(OUTPUT_HTML)
print(f"Saved visualization to {OUTPUT_HTML}")

flags_fig, axes = plt.subplots(2, 2, figsize=(11, 9))
#flags_fig.suptitle(
    #f"UMAP projection of query embeddings ({EMBEDDING_MODEL}): color = category, marker = annotation flag",
#    fontsize=12, y=0.99,
#)

for ax, flag in zip(axes.flat, FLAG_COLS):
    for category in CATEGORIES:
        cat_df = df[df[CATEGORY_COL] == category]
        off = cat_df[~cat_df[flag]]
        on = cat_df[cat_df[flag]]
        ax.scatter(off["x"], off["y"], marker="o", s=40, color=CATEGORY_COLORS[category],
                   alpha=0.55, edgecolors="white", linewidths=0.5)
        ax.scatter(on["x"], on["y"], marker="X", s=110, color=CATEGORY_COLORS[category],
                   alpha=1.0, edgecolors="black", linewidths=0.8)
    n_flagged = int(df[flag].sum())
    ax.set_title(f"{flag} (n={n_flagged})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.spines[["top", "right"]].set_visible(False)

legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=CATEGORY_COLORS[c], markersize=9, label=c)
    for c in CATEGORIES
] + [
    Line2D([0], [0], marker="o", color="gray", linestyle="None", alpha=0.55, markersize=7, label="Not flagged"),
    Line2D([0], [0], marker="X", color="gray", linestyle="None", markersize=9, label="Flagged"),
]
flags_fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))

flags_fig.tight_layout(rect=[0, 0.06, 1, 0.94])
flags_fig.savefig(OUTPUT_FLAGS_PNG, dpi=150)
print(f"Saved flag grid to {OUTPUT_FLAGS_PNG}")

flags_html_fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f"{flag} (n={int(df[flag].sum())})" for flag in FLAG_COLS],
)
positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

for (row, col), flag in zip(positions, FLAG_COLS):
    for category in CATEGORIES:
        cat_df = df[df[CATEGORY_COL] == category]
        for flagged, symbol, size, opacity, line in [
            (False, "circle", 8, 0.55, dict(width=0.8, color="white")),
            (True, "x", 12, 1.0, dict(width=1.2, color="black")),
        ]:
            sub = cat_df[cat_df[flag] == flagged]
            if sub.empty:
                continue
            flags_html_fig.add_trace(
                go.Scatter(
                    x=sub["x"], y=sub["y"],
                    mode="markers",
                    name=category,
                    legendgroup=category,
                    showlegend=False,
                    marker=dict(size=size, color=CATEGORY_COLORS[category], symbol=symbol, opacity=opacity, line=line),
                    text=hover_text_for(sub, category),
                    hoverinfo="text",
                ),
                row=row, col=col,
            )

# Legend-only dummy traces: category colors + flag shapes, each shown once regardless of subplot.
for category in CATEGORIES:
    flags_html_fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers", name=category, legendgroup=category,
                   marker=dict(size=10, color=CATEGORY_COLORS[category], symbol="circle")),
        row=1, col=1,
    )
for label, symbol, color in [("Not flagged", "circle", "gray"), ("Flagged", "x", "gray")]:
    flags_html_fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers", name=label,
                   marker=dict(size=10, color=color, symbol=symbol)),
        row=1, col=1,
    )

flags_html_fig.update_layout(
    #title=f"UMAP projection of query embeddings ({EMBEDDING_MODEL}) — color = category, marker = annotation flag",
    template="plotly_white",
    height=850,
    legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
)
flags_html_fig.update_xaxes(title_text="UMAP-1")
flags_html_fig.update_yaxes(title_text="UMAP-2")

flags_html_fig.write_html(OUTPUT_FLAGS_HTML)
print(f"Saved interactive flag grid to {OUTPUT_FLAGS_HTML}")
