"""
Spotify Data Analysis and Music Recommendation System
======================================================
Author: Arshad Ali Mohammed
GitHub: https://github.com/Arshad289

Analyzes 114,000 Spotify tracks to uncover audio feature trends,
genre distributions, and artist popularity patterns. Builds a
content-based recommendation engine using cosine similarity.

Dataset: Spotify Tracks Dataset (Kaggle)
https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="viridis")

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]


# ──────────────────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ──────────────────────────────────────────────────────────

def load_and_clean_data(filepath: str = "data/spotify_tracks.csv") -> pd.DataFrame:
    """Load the Spotify dataset and perform initial cleaning."""
    df = pd.read_csv(filepath, index_col=0)
    print(f"Raw dataset shape: {df.shape}")

    # Drop duplicates based on track_id
    df.drop_duplicates(subset=["track_id"], inplace=True)

    # Drop rows with missing critical fields
    df.dropna(subset=["track_name", "artists", "track_genre"], inplace=True)

    # Convert duration from ms to minutes
    df["duration_min"] = df["duration_ms"] / 60_000

    # Create popularity buckets
    df["popularity_tier"] = pd.cut(
        df["popularity"],
        bins=[0, 25, 50, 75, 100],
        labels=["Emerging", "Up & Coming", "Mainstream", "Chart Topper"],
        include_lowest=True,
    )

    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Null values remaining:\n{df[AUDIO_FEATURES].isnull().sum()}")
    return df


# ──────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ──────────────────────────────────────────────────────────

def plot_popularity_distribution(df: pd.DataFrame) -> None:
    """Histogram of track popularity scores."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["popularity"], bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(df["popularity"].median(), color="red", linestyle="--",
               label=f'Median: {df["popularity"].median():.0f}')
    ax.set_title("Distribution of Track Popularity", fontsize=14)
    ax.set_xlabel("Popularity Score")
    ax.set_ylabel("Number of Tracks")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/popularity_distribution.png", dpi=150)
    plt.close()
    print("[+] Saved popularity_distribution.png")


def plot_audio_feature_correlations(df: pd.DataFrame) -> None:
    """Heatmap of audio feature correlations."""
    corr = df[AUDIO_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Audio Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/feature_correlation_heatmap.png", dpi=150)
    plt.close()
    print("[+] Saved feature_correlation_heatmap.png")


def plot_top_genres(df: pd.DataFrame, n: int = 15) -> None:
    """Bar chart of the most common genres."""
    top = df["track_genre"].value_counts().head(n)
    fig, ax = plt.subplots(figsize=(10, 6))
    top.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", n))
    ax.set_title(f"Top {n} Genres by Track Count", fontsize=14)
    ax.set_xlabel("Number of Tracks")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/top_genres.png", dpi=150)
    plt.close()
    print("[+] Saved top_genres.png")


def plot_feature_by_popularity(df: pd.DataFrame) -> None:
    """Box plots of key audio features grouped by popularity tier."""
    features = ["danceability", "energy", "valence", "acousticness"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, feat in zip(axes.flatten(), features):
        sns.boxplot(data=df, x="popularity_tier", y=feat, ax=ax, palette="Set2")
        ax.set_title(f"{feat.title()} by Popularity Tier", fontsize=12)
    plt.suptitle("Audio Features Across Popularity Tiers", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/features_by_popularity.png", dpi=150)
    plt.close()
    print("[+] Saved features_by_popularity.png")


def plot_duration_analysis(df: pd.DataFrame) -> None:
    """Scatter plot of duration vs. popularity."""
    sample = df.sample(n=min(5000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sample["duration_min"], sample["popularity"], alpha=0.3, s=10)
    ax.set_title("Track Duration vs Popularity (sample n=5000)", fontsize=14)
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Popularity")
    ax.set_xlim(0, 10)
    plt.tight_layout()
    plt.savefig("outputs/duration_vs_popularity.png", dpi=150)
    plt.close()
    print("[+] Saved duration_vs_popularity.png")


def plot_genre_popularity(df: pd.DataFrame, n: int = 15) -> None:
    """Average popularity score per genre (top N)."""
    genre_pop = (df.groupby("track_genre")["popularity"]
                 .agg(["mean", "count"])
                 .query("count >= 50")
                 .nlargest(n, "mean"))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(genre_pop.index, genre_pop["mean"], color=sns.color_palette("magma", n))
    ax.set_title(f"Top {n} Genres by Average Popularity", fontsize=14)
    ax.set_xlabel("Average Popularity Score")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/genre_avg_popularity.png", dpi=150)
    plt.close()
    print("[+] Saved genre_avg_popularity.png")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """Distribution of each audio feature."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, feat in zip(axes.flatten(), AUDIO_FEATURES):
        ax.hist(df[feat].dropna(), bins=40, alpha=0.7, edgecolor="black")
        ax.set_title(feat.title(), fontsize=11)
        ax.set_ylabel("Count")
    plt.suptitle("Audio Feature Distributions", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig("outputs/feature_distributions.png", dpi=150)
    plt.close()
    print("[+] Saved feature_distributions.png")


# ──────────────────────────────────────────────────────────
# 3. K-MEANS CLUSTERING (with Elbow Method)
# ──────────────────────────────────────────────────────────

def find_optimal_clusters(X_scaled: np.ndarray, max_k: int = 12) -> int:
    """Use the elbow method to find optimal k."""
    inertias = []
    K_range = range(2, max_k + 1)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    # Plot elbow curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(K_range), inertias, "bo-", linewidth=2)
    ax.set_title("Elbow Method for Optimal k", fontsize=14)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    plt.tight_layout()
    plt.savefig("outputs/elbow_method.png", dpi=150)
    plt.close()
    print("[+] Saved elbow_method.png")

    # Simple elbow detection: find the k with the largest drop in improvement
    diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    ratios = [diffs[i] / diffs[i + 1] if diffs[i + 1] > 0 else 1
              for i in range(len(diffs) - 1)]
    optimal_k = list(K_range)[np.argmax(ratios) + 1]
    print(f"[+] Suggested optimal k = {optimal_k}")
    return optimal_k


def cluster_tracks(df: pd.DataFrame, scaler: MinMaxScaler, n_clusters: int = None) -> pd.DataFrame:
    """Apply K-Means clustering on scaled audio features."""
    X_scaled = scaler.transform(df[AUDIO_FEATURES])

    if n_clusters is None:
        n_clusters = find_optimal_clusters(X_scaled)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    print(f"[+] Clustered {len(df):,} tracks into {n_clusters} groups")
    print(df["cluster"].value_counts().sort_index())
    return df, km


def plot_clusters(df: pd.DataFrame, scaler: MinMaxScaler) -> None:
    """2D PCA visualization of clusters."""
    X_scaled = scaler.transform(df[AUDIO_FEATURES])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    sample_idx = np.random.RandomState(42).choice(len(df), size=min(8000, len(df)), replace=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        coords[sample_idx, 0], coords[sample_idx, 1],
        c=df["cluster"].iloc[sample_idx], cmap="tab10", alpha=0.4, s=8,
    )
    ax.set_title("Track Clusters (PCA Projection)", fontsize=14)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.colorbar(scatter, label="Cluster", ax=ax)
    plt.tight_layout()
    plt.savefig("outputs/cluster_pca.png", dpi=150)
    plt.close()
    print("[+] Saved cluster_pca.png")


def print_cluster_profiles(df: pd.DataFrame) -> None:
    """Print the mean audio features per cluster."""
    profiles = df.groupby("cluster")[AUDIO_FEATURES].mean().round(3)
    print("\nCluster Audio Profiles (mean values):")
    print(profiles.to_string())


# ──────────────────────────────────────────────────────────
# 4. CONTENT-BASED RECOMMENDATION ENGINE
# ──────────────────────────────────────────────────────────

def build_recommendation_engine(df: pd.DataFrame, scaler: MinMaxScaler):
    """
    Build a scalable cosine-similarity recommender.

    Instead of precomputing the full N x N similarity matrix (which would
    require ~200GB for 114K tracks), computes similarity on-the-fly for
    the query track against only its cluster, then optionally searches
    neighboring clusters.

    Returns
    -------
    recommend : callable
        recommend(track_name, n=10) -> DataFrame of similar tracks
    """
    feature_matrix = scaler.transform(df[AUDIO_FEATURES])

    # Build index: track name (lowercase) -> row positions
    track_lookup = {}
    for idx, name in enumerate(df["track_name"].str.lower()):
        if name not in track_lookup:
            track_lookup[name] = idx

    def recommend(track_name: str, n: int = 10, search_full: bool = False) -> pd.DataFrame:
        key = track_name.strip().lower()
        if key not in track_lookup:
            print(f"Track '{track_name}' not found. Try another title.")
            return pd.DataFrame()

        idx = track_lookup[key]
        query_vec = feature_matrix[idx].reshape(1, -1)

        if search_full or "cluster" not in df.columns:
            # Search all tracks (slower but exhaustive)
            candidates = df.index.tolist()
        else:
            # Search within same cluster + adjacent clusters for speed
            query_cluster = df["cluster"].iloc[idx]
            cluster_centers = df.groupby("cluster")[AUDIO_FEATURES].mean()
            center_dists = ((cluster_centers - df[AUDIO_FEATURES].iloc[idx]) ** 2).sum(axis=1)
            nearby_clusters = center_dists.nsmallest(3).index.tolist()
            candidates = df[df["cluster"].isin(nearby_clusters)].index.tolist()

        candidate_matrix = feature_matrix[candidates]
        scores = cosine_similarity(query_vec, candidate_matrix)[0]

        # Get top N+1 (skip self)
        top_indices = np.argsort(scores)[::-1]
        results = []
        for i in top_indices:
            if candidates[i] == idx:
                continue
            results.append((candidates[i], scores[i]))
            if len(results) >= n:
                break

        if not results:
            return pd.DataFrame()

        row_indices = [r[0] for r in results]
        similarities = [r[1] for r in results]
        out = df.iloc[row_indices][["track_name", "artists", "track_genre", "popularity"]].copy()
        out["similarity_score"] = [round(s, 4) for s in similarities]
        return out.reset_index(drop=True)

    return recommend


# ──────────────────────────────────────────────────────────
# 5. MAIN EXECUTION
# ──────────────────────────────────────────────────────────

def main():
    import os
    os.makedirs("outputs", exist_ok=True)

    # Load & clean
    df = load_and_clean_data()

    # Fit a single scaler for the whole pipeline
    scaler = MinMaxScaler()
    scaler.fit(df[AUDIO_FEATURES])

    # EDA visualizations
    print("\n--- Exploratory Data Analysis ---")
    plot_popularity_distribution(df)
    plot_audio_feature_correlations(df)
    plot_top_genres(df)
    plot_feature_by_popularity(df)
    plot_duration_analysis(df)
    plot_genre_popularity(df)
    plot_feature_distributions(df)

    # Clustering (with elbow method)
    print("\n--- K-Means Clustering ---")
    df, km = cluster_tracks(df, scaler, n_clusters=8)
    plot_clusters(df, scaler)
    print_cluster_profiles(df)

    # Recommendation demo
    print("\n--- Recommendation Engine ---")
    recommend = build_recommendation_engine(df, scaler)

    demo_tracks = ["Blinding Lights", "Shape of You", "Bohemian Rhapsody"]
    for track in demo_tracks:
        print(f"\nTop 5 recommendations for: '{track}'")
        recs = recommend(track, n=5)
        if not recs.empty:
            print(recs.to_string(index=False))

    # Save enriched dataset
    df.to_csv("outputs/spotify_tracks_enriched.csv", index=False)
    print("\n[+] Saved enriched dataset to outputs/spotify_tracks_enriched.csv")
    print("[+] All analyses complete!")


if __name__ == "__main__":
    main()
