# 🎵 Spotify Data Analysis & Music Recommendation System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-green?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

A comprehensive data science project that analyzes the **Spotify Tracks Dataset** (~114K rows raw; ~90K tracks after cleaning) to uncover audio feature trends, genre distributions, and popularity patterns — then builds a **content-based music recommendation engine** using cosine similarity and K-Means clustering.

---

## 📌 Project Highlights

| Area | Detail |
|---|---|
| **Dataset** | ~114K rows raw; ~90K tracks after cleaning (duplicates & nulls removed) |
| **EDA** | Correlation heatmaps, popularity distributions, genre analysis |
| **Clustering** | K-Means (k=8) on 9 normalized audio features |
| **Recommender** | Cosine-similarity engine returning top-N by audio-feature similarity |
| **Output** | 8 charts in `outputs/`, enriched CSV, console recommendations for 3 demo tracks |

---

## 🗂 Repository Structure

```
Spotify-Data-Analysis-and-MusicRecommendation/
├── spotify_analysis.py      # Main analysis & recommendation script
├── requirements.txt         # Python dependencies
├── DATASET.md               # Dataset schema & download instructions
├── .gitignore
├── Data/                    # Place spotify_tracks.csv here (not tracked)
│   └── .gitkeep
└── outputs/                 # Generated charts (8 PNGs) & enriched CSV
    └── .gitkeep
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/Arshad289/Spotify-Data-Analysis-and-MusicRecommendation.git
cd Spotify-Data-Analysis-and-MusicRecommendation
pip install -r requirements.txt
```

### Download the Dataset

1. Download from [Kaggle – Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset).
2. Rename the file to `spotify_tracks.csv` and place it inside the `data/` folder (or `Data/`; the script checks both).
3. See [DATASET.md](DATASET.md) for full schema details.

### Run the Analysis

```bash
python spotify_analysis.py
```

This will generate all visualizations in `outputs/` and print recommendation results to the console.

---

## 🎯 Sample Output

### Recommendation Results

The script runs the recommender for three demo tracks: **"Blinding Lights"**, **"Shape of You"**, and **"Bohemian Rhapsody"**. Each query returns a table with:

| Column | Description |
|--------|-------------|
| track_name | Name of the recommended track |
| artists | Artist(s) |
| track_genre | Genre label from the dataset |
| popularity | Popularity score (0–100) |
| similarity_score | Cosine similarity to the seed track (0–1) |

Recommendations are based **only on audio-feature similarity** (the 9 features). Genre and artist are not used, so recommended tracks can be from any genre that happens to have similar danceability, energy, loudness, etc. Scores are typically high (e.g. ≥ 0.99) when searching within nearby K-Means clusters.

### K-Means Cluster Profiles

The script prints **Cluster Audio Profiles (mean values)** for all 8 clusters. Exact numbers depend on your cleaned dataset. Illustrative profile types the clustering often reveals:

| Cluster | Typical profile (data-driven) |
|---------|-------------------------------|
| 0–1 | Mixed / upbeat (e.g. higher danceability, valence) |
| 2–3 | Acoustic, chill, or highly instrumental |
| 4 | High energy and/or instrumental |
| 5 | High liveness (live recordings), more speechiness |
| 6–7 | High energy, low acousticness; or high valence/danceability |

Run the script to see the full mean-value table and cluster counts for your data.

### Audio Feature Correlations (Top 5)

| Feature Pair | Correlation Coefficient | Interpretation |
|--------------|------------------------|----------------|
| Energy ↔ Loudness | +0.76 | Energetic songs are louder |
| Energy ↔ Acousticness | -0.73 | Acoustic songs tend to be calmer |
| Danceability ↔ Valence | +0.48 | Danceable songs are often happier |
| Speechiness ↔ Instrumentalness | -0.09 | More speech = less instrumental |
| Liveness ↔ Energy | +0.18 | Live tracks tend to be more energetic |

---

## 📊 Analysis Overview

### 1. Exploratory Data Analysis
- **Popularity Distribution** — Right-skewed; most tracks have low popularity, a small fraction go viral.
- **Feature Correlations** — Strong positive correlation between energy and loudness; strong negative correlation between energy and acousticness.
- **Top Genres** — Pop, rock, and hip-hop dominate the dataset by track count.
- **Features by Popularity Tier** — Chart-topping tracks tend to have higher danceability and energy but moderate acousticness.

### 2. K-Means Clustering
Tracks are clustered into **8 groups** based on 9 normalized audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo). Clusters reveal natural groupings like "high-energy dance tracks" vs. "acoustic ballads."

### 3. Content-Based Recommendation Engine
- Scales all audio features to [0, 1] using MinMaxScaler.
- For each query, computes **cosine similarity** between the seed track and candidates (cluster-scoped for speed).
- Returns the top-N tracks by similarity. Demo runs for "Blinding Lights", "Shape of You", and "Bohemian Rhapsody" (5 recommendations each).
- **Note:** Similarity is purely on the 9 audio features; genre and artist are not considered.

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core language |
| **Pandas / NumPy** | Data wrangling & numerical ops |
| **Matplotlib / Seaborn** | Data visualization |
| **scikit-learn** | MinMaxScaler, KMeans, cosine_similarity |

---

## ⚡ Performance Metrics

| Metric | Value |
|--------|-------|
| **Data Loading** | ~2-3 seconds (~114K raw rows) |
| **Cleaned Tracks** | ~90K (after dedup & null drop) |
| **Feature Scaling** | ~2 seconds (MinMaxScaler) |
| **Cosine Similarity** | Cluster-scoped (fast per-query) |
| **K-Means Clustering (k=8)** | ~45 seconds |
| **Recommendation Generation** | <1 second per query |
| **Visualization Creation** | ~10 seconds (8 PNGs in `outputs/`) |
| **Total Runtime** | ~2-3 minutes (full pipeline) |
| **Memory Usage** | ~250-350 MB peak |

### System Requirements
- **Minimum:** 4 GB RAM, Python 3.9+
- **Recommended:** 8 GB RAM for faster processing
- **Storage:** ~100 MB (dataset + outputs)

---

## 🎮 Interactive Demo (Optional)

Want to try the recommender interactively? You can extend this project with a web interface:

```python
# Install Streamlit (optional)
pip install streamlit

# Create a simple app (app.py)
# Then run:
streamlit run app.py
```

This would allow you to:
- Search for tracks by name
- Get instant recommendations
- Visualize audio features
- Explore cluster memberships

---

## 📈 Key Findings

1. **Energy and loudness** are the strongest correlated audio features (r ≈ 0.76).
2. Tracks in the **"Chart Topper"** tier tend to have higher danceability and energy than "Emerging" tracks.
3. **Pop and hip-hop** are among the top genres by track count and median popularity in the dataset.
4. The recommender returns the **top-N by audio-feature cosine similarity**; it does not use genre or artist, so recommended tracks can span many genres.
5. K-Means (k=8) reveals **8 distinct sonic profiles** from the 9 normalized features; cluster means are printed when you run the script.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-analysis`).
3. Commit your changes and push.
4. Open a Pull Request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Arshad Ali Mohammed**
- [GitHub](https://github.com/Arshad289)
- [LinkedIn](https://www.linkedin.com/in/arshad-ali-m-080110135)
- Email: Mohammedarshadali89@gmail.com
