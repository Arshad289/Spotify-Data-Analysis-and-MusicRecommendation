# Dataset Description

## Source
**Spotify Tracks Dataset** from Kaggle
https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

## Download Instructions
1. Visit the Kaggle link above and download `dataset.csv`.
2. Rename the file to `spotify_tracks.csv`.
3. Place it in the `data/` folder of this repository.

## Overview
The dataset contains **114,000 Spotify tracks** across 114 genres, retrieved via the Spotify Web API. Each row represents a unique track with metadata and audio features computed by Spotify's internal algorithms.

## Schema

| Column | Type | Description |
|---|---|---|
| `track_id` | string | Unique Spotify identifier for the track |
| `artists` | string | Artist name(s), comma-separated for collaborations |
| `album_name` | string | Name of the album containing the track |
| `track_name` | string | Name of the track |
| `popularity` | int (0–100) | Popularity score based on recent stream count |
| `duration_ms` | int | Track duration in milliseconds |
| `explicit` | bool | Whether the track contains explicit content |
| `danceability` | float (0–1) | How suitable a track is for dancing |
| `energy` | float (0–1) | Perceptual measure of intensity and activity |
| `key` | int (0–11) | Musical key (Pitch Class notation) |
| `loudness` | float (dB) | Overall loudness in decibels |
| `mode` | int (0/1) | Major (1) or Minor (0) modality |
| `speechiness` | float (0–1) | Presence of spoken words |
| `acousticness` | float (0–1) | Confidence the track is acoustic |
| `instrumentalness` | float (0–1) | Predicts whether a track has no vocals |
| `liveness` | float (0–1) | Probability of a live recording |
| `valence` | float (0–1) | Musical positiveness (happy vs. sad) |
| `tempo` | float (BPM) | Estimated tempo in beats per minute |
| `time_signature` | int | Estimated time signature (3–7) |
| `track_genre` | string | Genre label assigned to the track |

## Size
- **Rows:** ~114,000
- **Columns:** 20
- **File size:** ~20 MB (CSV)

## License
The dataset is publicly available on Kaggle under the standard Kaggle Dataset License. It is derived from the Spotify Web API and is intended for educational and research purposes.
