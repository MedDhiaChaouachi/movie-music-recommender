import pandas as pd

def load_movie_data(path="data/movies.csv"):
    """Load and clean movies dataset."""
    movies = pd.read_csv(path)
    if "title" not in movies.columns:
        raise KeyError("Movies dataset must contain a 'title' column.")
    movies.dropna(subset=["title"], inplace=True)
    return movies

def load_music_data(path="data/spotify.csv"):
    """Load and clean Spotify dataset."""
    music = pd.read_csv(path)

    # ✅ Handle datasets where the column is 'track_name' instead of 'name'
    if "track_name" in music.columns and "name" not in music.columns:
        music.rename(columns={"track_name": "name"}, inplace=True)

    if "name" not in music.columns:
        raise KeyError("Music dataset must contain a 'name' or 'track_name' column.")

    music.dropna(subset=["name"], inplace=True)
    return music

def print_recommendations(recs, item_type="Movie"):
    """Nicely format recommendations for console output."""
    print(f"\nTop {item_type} Recommendations:")
    for i, rec in enumerate(recs, 1):
        print(f"{i}. {rec}")
