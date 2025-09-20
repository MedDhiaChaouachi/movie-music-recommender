import sys
import os
import streamlit as st
import pandas as pd

# ✅ Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.content_based import ContentBasedRecommender
from src.music_recommender import MusicRecommender
from src.utils import load_movie_data, load_music_data

# Load data
movies = load_movie_data()
spotify = load_music_data()

# Init models
movie_rec = ContentBasedRecommender(movies)
music_rec = MusicRecommender(spotify)

st.title("🎬 Movie & 🎵 Music Recommendation System")

choice = st.radio("Choose a system:", ["Movie", "Music"])

if choice == "Movie":
    title = st.text_input("Enter a movie title:")
    if st.button("Recommend Movies"):
        recommendations = movie_rec.recommend(title)
        st.write(recommendations)

elif choice == "Music":
    track = st.text_input("Enter a music track name:")
    if st.button("Recommend Songs"):
        recommendations = music_rec.recommend(track)
        st.write(recommendations)
