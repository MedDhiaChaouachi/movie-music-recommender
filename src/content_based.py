import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, movies_df):
        self.movies = movies_df.copy()
        self.movies["title"] = self.movies["title"].astype(str)
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies['genres'].fillna(''))
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def recommend(self, query, top_n=5):
        # ✅ Case-insensitive partial match
        matches = self.movies[self.movies["title"].str.contains(query, case=False, na=False)]
        if matches.empty:
            return ["Movie not found!"]

        # Take first match
        idx = matches.index[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        top_indices = [i[0] for i in sim_scores[1:top_n+1]]
        return self.movies['title'].iloc[top_indices].tolist()
