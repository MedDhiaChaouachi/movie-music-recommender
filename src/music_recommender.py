import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class MusicRecommender:
    def __init__(self, music_df, n_clusters=10):
        self.music = music_df.copy()
        self.music["name"] = self.music["name"].astype(str)

        # Features we cluster on
        features = ['danceability', 'energy', 'tempo']
        self.X = StandardScaler().fit_transform(self.music[features])

        # Train clustering model
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.music['cluster'] = self.kmeans.fit_predict(self.X)

    def recommend(self, query, top_n=5):
        # ✅ Case-insensitive partial match
        matches = self.music[self.music["name"].str.contains(query, case=False, na=False)]

        if matches.empty:
            return ["Track not found!"]

        # Take first match
        track = matches.iloc[0]
        cluster = track["cluster"]

        cluster_tracks = self.music[self.music['cluster'] == cluster]
        recs = cluster_tracks.sample(min(top_n, len(cluster_tracks)))

        # ✅ Flexible return depending on available columns
        cols = ['name']
        if 'artists' in recs.columns:
            cols.append('artists')

        return recs[cols].to_dict(orient="records")
