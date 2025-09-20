import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeRecommender:
    def __init__(self, ratings_df, n_components=50):
        """
        ratings_df : pd.DataFrame with columns [user_id, item_id, rating]
        n_components : number of latent factors for SVD
        """
        self.user_mapping = {u: i for i, u in enumerate(ratings_df["user_id"].unique())}
        self.item_mapping = {m: i for i, m in enumerate(ratings_df["item_id"].unique())}
        self.user_inv_mapping = {i: u for u, i in self.user_mapping.items()}
        self.item_inv_mapping = {i: m for m, i in self.item_mapping.items()}

        # Build user-item matrix
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)
        self.user_item_matrix = np.zeros((n_users, n_items))
        for row in ratings_df.itertuples(index=False):
            u = self.user_mapping[row.user_id]
            m = self.item_mapping[row.item_id]
            self.user_item_matrix[u, m] = row.rating

        # Apply SVD
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = svd.fit_transform(self.user_item_matrix)
        self.item_factors = svd.components_.T

        # Precompute similarity
        self.sim_matrix = cosine_similarity(self.item_factors)

    def predict(self, user_id, item_id):
        """Estimate rating for a given user and item"""
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return np.nan  # unknown user/item

        u_idx = self.user_mapping[user_id]
        i_idx = self.item_mapping[item_id]

        user_vec = self.user_factors[u_idx]
        item_vec = self.item_factors[i_idx]

        return np.dot(user_vec, item_vec) / (np.linalg.norm(user_vec) * np.linalg.norm(item_vec) + 1e-8)
