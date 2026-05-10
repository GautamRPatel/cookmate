import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

from app.utils.clean_text import clean_text


class RecipeEngine:

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)

        self.df["combined"] = (
            self.df["Ingredients"].fillna("") + " " +
            self.df["Instructions"].fillna("")
        )

        self.df["combined"] = self.df["combined"].apply(clean_text)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings_path = os.path.join(
            os.path.dirname(__file__),
            "recipe_embeddings.pkl",
        )

        if os.path.exists(embeddings_path):
            self.recipe_embeddings = joblib.load(embeddings_path)
        else:
            self.recipe_embeddings = self.model.encode(
                self.df["combined"].tolist(),
                show_progress_bar=True
            )
            joblib.dump(self.recipe_embeddings, embeddings_path)

    def get_best_recipes(self, veg_list, top_k=3):

        query = clean_text(" ".join(veg_list))

        query_embedding = self.model.encode([query])

        similarity_scores = cosine_similarity(
            query_embedding,
            self.recipe_embeddings
        )[0]

        top_indices = similarity_scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]

            results.append({
                "recipe_name": str(row["RecipeName"]) if not pd.isna(row["RecipeName"]) else "",
                "translated_name": str(row["TranslatedRecipeName"]) if not pd.isna(row["TranslatedRecipeName"]) else "",
                "ingredients": str(row["Ingredients"]) if not pd.isna(row["Ingredients"]) else "",
                "translated_ingredients": str(row["TranslatedIngredients"]) if not pd.isna(row["TranslatedIngredients"]) else "",
                "instructions": str(row["Instructions"]) if not pd.isna(row["Instructions"]) else "",
                "prep_time": int(row["PrepTimeInMins"]) if not pd.isna(row["PrepTimeInMins"]) else 0,
                "cook_time": int(row["CookTimeInMins"]) if not pd.isna(row["CookTimeInMins"]) else 0,
                "total_time": int(row["TotalTimeInMins"]) if not pd.isna(row["TotalTimeInMins"]) else 0,
                "servings": str(row["Servings"]) if not pd.isna(row["Servings"]) else "",
                "cuisine": str(row["Cuisine"]) if not pd.isna(row["Cuisine"]) else "",
                "diet": str(row["Diet"]) if not pd.isna(row["Diet"]) else "",
                "similarity_score": float(similarity_scores[idx])
            })

            print(f"Recipe: {row['RecipeName']}, Similarity Score: {similarity_scores[idx]:.4f}")
            

        return results
        
