from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

app = FastAPI(title="Music Recommendation API")

# -------------------------------
# Load & preprocess dataset
# -------------------------------
df = pd.read_csv("last.fm_data.csv")
df.columns = df.columns.str.lower()
df["play_count"] = 1

TOP_USERS = 3000
TOP_TRACKS = 3000
top_users = df["username"].value_counts().head(TOP_USERS).index
top_tracks = df["track"].value_counts().head(TOP_TRACKS).index
df = df[df["username"].isin(top_users) & df["track"].isin(top_tracks)]

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

user_item = train_df.pivot_table(
    index="username", columns="track", values="play_count", aggfunc="sum", fill_value=0
)
user_item = user_item.div(user_item.sum(axis=1), axis=0)

user_sim = cosine_similarity(user_item)
user_sim_df = pd.DataFrame(user_sim, index=user_item.index, columns=user_item.index)

# -------------------------------
# Recommendation function
# -------------------------------
def recommend(user, k=10):
    if user not in user_item.index:
        return []

    similar_users = user_sim_df[user].sort_values(ascending=False).iloc[1:11].index
    listened = user_item.loc[user]
    listened_tracks = listened[listened > 0].index
    scores = user_item.loc[similar_users].mean(axis=0)
    scores = scores.drop(listened_tracks, errors="ignore")
    return scores.sort_values(ascending=False).head(k).index.tolist()

# -------------------------------
# Evaluation function (fixed)
# -------------------------------
def evaluate_at_k(k=10):
    hits = 0
    total_relevant = 0
    total_recommended = 0
    hit_users = 0  # For corrected Hit Rate

    for user in test_df["username"].unique():
        true_tracks = test_df[test_df["username"] == user]["track"].unique().tolist()
        if not true_tracks:
            continue

        recs = recommend(user, k)
        if not recs:
            continue

        hit_count = len(set(recs) & set(true_tracks))
        hits += hit_count
        total_relevant += len(true_tracks)
        total_recommended += k

        if hit_count > 0:
            hit_users += 1

    precision = hits / total_recommended if total_recommended else 0
    recall = hits / total_relevant if total_relevant else 0
    hit_rate = hit_users / len(test_df["username"].unique())  # Corrected Hit Rate
    
    return precision, recall, hit_rate

# -------------------------------
# API Models
# -------------------------------
class RecRequest(BaseModel):
    user: str
    k: int = 10

class EvalRequest(BaseModel):
    k: int = 10

# -------------------------------
# API Endpoints
# -------------------------------
@app.get("/")
def root():
    return {"message": "Music Recommender API is running!"}

@app.get("/users")
def get_users():
    return {"users": user_item.index.tolist()}

@app.post("/recommend")
def get_recommendations(req: RecRequest):
    recs = recommend(req.user, req.k)
    result = []
    for track in recs:
        artist = df[df["track"] == track]["artist"].iloc[0]
        result.append({"track": track, "artist": artist})
    return {"recommendations": result}

@app.post("/evaluate")
def evaluate_model(req: EvalRequest):
    p, r, h = evaluate_at_k(req.k)
    return {"precision": p, "recall": r, "hit_rate": h}