import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Music Recommender", layout="centered")
st.title("🎵 Music Recommendation System (Last.fm)")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("last.fm_data.csv")
    df.columns = df.columns.str.lower()

    required = {"username", "track", "artist"}
    if not required.issubset(df.columns):
        st.error("❌ Dataset must contain username, track, artist")
        st.stop()

    df["play_count"] = 1
    return df

df = load_data()

# --------------------------------------------------
# Reduce size (VERY IMPORTANT)
# --------------------------------------------------
TOP_USERS = 3000
TOP_TRACKS = 3000

top_users = df["username"].value_counts().head(TOP_USERS).index
top_tracks = df["track"].value_counts().head(TOP_TRACKS).index

df = df[df["username"].isin(top_users) & df["track"].isin(top_tracks)]

# --------------------------------------------------
# Train / Test Split (implicit feedback)
# --------------------------------------------------
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42
)

# --------------------------------------------------
# User-Item Matrix
# --------------------------------------------------
user_item = train_df.pivot_table(
    index="username",
    columns="track",
    values="play_count",
    aggfunc="sum",
    fill_value=0
)

user_item = user_item.div(user_item.sum(axis=1), axis=0)

# --------------------------------------------------
# Similarity
# --------------------------------------------------
user_sim = cosine_similarity(user_item)
user_sim_df = pd.DataFrame(
    user_sim,
    index=user_item.index,
    columns=user_item.index
)

# --------------------------------------------------
# Recommendation Function
# --------------------------------------------------
def recommend(user, k=10):
    if user not in user_item.index:
        return []

    similar_users = (
        user_sim_df[user]
        .sort_values(ascending=False)
        .iloc[1:11]
        .index
    )

    listened = user_item.loc[user]
    listened_tracks = listened[listened > 0].index

    scores = user_item.loc[similar_users].mean(axis=0)
    scores = scores.drop(listened_tracks, errors="ignore")

    return scores.sort_values(ascending=False).head(k).index.tolist()

# --------------------------------------------------
# UI
# --------------------------------------------------
user_list = user_item.index.tolist()
user_id = st.selectbox("Select User", user_list)

k = st.slider("Number of recommendations", 5, 20, 10)

if st.button("Get Recommendations"):
    recs = recommend(user_id, k)

    if not recs:
        st.warning("No recommendations available.")
    else:
        st.success("🎧 Recommended Tracks")
        for i, track in enumerate(recs, 1):
            artist = df[df["track"] == track]["artist"].iloc[0]
            st.write(f"{i}. **{track}** — *{artist}*")

# --------------------------------------------------
# METRICS
# --------------------------------------------------
st.subheader("📊 Recommendation Metrics")

def evaluate_at_k(k=10):
    hits = 0
    total_relevant = 0
    total_recommended = 0

    for user in test_df["username"].unique():
        true_tracks = (
            test_df[test_df["username"] == user]["track"]
            .unique()
            .tolist()
        )

        if len(true_tracks) == 0:
            continue

        recs = recommend(user, k)
        if not recs:
            continue

        hit_count = len(set(recs) & set(true_tracks))

        hits += hit_count
        total_relevant += len(true_tracks)
        total_recommended += k

    precision = hits / total_recommended if total_recommended else 0
    recall = hits / total_relevant if total_relevant else 0
    hit_rate = hits / len(test_df["username"].unique())

    return precision, recall, hit_rate

if st.button("Evaluate Model"):
    p, r, h = evaluate_at_k(k)

    st.metric("Precision@K", f"{p:.4f}")
    st.metric("Recall@K", f"{r:.4f}")
    st.metric("Hit Rate@K", f"{h:.4f}")
