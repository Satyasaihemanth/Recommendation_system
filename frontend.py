import streamlit as st
import requests

st.set_page_config(page_title="Music Recommender", layout="centered")
st.title("🎵 Music Recommendation System (FastAPI Backend)")

API_URL = "http://127.0.0.1:8000"

# -------------------------------
# Fetch users for dropdown
# -------------------------------
try:
    response = requests.get(f"{API_URL}/users")
    if response.status_code == 200:
        users = response.json().get("users", [])
    else:
        users = []
except:
    users = []

# -------------------------------
# Recommendations UI
# -------------------------------
st.subheader("🎧 Get Recommendations")
user_id = st.selectbox("Select User", users)
k_rec = st.slider("Number of recommendations", 5, 20, 10)

if st.button("Get Recommendations") and user_id:
    payload = {"user": user_id, "k": k_rec}
    response = requests.post(f"{API_URL}/recommend", json=payload)

    if response.status_code == 200:
        data = response.json()
        recs = data.get("recommendations", [])
        if recs:
            st.success("Recommended Tracks")
            for i, rec in enumerate(recs, 1):
                st.write(f"{i}. **{rec['track']}** — *{rec['artist']}*")
        else:
            st.warning("No recommendations available.")
    else:
        st.error("❌ Error connecting to backend")

# -------------------------------
# Evaluation Metrics UI
# -------------------------------
st.subheader("📊 Evaluate Model")
k_eval = st.slider("Top-K for evaluation", 5, 20, 10, key="eval_slider")

if st.button("Evaluate Model"):
    payload = {"k": k_eval}
    response = requests.post(f"{API_URL}/evaluate", json=payload)
    if response.status_code == 200:
        metrics = response.json()
        st.metric("Precision@K", f"{metrics['precision']:.4f}")
        st.metric("Recall@K", f"{metrics['recall']:.4f}")
        st.metric("Hit Rate@K", f"{metrics['hit_rate']:.4f}")
    else:
        st.error("❌ Error evaluating the model")