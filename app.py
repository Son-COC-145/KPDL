import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# CONFIG UI (FULL WIDTH)
# =============================
st.set_page_config(
    page_title="Movie Recommender",
    layout="wide"
)

# =============================
# CSS (FULL SCREEN + TO RÕ)
# =============================
st.markdown("""

<style>
.main {
    background-color: #0e1117;
}

/* Mở rộng full khung */
.block-container {
    padding-top: 2rem;
    padding-left: 5rem;
    padding-right: 5rem;
    max-width: 100% !important;
}

/* Title */
.title {
    font-size: 50px;
    font-weight: bold;
    color: #4dabf7;
    text-align: center;
    margin-bottom: 25px;
}

/* Input */
.stTextInput > div > div > input {
    height: 55px;
    font-size: 18px;
    border-radius: 12px;
}

/* Table */
.stDataFrame {
    width: 100%;
}
.stTextInput > div > div > input {
    height: 55px;
    font-size: 20px;
    border-radius: 12px;

    /* 🔥 CĂN GIỮA DỌC */
    padding-top: 0px;
    padding-bottom: 0px;
    line-height: 55px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# TITLE
# =============================
st.markdown('<div class="title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    return pd.read_csv("user_item_matrix_clean.csv", index_col=0)

df = load_data()

# =============================
# SIMILARITY
# =============================
@st.cache_data
def compute_similarity(matrix):
    sim = cosine_similarity(matrix)
    return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)

similarity_matrix = compute_similarity(df)

# =============================
# RECOMMEND
# =============================
def recommend_movies(user_id, top_n=5):
    if user_id not in df.index:
        return None

    # Similarity between the target user and all users
    sim_scores = similarity_matrix[user_id].copy()
    # Avoid leaking the user's own ratings into the score
    sim_scores.loc[user_id] = 0.0

    # Predicted rating for each movie:
    #   score(m) = sum_u sim(user,u) * rating(u,m) / sum_u |sim(user,u)| where rating(u,m) > 0
    numerator = df.mul(sim_scores, axis=0).sum(axis=0)
    denominator = df.ne(0).mul(sim_scores.abs(), axis=0).sum(axis=0)
    predicted = numerator / (denominator + 1e-9)

    watched = df.loc[user_id]
    predicted = predicted[watched == 0]

    return predicted.sort_values(ascending=False).head(top_n)

# =============================
# UI INPUT (AUTO)
# =============================
user_input = st.text_input("🔍 Nhập UserID")

# AUTO LOAD
if user_input.strip() != "":
    try:
        user_id = int(user_input)
        results = recommend_movies(user_id)

        if results is None:
            st.error("❌ UserID không tồn tại!")
        else:
            st.success(f"Top 5 phim đề xuất cho User {user_id}")

            result_df = pd.DataFrame({
                "Rank": range(1, len(results) + 1),
                "MovieName": results.index,
                "Predicted Score": np.round(results.values, 2),
            })

            st.dataframe(result_df, use_container_width=True, hide_index=True)

    except:
        st.warning("⚠️ UserID phải là số!")