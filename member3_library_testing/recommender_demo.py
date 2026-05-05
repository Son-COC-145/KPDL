"""
THÀNH VIÊN 3 - Sinh thử Top 5 phim gợi ý bằng mô hình SVD.

File này không thay thế giao diện Streamlit của Thành viên 4.
Nó chỉ tạo dữ liệu demo để minh họa kết quả mô hình thư viện.
"""

import pandas as pd
from surprise import SVD
from surprise.model_selection import train_test_split

from config import OUTPUT_DIR, RANDOM_STATE, TEST_SIZE, SVD_N_FACTORS, SVD_N_EPOCHS
from data_loader import build_surprise_dataset


def train_svd_model(ratings):
    data = build_surprise_dataset(ratings)
    trainset, _ = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = SVD(
        n_factors=SVD_N_FACTORS,
        n_epochs=SVD_N_EPOCHS,
        random_state=RANDOM_STATE,
    )

    model.fit(trainset)
    return model


def get_top_n_for_user(model, ratings, movies, user_id, n=5):
    """
    Gợi ý Top-N phim cho một user:
    - Lấy các phim user chưa đánh giá.
    - Dự đoán rating bằng SVD.
    - Sắp xếp giảm dần theo rating dự đoán.
    """
    all_movie_ids = set(movies["movieId"].unique())
    watched_movie_ids = set(ratings.loc[ratings["userId"] == user_id, "movieId"].unique())
    unseen_movie_ids = list(all_movie_ids - watched_movie_ids)

    predictions = []

    for movie_id in unseen_movie_ids:
        pred = model.predict(uid=user_id, iid=movie_id)
        predictions.append((movie_id, pred.est))

    top_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

    result = pd.DataFrame(top_predictions, columns=["movieId", "predictedRating"])
    result = result.merge(movies, on="movieId", how="left")
    result["predictedRating"] = result["predictedRating"].round(4)

    return result[["movieId", "title", "genres", "predictedRating"]]


def export_demo_recommendations(prepared_data, user_id=None):
    """
    Train SVD trên tập 100K sample để tạo demo nhanh.
    Nếu không truyền user_id, lấy user đầu tiên trong dữ liệu.
    """
    ratings = prepared_data["sample_100k"]
    movies = prepared_data["movies"]

    if user_id is None:
        user_id = int(ratings["userId"].iloc[0])

    model = train_svd_model(ratings)
    top5 = get_top_n_for_user(model, ratings, movies, user_id=user_id, n=5)

    OUTPUT_DIR.mkdir(exist_ok=True)
    top5.to_csv(OUTPUT_DIR / "top5_recommendations_svd_demo.csv", index=False, encoding="utf-8-sig")

    return top5
