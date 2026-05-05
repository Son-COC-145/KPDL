"""
THÀNH VIÊN 3 - Đọc dữ liệu và chuẩn bị dữ liệu cho Surprise.

- Đang dùng MovieLens 1M với các file ratings.dat, movies.dat, users.dat.
- Surprise không dùng trực tiếp ma trận User-Item rộng.
- Surprise cần dữ liệu dạng bảng dài gồm: userId, movieId, rating.
"""

import pandas as pd
from surprise import Dataset, Reader

from config import (
    RATINGS_PATH,
    MOVIES_PATH,
    OUTPUT_DIR,
    MIN_USER_RATINGS,
    MIN_MOVIE_RATINGS,
    SAMPLE_SIZE,
    RANDOM_STATE,
    RATING_SCALE,
)


def read_movielens_1m():
    """
    Đọc dữ liệu MovieLens 1M từ ratings.dat và movies.dat.

    ratings.dat có dạng:
        UserID::MovieID::Rating::Timestamp

    movies.dat có dạng:
        MovieID::Title::Genres
    """
    if not RATINGS_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {RATINGS_PATH}. Hãy đặt file Thành viên 3 cùng thư mục với ratings.dat."
        )

    if not MOVIES_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {MOVIES_PATH}. Hãy đặt file Thành viên 3 cùng thư mục với movies.dat."
        )

    ratings = pd.read_csv(
        RATINGS_PATH,
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1",
    )

    movies = pd.read_csv(
        MOVIES_PATH,
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1",
    )

    return ratings, movies


def filter_ratings_like_member1(ratings):
    """
    Lọc dữ liệu tương thích với data_prep.py của Thành viên 1:
    - Giữ phim có ít nhất MIN_MOVIE_RATINGS lượt đánh giá.
    - Giữ user có ít nhất MIN_USER_RATINGS lượt đánh giá.
    """
    movie_counts = ratings["movieId"].value_counts()
    valid_movies = movie_counts[movie_counts >= MIN_MOVIE_RATINGS].index

    filtered = ratings[ratings["movieId"].isin(valid_movies)].copy()

    user_counts = filtered["userId"].value_counts()
    valid_users = user_counts[user_counts >= MIN_USER_RATINGS].index

    filtered = filtered[filtered["userId"].isin(valid_users)].copy()

    return filtered


def create_sample_100k(ratings):
    """
    Tạo tập mẫu 100K từ MovieLens 1M sau tiền xử lý.

    Đây KHÔNG phải MovieLens 100K gốc.
    Đây là sample 100.000 dòng từ MovieLens 1M để kiểm thử ảnh hưởng quy mô dữ liệu.
    """
    if len(ratings) <= SAMPLE_SIZE:
        return ratings.copy()

    return ratings.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).copy()


def save_prepared_datasets(ratings_full, ratings_sample):
    """
    Lưu dữ liệu dạng bảng dài để phục vụ báo cáo và kiểm tra lại.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    ratings_full[["userId", "movieId", "rating"]].to_csv(
        OUTPUT_DIR / "ratings_1m_clean_long_format.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ratings_sample[["userId", "movieId", "rating"]].to_csv(
        OUTPUT_DIR / "ratings_100k_sample_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )


def build_surprise_dataset(ratings):
    """
    Chuyển DataFrame sang Dataset của Surprise.
    """
    reader = Reader(rating_scale=RATING_SCALE)
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    return data


def build_dataset_summary(raw_ratings, clean_ratings, sample_ratings):
    """
    Tạo bảng mô tả dữ liệu để đưa vào báo cáo.
    """
    rows = [
        {
            "Dataset": "MovieLens 1M raw",
            "Ratings": len(raw_ratings),
            "Users": raw_ratings["userId"].nunique(),
            "Movies": raw_ratings["movieId"].nunique(),
        },
        {
            "Dataset": "MovieLens 1M after filtering",
            "Ratings": len(clean_ratings),
            "Users": clean_ratings["userId"].nunique(),
            "Movies": clean_ratings["movieId"].nunique(),
        },
        {
            "Dataset": "100K sample from filtered MovieLens 1M",
            "Ratings": len(sample_ratings),
            "Users": sample_ratings["userId"].nunique(),
            "Movies": sample_ratings["movieId"].nunique(),
        },
    ]

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False, encoding="utf-8-sig")

    return summary


def prepare_all_datasets():
    """
    Hàm chính cho bước chuẩn bị dữ liệu của Thành viên 3.
    """
    raw_ratings, movies = read_movielens_1m()
    clean_ratings = filter_ratings_like_member1(raw_ratings)
    sample_ratings = create_sample_100k(clean_ratings)

    OUTPUT_DIR.mkdir(exist_ok=True)
    save_prepared_datasets(clean_ratings, sample_ratings)
    summary = build_dataset_summary(raw_ratings, clean_ratings, sample_ratings)

    return {
        "movies": movies,
        "raw_ratings": raw_ratings,
        "clean_1m": clean_ratings,
        "sample_100k": sample_ratings,
        "summary": summary,
    }
