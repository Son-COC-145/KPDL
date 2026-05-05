"""
THÀNH VIÊN 3 - Cấu hình thực nghiệm thư viện
Movie Recommendation System - Library & Testing
"""

from pathlib import Path

# Đường dẫn dữ liệu gốc MovieLens 1M.
RATINGS_PATH = Path("ratings.dat")
MOVIES_PATH = Path("movies.dat")

# Thư mục lưu kết quả phần Thành viên 3.
OUTPUT_DIR = Path("member3_outputs")

# Cấu hình tiền xử lý tương thích với data_prep.py của Thành viên 1.
MIN_USER_RATINGS = 20
MIN_MOVIE_RATINGS = 20

# Tập mẫu 100K được trích từ MovieLens 1M sau tiền xử lý.
SAMPLE_SIZE = 100_000
RANDOM_STATE = 42

# Chia train/test.
TEST_SIZE = 0.2

# Thang điểm rating của MovieLens.
RATING_SCALE = (1, 5)

# Cấu hình mô hình KNN.
KNN_K = 40

# Cấu hình mô hình SVD.
SVD_N_FACTORS = 100
SVD_N_EPOCHS = 20
