"""
=============================================================
THÀNH VIÊN 2: LẬP TRÌNH VIÊN LÕI - THUẬT TOÁN THUẦN (FROM SCRATCH)
Collaborative Filtering dùng Cosine Similarity & Pearson Correlation
Chỉ sử dụng thư viện cơ bản: NumPy, Pandas
=============================================================
"""

import numpy as np
import pandas as pd
import time


# ============================================================
# PHẦN 1: ĐỌC DỮ LIỆU ĐẦU VÀO
# ============================================================

print("=" * 60)
print("  COLLABORATIVE FILTERING - FROM SCRATCH")
print("=" * 60)

print("\n[1] Đang đọc ma trận User-Item từ file CSV...")

user_item_matrix = pd.read_csv("user_item_matrix_clean.csv", index_col="userId")

print(f"    Ma trận đã load: {user_item_matrix.shape[0]} users x {user_item_matrix.shape[1]} phim")
print(f"    Ví dụ 5 user đầu, 3 phim đầu:")
print(user_item_matrix.iloc[:5, :3].to_string())

matrix       = user_item_matrix.values.astype(float)
user_ids     = user_item_matrix.index.tolist()
movie_titles = user_item_matrix.columns.tolist()


# ============================================================
# PHẦN 2: XÂY DỰNG MA TRẬN ĐỘ TƯƠNG ĐỒNG (VECTORIZED - FROM SCRATCH)
# ============================================================

def build_cosine_similarity(matrix):
    """
    Tính ma trận Cosine Similarity giữa tất cả cặp user.

    Công thức:
        cos(A, B) = (A · B) / (||A|| * ||B||)

    Tối ưu: chuẩn hoá toàn bộ matrix thành unit vectors một lần,
    sau đó dùng phép nhân ma trận (dot product) để tính tất cả cặp
    cùng lúc — nhanh hơn vòng lặp O(n²) rất nhiều.

    Tham số:
        matrix: ma trận User-Item (numpy array, shape: n_users x n_movies)

    Trả về:
        sim_matrix: ma trận vuông (n_users x n_users)
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)  # (n_users, 1)
    norms[norms == 0] = 1e-10                               # tránh chia 0
    matrix_norm = matrix / norms                            # unit vectors
    sim_matrix  = matrix_norm @ matrix_norm.T               # dot product toàn bộ
    return np.clip(sim_matrix, -1.0, 1.0)


def build_pearson_similarity(matrix):
    """
    Tính ma trận Pearson Correlation giữa tất cả cặp user.

    Công thức:
        pearson(A, B) = Σ[(a_i - ā)(b_i - b̄)] / sqrt[Σ(a_i - ā)² * Σ(b_i - b̄)²]

    Tối ưu: mean-center toàn bộ matrix một lần (chỉ trên ô đã rated),
    sau đó áp dụng logic Cosine trên matrix đã center.
    Pearson loại bỏ bias chấm điểm (người hay cho cao/thấp hơn trung bình).

    Tham số:
        matrix: ma trận User-Item (numpy array, shape: n_users x n_movies)

    Trả về:
        sim_matrix: ma trận vuông (n_users x n_users)
    """
    centered = matrix.copy()
    for i in range(matrix.shape[0]):
        rated = matrix[i] != 0
        if rated.sum() > 0:
            centered[i, rated] -= matrix[i, rated].mean()  # trừ mean chỉ trên ô rated
        # Ô chưa rated giữ nguyên 0

    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    centered_norm = centered / norms
    sim_matrix    = centered_norm @ centered_norm.T
    return np.clip(sim_matrix, -1.0, 1.0)


def build_similarity_matrix(matrix, method="cosine"):
    """
    Wrapper gọi hàm tính ma trận tương đồng theo method.

    Tham số:
        matrix: ma trận User-Item (numpy array)
        method: "cosine" hoặc "pearson"

    Trả về:
        sim_matrix: ma trận vuông (n_users x n_users)
    """
    print(f"\n[2] Xây dựng ma trận tương đồng ({method}) cho {matrix.shape[0]} users...")
    start = time.time()

    if method == "cosine":
        sim = build_cosine_similarity(matrix)
    elif method == "pearson":
        sim = build_pearson_similarity(matrix)
    else:
        raise ValueError("method phải là 'cosine' hoặc 'pearson'")

    print(f"    Hoàn thành! Thời gian: {time.time() - start:.2f} giây")
    return sim


# ============================================================
# PHẦN 3: HÀM DỰ ĐOÁN ĐIỂM RATING
# ============================================================

def predict_rating(user_idx, movie_idx, matrix, sim_matrix, k=10):
    """
    Dự đoán điểm user sẽ chấm cho một bộ phim.

    Công thức dự đoán (User-based CF):
        r̂(u, i) = r̄_u + Σ[sim(u,v) * (r(v,i) - r̄_v)] / Σ|sim(u,v)|

    - r̄_u     : điểm trung bình của user u (chỉ tính phim đã xem)
    - sim(u,v) : độ tương đồng giữa user u và user v
    - r(v,i)   : điểm user v đã chấm phim i
    - Chỉ dùng K neighbors gần nhất đã xem phim i

    Tham số:
        user_idx  : chỉ số hàng của user trong matrix
        movie_idx : chỉ số cột của phim trong matrix
        matrix    : ma trận User-Item (numpy array)
        sim_matrix: ma trận độ tương đồng
        k         : số neighbors

    Trả về:
        float: điểm dự đoán trong [1, 5], hoặc 0 nếu không đủ dữ liệu
    """
    rated_mask = matrix[:, movie_idx] != 0
    rated_mask[user_idx] = False

    if rated_mask.sum() == 0:
        return 0.0

    neighbor_sims    = sim_matrix[user_idx][rated_mask]
    neighbor_ratings = matrix[rated_mask, movie_idx]

    # Tính mean của mỗi neighbor (vectorized)
    neighbor_matrix = matrix[rated_mask]
    rated_counts    = (neighbor_matrix != 0).sum(axis=1)
    rated_counts    = np.where(rated_counts == 0, 1, rated_counts)
    neighbor_means  = neighbor_matrix.sum(axis=1) / rated_counts

    # Lấy Top-K
    if len(neighbor_sims) > k:
        top_k = np.argpartition(neighbor_sims, -k)[-k:]
    else:
        top_k = np.arange(len(neighbor_sims))

    top_sims    = neighbor_sims[top_k]
    top_ratings = neighbor_ratings[top_k]
    top_means   = neighbor_means[top_k]

    user_rated = matrix[user_idx][matrix[user_idx] != 0]
    user_mean  = user_rated.mean() if len(user_rated) > 0 else 3.0

    numerator   = np.dot(top_sims, top_ratings - top_means)
    denominator = np.abs(top_sims).sum()

    if denominator == 0:
        return float(user_mean)

    return float(np.clip(user_mean + numerator / denominator, 1.0, 5.0))


# ============================================================
# PHẦN 4: HÀM GỢI Ý TOP-5 PHIM CHO MỘT USER
# ============================================================

def recommend_top5(user_id, matrix, user_ids, movie_titles, sim_matrix, k=10):
    """
    Gợi ý Top 5 phim mà user chưa xem và có khả năng thích nhất.

    Tham số:
        user_id     : ID của user cần gợi ý (int)
        matrix      : ma trận User-Item (numpy array)
        user_ids    : danh sách userId tương ứng hàng matrix
        movie_titles: danh sách tên phim tương ứng cột matrix
        sim_matrix  : ma trận độ tương đồng
        k           : số neighbors khi dự đoán

    Trả về:
        DataFrame: Top 5 phim với cột [Phim, Diem_Du_Doan]
    """
    if user_id not in user_ids:
        print(f"    Lỗi: UserID {user_id} không tồn tại trong dữ liệu!")
        return None

    user_idx    = user_ids.index(user_id)
    unseen_idxs = np.where(matrix[user_idx] == 0)[0]

    if len(unseen_idxs) == 0:
        print(f"    User {user_id} đã xem tất cả phim!")
        return None

    predictions = [
        {"Phim": movie_titles[m],
         "Diem_Du_Doan": round(predict_rating(user_idx, m, matrix, sim_matrix, k), 4)}
        for m in unseen_idxs
    ]
    predictions = [p for p in predictions if p["Diem_Du_Doan"] > 0]

    if not predictions:
        print(f"    Không thể dự đoán cho UserID {user_id}.")
        return None

    result_df        = pd.DataFrame(predictions)
    result_df        = result_df.sort_values("Diem_Du_Doan", ascending=False).head(5).reset_index(drop=True)
    result_df.index += 1
    return result_df


# ============================================================
# PHẦN 5: TÍNH RMSE ĐỂ ĐÁNH GIÁ MÔ HÌNH
# ============================================================

def compute_rmse(matrix, sim_matrix, n_samples=200, k=10, random_seed=42):
    """
    Tính RMSE theo phương pháp leave-one-out:
    1. Lấy ngẫu nhiên n_samples ô có rating thật
    2. Tạm xoá rating đó (đặt = 0)
    3. Dùng mô hình dự đoán lại
    4. So sánh với rating thật, tính sai số bình phương trung bình

    Công thức:  RMSE = sqrt( mean( (r_thật - r_dự_đoán)² ) )

    Tham số:
        matrix    : ma trận User-Item gốc
        sim_matrix: ma trận độ tương đồng
        n_samples : số điểm test
        k         : số neighbors

    Trả về:
        float: RMSE
    """
    print(f"\n[4] Tính RMSE trên {n_samples} mẫu ngẫu nhiên...")
    np.random.seed(random_seed)

    rated_rows, rated_cols = np.where(matrix != 0)
    n_samples  = min(n_samples, len(rated_rows))
    chosen     = np.random.choice(len(rated_rows), n_samples, replace=False)

    matrix_test = matrix.copy()
    errors      = []

    for idx in chosen:
        r, c               = rated_rows[idx], rated_cols[idx]
        actual             = matrix_test[r, c]
        matrix_test[r, c]  = 0
        predicted          = predict_rating(r, c, matrix_test, sim_matrix, k)
        matrix_test[r, c]  = actual
        if predicted > 0:
            errors.append((actual - predicted) ** 2)

    rmse = float(np.sqrt(np.mean(errors))) if errors else float("inf")
    print(f"    RMSE (K={k}): {rmse:.4f}")
    return rmse


# ============================================================
# PHẦN 6: CHẠY TOÀN BỘ PIPELINE
# ============================================================

if __name__ == "__main__":

    # Bước 1: Xây dựng ma trận tương đồng Cosine & Pearson
    sim_cosine  = build_similarity_matrix(matrix, method="cosine")
    sim_pearson = build_similarity_matrix(matrix, method="pearson")

    # Bước 2: Demo gợi ý phim cho 3 user đầu
    print("\n" + "=" * 60)
    print("  DEMO GỢI Ý PHIM")
    print("=" * 60)

    for uid in user_ids[:3]:
        print(f"\n[3] Top 5 phim gợi ý cho UserID = {uid} (Cosine):")
        top5 = recommend_top5(uid, matrix, user_ids, movie_titles, sim_cosine, k=10)
        if top5 is not None:
            print(top5.to_string())

    # Bước 3: Tính RMSE cho cả 2 phương pháp
    rmse_cosine  = compute_rmse(matrix, sim_cosine,  n_samples=200, k=10)
    rmse_pearson = compute_rmse(matrix, sim_pearson, n_samples=200, k=10)

    # Bước 4: So sánh Cosine vs Pearson trên 1 điểm mẫu
    print("\n" + "=" * 60)
    print("  SO SÁNH COSINE vs PEARSON")
    print("=" * 60)

    u_idx  = 0
    m_idx  = int(np.where(matrix[u_idx] != 0)[0][0])
    actual = matrix[u_idx, m_idx]
    matrix[u_idx, m_idx] = 0

    p_cos = predict_rating(u_idx, m_idx, matrix, sim_cosine,  k=10)
    p_pea = predict_rating(u_idx, m_idx, matrix, sim_pearson, k=10)

    matrix[u_idx, m_idx] = actual

    print(f"\n  UserID = {user_ids[u_idx]} | Phim = {movie_titles[m_idx]}")
    print(f"  Rating thật     : {actual:.1f}")
    print(f"  Dự đoán Cosine  : {p_cos:.4f}")
    print(f"  Dự đoán Pearson : {p_pea:.4f}")
    print(f"\n  RMSE Cosine     : {rmse_cosine:.4f}")
    print(f"  RMSE Pearson    : {rmse_pearson:.4f}")

    # Bước 5: Lưu ma trận tương đồng cho Thành viên 3
    print("\n[5] Lưu ma trận tương đồng ra file...")
    pd.DataFrame(sim_cosine,  index=user_ids, columns=user_ids).to_csv("similarity_matrix_cosine.csv")
    pd.DataFrame(sim_pearson, index=user_ids, columns=user_ids).to_csv("similarity_matrix_pearson.csv")
    print("    Đã lưu: similarity_matrix_cosine.csv")
    print("    Đã lưu: similarity_matrix_pearson.csv")

    print("\n" + "=" * 60)
    print("  HOÀN THÀNH!")
    print("  Output: similarity_matrix_cosine.csv | similarity_matrix_pearson.csv")
    print("=" * 60)