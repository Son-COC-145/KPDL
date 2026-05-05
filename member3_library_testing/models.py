"""
THÀNH VIÊN 3 - Khai báo các mô hình thư viện Surprise.
"""

from surprise import KNNBasic, KNNWithMeans, SVD

from config import KNN_K, SVD_N_FACTORS, SVD_N_EPOCHS, RANDOM_STATE


def get_library_models():
    """
    Trả về danh sách các mô hình thư viện cần chạy thực nghiệm.

    KNNBasic_Cosine:
        Mô hình KNN cơ bản dùng Cosine Similarity.

    KNNBasic_Pearson:
        Mô hình KNN cơ bản dùng Pearson Correlation.

    KNNWithMeans_Cosine:
        KNN có hiệu chỉnh theo trung bình rating của user.

    SVD:
        Mô hình phân rã ma trận.
    """
    models = [
        (
            "KNNBasic_Cosine",
            KNNBasic(
                k=KNN_K,
                sim_options={"name": "cosine", "user_based": True},
                verbose=False,
            ),
        ),
        (
            "KNNBasic_Pearson",
            KNNBasic(
                k=KNN_K,
                sim_options={"name": "pearson", "user_based": True},
                verbose=False,
            ),
        ),
        (
            "KNNWithMeans_Cosine",
            KNNWithMeans(
                k=KNN_K,
                sim_options={"name": "cosine", "user_based": True},
                verbose=False,
            ),
        ),
        (
            "SVD",
            SVD(
                n_factors=SVD_N_FACTORS,
                n_epochs=SVD_N_EPOCHS,
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    return models
