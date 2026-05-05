"""
THÀNH VIÊN 3 - Huấn luyện, kiểm thử, đo RMSE/MAE và thời gian chạy.
"""

import time
import pandas as pd
from surprise import accuracy
from surprise.model_selection import train_test_split

from config import TEST_SIZE, RANDOM_STATE, OUTPUT_DIR
from data_loader import build_surprise_dataset
from models import get_library_models


def evaluate_one_model(data, model, model_name, dataset_name):
    """
    Huấn luyện một mô hình và trả về RMSE, MAE, thời gian train/test.
    """
    trainset, testset = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    start_train = time.time()
    model.fit(trainset)
    train_time = time.time() - start_train

    start_test = time.time()
    predictions = model.test(testset)
    test_time = time.time() - start_test

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    return {
        "Dataset": dataset_name,
        "Model": model_name,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "TrainTimeSeconds": round(train_time, 4),
        "TestTimeSeconds": round(test_time, 4),
        "TotalTimeSeconds": round(train_time + test_time, 4),
        "TestSize": len(testset),
    }


def run_experiments_on_dataset(ratings, dataset_name):
    """
    Chạy toàn bộ mô hình thư viện trên một bộ dữ liệu.
    """
    data = build_surprise_dataset(ratings)
    results = []

    for model_name, model in get_library_models():
        print(f"Đang chạy {model_name} trên {dataset_name}...")
        result = evaluate_one_model(data, model, model_name, dataset_name)
        results.append(result)
        print(result)

    return results


def run_all_experiments(prepared_data):
    """
    Chạy thực nghiệm trên:
    - Tập mẫu 100K trích từ MovieLens 1M.
    - Tập MovieLens 1M sau tiền xử lý.
    """
    all_results = []

    all_results.extend(
        run_experiments_on_dataset(
            prepared_data["sample_100k"],
            "Sample 100K from MovieLens 1M",
        )
    )

    all_results.extend(
        run_experiments_on_dataset(
            prepared_data["clean_1m"],
            "MovieLens 1M filtered",
        )
    )

    results_df = pd.DataFrame(all_results)
    OUTPUT_DIR.mkdir(exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / "evaluation_results.csv", index=False, encoding="utf-8-sig")

    return results_df
