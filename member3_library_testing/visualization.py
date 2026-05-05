"""
THÀNH VIÊN 3 - Vẽ biểu đồ đánh giá thực nghiệm.
"""

import matplotlib.pyplot as plt

from config import OUTPUT_DIR


def _build_label(df):
    return df["Dataset"] + " - " + df["Model"]


def plot_rmse_comparison(results_df):
    results_df = results_df.copy()
    results_df["Label"] = _build_label(results_df)

    plt.figure(figsize=(13, 6))
    plt.bar(results_df["Label"], results_df["RMSE"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("RMSE")
    plt.title("So sánh RMSE giữa các mô hình thư viện")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rmse_comparison.png", dpi=200)
    plt.close()


def plot_mae_comparison(results_df):
    results_df = results_df.copy()
    results_df["Label"] = _build_label(results_df)

    plt.figure(figsize=(13, 6))
    plt.bar(results_df["Label"], results_df["MAE"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("MAE")
    plt.title("So sánh MAE giữa các mô hình thư viện")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mae_comparison.png", dpi=200)
    plt.close()


def plot_runtime_comparison(results_df):
    results_df = results_df.copy()
    results_df["Label"] = _build_label(results_df)

    plt.figure(figsize=(13, 6))
    plt.bar(results_df["Label"], results_df["TotalTimeSeconds"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Thời gian thực thi tổng cộng giây")
    plt.title("So sánh thời gian thực thi giữa các mô hình thư viện")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "runtime_comparison.png", dpi=200)
    plt.close()


def plot_scale_comparison(results_df):
    """
    So sánh thời gian thực thi khi tăng quy mô dữ liệu.
    """
    pivot = results_df.pivot(index="Model", columns="Dataset", values="TotalTimeSeconds")

    plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Thời gian thực thi tổng cộng giây")
    plt.title("Ảnh hưởng của quy mô dữ liệu đến thời gian thực thi")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scale_comparison.png", dpi=200)
    plt.close()


def draw_all_charts(results_df):
    OUTPUT_DIR.mkdir(exist_ok=True)

    plot_rmse_comparison(results_df)
    plot_mae_comparison(results_df)
    plot_runtime_comparison(results_df)
    plot_scale_comparison(results_df)
