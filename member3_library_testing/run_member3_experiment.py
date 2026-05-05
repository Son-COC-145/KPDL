"""
THÀNH VIÊN 3 - File chạy chính.

Cách chạy:
    python run_member3_experiment.py

Yêu cầu:
    pip install pandas numpy matplotlib scikit-surprise

File này cần được đặt cùng thư mục với:
    ratings.dat
    movies.dat
    users.dat

Kết quả được lưu trong thư mục:
    member3_outputs/
"""

from data_loader import prepare_all_datasets
from evaluator import run_all_experiments
from visualization import draw_all_charts
from recommender_demo import export_demo_recommendations
from report_notes import write_report_notes


def main():
    print("=== THÀNH VIÊN 3: TRIỂN KHAI THƯ VIỆN & THỰC NGHIỆM ===")

    print("\n[1] Đọc và chuẩn bị dữ liệu...")
    prepared_data = prepare_all_datasets()
    print(prepared_data["summary"])

    print("\n[2] Chạy thực nghiệm các mô hình Surprise...")
    results_df = run_all_experiments(prepared_data)
    print("\nKết quả thực nghiệm:")
    print(results_df)

    print("\n[3] Vẽ biểu đồ...")
    draw_all_charts(results_df)

    print("\n[4] Tạo ví dụ Top 5 phim gợi ý bằng SVD...")
    top5 = export_demo_recommendations(prepared_data)
    print(top5)

    print("\n[5] Tạo ghi chú báo cáo...")
    write_report_notes()

    print("\nHoàn thành. Kiểm tra thư mục member3_outputs/ để lấy kết quả.")


if __name__ == "__main__":
    main()
