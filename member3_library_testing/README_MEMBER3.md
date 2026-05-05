# Thành viên 3 - Triển khai thư viện & Thực nghiệm

## 1. Vai trò của phần này

Thư mục `member3_library_testing` là phần thực hiện của **Thành viên 3: Triển khai thư viện & Thực nghiệm** trong đề tài:

**Xây dựng hệ tư vấn phim (Movie Recommendation System)**

Nhiệm vụ chính của phần này:

- Triển khai mô hình gợi ý phim bằng thư viện `scikit-surprise`.
- Chạy các mô hình Collaborative Filtering bằng thư viện:
  - `KNNBasic_Cosine`
  - `KNNBasic_Pearson`
  - `KNNWithMeans_Cosine`
  - `SVD`
- Đánh giá mô hình bằng các chỉ số:
  - RMSE
  - MAE
  - thời gian huấn luyện
  - thời gian kiểm thử
  - tổng thời gian thực thi
- So sánh kết quả trên 2 quy mô dữ liệu:
  - `Sample 100K from MovieLens 1M`
  - `MovieLens 1M filtered`
- Xuất bảng kết quả, biểu đồ và ví dụ Top 5 phim gợi ý.

---

## 2. Cấu trúc thư mục

```text
member3_library_testing/
├── config.py
├── data_loader.py
├── evaluator.py
├── models.py
├── recommender_demo.py
├── report_notes.py
├── requirements.txt
├── run_member3_experiment.py
├── visualization.py
└── README_MEMBER3.md
```

Ý nghĩa các file:

| File | Chức năng |
|---|---|
| `config.py` | Lưu cấu hình đường dẫn, tham số lọc dữ liệu, tham số mô hình |
| `data_loader.py` | Đọc `ratings.dat`, `movies.dat`, lọc dữ liệu và tạo sample 100K |
| `models.py` | Khai báo các mô hình thư viện Surprise |
| `evaluator.py` | Huấn luyện, kiểm thử, tính RMSE, MAE và thời gian chạy |
| `visualization.py` | Vẽ biểu đồ so sánh RMSE, MAE và thời gian thực thi |
| `recommender_demo.py` | Sinh ví dụ Top 5 phim gợi ý bằng mô hình SVD |
| `report_notes.py` | Tạo file ghi chú phục vụ viết báo cáo |
| `run_member3_experiment.py` | File chạy chính của Thành viên 3 |
| `requirements.txt` | Danh sách thư viện cần cài |

---

## 3. Dữ liệu đầu vào

Phần này sử dụng cùng dữ liệu với Thành viên 1 và Thành viên 2.

Các file dữ liệu cần nằm ở thư mục gốc project, cùng cấp với thư mục `member3_library_testing`:

```text
ratings.dat
movies.dat
users.dat
```

Cấu trúc thư mục đúng:

```text
KPDL-dev2/
├── ratings.dat
├── movies.dat
├── users.dat
├── data_prep.py
├── user_item_matrix_clean.csv
├── Collaborative filtering scratch.py
│
└── member3_library_testing/
    ├── config.py
    ├── data_loader.py
    ├── evaluator.py
    ├── models.py
    ├── recommender_demo.py
    ├── report_notes.py
    ├── requirements.txt
    ├── run_member3_experiment.py
    └── visualization.py
```

Lưu ý: `scikit-surprise` không sử dụng trực tiếp ma trận User-Item dạng rộng `user_item_matrix_clean.csv`. Thư viện này cần dữ liệu dạng bảng dài gồm:

```text
userId, movieId, rating
```

Vì vậy phần Thành viên 3 đọc trực tiếp từ `ratings.dat`, sau đó tự chuyển sang định dạng phù hợp cho Surprise.

---

## 4. Lưu ý về tập dữ liệu 100K

Trong project hiện tại, nhóm đang sử dụng bộ dữ liệu **MovieLens 1M** với các file:

```text
ratings.dat
movies.dat
users.dat
```

Phần `Sample 100K from MovieLens 1M` trong thực nghiệm là **100.000 dòng rating được trích mẫu từ MovieLens 1M sau tiền xử lý**, không phải bộ MovieLens 100K gốc.

Khi viết báo cáo, nên diễn đạt là:

> Nhóm thực nghiệm trên bộ MovieLens 1M và một tập mẫu 100.000 lượt đánh giá được trích từ MovieLens 1M sau tiền xử lý nhằm đánh giá ảnh hưởng của quy mô dữ liệu đến độ chính xác và thời gian thực thi.

Không nên ghi nhầm là nhóm dùng MovieLens 100K gốc nếu chưa tải thêm bộ `u.data`, `u.item`, `u.user`.

---

## 5. Yêu cầu môi trường

Khuyến nghị dùng:

```text
Python 3.10
numpy < 2
pandas
matplotlib
scikit-surprise
```

Trên Windows, `scikit-surprise` có thể cần thêm:

```text
Microsoft Visual C++ Build Tools
```

Nếu cài bằng Python 3.14 hoặc NumPy 2.x, thư viện `scikit-surprise` có thể lỗi khi build.

---

## 6. Cài đặt môi trường

Tạo môi trường ảo bằng Python 3.10:

```powershell
py -3.10 -m venv .venv
```

Nếu PowerShell không cho activate do chính sách bảo mật, có thể chạy trực tiếp bằng Python trong `.venv` như sau:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install "numpy<2" pandas matplotlib
.\.venv\Scripts\python.exe -m pip install scikit-surprise
```

Kiểm tra thư viện Surprise:

```powershell
.\.venv\Scripts\python.exe -c "import surprise; print('surprise ok')"
```

Nếu hiện:

```text
surprise ok
```

là môi trường đã sẵn sàng.

---

## 7. Cách chạy thực nghiệm

Tại thư mục gốc project, tức thư mục có `ratings.dat`, chạy:

```powershell
.\.venv\Scripts\python.exe member3_library_testing/run_member3_experiment.py
```

Nếu dùng môi trường Python đã activate, có thể chạy:

```powershell
python member3_library_testing/run_member3_experiment.py
```

Sau khi chạy thành công, chương trình sẽ sinh thư mục:

```text
member3_outputs/
```

---

## 8. Kết quả đầu ra

Sau khi chạy xong, thư mục `member3_outputs` gồm:

```text
dataset_summary.csv
evaluation_results.csv
mae_comparison.png
ratings_100k_sample_clean.csv
ratings_1m_clean_long_format.csv
report_notes_member3.txt
rmse_comparison.png
runtime_comparison.png
scale_comparison.png
top5_recommendations_svd_demo.csv
```

Ý nghĩa các file:

| File | Ý nghĩa |
|---|---|
| `dataset_summary.csv` | Thống kê số lượng rating, user, movie của từng tập dữ liệu |
| `ratings_100k_sample_clean.csv` | Tập mẫu 100K sau tiền xử lý |
| `ratings_1m_clean_long_format.csv` | MovieLens 1M sau tiền xử lý ở dạng bảng dài |
| `evaluation_results.csv` | Bảng kết quả RMSE, MAE và thời gian chạy |
| `rmse_comparison.png` | Biểu đồ so sánh RMSE |
| `mae_comparison.png` | Biểu đồ so sánh MAE |
| `runtime_comparison.png` | Biểu đồ so sánh thời gian thực thi |
| `scale_comparison.png` | Biểu đồ ảnh hưởng của quy mô dữ liệu |
| `top5_recommendations_svd_demo.csv` | Ví dụ Top 5 phim gợi ý bằng SVD |
| `report_notes_member3.txt` | Ghi chú dùng để viết báo cáo |

---

## 9. Kết quả thực nghiệm đã chạy

Kết quả thực nghiệm hiện tại:

| Dataset | Model | RMSE | MAE | Total Time |
|---|---|---:|---:|---:|
| Sample 100K from MovieLens 1M | KNNBasic_Cosine | 1.0349 | 0.8185 | 2.3251s |
| Sample 100K from MovieLens 1M | KNNBasic_Pearson | 1.1536 | 0.9159 | 2.6909s |
| Sample 100K from MovieLens 1M | KNNWithMeans_Cosine | 1.0068 | 0.7852 | 2.4908s |
| Sample 100K from MovieLens 1M | SVD | 0.9492 | 0.7546 | 0.9242s |
| MovieLens 1M filtered | KNNBasic_Cosine | 0.9764 | 0.7727 | 145.0940s |
| MovieLens 1M filtered | KNNBasic_Pearson | 0.9617 | 0.7607 | 145.1781s |
| MovieLens 1M filtered | KNNWithMeans_Cosine | 0.9383 | 0.7319 | 129.2036s |
| MovieLens 1M filtered | SVD | 0.8725 | 0.6852 | 9.3916s |

Nhận xét chính:

> Mô hình SVD đạt RMSE thấp nhất trên cả tập Sample 100K và MovieLens 1M filtered. Đồng thời, SVD có thời gian thực thi thấp hơn đáng kể so với các mô hình KNN trên tập dữ liệu lớn. Điều này cho thấy SVD phù hợp hơn với dữ liệu rating thưa và quy mô lớn trong bài toán hệ tư vấn phim.

---

## 10. Ví dụ Top 5 phim gợi ý

File `top5_recommendations_svd_demo.csv` lưu ví dụ Top 5 phim được gợi ý bằng mô hình SVD.

Kết quả demo hiện tại:

| movieId | title | genres | predictedRating |
|---:|---|---|---:|
| 904 | Rear Window (1954) | Mystery\\|Thriller | 4.4114 |
| 1204 | Lawrence of Arabia (1962) | Adventure\\|War | 4.3751 |
| 1212 | Third Man, The (1949) | Mystery\\|Thriller | 4.3223 |
| 3429 | Creature Comforts (1990) | Animation\\|Comedy | 4.3218 |
| 913 | Maltese Falcon, The (1941) | Film-Noir\\|Mystery | 4.3213 |

---

## 11. Dùng kết quả cho báo cáo

Phần này phục vụ:

```text
Chương 3 - III. Cài đặt mô hình bằng thư viện chuyên dụng
Chương 3 - IV. Đánh giá và chứng minh thuật toán
```

Khi viết báo cáo, có thể sử dụng:

- `evaluation_results.csv` để lập bảng kết quả;
- `rmse_comparison.png` để minh họa so sánh sai số RMSE;
- `mae_comparison.png` để minh họa so sánh sai số MAE;
- `runtime_comparison.png` để minh họa thời gian thực thi;
- `scale_comparison.png` để chứng minh ảnh hưởng của quy mô dữ liệu;
- `top5_recommendations_svd_demo.csv` để minh họa kết quả gợi ý Top 5 phim.

---

## 12. Lưu ý khi đưa lên Git

Không upload thư mục môi trường ảo `.venv`.

Nên có file `.gitignore` ở thư mục gốc project với nội dung:

```gitignore
.venv/
__pycache__/
*.pyc
```

Các thư mục/file nên upload cho phần Thành viên 3:

```text
member3_library_testing/
member3_outputs/
.gitignore
```

Không cần upload lại dữ liệu `.dat` nếu dữ liệu đã có sẵn trên branch/repo chung của nhóm.
