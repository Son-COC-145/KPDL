import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# PHẦN 1: ĐỌC DỮ LIỆU TỪ FILE .DAT
print("Đang đọc dữ liệu từ file .dat...")

# 1. Đọc file ratings.dat
# File này cấu trúc là: UserID::MovieID::Rating::Timestamp
ratings_cols = ['userId', 'movieId', 'rating', 'timestamp']
ratings = pd.read_csv('ratings.dat', sep='::', engine='python', names=ratings_cols)

# 2. Đọc file movies.dat
# File này cấu trúc là: MovieID::Title::Genres
movies_cols = ['movieId', 'title', 'genres']
# Thêm encoding='latin-1' vì tên phim cũ có thể chứa ký tự đặc biệt
movies = pd.read_csv('movies.dat', sep='::', engine='python', names=movies_cols, encoding='latin-1')

# Ghép tên phim vào bảng ratings để dễ nhìn
df = pd.merge(ratings, movies, on='movieId')
print(f"Tổng số lượt đánh giá: {len(df)}")

# In thử 5 dòng đầu tiên để kiểm tra xem đã đọc đúng chưa
print(df.head())

# PHẦN 2: KHÁM PHÁ DỮ LIỆU (EDA) - LẤY HÌNH ĐƯA VÀO BÁO CÁO
# 1. Biểu đồ phân bố điểm số (Xem người dùng hay cho mấy sao)
plt.figure(figsize=(8, 5))
sns.countplot(x='rating', data=df, hue='rating', palette='viridis', legend=False)
plt.title('Phân bố điểm đánh giá (Ratings Distribution)')
plt.xlabel('Điểm số')
plt.ylabel('Số lượng')
plt.show() # -> Lưu hình này lại cho vào báo cáo!

# PHẦN 3: LÀM SẠCH VÀ GIẢM ĐỘ THƯA THỚT (SPARSITY)
# Giải thích cho báo cáo: Nếu giữ nguyên tất cả, ma trận sẽ rất rỗng (nhiều số 0) 
# vì có những phim chỉ 1 người xem, hoặc có người chỉ đánh giá 1 phim.
# Do đó, ta cần lọc bỏ các "nhiễu" này.

min_movie_ratings = 20 # Phim phải có ít nhất 20 lượt đánh giá mới giữ lại
min_user_ratings = 20  # Người dùng phải đánh giá ít nhất 20 phim mới giữ lại

# Lọc các phim phổ biến
movie_counts = df['movieId'].value_counts()
popular_movies = movie_counts[movie_counts >= min_movie_ratings].index
df_filtered = df[df['movieId'].isin(popular_movies)]

# Lọc các user tích cực
user_counts = df_filtered['userId'].value_counts()
active_users = user_counts[user_counts >= min_user_ratings].index
df_filtered = df_filtered[df_filtered['userId'].isin(active_users)]

print(f"Số lượt đánh giá sau khi lọc rác: {len(df_filtered)}")


# PHẦN 4: TẠO MA TRẬN USER-ITEM (ĐÍCH ĐẾN CUỐI CÙNG)
# Chuyển bảng dọc thành ma trận 2 chiều (Pivot Table)
user_item_matrix = df_filtered.pivot_table(index='userId', columns='title', values='rating')

# Những phim người dùng chưa xem sẽ có giá trị NaN. Ta điền số 0 vào đó.
user_item_matrix.fillna(0, inplace=True)

print("\nMa trận User-Item đã sẵn sàng!")
print(f"Kích thước ma trận (Số User x Số Phim): {user_item_matrix.shape}")

# In thử 5 hàng và 5 cột đầu tiên để kiểm tra
print(user_item_matrix.iloc[:5, :5])

# Xuất file này ra để Thành viên 2 và 3 dùng làm đầu vào chạy thuật toán
user_item_matrix.to_csv('user_item_matrix_clean.csv')
print("\nĐã lưu file 'user_item_matrix_clean.csv'")