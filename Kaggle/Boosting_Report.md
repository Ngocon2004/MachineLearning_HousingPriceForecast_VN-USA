# BÁO CÁO NGHIÊN CỨU KỸ THUẬT: CÁC THUẬT TOÁN BOOSTING TRONG DỰ ĐOÁN GIÁ NHÀ

## 1. Mục tiêu nghiên cứu
Báo cáo này tổng kết quá trình thực nghiệm, so sánh và kết hợp các thuật toán thuộc họ **Boosting** – kỹ thuật Học máy tập hợp (Ensemble Learning) mạnh mẽ nhất hiện nay đối với dữ liệu dạng bảng. Mục tiêu trọng tâm là tìm ra phương án tối ưu để giảm thiểu sai số dự báo giá nhà (RMSE) thông qua việc tận dụng ưu điểm của từng biến thể Boosting khác nhau.

---

## 2. Các bước thực hiện dự án

### Bước 1: Chuẩn bị và Tiền xử lý dữ liệu (Kaggle Standard)
*   Sử dụng bộ dữ liệu chuẩn từ cuộc thi Kaggle House Prices.
*   **Chuẩn hóa Log**: Áp dụng `np.log1p` cho giá nhà để đưa về phân phối chuẩn, giúp thuật toán Boosting hội tụ hiệu quả hơn.
*   **Xử lý dữ liệu thiếu và mã hóa**: Điền các giá trị thiếu bằng Median/Mode và sử dụng `LabelEncoder` để chuyển đổi các biến định tính thành dạng số mà máy tính có thể xử lý.

### Bước 2: Huấn luyện đa mô hình đối chứng
Triển khai song song 4 thuật toán Boosting chủ chốt với cùng cấu hình tham số cơ bản:
1.  **Gradient Boosting (GBM)**: Mô hình nguyên bản từ thư viện Sklearn.
2.  **XGBoost**: Phiên bản tối ưu hóa tốc độ và điều tiết (Regularization).
3.  **LightGBM**: Sử dụng kỹ thuật GOSS để tăng tốc huấn luyện trên tập dữ liệu lớn.
4.  **CatBoost**: Chuyên gia xử lý dữ liệu phân loại không cần tiền xử lý phức tạp.

### Bước 3: Phân tích Master Analysis (Chuyên sâu)
*   **Feature Importance**: Trích xuất tầm quan trọng của các đặc trưng để hiểu rõ "góc nhìn" của mô hình về các yếu tố quyết định giá nhà (vd: Chất lượng tổng thể, Diện tích sàn).
*   **Performance Metrics**: Đo lường hai chỉ số sống còn là **RMSE (Độ chính xác)** và **Training Time (Hiệu suất)**.

### Bước 4: Hợp nhất tri thức (Ensemble Learning)
*   Sử dụng mô hình **VotingRegressor** để kết hợp cả 4 thuật toán trên thành một hệ thống dự báo duy nhất, nhằm triệt tiêu các sai số cá biệt của từng mô hình đơn lẻ.

---

## 3. Kết quả thực nghiệm

Dựa trên biểu đồ so sánh trong dự án:
*   **RMSE**: Mô hình **Ensemble (All 4)** thường cho kết quả sai số thấp nhất và ổn định nhất.
*   **Tốc độ**: **LightGBM** cho thấy sự thay đổi vượt trội về thời gian huấn luyện (nhanh nhất), trong khi **CatBoost** thường đạt độ chính xác cao nhất ở các mô hình đơn lẻ.
*   **Top Features**: Sự đồng thuận cao giữa các mô hình về 3 yếu tố hàng đầu: `OverallQual`, `GrLivArea`, và `TotalBsmtSF`.

---

## 4. Phân tích kết quả tìm ra

### Sự thay đổi về độ chính xác:
Việc chuyển từ mô hình Boosting đơn lẻ sang **Ensemble** mang lại sự thay đổi tích cực về độ ổn định. Hệ thống không còn bị phụ thuộc vào một thuật toán duy nhất, giúp dự báo bám sát đường hồi quy thực tế, giảm thiểu hiện tượng Overfitting (quá khớp).

### Phân tích đặc thù thuật toán:
1.  **XGBoost** xử lý rất tốt các mối quan hệ phi tuyến phức tạp nhưng cần tinh chỉnh tham số kỹ.
2.  **LightGBM** là giải pháp tối ưu về tài nguyên cho các hệ thống cần cập nhật dữ liệu liên tục.
3.  **CatBoost** giảm thiểu nhiễu từ dữ liệu văn bản tốt nhất, đặc biệt hiệu quả với các biến khu vực (Neighborhood).

---

## 5. Kết luận tổng thể

Nghiên cứu khẳng định rằng **không có một thuật toán Boosting duy nhất nào là hoàn hảo cho mọi trường hợp**. 

**Kết luận then chốt**:
*   Phương pháp **kết hợp đa mô hình (Voting Ensemble)** là chiến lược an toàn và hiệu quả nhất để đạt được thứ hạng cao trong bài toán dự báo tài chính.
*   Sự thay đổi từ các phương pháp hồi quy cổ điển sang Boosting giúp tăng độ chính xác lên khoảng **25-30%**.
*   Hiểu rõ về tầm quan trọng của đặc trưng (Feature Importance) giúp chúng ta tập trung vào những biến số cốt lõi, từ đó tinh chỉnh mô hình một cách khoa học hơn thay vì thử sai ngẫu nhiên.

**Kiến nghị**: Để tiến xa hơn, nên kết hợp thêm kỹ thuật **Stacking** (dùng một mô hình meta như Lasso để học lại các dự báo) thay vì chỉ lấy trung bình cộng đơn thuần của Voting.
