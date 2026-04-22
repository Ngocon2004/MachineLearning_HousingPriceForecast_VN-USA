# BÁO CÁO DỰ ÁN: HỆ THỐNG DỰ BÁO GIÁ BẤT ĐỘNG SẢN VIỆT NAM (AI-BASED)

## 1. Giới thiệu tổng quan
Dự án này được xây dựng nhằm mục đích xây dựng một công cụ hỗ trợ định giá bất động sản tự động tại thị trường Việt Nam dựa trên các thuật toán Học máy (Machine Learning) tiên tiến. Thay vì dựa vào các bộ dữ liệu mẫu (như Kaggle), dự án trực tiếp khai thác dữ liệu thực tế từ nền tảng **Batdongsan.com.vn** để phản ánh chính xác hơi thở và sự biến động của thị trường nội địa.

---

## 2. Quy trình thực hiện dự án (Step-by-Step)

Dự án được triển khai qua 5 giai đoạn cốt lõi:

### Bước 1: Thu thập dữ liệu (Web Scraping)
*   **Công cụ**: Sử dụng thư viện `Selenium` kết hợp `undetected-chromedriver`.
*   **Cách thức**: Giả lập hành vi người dùng thật để vượt qua các lớp bảo vệ bot (Cloudflare). Hệ thống đã cào thành công **1.000 tin đăng** mới nhất từ mục "Bán nhà riêng" với các trường thông tin: Tiêu đề, Giá, Diện tích, Địa điểm và Mô tả.

### Bước 2: Tiền xử lý dữ liệu (Data Cleaning)
*   **Chuẩn hóa đơn vị**: Chuyển đổi các giá trị văn bản như "10,5 tỷ" hoặc "500 triệu" về một đơn vị số thực duy nhất là **Triệu VNĐ**.
*   **Làm sạch diện tích**: Loại bỏ các ký tự đặc biệt (m²), chuyển đổi về dạng số thực (float).
*   **Lọc nhiễu (Outlier Handling)**: Loại bỏ các tin đăng có giá trị phi lý (dưới 200 triệu hoặc trên 500 tỷ) để đảm bảo mô hình không học các "tin đăng ảo".

### Bước 3: Kỹ thuật đặc trưng (Feature Engineering)
*   **Định danh vị trí**: Sử dụng Biểu thức chính quy (Regex) để quét cột mô tả và tiêu đề nhằm trích xuất thông tin **Quận/Huyện**. Đây là yếu tố then chốt vì vị trí quyết định đến 80% giá trị bất động sản tại Việt Nam.
*   **Mã hóa**: Sử dụng `LabelEncoder` để chuyển đổi tên các Quận thành mã số mà máy tính có thể hiểu được.

### Bước 4: Xây dựng siêu mô hình (Ensemble Learning)
*   **Kiến trúc**: Sử dụng kỹ thuật **Stacking Regressor** (Học máy tích hợp).
*   **Mô hình thành phần**:
    *   **XGBoost**: Bắt các mối quan hệ phi tuyến phức tạp giữa diện tích và giá.
    *   **LightGBM**: Tối ưu hóa đặc biệt cho các biến phân loại khu vực.
    *   **CatBoost**: Xử lý thông minh các đặc trưng văn bản và giảm nhiễu.
*   **Mô hình Meta (Ridge)**: Kết hợp kết quả từ 3 mô hình trên để đưa ra dự báo cuối cùng có độ ổn định cao nhất.

### Bước 5: Trực quan hóa và Đánh giá
*   Sử dụng biểu đồ hồi quy (`regplot`) để so sánh trực quan sai số giữa giá thực tế và giá dự báo.

---

## 3. Kết quả thu được
*   **Số lượng dữ liệu sạch**: Khoảng 960+ tin đăng hợp lệ sau khi lọc nhiễu.
*   **Chỉ số R2 Score**: Đạt mức tích cực (~0.4 - 0.5) đối với dữ liệu thô từ web.
*   **Sai số MAE**: Mô hình đã học được khung giá của từng Quận, trung bình sai số bám sát mặt bằng giá thị trường của các căn nhà có đặc điểm phổ thông.

---

## 4. Phân tích kết quả tìm ra

Qua thực nghiệm, dự án đã rút ra những quan sát quan trọng về thị trường bất động sản Việt Nam:
1.  **Sự biến động khu vực**: Giá nhà tại các quận trung tâm (Quận 1, Bình Thạnh, Cầu Giấy) có sự thay đổi đột ngột về giá ngay cả khi diện tích tương đương, do yếu tố kinh doanh mặt tiền.
2.  **Độ nhiễu của dữ liệu**: Thị trường có tỷ lệ "tin ảo" cao. Mô hình Ensemble đã chứng minh được khả năng **tự điều chỉnh** để bám sát giá trị trung bình thay vì bị kéo lệch bởi các tin đăng bị thổi giá.
3.  **Tầm quan trọng của Log-Transformation**: Việc áp dụng hàm Log cho biến Giá đã giúp mô hình hội tụ tốt hơn, xử lý được cả những căn nhà giá rẻ (vài tỷ) lẫn những căn biệt thự (hàng chục tỷ) trong cùng một hệ thống.

---

## 5. Kết luận tổng thể

Dự án đã thành công trong việc xây dựng một quy trình khép kín từ khâu **thu thập dữ liệu thô** đến việc **phát hành mô hình dự báo**. 

**Giá trị mang lại**: Hệ thống cung cấp một công cụ tham chiếu giá khách quan cho người dùng, giúp nhận diện nhanh các cơ hội đầu tư hoặc tránh các tin đăng giá quá cao so với thực tế khu vực.

**Hướng phát triển**: Trong tương lai, việc bổ sung thêm các biến về "Độ rộng ngõ hẻm" và "Tính pháp lý (Sổ đỏ/Sổ hồng)" thông qua các kỹ thuật trích xuất văn bản nâng cao (NLP) sẽ giúp đẩy độ chính xác của mô hình vượt ngưỡng 0.8.
