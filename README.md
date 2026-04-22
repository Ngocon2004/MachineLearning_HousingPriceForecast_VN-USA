# Dự Án Dự Báo Giá Bất Động Sản: Việt Nam & Hoa Kỳ (Kaggle)
**Machine Learning Housing Price Forecast Project**

Dự án này là một nghiên cứu toàn diện về việc áp dụng các thuật toán Học máy (Machine Learning) tiên tiến để dự báo giá bất động sản tại hai thị trường khác nhau:
1.  **Thị trường Việt Nam**: Dữ liệu thực tế được thu thập trực tiếp từ Web.
2.  **Thị trường Hoa Kỳ (Ames, Iowa)**: Dữ liệu từ cuộc thi nổi tiếng trên Kaggle.

---

## 🚀 Tính Năng Chính
*   **Web Scraping**: Tự động thu thập tin đăng bất động sản từ Batdongsan.com.vn.
*   **Feature Engineering**: Xử lý dữ liệu thô, trích xuất vị trí, tính toán tuổi thọ nhà và các đặc trưng địa lý.
*   **Ensemble Learning**: Kết hợp các thuật toán mạnh mẽ nhất hiện nay: **XGBoost**, **LightGBM**, **CatBoost** và **Stacking Ensemble**.
*   **Optimization**: Chiến lược tối ưu hóa để đạt thứ hạng cao trên bảng xếp hạng Kaggle toàn cầu.

---

## 📂 Cấu Trúc Thư Mục
Dự án được chia làm 2 phần chính:

### 1. Phân tích Bất động sản Việt Nam (`/BDSVietNam`)
Tập trung vào dữ liệu thực tế tại TP.HCM, Hà Nội và các tỉnh thành khác.
*   `scraper_batdongsan.py`: Công cụ cào dữ liệu sử dụng Selenium.
*   `batdongsan_data.csv`: Bộ dữ liệu 1.000 tin đăng đã thu thập.
*   `BDSVietNam.ipynb`: **Notebook phân tích chính**. Thực hiện làm sạch dữ liệu, trích xuất Quận/Huyện và huấn luyện mô hình Stacking Ensemble.
*   `BDSVietNam_Report.md`: Báo cáo chi tiết quy trình thực hiện cho thị trường VN.

### 2. Dự báo Giá nhà Hoa Kỳ - Kaggle (`/Kaggle`)
Giải quyết bài toán Ames Housing Dataset (Mỹ) với mục tiêu tối ưu hóa sai số RMSE.
*   `2212363_DGHaHai.ipynb`: **Notebook giải pháp tối ưu**. Áp dụng kỹ thuật Feature Engineering bậc cao và Siêu pha trộn mô hình (Blending) để đạt thứ hạng Top đầu (Mục tiêu Top 1-5%).
*   `Boosting.ipynb`: Notebook nghiên cứu so sánh chuyên sâu 4 thuật toán Boosting (GBM, XGB, LGBM, CatBoost).
*   `Boosting_Report.md`: Báo cáo so sánh hiệu năng giữa các thuật toán.

---

## 🛠️ Hướng Dẫn Sử Dụng

### Cài đặt môi trường
Bạn cần cài đặt các thư viện Python sau để chạy dự án:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost selenium undetected-chromedriver scipy
```

### Cách vận hành phần Việt Nam
1.  (Tùy chọn) Chạy `python scraper_batdongsan.py` nếu bạn muốn thu thập dữ liệu mới nhất.
2.  Mở `BDSVietNam.ipynb` trong Jupyter Notebook hoặc VS Code.
3.  Nhấn **Run All** để xem quá trình xử lý dữ liệu thực tế và dự báo giá.

### Cách vận hành phần Kaggle (Mỹ)
1.  Đảm bảo file `train.csv` và `test.csv` nằm trong thư mục `/Kaggle`.
2.  Mở `2212363_DGHaHai.ipynb` và chạy toàn bộ các cell để tạo file nộp bài `submission_Grandmaster_Final.csv`.
3.  Chạy `Boosting.ipynb` nếu bạn muốn xem bản so sánh chi tiết giữa các thuật toán Boosting.

---

## 📊 Kết Quả Đạt Được
*   **Thị trường Mỹ**: Giải pháp đạt được độ khớp (R2 Score) cực cao và sai số RMSE tối thiểu, hướng tới Top 1-5% Leaderboard.
*   **Thị trường Việt Nam**: Xây dựng thành công quy trình xử lý dữ liệu thô từ web, cung cấp mức giá tham khảo tin cậy cho người dùng thực tế tại Việt Nam.

---

## 👤 Thông tin tác giả
*   **Họ tên**: Đồng Gur Hà Hải
*   **Mã sinh viên**: 2212363
*   **Lớp**: CTK46B-PM
*   **Học phần**: Phương pháp Học máy
