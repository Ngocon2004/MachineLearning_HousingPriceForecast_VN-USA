import pandas as pd
import time
import os
import random
from bs4 import BeautifulSoup
import undetected_chromedriver as uc

def setup_driver():
    """Thiết lập driver với cấu hình an toàn hơn."""
    options = uc.ChromeOptions()
    # Không dùng headless để tránh bị Cloudflare phát hiện
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--window-size=1920,1080")
    
    driver = uc.Chrome(options=options)
    return driver

def scrape_page(driver, url):
    """Cào dữ liệu từ một trang cụ thể với cơ chế retry."""
    for attempt in range(3): # Thử lại tối đa 3 lần
        try:
            driver.get(url)
            time.sleep(random.uniform(5, 8)) # Chờ ngẫu nhiên từ 5-8 giây
            
            # Kiểm tra xem có nội dung không
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            items = soup.find_all('div', class_='re__card-info')
            
            if items:
                page_data = []
                for item in items:
                    try:
                        title = item.find('h3', class_='re__card-title').text.strip()
                        config = item.find('div', class_='re__card-config')
                        price = config.find('span', class_='re__card-config-price').text.strip() if config else "N/A"
                        area = config.find('span', class_='re__card-config-area').text.strip() if config else "N/A"
                        location = item.find('div', class_='re__card-location').text.strip() if item.find('div', class_='re__card-location') else "N/A"
                        description = item.find('div', class_='re__card-description').text.strip() if item.find('div', class_='re__card-description') else ""
                        
                        page_data.append({
                            'Title': title,
                            'Price': price,
                            'Area': area,
                            'Location': location,
                            'Description': description,
                            'URL': "https://batdongsan.com.vn" + item.find('a')['href'] if item.find('a') else ""
                        })
                    except Exception:
                        continue
                return page_data
            else:
                print(f"Lần thử {attempt+1}: Không tìm thấy thẻ tin đăng. Đang thử lại...")
                time.sleep(5)
        except Exception as e:
            print(f"Lỗi truy cập: {e}")
            time.sleep(5)
            
    return []

def main():
    base_url = "https://batdongsan.com.vn/ban-nha-rieng"
    num_pages = 50
    output_file = r"D:\Nam4\PhuongPhapHocMay\Lab1\batdongsan_data.csv"
    
    driver = setup_driver()
    all_data = []
    
    try:
        for page in range(1, num_pages + 1):
            print(f"Đang xử lý trang {page}/50...")
            url = f"{base_url}/p{page}" if page > 1 else base_url
            data = scrape_page(driver, url)
            
            if not data:
                print(f"Dừng lại ở trang {page} do không lấy được dữ liệu.")
                break
                
            all_data.extend(data)
            print(f"Đã lấy {len(data)} tin. Tổng: {len(all_data)}")
            
            # Nghỉ ngẫu nhiên để tránh bị phát hiện pattern
            time.sleep(random.uniform(2, 4))
            
    finally:
        driver.quit()
        
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nTHÀNH CÔNG! Đã lưu {len(df)} tin đăng vào: {output_file}")
    else:
        print("\nTHẤT BẠI: Không thu thập được dữ liệu nào.")

if __name__ == "__main__":
    main()
