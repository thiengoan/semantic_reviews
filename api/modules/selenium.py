from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def get_reviews_with_selenium(url):
    options = webdriver.ChromeOptions()
    service = webdriver.ChromeService()
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # Mở trang web
        driver.get(url)
        time.sleep(3)

        # Scroll để tải thêm nội dung
        total_height = driver.execute_script("return document.body.scrollHeight")
        driver.execute_script(f"window.scrollTo(0, {total_height * 0.5});")
        time.sleep(2)

        reviews = []

        # Biến đếm số vòng lặp
        loop_count = 0  
        # Lặp qua các trang phân trang
        while loop_count < 5:
            # Tăng số vòng lặp
            loop_count += 1
            # Tìm các đánh giá trên trang hiện tại
            review_elements = driver.find_elements(By.CLASS_NAME, "review-comment__content")

            for review in review_elements:
                text = review.text.strip()
                if text:  # Chỉ thêm nếu nội dung không rỗng
                    reviews.append(text)

            time.sleep(2)  # Tạm dừng để tránh bị phát hiện như bot

            try:
                # Tìm nút "Next" cố định
                next_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "//li/a[contains(@class, 'btn') and contains(@class, 'next')]")
                    )
                )
                
                # Nhấp vào nút của trang tiếp theo
                next_button.click()

                time.sleep(2)  # Tạm dừng để tránh bị phát hiện như bot

            except Exception as e:

                print("Không tìm thấy nút 'Trang tiếp theo' hoặc gặp lỗi:", e)
                break

        # Trả về danh sách đánh giá hoặc thông báo nếu không có 
        if not reviews:
            return {"message": "Không tìm thấy đánh giá nào"}
        return reviews
    finally:
        driver.quit()