from flask import Flask, request, jsonify, render_template, session
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers import BertTokenizer, BertForSequenceClassification, RobertaForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import joblib
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# Giả sử bạn đã huấn luyện một mô hình và lưu vào file
model = joblib.load("./models/multinomial_nb_model.joblib")
vectorizer = joblib.load("./models/count_vectorizer.joblib")

@app.route('/')
def index():
    return render_template('index.html')

def get_reviews_with_selenium(url):
    options = webdriver.ChromeOptions()
    service = webdriver.ChromeService()
    driver = webdriver.Chrome(service=service, options=options)
    try:
        # Mở trang sản phẩm Tiki
        driver.get(url)
        time.sleep(3)  
        # Get the total height of the webpage
        total_height = driver.execute_script("return document.body.scrollHeight")
        
        # Scroll to 50% of the page
        scroll_height = total_height * 0.5
        driver.execute_script(f"window.scrollTo(0, {scroll_height});")
        time.sleep(2)
        reviews = []
        review_elements = driver.find_elements(By.CLASS_NAME, "review-comment__content")
        for review in review_elements:
            reviews.append(review.text.strip())  # Lấy nội dung đánh giá

        time.sleep(2)
        # Kiểm tra phân trang, nếu có "Next" mới tìm thẻ <li> tiếp theo
        # while True:
        #     try:
        #         # Chờ và tìm thẻ <a> có class "btn active" trong <li> - thẻ này chỉ thị trang hiện tại
        #         active_page_a = WebDriverWait(driver, 10).until(
        #             EC.presence_of_element_located(
        #             (By.XPATH, "//li/a[contains(@class, 'btn') and contains(@class, 'active')]")
        #             )
        #         )
        #         # Tìm thẻ "Next" trong trang tiếp theo, ngay sau thẻ "active" (thường là nút phân trang)
        #         next_button = active_page_a.find_element(By.XPATH, "following::li/a")

        #         # Kiểm tra xem nút "Next" có thể nhấp được không (nút không bị vô hiệu hóa)
        #         if next_button.is_enabled():
        #             # Nếu nút "Next" có thể nhấp, thực hiện nhấp vào nó để chuyển sang trang tiếp theo
        #             next_button.click()

        #             # Chờ trang tải lại sau khi nhấp vào nút "Next"
        #             WebDriverWait(driver, 10).until(EC.staleness_of(next_button))  # Đảm bảo trang đã tải xong

        #             # Lấy tất cả các đánh giá trên trang hiện tại
        #             review_elements = driver.find_elements(By.CLASS_NAME, "review-comment__content")
        #             for review in review_elements:
        #                 reviews.append(review.text.strip())  # Thêm đánh giá vào danh sách
        #         else:
        #             # Nếu nút "Next" không thể nhấp, nghĩa là không còn trang tiếp theo
        #             break

        #     except Exception as e:
        #     # Nếu có lỗi hoặc không tìm thấy trang tiếp theo, kết thúc vòng lặp
        #         print("Error:", e)
        #         break

        # Kiểm tra nếu không có đánh giá
        if not reviews:
            return {"message": "No reviews found"}
        
        return reviews

    finally:
        driver.quit() 

# Hàm dự đoán với mô hình Logistic Regression
def predict_lr(text):
    lr_model = joblib.load('logistic_regression_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    text_vectorized = vectorizer.transform([text])
    prediction = lr_model.predict(text_vectorized)[0]
    confidence = float(lr_model.predict_proba(text_vectorized).max())
    return {"prediction": int(prediction), "confidence": confidence}

# Hàm dự đoán với mô hình SVM
def predict_svm(text):
    svm_model = joblib.load('svm_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    text_vectorized = vectorizer.transform([text])
    prediction = svm_model.predict(text_vectorized)[0]
    confidence = float(svm_model.predict_proba(text_vectorized).max())
    return {"prediction": int(prediction), "confidence": confidence}

# Hàm dự đoán với mô hình BERT
def predict_bert(reviews):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    inputs = bert_tokenizer(reviews, return_tensors="pt", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1).max().item()
    return {"prediction": int(prediction), "confidence": confidence}
  
# Hàm dự đoán với PhoBERT
def predict_phobert(reviews):

    model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)
    
    predictions = []

    for review in reviews:
        input_ids = torch.tensor([tokenizer.encode(review)])
        
        with torch.no_grad():
            out = model(input_ids)
            results = out.logits.softmax(dim=-1).tolist()[0]
            max_prob = max(results)
            for i, j in enumerate(results):
                if j == max_prob:
                    predictions.append({"review": review, "prediction": i, "confidence": j })
                    break

    return jsonify({ "prediction": predictions })

# Hàm get_reviews
@app.route('/get_reviews', methods=['GET'])
def get_reviews():
    try:
        # Nhận dữ liệu từ request
        data = request.get_json()
        url = data.get("url", "")

        if not url or "tiki.vn" not in url:
            return jsonify({"error": "Invalid or missing Tiki URL"}), 400

        # Gọi hàm xử lý với Selenium
        result = get_reviews_with_selenium(url)

        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/predict-pho', methods=['GET'])
def predict_pho():
    data = request.get_json()
    texts = data.get("texts", [])
    
    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)
    
    predictions = []
    for text in texts:
        input_ids = torch.tensor([tokenizer.encode(text)])
        
        with torch.no_grad():
            out = model(input_ids)
            results = out.logits.softmax(dim=-1).tolist()[0]
            max_prob = max(results)
            for i, j in enumerate(results):
                if j == max_prob:
                    predictions.append({"text": text, "prediction": 'Positive' if i > 0 else 'Negative', "confidence": j })
                    break

    return jsonify({"predictions": predictions})

# Hàm dự đoán với mô hình đã chọn
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Nhận dữ liệu từ request
        data = request.get_json()

        # Lấy URL từ request
        url = data.get("url", "")

        # Mặc định là "pho_bert"
        model_type = data.get("model", "pho_bert")  

        if not url or "tiki.vn" not in url:
            return jsonify({"error": "Invalid or missing Tiki URL"}), 400

        # Gọi hàm để lấy danh sách đánh giá từ Tiki
        reviews = get_reviews_with_selenium(url)

        if reviews is None:
            return jsonify({"error": "No reviews"}), 400

        # Dự đoán với mô hình đã chọn
        if model_type == "lr":
            return predict_lr(reviews), None
        elif model_type == "svm":
            return predict_svm(reviews), None
        elif model_type == "bert":
            return predict_bert(reviews), None
        elif model_type == "pho_bert":
            return predict_phobert(reviews), None
        else:
            return None, "Invalid model type"
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
