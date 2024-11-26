from flask import Flask, request, jsonify, render_template
import joblib
from transformers import BertTokenizer, BertForSequenceClassification, RobertaForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
import time

app = Flask(__name__)

# Giả sử bạn đã huấn luyện một mô hình và lưu vào file
# model = joblib.load("./models/multinomial_nb_model.joblib")
# vectorizer = joblib.load("./models/count_vectorizer.joblib")

@app.route('/')

def index():
    return render_template('index.html')

def get_reviews_with_selenium(url):
    # options = webdriver.ChromeOptions()
    # options.add_argument('--disable-web-security')
    # options.add_argument('--allow-running-insecure-content')
    # driver = webdriver.Chrome(options=options)
    # Cấu hình Firefox để tắt SSL verification
    options = Options()
    options.set_preference("security.ssl.enable_ocsp_stapling", False)
    options.set_preference("security.ssl.enable_ocsp_must_staple", False)
    options.set_preference("browser.ssl_override_behavior", 2)

    driver = webdriver.Firefox(options=options)
    try:
        # Mở trang sản phẩm Tiki
        driver.get(url)
        time.sleep(7)  # Đợi trang tải xong (có thể tối ưu hơn bằng WebDriverWait)
        # Lấy tất cả đánh giá từ phần tử liên quan
        # element = WebDriverWait(driver, 10).until(
        #     EC.element_to_be_clickable((By.CLASS_NAME, 'review-comment__content'))
        # )
        # element.click()
        reviews = []
        review_elements = driver.find_elements(By.CLASS_NAME, "review-comment__content")
        print(review_elements)
        for review in review_elements:
            reviews.append(review.text.strip())  # Lấy nội dung đánh giá

        # Kiểm tra nếu không có đánh giá
        if not reviews:
            return {"message": "No reviews found"}
        
        return {"url": url, "reviews": reviews}
    
    finally:
        driver.quit()  # Đảm bảo driver được tắt để giải phóng tài nguyên

@app.route('/get_reviews', methods=['POST'])
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
def predict_bert(text):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1).max().item()
    return {"prediction": int(prediction), "confidence": confidence}
  
# Hàm dự đoán với PhoBERT
def predict_phobert(texts):
    model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    # Đưa dữ liệu vào mô hình PhoBERT
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).tolist()  # Lấy các dự đoán cho từng văn bản
    confidence = torch.softmax(logits, dim=1).max(dim=1).values.tolist()  # Lấy độ tin cậy của dự đoán

    return [{"prediction": pred, "confidence": conf} for pred, conf in zip(predictions, confidence)]
      
@app.route('/predict', methods=['POST'])

def get_reviews_from_url(url):
    """Lấy danh sách đánh giá từ Tiki qua Selenium."""
    reviews_data = get_reviews_with_selenium(url)
    if "reviews" not in reviews_data:
        return None, reviews_data.get("message", "Unknown error")
    return reviews_data["reviews"], None

def select_model_prediction(reviews, model_type):
    """Chọn mô hình dự đoán theo loại mô hình yêu cầu."""
    if model_type == "lr":
        return predict_lr(reviews)
    elif model_type == "svm":
        return predict_svm(reviews)
    elif model_type == "bert":
        return predict_bert(reviews)
    elif model_type == "pho_bert":
        return predict_phobert(reviews)
    else:
        return None, "Invalid model type"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Nhận dữ liệu từ request
        data = request.get_json()
        url = data.get("url", "")
        model_type = data.get("model", "pho_bert")  # Mặc định là "pho_bert"

        # Kiểm tra tính hợp lệ của URL
        if not url or "tiki.vn" not in url:
            return jsonify({"error": "Invalid or missing Tiki URL"}), 400

        # Lấy danh sách đánh giá từ Tiki
        reviews, error = {
            "url": "https://tiki.vn/",
            "reviews": [
                "Sản phẩm rất tốt, giao hàng nhanh chóng.",
                "Chất lượng không như mong đợi.",
                "Đóng gói cẩn thận, sẽ mua lại lần sau."
            ]
        }

        if reviews is None:
            return jsonify({"error": error}), 400

        # # Dự đoán với mô hình đã chọn
        result, error = select_model_prediction(reviews, model_type)
        if result is None:
            return jsonify({"error": error}), 400
        
        # Trả về kết quả dự đoán
        return jsonify({"predictions": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
