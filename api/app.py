import csv
from flask import Flask, request, jsonify, render_template, session
import re
from underthesea import text_normalize, word_tokenize, sentiment
import pandas as pd
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
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from flask_cors import CORS
import openai

app = Flask(__name__)

CORS(app)

def remove_duplicate_vowels(sentence):
    vowels = "aáàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụ"
    result = ""
    prev_char = None

    for char in sentence:
        if char in vowels:
            if prev_char is None or prev_char != char:
                result += char
        else:
            result += char
        prev_char = char 

    return result

def preprocess_text(text):
    text = text.lower()                                                                 # Convert to lowercase
    text = re.sub(r'<a\s+[^>]*>.*?</a>', '', text, flags=re.DOTALL | re.IGNORECASE)     # Remove hyperlinks
    text = re.sub(r'[^\w\s]', ' ', text)                                                # Remove punctuation
    text = re.sub(r'\d+', '', text)                                                     # Remove digits
    text = text_normalize(text)                                                         # Correct word syntax: baỏ -> bảo
    text = re.sub(r'\b\S\b', '', text, flags=re.UNICODE)                                # Keep only words with length greater than 1
    text = ' '.join(word.replace(' ', '_') for word in word_tokenize(text))             # Word segmentation
    text = remove_duplicate_vowels(text)                                                # Removes duplicate vowels: Ngoooooon -> Ngon
    return text

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

# Hàm dự đoán với mô hình Logistic Regression
def predict_bi_lstm(reviews):
    model = load_model('./models/lstm/models/sentiment_model.keras')
    with open('./models/lstm/models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    predictions = []

    for review in reviews:
        sentence = preprocess_text(review)
        input = tokenizer.texts_to_sequences([sentence])
        padded_sequence = pad_sequences(input, maxlen=150)
        prediction = model.predict(padded_sequence, verbose=0)[0][0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        prediction if prediction > 0.5 else 1 - prediction
        predictions.append({"review": review, "prediction": sentiment, "confidence": float(prediction)})

    return jsonify({ "prediction": predictions })

# Multinomial Naive Bayes
def predict_naive_bayes(reviews):
    model = joblib.load('./models/naive_bayes/multinomial_nb_model.joblib')
    vectorizer = joblib.load('./models/naive_bayes/count_vectorizer.joblib')

    predictions = []
    for review in reviews:
        sentence = preprocess_text(review)
        input = vectorizer.transform([sentence])
        prediction = model.predict(input)[0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        confidence = float(model.predict_proba(input).max())
        predictions.append({"review": review, "prediction": sentiment, "confidence": confidence})

    return jsonify({ "prediction": predictions })

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
                    predictions.append({"review": review, "prediction": "Positive" if i > 0.5 else "Negative", "confidence": j })

    return jsonify({ "prediction": predictions })

# Hàm dự đoán với mô hình gpt
def predict_gpt(reviews):
    # Set up OpenAI API key
    openai.api_key = ""
    predictions = []

    for review in reviews:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use gpt-3.5-turbo or gpt-4
            messages=[ 
                {"role": "system", "content": "You are a helpful assistant that classifies text sentiment."},
                {"role": "user", "content": f"Classify the sentiment of this review as Positive, Negative, or Neutral: {review}"}
            ],
            temperature=0
        )
        if response['choices'][0]['message']['content'] == "Positive" or response['choices'][0]['message']['content'] == "Negative":
            predictions.append({"review": review, "prediction": response['choices'][0]['message']['content']})

    return jsonify({ "prediction": predictions })
  
# Hàm get_reviews
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

        return jsonify({ "data" : result })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Hàm dự đoán với mô hình đã chọn
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Nhận dữ liệu từ request
        data = request.get_json()

        # Mặc định là "pho_bert"
        model_type = data.get("model", "pho_bert")  

        # Lấy danh sách đánh giá
        reviews = data.get("reviews", []) 

        if isinstance(reviews, str):
            reviews = reviews.split(',')

        if reviews is None:
            return jsonify({"error": "No reviews"}), 400

        # Dự đoán với mô hình đã chọn
        if model_type == "lstm":
            return predict_bi_lstm(reviews), None
        elif model_type == "naive_bayes":
            return predict_naive_bayes(reviews), None
        elif model_type == "pho_bert":
            return predict_phobert(reviews), None
        elif model_type == "gpt":
            return predict_gpt(reviews), None
        else:
            return None, "Invalid model type"
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/add-dataset', methods=['POST'])
def addDataset():   
    try:
        data = request.get_json()
        review = data.get("review", "")
        prediction = data.get("prediction", "")

        if not review or not prediction:
            return jsonify({"error": "Invalid data"}), 400

        prediction = "1" if prediction == "Positive" else "0"
        
        with open('output.csv', 'a', newline='', encoding='utf-8') as csvFile:
            writer = csv.writer(csvFile)
            review = preprocess_text(review)
            writer.writerow([review, prediction])

        return jsonify({ "status": "success" }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
