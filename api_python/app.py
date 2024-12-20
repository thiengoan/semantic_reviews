import csv
from flask import Flask, request, jsonify
from flask_cors import CORS
from modules.selenium import get_reviews_with_selenium # Import the function from the modules package
from modules.predict import predict_bi_lstm, predict_naive_bayes, predict_phobert, predict_gpt
from modules.preprocess import preprocess_text

app = Flask(__name__)

CORS(app)

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
        if model_type == "naive_bayes":
            return predict_naive_bayes(reviews), None
        elif model_type == "lstm":
            return predict_bi_lstm(reviews), None
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
