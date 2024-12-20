from transformers import RobertaForSequenceClassification, AutoTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from flask import jsonify
from modules.preprocess import preprocess_text
import torch
import joblib
import pickle
import openai

# Hàm dự đoán với mô hình Logistic Regression
def predict_bi_lstm(reviews):
    model = load_model('./models/lstm/model/sentiment_model.keras')
    with open('./models/lstm/model/tokenizer.pickle', 'rb') as handle:
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
    model = joblib.load('./models/naive_bayes/model/multinomial_nb_model.joblib')
    # load the model 
    vectorizer = joblib.load('./models/naive_bayes/model/count_vectorizer.joblib')

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
  