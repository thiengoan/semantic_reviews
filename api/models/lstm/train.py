import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

def load_data(file_path):
    df = pd.read_csv(file_path)
    # Convert 'text' column to string type
    df['text'] = df['text'].astype(str)
    return df['text'].values, df['label'].values

def preprocess_data(texts, labels, max_words=90000, max_len=150):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len)
    y = np.array(labels)
    return X, y, tokenizer

def create_model(max_words, max_len, embedding_dim=100):
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_len))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train, y_train, X_val, y_val, model, epochs=10, batch_size=64):
    checkpoint = ModelCheckpoint('models/best_model.keras', save_best_only=True, monitor='val_accuracy')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[checkpoint])
    return history

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred_classes)
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    class_report = classification_report(y_test, y_pred_classes)
    
    return accuracy, conf_matrix, class_report

def export_model(model, tokenizer, file_path):
    model.save(file_path)
    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_exported_model(model_path, tokenizer_path):
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

def predict(text, model, tokenizer, max_len=100):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, prediction

if __name__ == "__main__":
    # Load and preprocess data
    texts, labels = load_data('summary.csv')
    # X, y, tokenizer = preprocess_data(texts, labels)

    # Count labels 0 and 1
    label_counts = np.bincount(labels)
    print(f"Label 0 count: {label_counts[0]}")
    print(f"Label 1 count: {label_counts[1]}")

    # Filter out texts with label 1
    text_0 = texts[labels == 0]
    label_0 = labels[labels == 0]

    text_1 = texts[labels == 1]
    label_1 = labels[labels == 1]

    # Balance the dataset by taking the same number of samples from text_1 as in text_0
    text_1_balanced = text_1[:len(text_0)]
    label_1_balanced = label_1[:len(text_0)]

    # Combine the balanced datasets
    texts_balanced = np.concatenate([text_0, text_1_balanced])
    labels_balanced = np.concatenate([label_0, label_1_balanced])

    label_counts = np.bincount(labels_balanced)
    print(f"Label 0 count: {label_counts[0]}")
    print(f"Label 1 count: {label_counts[1]}")

    # Shuffle the combined dataset
    shuffled_indices = np.random.permutation(len(texts_balanced))
    texts_balanced = texts_balanced[shuffled_indices]
    labels_balanced = labels_balanced[shuffled_indices]

    # Preprocess the balanced data
    # X_balanced, y_balanced, tokenizer_balanced = preprocess_data(texts_balanced, labels_balanced)
    X, y, tokenizer = preprocess_data(texts_balanced, labels_balanced)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = create_model(max_words=90000, max_len=150)
    history = train_model(X_train, y_train, X_val, y_val, model)
    
    # Evaluate the model
    accuracy, conf_matrix, class_report = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    
    # Export the model
    export_model(model, tokenizer, 'models/sentiment_model.keras')