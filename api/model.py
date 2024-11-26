import re
import joblib
import sys
from underthesea import text_normalize, word_tokenize

def remove_duplicate_vowels(sentence):
    # Combined vowel set
    vowels = "aáàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụ"
    result = ""
    prev_char = None

    for char in sentence:
        if char in vowels:
            # Keep the vowel only if it's different from the previous character
            if prev_char is None or prev_char != char:
                result += char
        else:
            result += char
        prev_char = char  # Update previous character

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

# Load the model and vectorizer
nb_model_loaded = joblib.load('./models/multinomial_nb_model.joblib')
vectorizer_loaded = joblib.load('./models/count_vectorizer.joblib')


def predict_sentiment(sentence):

    # Preprocess the sentence
    preprocessed_sentence = preprocess_text(sentence)

    # Transform the sentence using the loaded vectorizer
    sentence_vec = vectorizer_loaded.transform([preprocessed_sentence])

    # Predict the sentiment
    sentiment = nb_model_loaded.predict(sentence_vec)[0]
    probabilities = nb_model_loaded.predict_proba(sentence_vec)[0]

    # Calculate confidence percentage
    confidence = probabilities[sentiment]

    return sentiment, confidence


# Example usage
# example_sentence = "Cuộc sống có những lúc thật khó khăn và đầy thử thách, nhưng tôi luôn cảm thấy hạnh phúc và biết ơn vì những gì mình có."
# prediction, confidence = predict_sentiment(example_sentence)
# print(f"Sentiment of the sentence: {example_sentence}")
# print(f"=> {'Positive' if prediction == True else 'Negative'}")
# print(f"Confidence: {confidence:.2f}%\n")

# Main function to execute the script
# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python main.py '<sentence>'")
#         sys.exit(1)

#     input_sentence = sys.argv[1]
#     prediction, confidence = predict_sentiment(input_sentence)

#     # Print 1 for positive sentiment, 0 for negative sentiment
#     print(f"Sentiment: {'Positive' if prediction == True else 'Negative'}")
#     print(f"Confidence: {confidence:.2f}")
