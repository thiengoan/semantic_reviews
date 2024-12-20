from underthesea import text_normalize, word_tokenize
import regex
import re

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

def remove_stopword(text):
    file = open('files/vietnamese-stopwords.txt', 'r', encoding = 'utf8')
    stopwords_list = file.read().split('\n')
    file.close()
    document = ' '.join('' if word in stopwords_list else word for word in text.split())
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()                                                                 # Convert to lowercase
    text = re.sub(r'<a\s+[^>]*>.*?</a>', '', text, flags=re.DOTALL | re.IGNORECASE)     # Remove hyperlinks
    text = re.sub(r'[^\w\s]', ' ', text)                                                # Remove punctuation
    text = re.sub(r'\d+', '', text)                                                     # Remove digits
    text = text_normalize(text)                                                         # Correct word syntax: baỏ -> bảo
    text = re.sub(r'\b\S\b', '', text, flags=re.UNICODE)                                # Keep only words with length greater than 1
    text = remove_duplicate_vowels(text)                                                # Removes duplicate vowels: Ngoooooon -> Ngon
    text = remove_stopword(text)                                                        # Remove stopwords      
    text = ' '.join(word.replace(' ', '_') for word in word_tokenize(text))             # Word segmentation
    return text