import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('../../datasets/data.csv', encoding='utf-8')  # Replace with your dataset path

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data['text'].values.astype('U'), data['label'], test_size=0.2, random_state=42)

# Convert text data to feature vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Make predictions
X_test_vec = vectorizer.transform(X_test)
y_pred = nb_model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# Visualize confusion matrix
# plt.figure(figsize=(10,7))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()

# Save model
joblib.dump(nb_model, './models/multinomial_nb_model.joblib')
joblib.dump(vectorizer, './models/count_vectorizer.joblib')
