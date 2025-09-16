import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from utils import clean_text
import os

# === Load dataset ===
df = pd.read_csv("data/train.csv")  # Kaggle Fake News dataset
df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')
df['content'] = df['content'].apply(clean_text)

X = df['content']
y = df['label']

# === Feature extraction ===
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(X)

# === Train/Test split ===
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# === Train model ===
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Save model and vectorizer ===
# === Save model and vectorizer ===
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(tfidf, "models/tfidf.pkl")
print("✅ Model and vectorizer saved in 'models/' folder")

