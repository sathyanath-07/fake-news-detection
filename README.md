# 📰 Fake News Detection

A machine learning project to classify news articles as **Real** or **Fake** using NLP techniques.

## 🚀 Features
- Preprocessing with stopword removal & cleaning
- TF-IDF feature extraction
- Logistic Regression (baseline)
- Extendable to Random Forest, SVM, LSTM, BERT
- Streamlit web app for live predictions

## 📂 Project Structure
```
fake-news-detection/
│── data/ (dataset CSVs)
│── models/ (saved ML models)
│── notebooks/ (optional Jupyter experiments)
│── app.py (Streamlit app)
│── train.py (train & save model)
│── utils.py (text preprocessing)
│── requirements.txt
│── README.md
```

## ⚙️ Setup

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
```

## 🏋️ Train the Model
```bash
python train.py
```

## 🌐 Run the Streamlit App
```bash
streamlit run app.py
```
# fake-news-detection
