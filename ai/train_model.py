# -*- coding: utf-8 -*-
# تدريب نموذج تصنيف السير الذاتية

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, 'cv_dataset.csv')
MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'cv_classifier.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u0600-\u06FF\u0000-\u007F]', ' ', text)
    return text.strip()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print('تحميل البيانات...')
    df = pd.read_csv(DATASET_PATH, encoding='utf-8')
    
    print('تنظيف البيانات...')
    df = df.dropna()
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 10]
    
    X = df['text']
    y = df['label']
    
    print('تحويل النصوص إلى تمثيلات رقمية (TF-IDF)...')
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    X_tfidf = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print('تدريب نموذج Logistic Regression...')
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nدقة النموذج: {accuracy:.4f}')
    print('\nتقرير التصنيف:')
    print(classification_report(y_test, y_pred))
    
    print('\nحفظ النموذج والمُجهِّز...')
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f'تم حفظ النموذج في: {MODEL_PATH}')
    print(f'تم حفظ المُجهِّز في: {VECTORIZER_PATH}')


if __name__ == '__main__':
    main()
