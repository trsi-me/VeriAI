# -*- coding: utf-8 -*-
# تحليل السير الذاتية وتصنيفها

import os
import re
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_DIR, 'model', 'cv_classifier.pkl')
VECTORIZER_PATH = os.path.join(PROJECT_DIR, 'model', 'vectorizer.pkl')


def clean_text(text):
    if not text:
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u0600-\u06FF\u0000-\u007F]', ' ', text)
    return text.strip()


def extract_text_from_pdf(file_path):
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception:
        pass
    if len(text.strip()) < 20:
        text = ""
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        except Exception:
            pass
    return text


def extract_text_from_docx(file_path):
    try:
        from docx import Document
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""


def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ('.docx', '.doc'):
        return extract_text_from_docx(file_path)
    return ""


def analyze_cv(file_path):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return {
            'result': 'unknown',
            'confidence': 0,
            'text_preview': '',
            'error': 'النموذج غير مدرب. قم بتشغيل ai/train_model.py أولاً.'
        }
    
    text = extract_text(file_path)
    if not text or len(text.strip()) < 20:
        return {
            'result': 'unknown',
            'confidence': 0,
            'text_preview': text[:200] if text else '',
            'error': 'لم يتم استخراج نص كافٍ من الملف.'
        }
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    cleaned = clean_text(text)
    if len(cleaned) < 15:
        return {
            'result': 'unknown',
            'confidence': 0,
            'text_preview': text[:200] if text else '',
            'error': 'النص المستخرج قصير جداً أو غير واضح.'
        }
    
    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    idx = int(proba.argmax())
    result = str(classes[idx]).strip().lower()
    confidence = float(proba[idx])
    
    if result not in ('ai', 'human'):
        result = 'human' if idx == 0 else 'ai'
    
    return {
        'result': result,
        'confidence': round(confidence * 100, 1),
        'text_preview': text[:300].strip(),
        'error': None
    }
