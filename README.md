# VeriAI - تقرير مفصل للمشروع

## 1. نظرة عامة

**VeriAI** مشروع ويب متكامل لتحليل السير الذاتية (CV) واكتشاف إن كانت مكتوبة بواسطة الذكاء الاصطناعي أو بشكل بشري. يعتمد على تقنيات **معالجة اللغة الطبيعية (NLP)** و**التعلم الآلي (Machine Learning)** للتمييز بين النصوص البشرية والمولدة آلياً.

---

## 2. المصطلحات التقنية

| المصطلح | الشرح |
|---------|-------|
| **Flask** | إطار عمل ويب خفيف مكتوب بلغة Python لبناء تطبيقات الويب |
| **SQLite** | قاعدة بيانات خفيفة تُخزّن في ملف واحد، لا تحتاج خادم منفصل |
| **NLP** | Natural Language Processing - معالجة اللغة الطبيعية لتحليل النصوص |
| **TF-IDF** | Term Frequency-Inverse Document Frequency - طريقة لتحويل النص إلى أرقام حسب أهمية الكلمات |
| **Logistic Regression** | خوارزمية تصنيف إحصائية تتنبأ باحتمالية انتماء النص لفئة معينة |
| **Vectorizer** | أداة تحول النص إلى مصفوفة رقمية (تمثيل رياضي للنص) |
| **joblib** | مكتبة لحفظ وتحميل النماذج المدربة في ملفات |
| **Session** | جلسة المستخدم - تخزين مؤقت لبيانات تسجيل الدخول في الخادم |
| **Flash Message** | رسالة مؤقتة تظهر للمستخدم بعد تنفيذ عملية (نجاح/خطأ) |
| **Decorator** | دالة تغلف دالة أخرى لتعديل سلوكها (مثل @login_required) |
| **Route** | مسار URL يرتبط بدالة معينة في التطبيق |
| **Template** | قالب HTML ديناميكي يستخدم متغيرات من Python |
| **PBKDF2** | خوارزمية تشفير كلمات المرور بشكل آمن |

---

## 3. هيكل المشروع والملفات

```
VeriAI/
├── app.py                 # نقطة الدخول الرئيسية - تطبيق Flask
├── requirements.txt       # قائمة المكتبات المطلوبة
├── README.md              # هذا التقرير
│
├── assets/                # الموارد الثابتة
│   ├── images/            # الصور والشعار (Logo.png)
│   └── fonts/             # خط IBM Plex Sans Arabic
│
├── templates/             # قوالب HTML
│   ├── base.html          # القالب الأساسي (الهيدر، الفوتر، القوائم)
│   ├── index.html         # الصفحة الرئيسية
│   ├── login.html         # تسجيل الدخول
│   ├── register.html      # التسجيل
│   ├── profile.html       # تعديل بيانات المستخدم
│   ├── dashboard.html     # لوحة التحكم (قائمة السير الذاتية)
│   ├── upload.html        # رفع ملف CV
│   └── about.html         # نبذة عن المشروع
│
├── static/                # الملفات الثابتة
│   ├── css/style.css      # تنسيقات الواجهة
│   └── js/main.js         # حركة الشعار
│
├── database/              # قاعدة البيانات
│   └── veriai.db          # ملف SQLite (يُنشأ تلقائياً)
│
├── model/                 # النموذج المدرب
│   ├── cv_classifier.pkl  # نموذج التصنيف (يُنشأ بعد التدريب)
│   └── vectorizer.pkl     # مُجهِّز TF-IDF (يُنشأ بعد التدريب)
│
├── ai/                    # وحدة الذكاء الاصطناعي
│   ├── train_model.py     # سكربت تدريب النموذج
│   ├── cv_analyzer.py     # تحليل السير الذاتية
│   └── cv_dataset.csv     # بيانات التدريب (نصوص + تسميات)
│
└── uploads/               # مجلد مؤقت لملفات الرفع (تُحذف بعد المعالجة)
```

---

## 4. شرح الأكواد المهمة بالتفصيل

### 4.1 ملف app.py - التطبيق الرئيسي

#### الاستيرادات والإعدادات (السطور 1-28)

```python
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from ai.cv_analyzer import analyze_cv
```
- **Flask**: إنشاء التطبيق، المسارات، القوالب
- **request**: قراءة بيانات النموذج والملفات المرسلة
- **session**: تخزين معرف المستخدم بعد تسجيل الدخول
- **flash**: رسائل مؤقتة (نجاح/خطأ)
- **secure_filename**: تأمين اسم الملف من الأحرف الخطرة
- **generate_password_hash / check_password_hash**: تشفير ومقارنة كلمات المرور
- **analyze_cv**: الدالة التي تحلل ملف CV وتعيد النتيجة

```python
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database', 'veriai.db')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 ميجابايت
```
- **secret_key**: مفتاح عشوائي لتوقيع الجلسات وحمايتها
- **DB_PATH**: مسار ملف قاعدة البيانات SQLite
- **UPLOAD_FOLDER**: مجلد حفظ الملفات المرفوعة مؤقتاً
- **ALLOWED_EXTENSIONS**: الصيغ المسموح بها فقط
- **MAX_CONTENT_LENGTH**: حد حجم الملف المرفوع

---

#### دالة get_db (السطور 36-39)

```python
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
```
- تفتح اتصالاً بملف `veriai.db`
- **row_factory = sqlite3.Row**: يجعل كل صف يُعاد كقاموس (يمكن الوصول بـ `row['name']`)
- يجب استدعاء `conn.close()` بعد الانتهاء

---

#### دالة init_db (السطور 42-61)

```python
def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS cvs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            result TEXT,
            confidence REAL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()
```
- **CREATE TABLE IF NOT EXISTS**: ينشئ الجدول فقط إن لم يكن موجوداً
- **users**: id تلقائي، name، email فريد، password مشفر
- **cvs**: ربط بـ user_id، اسم الملف، تاريخ الرفع، النتيجة، نسبة الثقة
- **FOREIGN KEY**: يربط cvs بالمستخدم في جدول users

---

#### ديكوراتور login_required (السطور 65-72)

```python
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('يجب تسجيل الدخول أولاً', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated
```
- يلف أي دالة تُكتب فوقها `@login_required`
- إذا لم يكن المستخدم مسجلاً (`user_id` غير موجود في session)، يُعاد توجيهه لصفحة تسجيل الدخول
- **@wraps(f)**: يحافظ على اسم الدالة الأصلية (مهم للتصحيح)

---

#### مسار التسجيل register (السطور 85-115)

```python
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        if not name or not email or not password:
            flash('جميع الحقول مطلوبة', 'error')
            return redirect(url_for('register'))
        
        if len(password) < 6:
            flash('كلمة المرور يجب أن تكون 6 أحرف على الأقل', 'error')
            return redirect(url_for('register'))
        
        conn = get_db()
        if conn.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone():
            conn.close()
            flash('البريد الإلكتروني مسجل مسبقاً', 'error')
            return redirect(url_for('register'))
        
        hashed = generate_password_hash(password, method='pbkdf2:sha256')
        conn.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                     (name, email, hashed))
        conn.commit()
        conn.close()
        flash('تم التسجيل بنجاح. يمكنك تسجيل الدخول الآن', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')
```
- **request.method == 'POST'**: الطلب من نموذج الإرسال
- **request.form.get()**: قراءة القيم من الحقول
- **? في SQL**: مكان آمن للقيم (يمنع حقن SQL)
- **generate_password_hash**: يحول كلمة المرور إلى هاش لا يمكن عكسه
- **render_template**: يعرض صفحة HTML

---

#### مسار تسجيل الدخول login (السطور 118-141)

```python
if user and check_password_hash(user['password'], password):
    session['user_id'] = user['id']
    session['user_name'] = user['name']
    session['user_email'] = user['email']
    flash(f'مرحباً {user["name"]}', 'success')
    return redirect(url_for('dashboard'))
```
- **check_password_hash**: يقارن كلمة المرور المدخلة بالهاش المخزن
- **session['user_id']**: يحفظ معرف المستخدم في الجلسة لاستخدامه في الصفحات المحمية
- بعد النجاح يتم التوجيه إلى لوحة التحكم

---

#### مسار رفع CV upload (السطور 223-271)

```python
filename = secure_filename(file.filename)
unique_name = f"{secrets.token_hex(8)}_{filename}"
filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
file.save(filepath)
```
- **secure_filename**: يزيل الأحرف الخطرة من اسم الملف
- **secrets.token_hex(8)**: يولد 8 بايت عشوائية كبادئة لتفادي تكرار الأسماء
- **file.save()**: يحفظ الملف على القرص

```python
analysis = analyze_cv(filepath)
result = analysis.get('result', 'unknown')
confidence = analysis.get('confidence', 0)
os.remove(filepath)
```
- **analyze_cv()**: يستخرج النص، ينظفه، ويصنفه
- **os.remove()**: يحذف الملف المؤقت بعد التحليل

```python
if result in ('ai', 'human'):
    conn.execute('INSERT INTO cvs (user_id, filename, result, confidence) VALUES (?, ?, ?, ?)',
                 (session['user_id'], filename, result, confidence))
```
- يتم الحفظ في قاعدة البيانات فقط عند نجاح التحليل (ai أو human)

---

### 4.2 ملف ai/cv_analyzer.py - وحدة التحليل

#### دالة clean_text (السطور 14-20)

```python
def clean_text(text):
    if not text:
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u0600-\u06FF\u0000-\u007F]', ' ', text)
    return text.strip()
```
- **\s+**: مسافات أو أسطر جديدة متتالية → تُستبدل بمسافة واحدة
- **\w**: حروف وأرقام
- **\u0600-\u06FF**: نطاق الأحرف العربية
- **\u0000-\u007F**: نطاق ASCII (الإنجليزية)
- أي حرف خارج هذه المجموعات يُستبدل بمسافة

---

#### دالة extract_text_from_pdf (السطور 23-44)

```python
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
```
- **pdfplumber**: يُستخدم أولاً لأنه أدق في استخراج النص
- إذا كان النص أقل من 20 حرفاً، يُجرّب **PyPDF2** كبديل
- **'rb'**: فتح الملف بوضع القراءة الثنائية (للملفات الثنائية مثل PDF)

---

#### دالة extract_text_from_docx (السطور 47-52)

```python
def extract_text_from_docx(file_path):
    try:
        from docx import Document
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""
```
- **Document()**: يفتح ملف Word
- **doc.paragraphs**: قائمة بكل الفقرات
- **p.text**: نص كل فقرة
- النتيجة: نص واحد مفصول بأسطر جديدة

---

#### دالة analyze_cv (السطور 65-110)

```python
X = vectorizer.transform([cleaned])
proba = model.predict_proba(X)[0]
classes = model.classes_
idx = int(proba.argmax())
result = str(classes[idx]).strip().lower()
confidence = float(proba[idx])
```
- **vectorizer.transform()**: يحول النص إلى مصفوفة رقمية بنفس البنية المستخدمة في التدريب
- **predict_proba()**: يعيد احتمالات الفئات (مثلاً [0.3, 0.7] لـ human و ai)
- **proba.argmax()**: index الاحتمال الأعلى
- **classes[idx]**: اسم الفئة (ai أو human)
- **confidence**: الاحتمال كنسبة مئوية (0–100)

---

### 4.3 ملف ai/train_model.py - تدريب النموذج

#### تحميل وتنظيف البيانات (السطور 34-42)

```python
df = pd.read_csv(DATASET_PATH, encoding='utf-8')
df = df.dropna()
df['text'] = df['text'].apply(clean_text)
df = df[df['text'].str.len() > 10]
X = df['text']
y = df['label']
```
- **read_csv**: يقرأ ملف CSV
- **dropna()**: يحذف الصفوف التي تحتوي قيماً ناقصة
- **apply(clean_text)**: ينظف كل نص
- **str.len() > 10**: يحذف النصوص القصيرة جداً
- **X**: النصوص، **y**: التسميات (ai أو human)

---

#### TF-IDF Vectorizer (السطور 45-53)

```python
vectorizer = TfidfVectorizer(
    max_features=5000,    # أقصى 5000 ميزة (كلمة أو عبارة)
    ngram_range=(1, 2),   # كلمات مفردة + عبارات من كلمتين
    min_df=1,             # تظهر في وثيقة واحدة على الأقل
    max_df=0.95,          # لا تظهر في أكثر من 95% من الوثائق
    sublinear_tf=True     # استخدام log لتردد الكلمة
)
X_tfidf = vectorizer.fit_transform(X)
```
- **fit_transform**: يتعلم المفردات من النصوص ويحولها إلى مصفوفة رقمية
- كل صف = وثيقة، كل عمود = ميزة (كلمة أو عبارة)
- القيم = أوزان TF-IDF (أهمية الكلمة في الوثيقة)

---

#### تقسيم البيانات والتدريب (السطور 55-66)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
model = LogisticRegression(max_iter=1000, random_state=42, C=1.0, solver='lbfgs')
model.fit(X_train, y_train)
```
- **test_size=0.2**: 20% للاختبار، 80% للتدريب
- **stratify=y**: يحافظ على نسبة الفئات في كل مجموعة
- **LogisticRegression**: خوارزمية تصنيف خطية
- **fit()**: يدرّب النموذج على بيانات التدريب

---

#### حفظ النموذج (السطور 74-75)

```python
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
```
- **joblib.dump()**: يحفظ الكائن في ملف (النموذج والمُجهِّز)
- يُحمّل لاحقاً بـ `joblib.load()` في cv_analyzer

---

### 4.4 قاعدة البيانات (SQLite)

**جدول users:**
| العمود | النوع | الوصف |
|--------|-------|-------|
| id | INTEGER | معرف فريد (تلقائي) |
| name | TEXT | اسم المستخدم |
| email | TEXT | البريد (فريد) |
| password | TEXT | كلمة المرور مشفرة |
| created_at | TIMESTAMP | تاريخ الإنشاء |

**جدول cvs:**
| العمود | النوع | الوصف |
|--------|-------|-------|
| id | INTEGER | معرف فريد |
| user_id | INTEGER | ربط بالمستخدم |
| filename | TEXT | اسم الملف |
| upload_date | TIMESTAMP | تاريخ الرفع |
| result | TEXT | النتيجة (ai أو human) |
| confidence | REAL | نسبة الثقة (0-100) |

---

## 5. طريقة عمل المشروع (التدفق)

```
المستخدم يفتح الموقع
        ↓
الصفحة الرئيسية → التسجيل أو تسجيل الدخول
        ↓
بعد تسجيل الدخول → لوحة التحكم
        ↓
رفع CV → اختيار ملف PDF أو DOCX
        ↓
الخادم: استخراج النص → تنظيف → تحميل النموذج
        ↓
النموذج: تحويل النص (TF-IDF) → التصنيف (Logistic Regression)
        ↓
النتيجة: بشري X% أو مولد بواسطة AI X%
        ↓
حفظ في قاعدة البيانات → عرض في لوحة التحكم
```

---

## 6. طريقة التشغيل للمبتدئين

### الخطوة 1: تثبيت Python

1. حمّل Python 3.10.6 من الموقع الرسمي
2. أثناء التثبيت، فعّل خيار **"Add Python to PATH"**
3. تحقق من التثبيت بفتح **Command Prompt** أو **PowerShell** وكتابة:
   ```
   python --version
   ```
   يجب أن يظهر: `Python 3.10.6`

### الخطوة 2: فتح مجلد المشروع

في الطرفية، انتقل لمجلد المشروع:
```
cd D:\VSCode\Projects\VeriAI
```

### الخطوة 3: تثبيت المكتبات

نفّذ الأمر:
```
pip install -r requirements.txt
```
انتظر حتى تكتمل عملية التثبيت.

### الخطوة 4: تدريب النموذج (مرة واحدة فقط)

قبل أول تشغيل، درّب النموذج:
```
python ai/train_model.py
```
ستظهر رسائل عن تحميل البيانات والتدريب، وفي النهاية يُنشأ مجلد `model/` بملفين.

### الخطوة 5: تشغيل التطبيق

نفّذ:
```
python app.py
```
ستظهر رسالة مثل: `Running on http://127.0.0.1:5000`

### الخطوة 6: فتح الموقع

افتح المتصفح واكتب في شريط العنوان:
```
http://127.0.0.1:5000
```

### الخطوة 7: التجربة

1. اضغط **التسجيل** وأنشئ حساباً
2. سجّل الدخول
3. ادخل إلى **رفع CV** واختر ملف PDF أو DOCX
4. انتظر التحليل وستظهر النتيجة مع النسبة

---

## 7. المكتبات المستخدمة

| المكتبة | الإصدار | الاستخدام |
|---------|---------|-----------|
| Flask | 3.0.0 | إطار الويب |
| Werkzeug | 3.0.1 | أمان كلمات المرور، التعامل مع الملفات |
| PyPDF2 | 3.0.1 | قراءة PDF |
| pdfplumber | 0.10.3 | قراءة PDF (بديل أدق) |
| python-docx | 1.1.0 | قراءة DOCX |
| scikit-learn | 1.3.2 | TF-IDF، Logistic Regression |
| numpy | 1.26.2 | عمليات رياضية |
| pandas | 2.1.3 | قراءة CSV ومعالجة البيانات |
| joblib | 1.3.2 | حفظ وتحميل النماذج |

---

## 8. ملاحظات مهمة

- **الملفات المطلوبة**: تأكد من وجود `assets/images/Logo.png` و `assets/fonts/IBMPlexSansArabic-Medium.ttf`
- **حجم الرفع**: الحد الأقصى 16 ميجابايت لكل ملف
- **الصيغ المدعومة**: PDF، DOCX، DOC فقط
- **قاعدة البيانات**: تُنشأ تلقائياً عند أول تشغيل في `database/veriai.db`
- **إيقاف التطبيق**: اضغط `Ctrl+C` في الطرفية

---

## 9. ربط النظام بأجهزة خارجية

لتشغيل الخادم على الشبكة المحلية (للوصول من أجهزة أخرى):

في `app.py` غيّر السطر الأخير إلى:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```
ثم استخدم IP جهازك (مثل `192.168.1.5:5000`) للوصول من أجهزة أخرى على نفس الشبكة.
