# -*- coding: utf-8 -*-
# تطبيق VeriAI

import os
import sqlite3
import hashlib
import secrets
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from ai.cv_analyzer import analyze_cv

app = Flask(__name__)
# على Render: عيّن SECRET_KEY في Environment (مستقر بين إعادة التشغيل)
app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_hex(32)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database', 'veriai.db')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'database'), exist_ok=True)
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')


@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory(ASSETS_DIR, filename)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('يجب تسجيل الدخول أولاً', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


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
        conn.execute(
            'INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
            (name, email, hashed)
        )
        conn.commit()
        conn.close()
        flash('تم التسجيل بنجاح. يمكنك تسجيل الدخول الآن', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        conn = get_db()
        user = conn.execute(
            'SELECT id, name, email, password FROM users WHERE email = ?',
            (email,)
        ).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            session['user_email'] = user['email']
            flash(f'مرحباً {user["name"]}', 'success')
            return redirect(url_for('dashboard'))
        
        flash('البريد الإلكتروني أو كلمة المرور غير صحيحة', 'error')
        return redirect(url_for('login'))
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('تم تسجيل الخروج', 'info')
    return redirect(url_for('index'))


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        if not name or not email:
            flash('الاسم والبريد الإلكتروني مطلوبان', 'error')
            return redirect(url_for('profile'))
        
        conn = get_db()
        existing = conn.execute(
            'SELECT id FROM users WHERE email = ? AND id != ?',
            (email, session['user_id'])
        ).fetchone()
        
        if existing:
            conn.close()
            flash('البريد الإلكتروني مستخدم من قبل مستخدم آخر', 'error')
            return redirect(url_for('profile'))
        
        if password:
            if len(password) < 6:
                conn.close()
                flash('كلمة المرور يجب أن تكون 6 أحرف على الأقل', 'error')
                return redirect(url_for('profile'))
            hashed = generate_password_hash(password, method='pbkdf2:sha256')
            conn.execute(
                'UPDATE users SET name = ?, email = ?, password = ? WHERE id = ?',
                (name, email, hashed, session['user_id'])
            )
        else:
            conn.execute(
                'UPDATE users SET name = ?, email = ? WHERE id = ?',
                (name, email, session['user_id'])
            )
        
        conn.commit()
        conn.close()
        session['user_name'] = name
        session['user_email'] = email
        flash('تم تحديث البيانات بنجاح', 'success')
        return redirect(url_for('profile'))
    
    conn = get_db()
    user = conn.execute(
        'SELECT name, email FROM users WHERE id = ?',
        (session['user_id'],)
    ).fetchone()
    conn.close()
    
    return render_template('profile.html', user=dict(user))


@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db()
    cvs = conn.execute(
        '''SELECT id, filename, upload_date, result, confidence 
           FROM cvs WHERE user_id = ? ORDER BY upload_date DESC''',
        (session['user_id'],)
    ).fetchall()
    conn.close()
    
    return render_template('dashboard.html', cvs=[dict(r) for r in cvs])


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'cv_file' not in request.files:
            flash('لم يتم اختيار ملف', 'error')
            return redirect(url_for('upload'))
        
        file = request.files['cv_file']
        if file.filename == '':
            flash('لم يتم اختيار ملف', 'error')
            return redirect(url_for('upload'))
        
        if not allowed_file(file.filename):
            flash('الملف يجب أن يكون PDF أو DOCX', 'error')
            return redirect(url_for('upload'))
        
        filename = secure_filename(file.filename)
        unique_name = f"{secrets.token_hex(8)}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(filepath)
        
        try:
            analysis = analyze_cv(filepath)
            result = analysis.get('result', 'unknown')
            confidence = analysis.get('confidence', 0)
            
            os.remove(filepath)
            
            if result in ('ai', 'human'):
                result_text = 'مولد بواسطة AI' if result == 'ai' else 'بشري'
                flash(f'{result_text} {confidence}%', 'success')
                conn = get_db()
                conn.execute(
                    'INSERT INTO cvs (user_id, filename, result, confidence) VALUES (?, ?, ?, ?)',
                    (session['user_id'], filename, result, confidence)
                )
                conn.commit()
                conn.close()
            else:
                flash('لم يتم استخراج نص كافٍ من الملف', 'error')
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            flash(f'حدث خطأ أثناء التحليل: {str(e)}', 'error')
        
        return redirect(url_for('dashboard'))
    
    return render_template('upload.html')


@app.route('/about')
def about():
    return render_template('about.html')


init_db()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    debug = os.environ.get('FLASK_DEBUG', '').lower() in ('1', 'true', 'yes')
    app.run(debug=debug, host='0.0.0.0', port=port)
