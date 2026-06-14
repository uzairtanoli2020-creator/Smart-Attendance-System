import os
import cv2
import dlib
import numpy as np
import sqlite3
import pickle
from datetime import datetime, time, timedelta
import pytz
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import base64

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ENCODINGS_FILE = 'face_encodings.pkl'
DB_FILE = 'attendance.db'
PAKISTAN_TZ = pytz.timezone('Asia/Karachi')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize dlib face detector and recognizer
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Database initialization
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Admin table
    c.execute('''CREATE TABLE IF NOT EXISTS admins
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Students table with password
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  roll_number TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  encoding BLOB NOT NULL,
                  photo_path TEXT,
                  registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        # Teachers table
    c.execute('''CREATE TABLE IF NOT EXISTS teachers
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    # Subjects table
    c.execute('''CREATE TABLE IF NOT EXISTS subjects
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  code TEXT NOT NULL,
                  day_of_week TEXT NOT NULL,
                  start_time TEXT NOT NULL,
                  end_time TEXT NOT NULL,
                  attendance_window INTEGER DEFAULT 15,
                  teacher_id INTEGER,
                  created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (teacher_id) REFERENCES teachers(id))''')
    c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='subjects'")
    row = c.fetchone()
    if row and 'code TEXT UNIQUE' in row[0]:
        c.execute("ALTER TABLE subjects RENAME TO subjects_old")
        c.execute('''CREATE TABLE subjects
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT NOT NULL,
                      code TEXT NOT NULL,
                      day_of_week TEXT NOT NULL,
                      start_time TEXT NOT NULL,
                      end_time TEXT NOT NULL,
                      attendance_window INTEGER DEFAULT 15,
                      teacher_id INTEGER,
                      created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (teacher_id) REFERENCES teachers(id))''')
        c.execute("""INSERT INTO subjects (id, name, code, day_of_week, start_time, end_time, attendance_window, teacher_id, created_date)
                     SELECT id, name, code, day_of_week, start_time, end_time, attendance_window, teacher_id, created_date FROM subjects_old""")
        c.execute("DROP TABLE subjects_old")
    # Courses table (teacher creates courses separately)
    c.execute('''CREATE TABLE IF NOT EXISTS courses
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  code TEXT UNIQUE NOT NULL,
                  teacher_id INTEGER,
                  created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (teacher_id) REFERENCES teachers(id))''')

    # Enrollments table (which students are in which course)
    c.execute('''CREATE TABLE IF NOT EXISTS enrollments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  student_id INTEGER NOT NULL,
                  course_id INTEGER NOT NULL,
                  enrolled_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  UNIQUE(student_id, course_id),
                  FOREIGN KEY (student_id) REFERENCES students(id),
                  FOREIGN KEY (course_id) REFERENCES courses(id))''')
    # Location/WiFi settings
    c.execute('''CREATE TABLE IF NOT EXISTS locations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  wifi_ssid TEXT,
                  ip_range TEXT,
                  latitude REAL,
                  longitude REAL,
                  radius INTEGER DEFAULT 100,
                  is_active INTEGER DEFAULT 1)''')
    
    # Attendance table with subject
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  student_id INTEGER,
                  subject_id INTEGER,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  status TEXT DEFAULT 'Present',
                  location_info TEXT,
                  FOREIGN KEY (student_id) REFERENCES students(id),
                  FOREIGN KEY (subject_id) REFERENCES subjects(id))''')
    
    # Create default admin if not exists
    c.execute("SELECT * FROM admins WHERE username = ?", ('admin',))
    if not c.fetchone():
        hashed_password = generate_password_hash('admin123')
        c.execute("INSERT INTO admins (username, password) VALUES (?, ?)", ('admin', hashed_password))
    
    conn.commit()
    conn.close()

init_db()

def get_pakistan_time():
    """Get current time in Pakistan timezone"""
    return datetime.now(PAKISTAN_TZ)

def get_current_subject():
    """Get the current active subject based on Pakistan time"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    current_time = get_pakistan_time()
    day_name = current_time.strftime('%A')
    current_time_str = current_time.strftime('%H:%M')
    
    c.execute("""SELECT id, name, code, start_time, end_time, attendance_window 
                 FROM subjects 
                 WHERE day_of_week = ? 
                 AND time(?) BETWEEN time(start_time) 
                 AND time(end_time, '+' || attendance_window || ' minutes')
                 ORDER BY start_time""", 
              (day_name, current_time_str))
    
    subject = c.fetchone()
    conn.close()
    
    if subject:
        return {
            'id': subject[0],
            'name': subject[1],
            'code': subject[2],
            'start_time': subject[3],
            'end_time': subject[4],
            'attendance_window': subject[5]
        }
    return None

# Face encoding functions
def get_face_encoding(image):
    """Extract face encoding from image"""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector(rgb, 1)
    
    if len(faces) == 0:
        return None
    
    shape = predictor(rgb, faces[0])
    encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))
    return encoding

def compare_faces(encoding1, encoding2, tolerance=0.6):
    """Compare two face encodings"""
    distance = np.linalg.norm(encoding1 - encoding2)
    return distance < tolerance

# Authentication decorators
def login_required(f):
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

def admin_required(f):
    def wrapper(*args, **kwargs):
        if 'user_type' not in session or session['user_type'] != 'admin':
            return jsonify({'success': False, 'message': 'Admin access required'}), 403
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper
def teacher_required(f):
    def wrapper(*args, **kwargs):
        if 'user_type' not in session or session['user_type'] != 'teacher':
            return jsonify({'success': False, 'message': 'Teacher access required'}), 403
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper
# Password reset tokens (in production, use Redis or database)
password_reset_tokens = {}

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        if session['user_type'] == 'admin':
            return redirect(url_for('admin_dashboard'))
        elif session['user_type'] == 'teacher':
            return redirect(url_for('teacher_dashboard'))
        else:
            return redirect(url_for('student_dashboard'))
    return redirect(url_for('login_page'))

@app.route('/login')
def login_page():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/forgot-password')
def forgot_password_page():
    return render_template('forgot_password.html')

@app.route('/auth/forgot-password', methods=['POST'])
def request_password_reset():
    """Request password reset - generates a reset code"""
    data = request.json
    identifier = data.get('identifier')  # roll_number or username
    user_type = data.get('user_type')
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    user_found = False
    user_name = None
    
    if user_type == 'student':
        c.execute("SELECT id, name, roll_number FROM students WHERE roll_number = ?", (identifier,))
        user = c.fetchone()
        if user:
            user_found = True
            user_name = user[1]
            # Generate 6-digit reset code
            import random
            reset_code = str(random.randint(100000, 999999))
            password_reset_tokens[identifier] = {
                'code': reset_code,
                'type': 'student',
                'id': user[0],
                'expires': datetime.now() + timedelta(minutes=15)
            }
    else:
        c.execute("SELECT id, username FROM admins WHERE username = ?", (identifier,))
        user = c.fetchone()
        if user:
            user_found = True
            user_name = user[1]
            import random
            reset_code = str(random.randint(100000, 999999))
            password_reset_tokens[identifier] = {
                'code': reset_code,
                'type': 'admin',
                'id': user[0],
                'expires': datetime.now() + timedelta(minutes=15)
            }
    
    conn.close()
    
    if user_found:
        # In production, send this via email/SMS
        # For now, return it in response (DEVELOPMENT ONLY)
        return jsonify({
            'success': True, 
            'message': f'Reset code generated for {user_name}',
            'reset_code': reset_code,  # Remove this in production
            'note': 'In production, this code would be sent via email/SMS'
        })
    
    return jsonify({'success': False, 'message': 'User not found'})

@app.route('/auth/reset-password', methods=['POST'])
def reset_password():
    """Reset password using code"""
    data = request.json
    identifier = data.get('identifier')
    reset_code = data.get('reset_code')
    new_password = data.get('new_password')
    
    if identifier not in password_reset_tokens:
        return jsonify({'success': False, 'message': 'Invalid or expired reset code'})
    
    token_data = password_reset_tokens[identifier]
    
    # Check if expired
    if datetime.now() > token_data['expires']:
        del password_reset_tokens[identifier]
        return jsonify({'success': False, 'message': 'Reset code expired'})
    
    # Verify code
    if token_data['code'] != reset_code:
        return jsonify({'success': False, 'message': 'Invalid reset code'})
    
    # Update password
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    hashed_password = generate_password_hash(new_password)
    
    if token_data['type'] == 'student':
        c.execute("UPDATE students SET password = ? WHERE id = ?", (hashed_password, token_data['id']))
    else:
        c.execute("UPDATE admins SET password = ? WHERE id = ?", (hashed_password, token_data['id']))
    
    conn.commit()
    conn.close()
    
    # Remove used token
    del password_reset_tokens[identifier]
    
    return jsonify({'success': True, 'message': 'Password reset successfully'})

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    user_type = data.get('user_type')
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    if user_type == 'admin':
        c.execute("SELECT * FROM admins WHERE username = ?", (username,))
        user = c.fetchone()
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['user_type'] = 'admin'
            conn.close()
            return jsonify({'success': True, 'redirect': '/admin'})
    elif user_type == 'teacher':
        c.execute("SELECT * FROM teachers WHERE username = ?", (username,))
        user = c.fetchone()
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[2]
            session['user_type'] = 'teacher'
            conn.close()
            return jsonify({'success': True, 'redirect': '/teacher'})
    else:
        c.execute("SELECT * FROM students WHERE roll_number = ?", (username,))
        user = c.fetchone()
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['user_type'] = 'student'
            session['roll_number'] = user[2]
            conn.close()
            return jsonify({'success': True, 'redirect': '/student'})
    
    conn.close()
    return jsonify({'success': False, 'message': 'Invalid credentials'})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

@app.route('/admin')
@login_required
def admin_dashboard():
    if session.get('user_type') != 'admin':
        return redirect(url_for('student_dashboard'))
    return render_template('admin.html')

@app.route('/student')
@login_required
def student_dashboard():
    if session.get('user_type') != 'student':
        return redirect(url_for('admin_dashboard'))
    return render_template('student.html', roll_number=session.get('roll_number'))
@app.route('/teacher')
@login_required
def teacher_dashboard():
    if session.get('user_type') != 'teacher':
        return redirect(url_for('index'))
    return render_template('teacher.html', username=session.get('username'))

@app.route('/register', methods=['POST'])
@admin_required
def register_student():
    """Register a new student with face encoding"""
    try:
        name = request.form.get('name')
        roll_number = request.form.get('roll_number')
        password = request.form.get('password')
        image_data = request.form.get('image')
        
        # Decode base64 image
        img_data = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get face encoding
        encoding = get_face_encoding(img)
        if encoding is None:
            return jsonify({'success': False, 'message': 'No face detected in image'})
        
        # Save photo
        photo_path = os.path.join(UPLOAD_FOLDER, f"{roll_number}.jpg")
        cv2.imwrite(photo_path, img)
        
        # Hash password
        hashed_password = generate_password_hash(password)
        
        # Save to database
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO students (name, roll_number, password, encoding, photo_path) VALUES (?, ?, ?, ?, ?)",
                  (name, roll_number, hashed_password, pickle.dumps(encoding), photo_path))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Student registered successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': 'Roll number already exists'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/teacher/students/list')
@teacher_required
def teacher_get_students():
    """Return only students enrolled in this teacher's courses"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""SELECT DISTINCT s.id, s.name, s.roll_number, s.registered_date, s.photo_path
                 FROM students s
                 JOIN enrollments e ON e.student_id = s.id
                 JOIN courses co ON co.id = e.course_id
                 WHERE co.teacher_id = ?
                 ORDER BY s.name""", (session['user_id'],))
    students = c.fetchall()
    conn.close()
    return jsonify([{'id': s[0], 'name': s[1], 'roll_number': s[2], 'registered_date': s[3], 'photo_path': s[4] or ''} for s in students])

@app.route('/teacher/all_students')
@teacher_required
def teacher_all_students():
    """Full student list (admin-registered) for enrollment picker"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name, roll_number FROM students ORDER BY name")
    students = c.fetchall()
    conn.close()
    return jsonify([{'id': s[0], 'name': s[1], 'roll_number': s[2]} for s in students])

@app.route('/teachers/add', methods=['POST'])
@admin_required
def add_teacher():
    try:
        data = request.json
        hashed_password = generate_password_hash(data['password'])
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO teachers (name, username, password) VALUES (?, ?, ?)",
                  (data['name'], data['username'], hashed_password))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Teacher added successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': 'Username already exists'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
@app.route('/admin/courses/list')
@admin_required
def admin_list_courses():
    """Admin view: all courses with their teacher"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""SELECT co.id, co.name, co.code, t.name, t.username, co.created_date
                 FROM courses co
                 LEFT JOIN teachers t ON co.teacher_id = t.id
                 ORDER BY co.name""")
    rows = c.fetchall()
    conn.close()
    return jsonify([{
        'id': r[0], 'name': r[1], 'code': r[2],
        'teacher_name': r[3], 'teacher_username': r[4], 'created_date': r[5]
    } for r in rows])
@app.route('/admin/courses/add', methods=['POST'])
@admin_required
def admin_add_course():
    try:
        data = request.json
        teacher_id = data.get('teacher_id') or None
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO courses (name, code, teacher_id) VALUES (?, ?, ?)",
                  (data['name'], data['code'], teacher_id))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Course added successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': 'Course code already exists'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/admin/courses/update/<int:course_id>', methods=['PUT'])
@admin_required
def admin_update_course(course_id):
    try:
        data = request.json
        teacher_id = data.get('teacher_id') or None
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("UPDATE courses SET name = ?, code = ?, teacher_id = ? WHERE id = ?",
                  (data['name'], data['code'], teacher_id, course_id))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Course updated successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': 'Course code already exists'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/admin/courses/delete/<int:course_id>', methods=['DELETE'])
@admin_required
def admin_delete_course(course_id):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT code FROM courses WHERE id = ?", (course_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return jsonify({'success': False, 'message': 'Course not found'})
        c.execute("DELETE FROM subjects WHERE code = ?", (row[0],))
        c.execute("DELETE FROM enrollments WHERE course_id = ?", (course_id,))
        c.execute("DELETE FROM courses WHERE id = ?", (course_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Course deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/teachers/list')
@admin_required
def list_teachers():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name, username, registered_date FROM teachers")
    teachers = c.fetchall()
    conn.close()
    return jsonify([{'id': t[0], 'name': t[1], 'username': t[2], 'registered_date': t[3]} for t in teachers])

@app.route('/teachers/delete/<int:teacher_id>', methods=['DELETE'])
@admin_required
def delete_teacher(teacher_id):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("UPDATE subjects SET teacher_id = NULL WHERE teacher_id = ?", (teacher_id,))
        c.execute("UPDATE courses SET teacher_id = NULL WHERE teacher_id = ?", (teacher_id,))
        c.execute("DELETE FROM teachers WHERE id = ?", (teacher_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Teacher deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/teachers/update/<int:teacher_id>', methods=['PUT'])
@admin_required
def update_teacher(teacher_id):
    try:
        data = request.json
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("UPDATE teachers SET name = ?, username = ? WHERE id = ?",
                  (data['name'], data['username'], teacher_id))
        if data.get('password'):
            hashed_password = generate_password_hash(data['password'])
            c.execute("UPDATE teachers SET password = ? WHERE id = ?", (hashed_password, teacher_id))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Teacher updated successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': 'Username already exists'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# ── Teacher: Course CRUD ──────────────────────────────────────────────────────

@app.route('/teacher/courses/add', methods=['POST'])
@teacher_required
def teacher_add_course():
    """Teacher creates a new course (name + code)"""
    try:
        data = request.json
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO courses (name, code, teacher_id) VALUES (?, ?, ?)",
                  (data['name'], data['code'], session['user_id']))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Course created successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': 'Course code already exists'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/teacher/courses/list')
@teacher_required
def teacher_list_courses():
    """List teacher's courses with their class slots"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""SELECT id, name, code FROM courses
                 WHERE teacher_id = ? ORDER BY name""", (session['user_id'],))
    course_rows = c.fetchall()

    result = []
    for co in course_rows:
        c.execute("""SELECT id, day_of_week, start_time, end_time, attendance_window
                     FROM subjects WHERE code = ? AND teacher_id = ?
                     ORDER BY day_of_week, start_time""", (co[2], session['user_id']))
        slots = c.fetchall()
        result.append({
            'id': co[0],
            'name': co[1],
            'code': co[2],
            'slots': [{'id': s[0], 'day_of_week': s[1], 'start_time': s[2],
                        'end_time': s[3], 'attendance_window': s[4]} for s in slots]
        })
    conn.close()
    return jsonify(result)

@app.route('/teacher/courses/delete/<int:course_id>', methods=['DELETE'])
@teacher_required
def teacher_delete_course(course_id):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        # Verify ownership
        c.execute("SELECT code FROM courses WHERE id = ? AND teacher_id = ?",
                  (course_id, session['user_id']))
        row = c.fetchone()
        if not row:
            conn.close()
            return jsonify({'success': False, 'message': 'Course not found'})
        course_code = row[0]
        c.execute("DELETE FROM subjects WHERE code = ? AND teacher_id = ?",
                  (course_code, session['user_id']))
        c.execute("DELETE FROM enrollments WHERE course_id = ?", (course_id,))
        c.execute("DELETE FROM courses WHERE id = ?", (course_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Course deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# ── Teacher: Class Slots (linked to a course) ─────────────────────────────────

@app.route('/teacher/subjects/add', methods=['POST'])
@teacher_required
def teacher_add_subject():
    """Add class slots for an existing course"""
    try:
        data = request.json
        course_id = data.get('course_id')
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT name, code FROM courses WHERE id = ? AND teacher_id = ?",
                  (course_id, session['user_id']))
        course = c.fetchone()
        if not course:
            conn.close()
            return jsonify({'success': False, 'message': 'Course not found'})
        for slot in data.get('slots', []):
            c.execute("""INSERT INTO subjects (name, code, day_of_week, start_time, end_time,
                         attendance_window, teacher_id) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                      (course[0], course[1], slot['day_of_week'], slot['start_time'],
                       slot['end_time'], data['attendance_window'], session['user_id']))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Class slots added successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/teacher/subjects/list')
@teacher_required
def teacher_list_subjects():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""SELECT id, name, code, day_of_week, start_time, end_time, attendance_window
                 FROM subjects WHERE teacher_id = ? ORDER BY day_of_week, start_time""",
              (session['user_id'],))
    subjects = c.fetchall()
    conn.close()
    return jsonify([{
        'id': s[0], 'name': s[1], 'code': s[2], 'day_of_week': s[3],
        'start_time': s[4], 'end_time': s[5], 'attendance_window': s[6]
    } for s in subjects])

# ── Teacher: Enrollment Management ───────────────────────────────────────────

@app.route('/teacher/courses/<int:course_id>/students')
@teacher_required
def teacher_course_students(course_id):
    """Students enrolled in a specific course"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM courses WHERE id = ? AND teacher_id = ?",
              (course_id, session['user_id']))
    if not c.fetchone():
        conn.close()
        return jsonify({'success': False, 'message': 'Course not found'}), 404
    c.execute("""SELECT s.id, s.name, s.roll_number, s.photo_path
                 FROM students s JOIN enrollments e ON e.student_id = s.id
                 WHERE e.course_id = ? ORDER BY s.name""", (course_id,))
    students = c.fetchall()
    conn.close()
    return jsonify([{'id': s[0], 'name': s[1], 'roll_number': s[2], 'photo_path': s[3] or ''} for s in students])

@app.route('/teacher/courses/<int:course_id>/enroll', methods=['POST'])
@teacher_required
def teacher_enroll_student(course_id):
    """Enroll a student in a course"""
    try:
        data = request.json
        student_id = data.get('student_id')
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id FROM courses WHERE id = ? AND teacher_id = ?",
                  (course_id, session['user_id']))
        if not c.fetchone():
            conn.close()
            return jsonify({'success': False, 'message': 'Course not found'})
        c.execute("INSERT INTO enrollments (student_id, course_id) VALUES (?, ?)",
                  (student_id, course_id))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Student enrolled successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': 'Student already enrolled'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/teacher/courses/<int:course_id>/unenroll/<int:student_id>', methods=['DELETE'])
@teacher_required
def teacher_unenroll_student(course_id, student_id):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id FROM courses WHERE id = ? AND teacher_id = ?",
                  (course_id, session['user_id']))
        if not c.fetchone():
            conn.close()
            return jsonify({'success': False, 'message': 'Course not found'})
        c.execute("DELETE FROM enrollments WHERE course_id = ? AND student_id = ?",
                  (course_id, student_id))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Student unenrolled successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    
@app.route('/teacher/subjects/delete/<int:subject_id>', methods=['DELETE'])
@teacher_required
def teacher_delete_subject(subject_id):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM subjects WHERE id = ? AND teacher_id = ?", (subject_id, session['user_id']))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Course deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/teacher/subjects/update/<int:subject_id>', methods=['PUT'])
@teacher_required
def teacher_update_subject(subject_id):
    try:
        data = request.json
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""UPDATE subjects SET name = ?, code = ?, day_of_week = ?, start_time = ?, end_time = ?, attendance_window = ?
                     WHERE id = ? AND teacher_id = ?""",
                  (data['name'], data['code'], data['day_of_week'],
                   data['start_time'], data['end_time'], data['attendance_window'], subject_id, session['user_id']))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Course updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/teacher/attendance')
@teacher_required
def teacher_get_attendance():
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    subject_id = request.args.get('subject_id')

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    if subject_id:
        c.execute("""SELECT s.name, s.roll_number, a.timestamp, a.status, sub.name, sub.code
                     FROM attendance a
                     JOIN students s ON a.student_id = s.id
                     JOIN subjects sub ON a.subject_id = sub.id
                     WHERE sub.teacher_id = ? AND a.subject_id = ? AND DATE(a.timestamp) = ?
                     ORDER BY a.timestamp DESC""", (session['user_id'], subject_id, date))
    else:
        c.execute("""SELECT s.name, s.roll_number, a.timestamp, a.status, sub.name, sub.code
                     FROM attendance a
                     JOIN students s ON a.student_id = s.id
                     JOIN subjects sub ON a.subject_id = sub.id
                     WHERE sub.teacher_id = ? AND DATE(a.timestamp) = ?
                     ORDER BY a.timestamp DESC""", (session['user_id'], date))
    records = c.fetchall()
    conn.close()

    return jsonify([{
        'name': r[0], 'roll_number': r[1], 'timestamp': r[2], 'status': r[3],
        'subject_name': r[4], 'subject_code': r[5]
    } for r in records])

# Subject Management
@app.route('/subjects/add', methods=['POST'])
@admin_required
def add_subject():
    try:
        data = request.json
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""INSERT INTO subjects (name, code, day_of_week, start_time, end_time, attendance_window)
                     VALUES (?, ?, ?, ?, ?, ?)""",
                  (data['name'], data['code'], data['day_of_week'], 
                   data['start_time'], data['end_time'], data['attendance_window']))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Subject added successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': 'Subject code already exists'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/subjects/list')
@admin_required
def list_subjects():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM subjects ORDER BY day_of_week, start_time")
    subjects = c.fetchall()
    conn.close()
    
    return jsonify([{
        'id': s[0],
        'name': s[1],
        'code': s[2],
        'day_of_week': s[3],
        'start_time': s[4],
        'end_time': s[5],
        'attendance_window': s[6]
    } for s in subjects])

@app.route('/subjects/delete/<int:subject_id>', methods=['DELETE'])
@admin_required
def delete_subject(subject_id):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Subject deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/subjects/update/<int:subject_id>', methods=['PUT'])
@admin_required
def update_subject(subject_id):
    try:
        data = request.json
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""UPDATE subjects 
                     SET name = ?, code = ?, day_of_week = ?, start_time = ?, end_time = ?, attendance_window = ?
                     WHERE id = ?""",
                  (data['name'], data['code'], data['day_of_week'], 
                   data['start_time'], data['end_time'], data['attendance_window'], subject_id))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Subject updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# Location Management
@app.route('/locations/add', methods=['POST'])
@admin_required
def add_location():
    try:
        data = request.json
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""INSERT INTO locations (name, wifi_ssid, ip_range, latitude, longitude, radius)
                     VALUES (?, ?, ?, ?, ?, ?)""",
                  (data['name'], data.get('wifi_ssid'), data.get('ip_range'),
                   data.get('latitude'), data.get('longitude'), data.get('radius', 100)))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Location added successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/locations/list')
@admin_required
def list_locations():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM locations WHERE is_active = 1")
    locations = c.fetchall()
    conn.close()
    
    return jsonify([{
        'id': l[0],
        'name': l[1],
        'wifi_ssid': l[2],
        'ip_range': l[3],
        'latitude': l[4],
        'longitude': l[5],
        'radius': l[6]
    } for l in locations])

@app.route('/locations/delete/<int:location_id>', methods=['DELETE'])
@admin_required
def delete_location(location_id):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("UPDATE locations SET is_active = 0 WHERE id = ?", (location_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Location deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/locations/update/<int:location_id>', methods=['PUT'])
@admin_required
def update_location(location_id):
    try:
        data = request.json
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""UPDATE locations 
                     SET name = ?, wifi_ssid = ?, ip_range = ?, latitude = ?, longitude = ?, radius = ?
                     WHERE id = ?""",
                  (data['name'], data.get('wifi_ssid'), data.get('ip_range'),
                   data.get('latitude'), data.get('longitude'), data.get('radius', 100), location_id))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Location updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/get_current_subject')
@login_required
def get_active_subject():
    """Get currently active subject for attendance"""
    subject = get_current_subject()
    if subject:
        return jsonify({'success': True, 'subject': subject})
    return jsonify({'success': False, 'message': 'No active class at this time'})

@app.route('/mark_attendance', methods=['POST'])
@login_required
def mark_attendance():
    """Mark attendance using face recognition with time and location validation"""
    try:
        image_data = request.form.get('image')
        location_info = request.form.get('location_info', '')
        
        # Check if there's an active subject
        current_subject = get_current_subject()
        if not current_subject and session['user_type'] == 'student':
            return jsonify({'success': False, 'message': 'No active class at this time. Attendance window closed.'})
        
        # Decode image
        img_data = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get face encoding
        encoding = get_face_encoding(img)
        if encoding is None:
            return jsonify({'success': False, 'message': 'No face detected'})
        
        # Compare with database
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # For students, only verify their own identity
        if session['user_type'] == 'student':
            c.execute("SELECT id, name, roll_number, encoding FROM students WHERE id = ?", (session['user_id'],))
            student = c.fetchone()
            if student:
                stored_encoding = pickle.loads(student[3])
                if not compare_faces(encoding, stored_encoding):
                    conn.close()
                    return jsonify({'success': False, 'message': 'Face does not match your registered profile'})
                matched_student = student
            else:
                conn.close()
                return jsonify({'success': False, 'message': 'Student not found'})
        else:
            # For admin marking attendance
            c.execute("SELECT id, name, roll_number, encoding FROM students")
            students = c.fetchall()
            
            matched_student = None
            for student in students:
                stored_encoding = pickle.loads(student[3])
                if compare_faces(encoding, stored_encoding):
                    matched_student = student
                    break
            
            if not matched_student:
                conn.close()
                return jsonify({'success': False, 'message': 'Face not recognized'})
        
        student_id = matched_student[0]
        subject_id = current_subject['id'] if current_subject else None
        
        # Check if already marked for this subject today
        if subject_id:
            c.execute("""SELECT * FROM attendance 
                        WHERE student_id = ? 
                        AND subject_id = ?
                        AND DATE(timestamp) = DATE('now')""", (student_id, subject_id))
            if c.fetchone():
                conn.close()
                subject_name = current_subject['name']
                return jsonify({'success': False, 'message': f'Attendance already marked for {subject_name} today'})
        
        # Mark attendance
        c.execute("INSERT INTO attendance (student_id, subject_id, location_info) VALUES (?, ?, ?)", 
                  (student_id, subject_id, location_info))
        conn.commit()
        conn.close()
        
        subject_info = f" for {current_subject['name']}" if current_subject else ""
        return jsonify({
            'success': True,
            'message': f'Attendance marked for {matched_student[1]} ({matched_student[2]}){subject_info}'
        })
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/get_attendance')
@login_required
def get_attendance():
    """Get attendance records"""
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    subject_id = request.args.get('subject_id')
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    if session['user_type'] == 'student':
        # Students see only their own attendance
        if subject_id:
            c.execute("""SELECT s.name, s.roll_number, a.timestamp, a.status, sub.name, sub.code
                         FROM attendance a
                         JOIN students s ON a.student_id = s.id
                         LEFT JOIN subjects sub ON a.subject_id = sub.id
                         WHERE s.id = ? AND a.subject_id = ? AND DATE(a.timestamp) = ?
                         ORDER BY a.timestamp DESC""", (session['user_id'], subject_id, date))
        else:
            c.execute("""SELECT s.name, s.roll_number, a.timestamp, a.status, sub.name, sub.code
                         FROM attendance a
                         JOIN students s ON a.student_id = s.id
                         LEFT JOIN subjects sub ON a.subject_id = sub.id
                         WHERE s.id = ? AND DATE(a.timestamp) = ?
                         ORDER BY a.timestamp DESC""", (session['user_id'], date))
    else:
        # Admin sees all attendance
        if subject_id:
            c.execute("""SELECT s.name, s.roll_number, a.timestamp, a.status, sub.name, sub.code
                         FROM attendance a
                         JOIN students s ON a.student_id = s.id
                         LEFT JOIN subjects sub ON a.subject_id = sub.id
                         WHERE a.subject_id = ? AND DATE(a.timestamp) = ?
                         ORDER BY a.timestamp DESC""", (subject_id, date))
        else:
            c.execute("""SELECT s.name, s.roll_number, a.timestamp, a.status, sub.name, sub.code
                         FROM attendance a
                         JOIN students s ON a.student_id = s.id
                         LEFT JOIN subjects sub ON a.subject_id = sub.id
                         WHERE DATE(a.timestamp) = ?
                         ORDER BY a.timestamp DESC""", (date,))
    
    records = c.fetchall()
    conn.close()
    
    return jsonify([{
        'name': r[0],
        'roll_number': r[1],
        'timestamp': r[2],
        'status': r[3],
        'subject_name': r[4] if r[4] else 'N/A',
        'subject_code': r[5] if r[5] else 'N/A'
    } for r in records])

@app.route('/get_my_attendance')
@login_required
def get_my_attendance():
    """Get student's full attendance history"""
    if session['user_type'] != 'student':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""SELECT DATE(a.timestamp) as date, a.timestamp, a.status, sub.name, sub.code
                 FROM attendance a
                 LEFT JOIN subjects sub ON a.subject_id = sub.id
                 WHERE a.student_id = ?
                 ORDER BY a.timestamp DESC""", (session['user_id'],))
    records = c.fetchall()
    conn.close()
    
    return jsonify([{
        'date': r[0],
        'timestamp': r[1],
        'status': r[2],
        'subject_name': r[3] if r[3] else 'N/A',
        'subject_code': r[4] if r[4] else 'N/A'
    } for r in records])

@app.route('/get_students')
@admin_required
def get_students():
    """Get all registered students"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name, roll_number, registered_date, photo_path FROM students")
    students = c.fetchall()
    conn.close()
    
    return jsonify([{
        'id': s[0],
        'name': s[1],
        'roll_number': s[2],
        'registered_date': s[3],
        'photo_path': s[4] or ''
    } for s in students])

@app.route('/students/delete/<int:student_id>', methods=['DELETE'])
@admin_required
def delete_student(student_id):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        # Also delete their attendance records
        c.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        c.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Student deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/students/update/<int:student_id>', methods=['PUT'])
@admin_required
def update_student(student_id):
    try:
        data = request.json
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Update basic info
        c.execute("""UPDATE students SET name = ?, roll_number = ? WHERE id = ?""",
                  (data['name'], data['roll_number'], student_id))
        
        # Update password if provided
        if data.get('password'):
            hashed_password = generate_password_hash(data['password'])
            c.execute("UPDATE students SET password = ? WHERE id = ?", (hashed_password, student_id))
        
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Student updated successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': 'Roll number already exists'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/get_stats')
@admin_required
def get_stats():
    """Get attendance statistics"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Total students
    c.execute("SELECT COUNT(*) FROM students")
    total_students = c.fetchone()[0]
    
    # Today's attendance
    c.execute("""SELECT COUNT(DISTINCT student_id) FROM attendance 
                 WHERE DATE(timestamp) = DATE('now')""")
    today_present = c.fetchone()[0]
    
    # Total subjects
    c.execute("SELECT COUNT(*) FROM subjects")
    total_subjects = c.fetchone()[0]
    
    # This month's total attendance records
    c.execute("""SELECT COUNT(*) FROM attendance
                 WHERE strftime('%Y-%m', timestamp) = strftime('%Y-%m', 'now')""")
    month_attendance = c.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        'total_students': total_students,
        'today_present': today_present,
        'today_absent': total_students - today_present,
        'total_subjects': total_subjects,
        'month_attendance': month_attendance
    })
@app.route('/student/profile')
@login_required
def student_profile():
    if session.get('user_type') != 'student':
        return jsonify({'success': False}), 403
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name, roll_number, photo_path, registered_date FROM students WHERE id = ?", (session['user_id'],))
    s = c.fetchone()
    conn.close()
    if not s:
        return jsonify({'success': False, 'message': 'Not found'}), 404
    return jsonify({'id': s[0], 'name': s[1], 'roll_number': s[2], 'photo_path': s[3] or '', 'registered_date': s[4]})
@app.route('/students/update_photo/<int:student_id>', methods=['POST'])
@admin_required
def update_student_photo(student_id):
    try:
        image_data = request.form.get('image')
        if not image_data:
            return jsonify({'success': False, 'message': 'No image provided'})

        # Decode base64 image
        img_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'success': False, 'message': 'Invalid image data'})

        # Get face encoding using dlib (same as register_student)
        encoding = get_face_encoding(img)
        if encoding is None:
            return jsonify({'success': False, 'message': 'No face detected in photo. Please try again with a clearer image.'})

        # Fetch student roll number to keep same filename convention
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT roll_number FROM students WHERE id = ?", (student_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return jsonify({'success': False, 'message': 'Student not found'})

        # Save new photo (overwrites old one)
        photo_path = os.path.join(UPLOAD_FOLDER, f"{row[0]}.jpg")
        cv2.imwrite(photo_path, img)

        # Update photo path and face encoding in DB
        c.execute(
            "UPDATE students SET photo_path = ?, encoding = ? WHERE id = ?",
            (photo_path, pickle.dumps(encoding), student_id)
        )
        conn.commit()
        conn.close()

        return jsonify({'success': True, 'message': 'Photo and face encoding updated successfully'})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)