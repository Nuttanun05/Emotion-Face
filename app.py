"""
Emotion Face - Flask Backend (เซิร์ฟเวอร์หลังบ้านของโปรเจกต์)

หน้าที่หลัก 3 ส่วน:
  1) ระบบสมาชิก: สมัครสมาชิก / เข้าสู่ระบบ / ออกจากระบบ (จัดเก็บในฐานข้อมูล SQLite)
  2) ระบบคะแนนมินิเกม: บันทึกคะแนนการเล่นและดึงตารางอันดับสูงสุด (Leaderboard)
  3) ระบบประมวลผล AI (/api/analyze): รับภาพจากกล้องเว็บแคม -> ตรวจจับใบหน้า (Haar Cascade)
     -> วิเคราะห์อารมณ์จากใบหน้าสี (ResNet18) -> ตรวจจับสิ่งของ (YOLO) -> ส่งผลลัพธ์กลับเป็น JSON
"""

# =====================================================================
# 1. นำเข้าไลบรารีและโมดูลที่จำเป็น (Imports)
# =====================================================================
import json                         # สำหรับอ่านไฟล์ labels.json ที่เก็บชื่อคลาสอารมณ์
import os                           # สำหรับจัดการ Path ไฟล์และอ่านค่า Environment Variables
import re                           # สำหรับตรวจสอบรูปแบบตัวอักษร (Regular Expression) ตอนสมัครสมาชิก
import sqlite3                      # ระบบฐานข้อมูล SQLite แบบไฟล์เดียวในเครื่อง
import threading                    # สำหรับสร้าง Lock ป้องกันไม่ให้หลาย Request เรียกใช้โมเดล AI ชนกัน
from functools import wraps         # ช่วยรักษาชื่อและข้อมูลของฟังก์ชันเดิมเวลาสร้าง Decorator

import cv2                          # OpenCV สำหรับแปลงไฟล์ภาพ, ปรับสี, ย่อขนาด และหาใบหน้า
import numpy as np                  # สำหรับจัดการอาเรย์ของพิกเซลภาพและคำนวณค่า Normalize
import torch                        # PyTorch สำหรับรันโมเดล ResNet18 ทำนายอารมณ์
from flask import (                 # เครื่องมือหลักของเว็บเซิร์ฟเวอร์ Flask
    Flask, flash, g, jsonify, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash # เข้ารหัสและตรวจรหัสผ่าน

# นำเข้าโครงสร้างโมเดลและค่ามาตรฐานการปรับภาพ (112x112, MEAN, STD) จากไฟล์ emotions.py
# เพื่อให้ขั้นตอนการเตรียมภาพตอนใช้งานจริงตรงกับตอนเทรนโมเดล RAF-DB 100%
from emotions import build_model, IMG_SIZE, MEAN, STD


# =====================================================================
# 2. การตั้งค่าแอปและพาธของไฟล์ต่างๆ (Configuration & Paths)
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))         # โฟลเดอร์หลักที่ไฟล์ app.py ตั้งอยู่
DB_PATH = os.path.join(BASE_DIR, "emotionface.db")            # ไฟล์ฐานข้อมูล SQLite (สร้างอัตโนมัติถ้าย้งไม่มี)
MODEL_PATH = os.path.join(BASE_DIR, "emotion_resnet18.pth")   # ไฟล์น้ำหนักโมเดล ResNet18 ที่เทรนเสร็จแล้ว
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")           # ไฟล์จับคู่เลขคลาส (0-6) กับชื่ออารมณ์

# เปิด/ปิดการปรับค่าน้ำหนักอารมณ์ (ค่าเริ่มต้น "0" คือปิด เพื่อใช้ค่าทำนายตรงๆ จากโมเดล RAF-DB)
USE_BIAS = os.environ.get("EMOTION_BIAS", "0") == "1"

# ชื่อไฟล์โมเดล YOLO สำหรับตรวจจับสิ่งของ (ถ้าโหลด yolo26n.pt ไม่ได้ จะสลับไปใช้ yolo11n.pt อัตโนมัติ)
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolo26n.pt")

app = Flask(__name__)
# กุญแจลับสำหรับเข้ารหัส Cookie ของ Session (ใช้จำสถานะการ Login ของผู้ใช้)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")


# =====================================================================
# 3. การจัดการฐานข้อมูล SQLite (Database Setup & Helpers)
# =====================================================================
# คำสั่ง SQL สำหรับสร้างตาราง users (เก็บข้อมูลบัญชี) และ scores (เก็บประวัติคะแนนมินิเกม)
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    score INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

def get_db():
    """เปิดการเชื่อมต่อฐานข้อมูลต่อ 1 HTTP Request และเก็บไว้ในตัวแปร g เพื่อใช้ซ้ำ"""
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row  # ทำให้ดึงข้อมูลแบบระบุชื่อคอลัมน์ได้ เช่น row["username"]
    return g.db

@app.teardown_appcontext
def close_db(_exc):
    """ปิดการเชื่อมต่อฐานข้อมูลอัตโนมัติทุกครั้งเมื่อประมวลผล Request เสร็จสิ้น"""
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    """สร้างไฟล์ฐานข้อมูลและตารางเริ่มต้นตาม SCHEMA (ทำงานตอนเปิดเซิร์ฟเวอร์)"""
    with sqlite3.connect(DB_PATH) as db:
        db.executescript(SCHEMA)


# =====================================================================
# 4. ระบบตรวจสอบสิทธิ์การเข้าใช้งาน (Authentication Decorator)
# =====================================================================
def login_required(view):
    """Decorator บังคับให้ผู้ใช้ต้องเข้าสู่ระบบก่อนเข้าถึงหน้าเว็บหรือ API นั้นๆ"""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            # ถ้าเรียกผ่าน API (/api/...) ให้ตอบกลับเป็น JSON Error 401
            if request.path.startswith("/api/"):
                return jsonify(error="กรุณาเข้าสู่ระบบ"), 401
            # ถ้าเปิดผ่านหน้าเว็บปกติ ให้เด้งไปหน้า Login
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


# =====================================================================
# 5. เส้นทางหน้าเว็บและระบบสมาชิก (Auth & Page Routes)
# =====================================================================
@app.route("/")
def index():
    """หน้าแรกสุด: ถ้าล็อกอินแล้วส่งไปหน้ากระจกวิเศษ (/mirror) ถ้ายังให้ไปหน้าล็อกอิน"""
    return redirect(url_for("mirror" if "user_id" in session else "login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    """หน้าสมัครสมาชิก: ตรวจสอบความถูกต้องของชื่อผู้ใช้/รหัสผ่าน แล้วบันทึกลงฐานข้อมูล"""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        # ตรวจสอบเงื่อนไขข้อมูลก่อนบันทึก
        if not re.fullmatch(r"[A-Za-z0-9_]{3,20}", username):
            flash("ชื่อผู้ใช้ต้องเป็น a-z, 0-9 หรือ _ ยาว 3-20 ตัวอักษร")
        elif len(password) < 6:
            flash("รหัสผ่านต้องยาวอย่างน้อย 6 ตัวอักษร")
        elif password != confirm:
            flash("รหัสผ่านทั้งสองช่องไม่ตรงกัน")
        else:
            try:
                db = get_db()
                # เข้ารหัสรหัสผ่านด้วย generate_password_hash ก่อนบันทึก (ไม่เก็บรหัสผ่านดิบ)
                db.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, generate_password_hash(password))
                )
                db.commit()
                flash("สมัครสมาชิกสำเร็จ เข้าสู่ระบบได้เลย")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                # เกิดขึ้นเมื่อ username ซ้ำกับที่มีอยู่แล้วในตาราง (ติดเงื่อนไข UNIQUE)
                flash("ชื่อผู้ใช้นี้ถูกใช้แล้ว")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """หน้าเข้าสู่ระบบ: ตรวจสอบชื่อผู้ใช้และเทียบ Hash รหัสผ่าน"""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = get_db().execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]         # เก็บ ID ผู้ใช้ไว้ใน Session
            session["username"] = user["username"]  # เก็บชื่อผู้ใช้ไว้แสดงผลบนหน้าเว็บ
            return redirect(url_for("mirror"))
        flash("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")
    return render_template("login.html")

@app.route("/logout")
def logout():
    """ออกจากระบบ: ล้างข้อมูล Session ทั้งหมดแล้วพากลับหน้า Login"""
    session.clear()
    return redirect(url_for("login"))

@app.route("/mirror")
@login_required
def mirror():
    """แสดงหน้าเว็บโหมดกระจกวิเศษ (ต้องล็อกอินก่อน)"""
    return render_template("mirror.html")

@app.route("/game")
@login_required
def game():
    """แสดงหน้าเว็บโหมดมินิเกมทำท่าทางและหาสิ่งของ (ต้องล็อกอินก่อน)"""
    return render_template("game.html")


# =====================================================================
# 6. API ระบบคะแนนมินิเกมและตารางอันดับ (Scores & Leaderboard API)
# =====================================================================
@app.post("/api/score")
@login_required
def save_score():

    """รับคะแนนจากฝั่งหน้าเว็บ (game.js) เมื่อเล่นจบเกม แล้วบันทึกลงฐานข้อมูล"""
    data = request.get_json(silent=True) or {}
    try:
        score = int(data.get("score", 0))
    except (TypeError, ValueError):
        return jsonify(error="score ไม่ถูกต้อง"), 400

    # ป้องกันการส่งค่าคะแนนติดลบหรือสูงผิดปกติ
    if not 0 <= score <= 10000:
        return jsonify(error="score ไม่ถูกต้อง"), 400

    db = get_db()
    db.execute("INSERT INTO scores (user_id, score) VALUES (?, ?)", (session["user_id"], score))
    db.commit()
    return jsonify(ok=True, best=_best(session["user_id"]))

def _best(user_id):
    """ฟังก์ชันช่วยดึงคะแนนสูงสุด (High Score) ของผู้ใช้ตาม user_id"""
    row = get_db().execute("SELECT MAX(score) AS b FROM scores WHERE user_id = ?", (user_id,)).fetchone()
    return row["b"] or 0

@app.get("/api/leaderboard")
@login_required
def leaderboard():
    """ดึงข้อมูลอันดับผู้เล่นที่ได้คะแนนสูงสุด 10 อันดับแรก พร้อมคะแนนสูงสุดของผู้เล่นปัจจุบัน"""
    rows = get_db().execute(
        """SELECT u.username, MAX(s.score) AS best
           FROM scores s JOIN users u ON u.id = s.user_id
           GROUP BY u.id ORDER BY best DESC LIMIT 10"""
    ).fetchall()
    return jsonify(top=[dict(r) for r in rows], mine=_best(session["user_id"]))

@app.get("/api/labels")
@login_required
def labels_route():
    """ส่งรายชื่ออารมณ์ทั้งหมดที่โมเดลรองรับให้หน้าเว็บ (ใช้กรองโจทย์ในมินิเกม)"""
    if not os.path.exists(LABELS_PATH):
        return jsonify(labels=[])
    with open(LABELS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return jsonify(labels=[raw[k] for k in sorted(raw, key=int)])


# =====================================================================
# 7. ส่วนเตรียมโมเดล AI (Face Detection, Emotion ResNet18, YOLO)
# =====================================================================
_lock = threading.Lock()     # ตัวล็อกคิวป้องกันการเรียกใช้โมเดลพร้อมกันจากหลาย Request
_models = {"yolo": None}     # แคชเก็บโมเดล YOLO (โหลดเมื่อถูกเรียกใช้ครั้งแรกแบบ Lazy Loading)
_cnn = None                  # แคชเก็บโมเดล ResNet18
_labels = None               # แคชเก็บ Dictionary แปลงเลขคลาสเป็นชื่ออารมณ์

# โหลดตัวตรวจจับใบหน้า Haar Cascade ของ OpenCV 2 ชุด (ตัวหลัก + ตัวสำรอง)
face_cascades = [
    cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, n))
    for n in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml")
]

def find_face(gray):
    """ค้นหาตำแหน่งใบหน้าจากภาพขาวดำ แล้วคืนค่าพิกัดกรอบใบหน้าที่ใหญ่ที่สุด (x, y, w, h)"""
    eq = cv2.equalizeHist(gray)  # ปรับสมดุลแสงให้เห็นโครงหน้าชัดขึ้นในที่แสงน้อย
    for c in face_cascades:
        faces = c.detectMultiScale(eq, scaleFactor=1.25, minNeighbors=4, minSize=(40, 40))
        if len(faces):
            return max(faces, key=lambda b: b[2] * b[3])  # เลือกใบหน้าที่อยู่ใกล้กล้องที่สุด (พื้นที่ใหญ่สุด)
    return None

def get_cnn():
    """โหลดโมเดล ResNet18 และไฟล์ labels.json เข้าหน่วยความจำ (ทำครั้งเดียวตอนเรียกใช้ครั้งแรก)"""
    global _cnn, _labels
    if _cnn is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
            raise FileNotFoundError("ไม่พบ emotion_resnet18.pth หรือ labels.json - รัน `python emotions.py` เพื่อเทรนก่อน")
        with open(LABELS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        _labels = {int(k): v for k, v in raw.items()}
        model = build_model(len(_labels), pretrained=False)  # สร้างโครงสร้าง ResNet18
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))  # โหลดน้ำหนักที่เทรนไว้
        model.eval()  # ตั้งค่าเป็นโหมดทำนายผล (ปิด Dropout)
        _cnn = model
    return _cnn, _labels

def adjust_emotion_bias(scores):
    """ปรับถ่วงน้ำหนักความน่าจะเป็นของแต่ละอารมณ์ (ทำงานเฉพาะเมื่อตั้งค่า USE_BIAS=True)"""
    weights = {
        "neutral": 0.85,
        "disgusted": 0.90,
        "happy": 1.05,
        "surprised": 1.15,
        "angry": 1.15,
        "sad": 1.15,
        "fearful": 1.10,
    }
    adjusted = {k: v * weights.get(k, 1.0) for k, v in scores.items()}
    total = sum(adjusted.values()) or 1.0
    return {k: v / total for k, v in adjusted.items()}  # ปรับสัดส่วนให้ผลรวมความน่าจะเป็นกลับเป็น 1.0

def predict_torch(face_bgr):
    """นำภาพสีเฉพาะส่วนใบหน้า (BGR) มาเตรียมข้อมูลให้ตรงกับชุด RAF-DB แล้วทำนายอารมณ์"""
    cnn, labels = get_cnn()
    # แปลงระบบสี BGR (ของ OpenCV) เป็น RGB -> ย่อขนาด 112x112 -> ปรับค่าพิกเซลเป็น 0.0 - 1.0
    rgb = cv2.cvtColor(cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    # Normalize ด้วยค่า MEAN และ STD เดียวกับตอนเทรน
    rgb = (rgb - np.array(MEAN, dtype="float32")) / np.array(STD, dtype="float32")
    # สลับแกนภาพจาก (Height, Width, Channel) เป็น (Batch=1, Channel=3, Height=112, Width=112)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

    with torch.no_grad():  # ปิดการคำนวณ Gradient เพื่อความเร็วและประหยัด RAM
        p = torch.softmax(cnn(tensor), dim=1)[0].numpy()  # แปลงค่าเอาต์พุตเป็นความน่าจะเป็น (0-1)

    scores = {labels[i]: float(v) for i, v in enumerate(p)}
    return adjust_emotion_bias(scores) if USE_BIAS else scores

def get_yolo():
    """โหลดโมเดล YOLO สำหรับตรวจจับสิ่งของ (ถ้าโหลดตัวหลักไม่ได้จะใช้ yolo11n.pt สำรอง)"""
    if _models["yolo"] is None:
        from ultralytics import YOLO
        try:
            _models["yolo"] = YOLO(YOLO_WEIGHTS)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] โหลด {YOLO_WEIGHTS} ไม่ได้ ({exc}) -> ใช้ yolo11n.pt แทน")
            _models["yolo"] = YOLO("yolo11n.pt")
    return _models["yolo"]


# =====================================================================
# 8. API หลักสำหรับวิเคราะห์ภาพจากกล้องเว็บแคม (/api/analyze)
# =====================================================================
@app.post("/api/analyze")
@login_required
def analyze():
    """รับภาพจากหน้าเว็บ -> ตรวจจับใบหน้าและอารมณ์ + ตรวจจับสิ่งของ -> ส่งผลลัพธ์กลับเป็น JSON"""
    file = request.files.get("image")
    if file is None:
        return jsonify(error="ไม่พบไฟล์ภาพ (field ชื่อ image)"), 400

    # แปลงไฟล์ภาพที่อัปโหลดมาเป็นภาพสี OpenCV (BGR)
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="อ่านไฟล์ภาพไม่ได้"), 400

    # เช็กพารามิเตอร์ ?objects=0 (ถ้าเป็น 0 จะข้ามการรัน YOLO เพื่อให้ประมวลผลเร็วขึ้น)
    want_objects = request.args.get("objects", "1") != "0"
    h, w = img.shape[:2]
    result = {"face": False, "objects": []}

    try:
        with _lock:  # ล็อกคิวให้ประมวลผลทีละภาพ ป้องกันโมเดลชนกัน
            # --- ขั้นตอนที่ 1: ค้นหาใบหน้าและวิเคราะห์อารมณ์ ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = find_face(gray)
            if face is not None:
                x, y, fw, fh = [int(v) for v in face]
                # ครอปใบหน้าจากภาพสี (img) ส่งเข้า ResNet18
                probs = predict_torch(img[y:y + fh, x:x + fw])
                emotion = max(probs, key=probs.get)  # เลือกอารมณ์ที่ได้ความน่าจะเป็นสูงสุด
                result.update(
                    face=True,
                    emotion=emotion,
                    confidence=probs[emotion],
                    probs=probs,
                    # แปลงพิกัดกรอบใบหน้าเป็นสัดส่วน 0.0 - 1.0 เพื่อนำไปวาดกรอบบนหน้าเว็บได้ทุกขนาดจอ
                    box={"x": x / w, "y": y / h, "w": fw / w, "h": fh / h},
                )

            # --- ขั้นตอนที่ 2: ตรวจจับสิ่งของในเฟรมด้วย YOLO ---
            if want_objects:
                det = get_yolo().predict(img, imgsz=320, conf=0.35, verbose=False)[0]
                best = {}
                for b in det.boxes:
                    name = det.names[int(b.cls)]
                    conf = float(b.conf)
                    # กรองคลาส "person" ออก และเก็บเฉพาะค่าความมั่นใจสูงสุดของวัตถุแต่ละชนิด
                    if name != "person" and conf > best.get(name, 0):
                        best[name] = conf
                # เรียงลำดับสิ่งของที่พบตามค่าความมั่นใจจากมากไปน้อย
                result["objects"] = [
                    {"name": n, "confidence": c}
                    for n, c in sorted(best.items(), key=lambda kv: -kv[1])
                ]

    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 503
    except ImportError as exc:
        return jsonify(error=f"ไลบรารีไม่พร้อม: {exc}"), 503
    except Exception as exc:  # noqa: BLE001
        print(f"[AI Error] {exc}")
        return jsonify(error=str(exc)), 500

    return jsonify(result)

if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)