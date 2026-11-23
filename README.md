# Emotion Face 😉

เว็บ Flask: Login/Register + กระจกวิเศษ (Real-time Emoji Mirror) + มินิเกม Emoji Challenge
Backend: OpenCV Haar Cascade → CNN 9 คลาส (7 FER2013 + wink + kiss) + YOLO (วัตถุ) → JSON, เก็บ user/score ด้วย sqlite3

## 1) ติดตั้ง
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2) เตรียมข้อมูลเทรน
1. โหลด FER2013 (Kaggle: `msambare/fer2013`) แล้ววางเป็น `data/train/<class>/` และ `data/test/<class>/`
   (angry, disgusted, fearful, happy, neutral, sad, surprised)
2. เก็บภาพ 2 คลาสใหม่จากกล้องตัวเอง (ขยับมุม/แสงให้หลากหลาย)
```bash
python collect_data.py --label wink --count 300
python collect_data.py --label kiss --count 300
```

## 3) เทรนโมเดล → ได้ model.h5 + labels.json
```bash
python emotions.py --mode train --epochs 50
python emotions.py --mode display      # ทดสอบกับ webcam (กด q ออก)
```

## 4) รันเว็บ
```bash
python app.py        # เปิด http://127.0.0.1:5000
```
ครั้งแรก YOLO จะโหลดน้ำหนักอัตโนมัติ (yolo26n.pt; ถ้าไม่ได้จะ fallback เป็น yolo11n.pt)

## ทดสอบ API ด้วย Postman / Thunder Client
API ต้อง login ก่อน (ใช้ session cookie):
1. `POST /login` แบบ x-www-form-urlencoded: `username`, `password` (ให้ client เก็บ cookie)
2. `POST /api/analyze` แบบ form-data: key `image` (type File) → ตัวอย่างผลลัพธ์
```json
{"face": true, "emotion": "wink", "confidence": 0.91,
 "box": {"x": 0.3, "y": 0.2, "w": 0.25, "h": 0.33},
 "objects": [{"name": "bottle", "confidence": 0.77}]}
```
ใส่ `?objects=0` เพื่อข้าม YOLO
3. `POST /api/score` JSON `{"score": 120}` และ `GET /api/leaderboard`

## ปรับแต่ง
- เวลาต่อโจทย์/ชีวิต/โจทย์ทั้งหมด: ต้นไฟล์ `static/js/game.js`
- ตั้ง `SECRET_KEY` เป็นค่าสุ่มก่อนใช้งานจริง: `export SECRET_KEY=...`
- หมายเหตุ: คะแนนส่งมาจากฝั่ง client จึงยังโกงได้ ถ้าจะจริงจังควรตรวจฝั่ง server
