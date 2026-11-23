"""เก็บภาพใบหน้าของคลาสใหม่ (wink / kiss) จาก webcam -> data/train และ data/test (80/20)

  python collect_data.py --label wink --count 300
  python collect_data.py --label kiss --count 300

เคล็ดลับ: ขยับหน้าเล็กน้อย เปลี่ยนมุม แสง และขยิบทั้งซ้าย/ขวา เพื่อให้โมเดลแม่นขึ้น

หลังเก็บภาพครบทั้ง 2 คลาสแล้ว ต้องเทรนโมเดลใหม่ด้วย `python emotions.py`
(จะได้ emotion_resnet18.pth + labels.json ใหม่ที่มี 9 คลาส) แล้วรัน app.py ใหม่
"""
# argparse = อ่านพารามิเตอร์จากบรรทัดคำสั่ง, os = สร้างโฟลเดอร์, time = จับเวลา, cv2 = OpenCV (กล้อง + ตรวจหน้า)
import argparse
import os
import time

import cv2

# พารามิเตอร์: --label (wink หรือ kiss เท่านั้น) และ --count (จำนวนภาพที่จะเก็บ ค่าเริ่มต้น 300)
ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True, choices=["wink", "kiss"])
ap.add_argument("--count", type=int, default=300)
args = ap.parse_args()

# สร้างโฟลเดอร์ปลายทาง data/train/<label> และ data/test/<label> (ถ้ามีอยู่แล้วก็ข้าม)
for split in ("train", "test"):
    os.makedirs(f"data/{split}/{args.label}", exist_ok=True)

# ตัวตรวจจับใบหน้า Haar (ไฟล์ xml มาพร้อม OpenCV) ใช้หาตำแหน่งหน้าเพื่อตัดเก็บเฉพาะใบหน้า
cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,
                                             "haarcascade_frontalface_default.xml"))
# เปิดกล้องตัวแรก (หมายเลข 0)
cap = cv2.VideoCapture(0)
# start = เวลาเริ่มโปรแกรม (ใช้นับถอยหลัง 3 วินาที)
# saved = จำนวนภาพที่เซฟแล้ว, last = เวลาที่เซฟภาพล่าสุด (ไว้เว้นช่วงระหว่างภาพ)
start, saved, last = time.time(), 0, 0.0
# วนซ้ำจนเก็บครบจำนวน (หรือกด q ออกก่อน)
while saved < args.count:
    # อ่านภาพ 1 เฟรมจากกล้อง; ถ้าอ่านไม่ได้ (กล้องหลุด / ถูกโปรแกรมอื่นใช้อยู่) ให้หยุด
    ok, frame = cap.read()
    if not ok:
        break
    # แปลงเป็นภาพเทาแล้วหาใบหน้า (หน้าที่เล็กกว่า 80x80 พิกเซลถือว่าไกลเกินไป ข้าม)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
    # 3 วินาทีแรกเป็นช่วงเตรียมตัว: แสดงตัวเลขนับถอยหลัง ยังไม่เซฟภาพ
    wait = 3 - (time.time() - start)
    if wait > 0:
        cv2.putText(frame, f"get ready: {wait:0.0f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
    # พ้นช่วงเตรียมตัวแล้ว: ถ้าเจอหน้า และห่างจากภาพที่เซฟล่าสุดเกิน 0.08 วินาที (~12 ภาพ/วินาที) ให้เซฟภาพ
    elif len(faces) and time.time() - last > 0.08:
        # เลือกใบหน้าที่ใหญ่ที่สุดในภาพ ตัดเฉพาะหน้า ย่อเป็น 48x48 พิกเซลภาพเทา (ขนาดเดียวกับ FER2013)
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        face = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
        # แบ่งชุดข้อมูล: ทุกภาพที่ 5 (20%) ไปชุด test ที่เหลือ (80%) ไปชุด train
        split = "test" if saved % 5 == 0 else "train"
        # บันทึกเป็น .png ตั้งชื่อด้วยเวลาปัจจุบัน (มิลลิวินาที) กันชื่อซ้ำ
        cv2.imwrite(f"data/{split}/{args.label}/{args.label}_{int(time.time() * 1000)}.png", face)
        # นับจำนวนที่เซฟแล้ว และจดเวลาล่าสุด
        saved, last = saved + 1, time.time()
    # แสดงความคืบหน้าที่มุมล่างของหน้าต่าง
    cv2.putText(frame, f"{args.label}: {saved}/{args.count}  (q = quit)", (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # แสดงภาพสดให้เห็นระหว่างเก็บ
    cv2.imshow("collect", frame)
    # กด q เพื่อหยุดก่อนครบจำนวน
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# คืนกล้อง ปิดหน้าต่าง และสรุปจำนวนภาพที่เก็บได้
cap.release()
cv2.destroyAllWindows()
print(f"saved {saved} images for '{args.label}'")