"""โครงสร้าง CNN จำแนกอารมณ์ (ใช้ร่วมกันระหว่าง emotions.py และ app.py)

หมายเหตุ: ปัจจุบัน app.py และ emotions.py ใช้ ResNet18 (ฟังก์ชัน build_model ใน emotions.py) แทนแล้ว
ไฟล์นี้จึงไม่ได้ถูก import จากที่ไหน เก็บไว้เป็นตัวอย่าง CNN ที่ออกแบบเอง (ลบทิ้งได้ถ้าไม่ต้องใช้)
"""
# torch.nn = ชั้นต่างๆ ของโครงข่ายประสาทเทียมใน PyTorch
import torch.nn as nn


# CNN ขนาดเล็ก แบ่งเป็น 2 ส่วน: features (ดึงลักษณะจากภาพ) และ classifier (ตัดสินว่าเป็นอารมณ์ไหน)
class EmotionCNN(nn.Module):
    """อินพุต 1x48x48 (ภาพเทา) -> logits ตามจำนวนคลาส (7 หรือ 9)"""

    # num_classes = จำนวนอารมณ์ที่จะจำแนก (นับจากโฟลเดอร์ใน data/train)
    def __init__(self, num_classes):
        super().__init__()
        # ส่วนดึงลักษณะ: Conv = ตัวกรองหาลายเส้น/รูปทรง, ReLU = ตัดค่าลบทิ้ง,
        #   MaxPool = ย่อภาพลงครึ่งหนึ่งโดยเก็บค่าเด่น, Dropout = สุ่มปิดบางส่วนขณะเทรนกันจำข้อมูลแบบท่องจำ (overfit)
        # ขนาดภาพที่ไหลผ่าน (ไม่มี padding: Conv 3x3 ลดลง 2 พิกเซล):
        #   48 -> 46 -> 44 -> (pool) 22 -> 20 -> (pool) 10 -> 8 -> (pool) 4
        self.features = nn.Sequential(
            # Conv2d(ช่องขาเข้า, ช่องขาออก, ขนาดตัวกรอง): ภาพเทา 1 ช่อง -> 32 ช่อง แล้วเพิ่มเป็น 64, 128 ตามลำดับ
            nn.Conv2d(1, 32, 3), nn.ReLU(),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        # ส่วนตัดสินใจ: แบนข้อมูลเป็นเวกเตอร์ -> ชั้นเชื่อมเต็ม 1024 โหนด -> ชั้นสุดท้ายให้คะแนนของแต่ละอารมณ์
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 128 ช่อง x 4 x 4 พิกเซล = 2048 ค่า เข้าชั้นเชื่อมเต็ม 1024 โหนด; Dropout 0.5 ลดการ overfit
            nn.Linear(128 * 4 * 4, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, num_classes),   # ชั้น Dense สุดท้าย: จำนวนคลาสตามโฟลเดอร์ใน data/train
        )

    # ลำดับคำนวณ: ภาพ -> features -> classifier -> คะแนนดิบ (logits) ของแต่ละอารมณ์
    # (ยังไม่ใช่ความน่าจะเป็น ต้องผ่าน softmax อีกที)
    def forward(self, x):
        return self.classifier(self.features(x))