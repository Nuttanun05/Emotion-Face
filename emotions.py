# =====================================================================
# 1. นำเข้าไลบรารีที่จำเป็น (Imports)
# =====================================================================
import json                         # สำหรับอ่าน/เขียนไฟล์ JSON (เก็บชื่อคลาสอารมณ์)
from pathlib import Path            # สำหรับจัดการ Path ของไฟล์และโฟลเดอร์ให้รองรับทุก OS
import numpy as np                  # สำหรับคำนวณทางคณิตศาสตร์และจัดการอาเรย์
import torch                        # ไลบรารีหลักของ PyTorch สำหรับสร้างและเทรนโมเดล AI
import torch.nn as nn               # โมดูลสำหรับสร้างเลเยอร์ของโครงข่ายประสาทเทียม (Neural Network)
from torch.utils.data import DataLoader           # ตัวแบ่งข้อมูลเป็นชุดย่อย (Batch) เพื่อป้อนเข้าโมเดล
from torchvision import datasets, models, transforms # เครื่องมือจัดการภาพ, โมเดลสำเร็จรูป และการแปลงภาพ

# =====================================================================
# 2. กำหนดค่าคงที่และการตั้งค่าหลัก (Configuration)
# =====================================================================
DATA_DIR = Path("data")                       # โฟลเดอร์หลักที่เก็บภาพ train และ test
MODEL_PATH = Path("emotion_resnet18.pth")     # ชื่อไฟล์ที่จะบันทึกน้ำหนัก (Weights) ของโมเดลที่เก่งที่สุด
LABELS_PATH = Path("labels.json")             # ชื่อไฟล์ที่จะบันทึกการจับคู่ตัวเลขคลาสกับชื่ออารมณ์
IMG_SIZE = 112                                # ขนาดภาพที่จะย่อ/ขยายก่อนส่งเข้าโมเดล (112x112 พิกเซล)
BATCH_SIZE = 32                               # จำนวนรูปภาพที่โมเดลจะประมวลผลพร้อมกันใน 1 รอบย่อย
EPOCHS = 20                                   # จำนวนรอบทั้งหมดที่จะให้โมเดลเรียนรู้ข้อมูลทั้งชุด

# ค่าเฉลี่ย (MEAN) และส่วนเบี่ยงเบนมาตรฐาน (STD) มาตรฐานของชุดข้อมูล ImageNet
# ใช้ปรับสเกลสีของภาพ (Normalize) ให้เข้ากับโมเดล ResNet18 ที่ผ่านการ Pre-train มาแล้ว
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# =====================================================================
# 3. ฟังก์ชันสร้างโครงสร้างโมเดล (Model Architecture)
# =====================================================================
def build_model(num_classes: int, pretrained: bool = True):
    # โหลดโมเดล ResNet18 ที่เรียนรู้ลักษณะพื้นฐานของภาพ (ขอบ, เส้น, รูปทรง) มาแล้ว (Transfer Learning)
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    # แก้ไขเลเยอร์สุดท้าย (Fully Connected Layer: fc) ให้ทำนายผลตามจำนวนอารมณ์ของเรา (7 คลาส)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),                               # สุ่มปิดการทำงานของโหนด 30% เพื่อลดการจำข้อสอบ (Overfitting)
        nn.Linear(model.fc.in_features, num_classes)   # แปลงฟีเจอร์จาก ResNet18 ออกมาเป็นคะแนนของแต่ละอารมณ์
    )
    return model

# =====================================================================
# 4. ฟังก์ชันหลักสำหรับเทรนและทดสอบโมเดล (Training Pipeline)
# =====================================================================
def train():
    # ตรวจสอบว่าเครื่องมีการ์ดจอ NVIDIA (CUDA) หรือไม่ ถ้ามีให้ใช้ GPU เพื่อความเร็ว ถ้าไม่มีให้ใช้ CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # --- 4.1 การเตรียมและเพิ่มความหลากหลายของภาพ (Data Augmentation & Preprocessing) ---
    # สำหรับชุด Train: มีการสุ่มพลิกภาพ, หมุนภาพ, ปรับแสง เพื่อช่วยให้โมเดลรับมือกับสภาพแสงและมุมกล้องจริงได้ดีขึ้น
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),             # ปรับขนาดภาพเป็น 112x112
        transforms.RandomHorizontalFlip(),                   # สุ่มกลับด้านภาพซ้าย-ขวา
        transforms.RandomRotation(10),                       # สุ่มเอียงภาพไม่เกิน +-10 องศา
        transforms.ColorJitter(brightness=0.2, contrast=0.2),# สุ่มปรับความสว่างและคอนทราสต์เล็กน้อย
        transforms.ToTensor(),                               # แปลงภาพเป็น Tensor (ค่าพิกเซล 0.0 - 1.0)
        transforms.Normalize(MEAN, STD),                     # ปรับค่าสีตามมาตรฐาน ImageNet
    ])

    # สำหรับชุด Test: ไม่ต้องสุ่มพลิกหรือหมุนภาพ แค่ปรับขนาดและ Normalize เพื่อวัดผลความแม่นยำตามจริง
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # --- 4.2 โหลดชุดข้อมูลจากโฟลเดอร์ ---
    # ImageFolder จะอ่านชื่อโฟลเดอร์ย่อย (angry, happy, ...) มาเป็นชื่อคลาสให้อัตโนมัติ
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=test_tf)

    # บันทึกตารางเทียบลำดับตัวเลขกับชื่ออารมณ์ลงไฟล์ labels.json (เช่น {"0": "angry", "1": "disgusted", ...})
    labels = {str(v): k for k, v in train_ds.class_to_idx.items()}
    LABELS_PATH.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    # --- 4.3 คำนวณน้ำหนักของแต่ละคลาส (Class Weights) ---
    # แก้ปัญหาข้อมูลแต่ละอารมณ์มีจำนวนรูปไม่เท่ากัน (เช่น disgusted มีรูปน้อยกว่า happy)
    counts = np.bincount(train_ds.targets)                         # นับจำนวนรูปในแต่ละคลาส
    weights = (counts.sum() / (len(counts) * counts)) ** 0.5       # คำนวณน้ำหนักแบบถอดรากที่สองเพื่อไม่ให้ค่าน้ำหนักพุ่งสูงเกินไป
    weights = np.clip(weights, 0.5, 2.5)                           # จำกัดเพดานน้ำหนักให้อยู่ระหว่าง 0.5 ถึง 2.5 เท่า
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # สร้างตัวโหลดข้อมูลแบบแบ่ง Batch (ชุด Train ให้สลับลำดับรูปทุกรอบ, ชุด Test ไม่ต้องสลับ)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- 4.4 ตั้งค่าโมเดล ฟังก์ชันคำนวณความผิดพลาด และตัวปรับค่าพารามิเตอร์ ---
    model = build_model(len(labels)).to(device)                    # สร้างโมเดลและย้ายไปประมวลผลบน GPU/CPU
    criterion = nn.CrossEntropyLoss(weight=class_weights)          # ตัววัดค่าความผิดพลาด (Loss) โดยนำน้ำหนักคลาสมาถ่วงดุล
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4) # ตัวปรับน้ำหนักโมเดลให้ทายแม่นขึ้น
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS) # ค่อยๆ ลดอัตราการเรียนรู้ (Learning Rate) ลงเมื่อใกล้จบรอบ

    # --- 4.5 ลูปการเทรนและวัดผล (Training & Evaluation Loop) ---
    best_acc = 0.0                                                 # ตัวแปรเก็บสถิติความแม่นยำสูงสุดที่ทำได้
    for epoch in range(1, EPOCHS + 1):
        # โหมดฝึกสอน (เปิด Dropout และ BatchNorm)
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)                      # ส่งภาพ (x) และเฉลย (y) เข้า GPU/CPU
            optimizer.zero_grad()                                  # ล้างค่าความชัน (Gradient) เก่าจากรอบก่อนหน้า
            loss = criterion(model(x), y)                          # ให้โมเดลทายผลแล้วคำนวณค่าความผิดพลาดเทียบกับเฉลย
            loss.backward()                                        # คำนวณย้อนกลับ (Backpropagation) เพื่อหาว่าต้องปรับจุดไหนบ้าง
            optimizer.step()                                       # อัปเดตน้ำหนักของโมเดลให้ฉลาดขึ้น
        scheduler.step()                                           # ปรับลด Learning Rate ประจำรอบ Epoch

        # โหมดทดสอบ (ปิด Dropout และไม่ต้องคำนวณ Gradient เพื่อความเร็ว)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)                     # เลือกคลาสที่โมเดลให้คะแนนสูงสุดเป็นคำตอบ
                correct += (preds == y).sum().item()               # นับจำนวนข้อที่ทายตรงกับเฉลย
                total += y.size(0)                                 # นับจำนวนภาพทั้งหมดที่ทดสอบ

        # คำนวณเปอร์เซ็นต์ความแม่นยำ (แก้ไขจุดที่บรรทัดตกแล้ว)
        acc = (correct / total) * 100
        print(f"Epoch {epoch:02d}/{EPOCHS} | Test Acc: {acc:.2f}%")

        # ถ้าในรอบนี้โมเดลทำคะแนนบนชุด Test ได้ดีกว่าสถิติเดิม ให้บันทึกทับไฟล์โมเดลไว้
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"Done! Best Test Accuracy: {best_acc:.2f}% -> saved to {MODEL_PATH}")

# จุดเริ่มต้นการทำงานเมื่อสั่งรันไฟล์นี้โดยตรง (เช่น python emotions.py)
if __name__ == "__main__":
    train()