/* =====================================================================
 * mirror.js  -  หน้า "กระจกวิเศษ" (/mirror)
 * เปิดกล้อง -> ส่งภาพให้เซิร์ฟเวอร์วิเคราะห์ต่อเนื่อง -> วาดอีโมจิทับหน้า + แสดงผลในแผงด้านข้าง
 * งานหนักทั้งหมดอยู่ใน EmojiCam (common.js) ไฟล์นี้ทำหน้าที่ต่อสายกับหน้าเว็บเท่านั้น
 * ===================================================================== */

// ตัวควบคุมกล้อง: รับ <video> (ภาพดิบจากกล้อง) กับ <canvas> (ภาพที่ผู้ใช้เห็น)
const cam = new EmojiCam(document.getElementById('video'), document.getElementById('canvas'));
const startBtn = document.getElementById('startBtn');   // ปุ่ม "เปิดกล้อง"
const errBox = document.getElementById('err');          // ช่องแสดงข้อความ error สีแดง

// เซิร์ฟเวอร์ตอบ error (เช่น ยังไม่มีโมเดล) -> โชว์ในแผง ส่งข้อความว่าง = ล้าง error เก่า
cam.onError = msg => { errBox.textContent = msg; };

// ได้ผลวิเคราะห์ใหม่ทุกเฟรม -> อัปเดตช่อง "อารมณ์" และ "สิ่งของ" ในแผงด้านข้าง
cam.onResult = d => {
  const t = describe(d);                                       // แปลงผลเป็นข้อความ (จาก common.js)
  document.getElementById('emoLabel').textContent = t.emo;     // เช่น "😄 ยิ้ม (87%)"
  document.getElementById('objLabel').textContent = t.objs;    // เช่น "🥤 แก้วน้ำ"
};

// กดปุ่ม: ขอสิทธิ์กล้อง -> ซ่อนปุ่ม -> เริ่มวาดภาพ (draw) และเริ่มส่งภาพไปวิเคราะห์ (loop)
startBtn.addEventListener('click', async () => {
  try {
    await cam.start();        // ถ้าผู้ใช้ไม่อนุญาต/ไม่มีกล้อง จะ error และไปที่ catch
    startBtn.hidden = true;
    cam.draw();               // ลูปวาดลง canvas (ภาพกล้อง + อีโมจิ)
    cam.loop();               // ลูปส่งภาพไปเซิร์ฟเวอร์ (ไม่ต้อง await เพราะวนไม่จบ)
  } catch (e) {
    errBox.textContent = 'เปิดกล้องไม่ได้ - อนุญาตการใช้กล้องแล้วลองใหม่ (ต้องเปิดผ่าน localhost หรือ https)';
  }
});