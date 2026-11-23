/* =====================================================================
 * common.js  -  โค้ดฝั่งเบราว์เซอร์ที่ใช้ร่วมกันทั้งหน้า "กระจกวิเศษ" และ "มินิเกม"

 * หน้าที่หลัก:
 *   1) เก็บตารางแปลงผลลัพธ์จากเซิร์ฟเวอร์ -> อีโมจิ / ชื่อภาษาไทย
 *   2) EmojiCam: เปิดกล้อง, ส่งภาพไปให้เซิร์ฟเวอร์วิเคราะห์ซ้ำๆ, วาดอีโมจิทับใบหน้า
 *   3) describe: แปลงผลลัพธ์เป็นข้อความโชว์ในแผงด้านข้าง
 * ===================================================================== */

const EMOTION_EMOJI = {
  angry: '😠', disgusted: '🤢', fearful: '😨', happy: '😄',
  neutral: '😐', sad: '😢', surprised: '😮'
};


const EMOTION_TH = {
  angry: 'โกรธ',
  disgusted: 'ขยะแขยง',
  fearful: 'กลัว',
  happy: 'ยิ้ม',
  neutral: 'เฉยๆ',
  sad: 'เศร้า',
  surprised: 'ตกใจ',
};

// วัตถุที่ YOLO (COCO 80 คลาส) รู้จัก -> อีโมจิ / ชื่อไทย
// ชื่อคีย์เป็นภาษาอังกฤษตามที่ YOLO ตอบกลับ ถ้าสิ่งของไม่อยู่ในตารางนี้ จะไม่ถูกวาดอีโมจิ
const OBJECT_EMOJI = {
  bottle: '🍺', cup: '🥤', 'wine glass': '🍷', fork: '🍴', knife: '🔪', spoon: '🥄',
  'cell phone': '📱', laptop: '💻', keyboard: '⌨️', mouse: '🖱️', remote: '🀞', tv: '📺',
  book: '📖', clock: '⏰', backpack: '🎒', umbrella: '☂️', scissors: '✂️'
};
const OBJECT_TH = {
  bottle: 'ขวดน้ำ', cup: 'แก้วน้ำ', 'wine glass': 'แก้วไวน์', fork: 'ส้อม', knife: 'มีด', spoon: 'ช้อน',
  'cell phone': 'มือถือ', laptop: 'โน้ตบุ๊ก', keyboard: 'คีย์บอร์ด', mouse: 'เมาส์', remote: 'รีโมท', tv: 'ทีวี',
  book: 'หนังสือ', clock: 'นาฬิกา', backpack: 'กระเป๋าเป้', umbrella: 'ร่ม', scissors: 'กรรไกร'
};

const EMOJI_FONT = '"Noto Color Emoji","Apple Color Emoji","Segoe UI Emoji",sans-serif';

const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---- กล้อง + วนส่งภาพไป /api/analyze + วาดผลลง canvas ------------------------
class EmojiCam {
  /**
   * @param video  <video> ที่รับภาพจากกล้อง (ซ่อนไว้ ไม่แสดงตรงๆ)
   * @param canvas <canvas> ที่ผู้ใช้เห็นจริง: วาดภาพกล้อง + อีโมจิทับ
   */
  constructor(video, canvas) {
    this.video = video;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.grab = document.createElement('canvas'); 
    this.latest = null;       
    this.running = false;   
    this.wantObjects = true;   
    this.onResult = null;      
    this.onError = null;      
  }

  // ขอสิทธิ์ใช้กล้อง แล้วต่อภาพเข้า <video> (ถ้าผู้ใช้ไม่อนุญาตจะ throw error ให้หน้าเพจจัดการ)
  async start() {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }, audio: false
    });
    this.video.srcObject = stream;
    await this.video.play();
  }

  // จับภาพ 1 เฟรมจากกล้อง ย่อให้กว้าง 400px (คงสัดส่วน) แล้วแปลงเป็นไฟล์ JPEG
  // ย่อก่อนส่งเพื่อให้อัปโหลดและประมวลผลเร็วขึ้น คุณภาพ 0.7 พอสำหรับตรวจหน้า
  grabBlob() {
    const w = 400, h = Math.round(w * this.video.videoHeight / this.video.videoWidth);
    this.grab.width = w; this.grab.height = h;
    this.grab.getContext('2d').drawImage(this.video, 0, 0, w, h);
    return new Promise(res => this.grab.toBlob(res, 'image/jpeg', 0.7));
  }

  // วนลูปส่งภาพทีละเฟรม (รอผลกลับมาก่อนค่อยส่งเฟรมถัดไป ไม่ให้ซ้อนกันจนเซิร์ฟเวอร์ค้าง)
  async loop() {
    this.running = true;
    while (this.running) {
      const t0 = performance.now();
      try {
        // ส่งภาพเป็น form-data ชื่อ "image" (ตรงกับ request.files["image"] ฝั่ง Flask)
        // พารามิเตอร์ objects=0 บอกให้เซิร์ฟเวอร์ข้าม YOLO
        const fd = new FormData();
        fd.append('image', await this.grabBlob(), 'frame.jpg');
        const res = await fetch('/api/analyze?objects=' + (this.wantObjects ? 1 : 0), { method: 'POST', body: fd });

        // 401 = session หมดอายุ / ยังไม่ login -> เด้งกลับหน้า login
        if (res.status === 401) { location.href = '/login'; return; }

        const data = await res.json();
        if (res.ok) {
          this.latest = data;                         // เก็บไว้ให้ draw() วาด
          if (this.onError) this.onError('');         // เคลียร์ข้อความ error เก่า
          if (this.onResult) this.onResult(data);     // แจ้งหน้าเพจ (อัปเดตแผงข้อมูล / ตรวจโจทย์เกม)
        } else if (this.onError) {
          this.onError(data.error || 'เกิดข้อผิดพลาด'); // เช่น "ยังไม่มีโมเดล"
          await sleep(1500);                           // เซิร์ฟเวอร์มีปัญหา: รอสักครู่ ไม่ยิงถี่ๆ
        }
      } catch (e) {
        console.error(e);                              // เน็ต/เซิร์ฟเวอร์หลุดชั่วคราว
        await sleep(800);
      }
      // เว้นอย่างน้อย ~60 มิลลิวินาทีระหว่างรอบ (ถ้ารอบนั้นใช้เวลานานกว่านี้อยู่แล้วก็ไม่ต้องรอเพิ่ม)
      await sleep(Math.max(0, 60 - (performance.now() - t0)));
    }
  }

  // สั่งหยุดลูปส่งภาพ (ตอนนี้ยังไม่มีหน้าไหนเรียกใช้ เผื่อไว้ใช้ภายหลัง)
  stop() { this.running = false; }

  // วาดภาพลง canvas ต่อเนื่อง (ผ่าน requestAnimationFrame ราว 60 ครั้ง/วินาที)
  // แยกจาก loop(): ภาพวิดีโอลื่นเสมอ ส่วนอีโมจิจะอัปเดตตามผลล่าสุดที่เซิร์ฟเวอร์ตอบ
  draw() {
    const { ctx, canvas, video } = this;
    const W = canvas.width, H = canvas.height;

    // วาดภาพกล้องแบบกลับซ้าย-ขวา (เหมือนส่องกระจก)
    ctx.save(); ctx.translate(W, 0); ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, W, H);
    ctx.restore();

    const d = this.latest;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';

    // อีโมจิอารมณ์: วาดทับตรงกลางกรอบใบหน้า
    if (d && d.face) {
      const b = d.box;  // กรอบหน้าจากเซิร์ฟเวอร์ เป็นสัดส่วน 0-1 ของภาพ
      // ภาพถูกพลิกซ้าย-ขวาแล้ว จึงต้องใช้ 1 - (x กลางกรอบ) เพื่อให้อีโมจิตรงกับหน้า
      const cx = (1 - (b.x + b.w / 2)) * W, cy = (b.y + b.h / 2) * H;
      const size = Math.max(b.w * W, b.h * H) * 1.3;   // ใหญ่กว่ากรอบหน้า 30% ให้คลุมทั้งหน้า
      ctx.font = `${size}px ${EMOJI_FONT}`;
      ctx.fillText(EMOTION_EMOJI[d.emotion] || '🙂', cx, cy + size * 0.05);  // ไม่รู้จัก -> 🙂
    }

    // อีโมจิสิ่งของ: เรียงซ้อนขึ้นไปที่มุมขวาล่าง สูงสุด 3 ชิ้น (ตัวที่มั่นใจสุดอยู่ล่างสุด)
    if (d && d.objects) {
      d.objects.slice(0, 3).forEach((o, i) => {
        const em = OBJECT_EMOJI[o.name];
        if (!em) return;                                // สิ่งของที่ไม่มีอีโมจิในตาราง ข้ามไป
        ctx.font = `64px ${EMOJI_FONT}`;
        ctx.fillText(em, W - 50, H - 50 - i * 76);
      });
    }
    requestAnimationFrame(() => this.draw());          // เรียกตัวเองซ้ำเพื่อวาดเฟรมถัดไป
  }
}

// แปลงผลวิเคราะห์เป็นข้อความสำหรับแผงด้านข้าง
// คืนค่า { emo: "😄 ยิ้ม (87%)" หรือ "ไม่พบใบหน้า", objs: "🥤 แก้วน้ำ, 📱 มือถือ" หรือ "-" }
function describe(d) {
  const emo = d && d.face ? `${EMOTION_EMOJI[d.emotion]} ${EMOTION_TH[d.emotion]} (${Math.round(d.confidence * 100)}%)` : 'ไม่พบใบหน้า';
  const objs = d && d.objects && d.objects.length
    ? d.objects.slice(0, 3).map(o => `${OBJECT_EMOJI[o.name] || '▫️'} ${OBJECT_TH[o.name] || o.name}`).join(', ')
    : '-';   // ไม่มีสิ่งของ แสดงขีด
  return { emo, objs };
}