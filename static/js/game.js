/* =====================================================================
 * game.js  -  ตรรกะมินิเกม "Emoji Challenge" (หน้า /game)
 *
 * วิธีเล่น: ระบบสุ่มโจทย์ (อีโมจิ + สีหน้า + สิ่งของ) -> ผู้เล่นทำหน้า/ถือของให้ตรง
 *           ก่อนหมดเวลา ได้คะแนน ถ้าไม่ทันเสียชีวิต 1 ดวง ชีวิตหมดเกมจบ แล้วบันทึกคะแนน
 *
 * พึ่งพา: common.js (EmojiCam, describe) และ id ต่างๆ ใน templates/game.html
 * ===================================================================== */

// ---- ตั้งค่าเกม -----------------------------------------------------------
const ROUND_SECONDS = 15;   // เวลาต่อโจทย์ (ปรับเป็น 5 ได้ถ้าอยากโหดขึ้น)
const MAX_LIVES = 3;        // จำนวนชีวิตตอนเริ่มเกม
const MIN_CONF = 0.5;       // ความมั่นใจขั้นต่ำของอารมณ์
const NEED_HITS = 2;        // ต้องตรงติดกันกี่เฟรมถึงนับ (กันเดาผิดแวบเดียว)

// โจทย์: ของทุกชิ้นหาได้ในชีวิตประจำวัน และอยู่ใน 80 คลาสของ YOLO
//   emoji   = อีโมจิใหญ่ที่แสดงบนการ์ดโจทย์
//   emotion = ชื่ออารมณ์ที่ต้องทำ (ต้องตรงกับ labels.json ของโมเดล)
//   object  = ชื่อสิ่งของตาม YOLO ที่ต้องถือ หรือ null ถ้าโจทย์นั้นใช้แค่สีหน้า
//   text    = ข้อความคำสั่งที่ผู้เล่นเห็น
const CHALLENGES = [
  // ---- มือถือ (cell phone) ----
  { emoji: '📱', emotion: 'happy',     object: 'cell phone',  text: 'ยิ้ม + ถือมือถือ' },
  { emoji: '📱', emotion: 'surprised', object: 'cell phone',  text: 'ทำหน้าตกใจ + ถือมือถือ' },
  { emoji: '📱', emotion: 'neutral',   object: 'cell phone',  text: 'หน้านิ่ง + ถือมือถือ' },
  { emoji: '📱', emotion: 'sad',       object: 'cell phone',  text: 'ทำหน้าเศร้า + ถือมือถือ' },
  { emoji: '📱', emotion: 'angry',     object: 'cell phone',  text: 'ทำหน้าโกรธ + ถือมือถือ' },
  // ---- เมาส์ (mouse) ----
  { emoji: '🖱️', emotion: 'happy',     object: 'mouse',       text: 'ยิ้ม + ถือเมาส์' },
  { emoji: '🖱️', emotion: 'surprised', object: 'mouse',       text: 'ทำหน้าตกใจ + ถือเมาส์' },
  { emoji: '🖱️', emotion: 'neutral',   object: 'mouse',       text: 'หน้านิ่ง + ถือเมาส์' },
  { emoji: '🖱️', emotion: 'sad',       object: 'mouse',       text: 'ทำหน้าเศร้า + ถือเมาส์' },
  { emoji: '🖱️', emotion: 'angry',     object: 'mouse',       text: 'ทำหน้าโกรธ + ถือเมาส์' },
  // ---- คีย์บอร์ด (keyboard) ----
  { emoji: '⌨️', emotion: 'happy',     object: 'keyboard',    text: 'ยิ้ม + ยกคีย์บอร์ดโชว์' },
  { emoji: '⌨️', emotion: 'surprised', object: 'keyboard',    text: 'ทำหน้าตกใจ + ยกคีย์บอร์ดโชว์' },
  { emoji: '⌨️', emotion: 'neutral',   object: 'keyboard',    text: 'หน้านิ่ง + ยกคีย์บอร์ดโชว์' },
  { emoji: '⌨️', emotion: 'sad',       object: 'keyboard',    text: 'ทำหน้าเศร้า + ยกคีย์บอร์ดโชว์' },
  { emoji: '⌨️', emotion: 'angry',     object: 'keyboard',    text: 'ทำหน้าโกรธ + ยกคีย์บอร์ดโชว์' },
  // ---- ของใช้อื่นๆ ----
  { emoji: '🍺', emotion: 'happy',     object: 'bottle',      text: 'ยิ้ม + ถือขวดน้ำ' },
  { emoji: '🍺', emotion: 'surprised', object: 'bottle',      text: 'ทำหน้าตกใจ + ถือขวดน้ำ' },
  { emoji: '🥤', emotion: 'happy',     object: 'cup',         text: 'ยิ้ม + ถือแก้วน้ำ' },
  { emoji: '🥤', emotion: 'sad',       object: 'cup',         text: 'ทำหน้าเศร้า + ถือแก้วน้ำ' },
  { emoji: '📖', emotion: 'neutral',   object: 'book',        text: 'หน้านิ่ง + ถือหนังสือ' },
  { emoji: '📖', emotion: 'sad',       object: 'book',        text: 'ทำหน้าเศร้า + ถือหนังสือ' },
  { emoji: '📺', emotion: 'neutral',   object: 'remote',      text: 'หน้านิ่ง + ถือรีโมท' },
  { emoji: '📺', emotion: 'angry',     object: 'remote',      text: 'ทำหน้าโกรธ + ถือรีโมท' },
  { emoji: '🥄', emotion: 'surprised', object: 'spoon',       text: 'ทำหน้าตกใจ + ถือช้อน' },
  { emoji: '☂️', emotion: 'happy',     object: 'umbrella',    text: 'ยิ้ม + ถือร่ม' },
  // ---- สีหน้าอย่างเดียว ----
  { emoji: '😮', emotion: 'surprised', object: null,          text: 'ทำหน้าตกใจ' },
  { emoji: '😄', emotion: 'happy',     object: null,          text: 'ยิ้มกว้างๆ' },
  { emoji: '😐', emotion: 'neutral',   object: null,          text: 'หน้านิ่งๆ ห้ามขยับ' },
  { emoji: '😢', emotion: 'sad',       object: null,          text: 'ทำหน้าเศร้า' },
  { emoji: '😠', emotion: 'angry',     object: null,          text: 'ทำหน้าโกรธ' },
];

// ---- ตัวแปรสถานะของเกม -----------------------------------------------------
const $ = id => document.getElementById(id);               // ตัวช่วยย่อ: หา element จาก id
const cam = new EmojiCam($('video'), $('canvas'));         // ตัวควบคุมกล้อง (จาก common.js)

// state: 'idle' = ยังไม่เริ่ม | 'playing' = กำลังนับถอยหลัง | 'between' = พักหลังผ่าน/ไม่ผ่าน | 'over' = จบเกม
let state = 'idle', score = 0, lives = MAX_LIVES, timeLeft = ROUND_SECONDS;
// current = โจทย์ปัจจุบัน, timer = ตัวนับเวลา (setInterval), hits = จำนวนเฟรมติดกันที่ตอบถูก,
// started = เปิดกล้องและเริ่มลูปส่งภาพไปแล้วหรือยัง (ทำครั้งเดียวต่อหน้า)
let current = null, timer = null, hits = 0, started = false;
let availableEmotions = null;   // ได้จาก /api/labels ว่าโมเดลมีอารมณ์อะไรบ้าง

// เลือกเฉพาะโจทย์ที่โมเดลทำได้จริง (เช่น โมเดล 7 คลาสไม่มี wink/kiss จึงตัดโจทย์ขยิบตา/จูบออก)
// ถ้ากรองแล้วว่างเปล่า ใช้โจทย์ทั้งหมดแทน
function pool() {
  const p = availableEmotions ? CHALLENGES.filter(c => availableEmotions.includes(c.emotion)) : CHALLENGES;
  return p.length ? p : CHALLENGES;
}

// ---- รับผลจากกล้อง --------------------------------------------------------
cam.onError = msg => { $('err').textContent = msg; };      // โชว์ error จากเซิร์ฟเวอร์ใต้แผง

// ถูกเรียกทุกครั้งที่เซิร์ฟเวอร์ตอบผลวิเคราะห์ภาพ 1 เฟรม
cam.onResult = d => {
  const t = describe(d);
  $('seen').textContent = `${t.emo} | ${t.objs}`;          // แสดงสิ่งที่ระบบตรวจเจอ ณ ตอนนี้
  if (state !== 'playing' || !current) return;              // นอกช่วงเล่น ไม่ต้องตรวจโจทย์

  // เงื่อนไขผ่านโจทย์: เจอหน้า + อารมณ์ตรง + มั่นใจพอ + (ถ้าโจทย์มีของ) เจอของนั้นในภาพ
  const ok = d.face && d.emotion === current.emotion && d.confidence >= MIN_CONF &&
    (!current.object || d.objects.some(o => o.name === current.object));
  hits = ok ? hits + 1 : 0;                                 // ตรงติดกันถึงนับ หลุดเมื่อไรเริ่มนับใหม่
  if (hits >= NEED_HITS) success();
};

// ---- ส่วนแสดงผล ------------------------------------------------------------
// อัปเดตแถบคะแนน / ชีวิต / เวลา / แถบเวลาที่หดลง
function renderHud() {
  $('score').textContent = score;
  $('lives').textContent = '❤️'.repeat(lives) + '🖤'.repeat(MAX_LIVES - lives);   // หัวใจเต็ม + หัวใจดำที่เสียไป
  $('time').textContent = state === 'playing' ? timeLeft + ' วิ' : '-';
  $('bar').style.width = (state === 'playing' ? (timeLeft / ROUND_SECONDS) * 100 : 0) + '%';
}

// แสดงข้อความสั้นๆ บนภาพกล้อง (เช่น ✅ +22 หรือ ⏰ หมดเวลา) แล้วซ่อนเองหลัง ms มิลลิวินาที
function flash(msg, ms = 900) {
  const el = $('flash');
  el.textContent = msg; el.hidden = false;
  setTimeout(() => { el.hidden = true; }, ms);
}

// ---- วงจรของเกม -------------------------------------------------------------
// เริ่มโจทย์ใหม่: สุ่มโจทย์ (ไม่ซ้ำอันเดิมติดกัน) แล้วเริ่มนับถอยหลัง
function nextChallenge() {
  let c;
  const list = pool();
  do { c = list[Math.floor(Math.random() * list.length)]; } while (c === current && list.length > 1);
  current = c; hits = 0; timeLeft = ROUND_SECONDS; state = 'playing';
  cam.wantObjects = !!c.object;   // โจทย์ที่ไม่มีของ ไม่ต้องรัน YOLO -> เร็วขึ้น
  $('chEmoji').textContent = c.emoji;
  $('chText').textContent = c.text;
  renderHud();
  clearInterval(timer);           // เคลียร์ตัวนับเก่าก่อน กันนับซ้อนสองตัว
  timer = setInterval(() => { timeLeft--; renderHud(); if (timeLeft <= 0) fail(); }, 1000);   // ทุก 1 วินาที
}

// ผ่านโจทย์: คะแนน = 10 + เวลาที่เหลือ (ยิ่งเร็วยิ่งได้เยอะ) แล้วรอ 1 วินาทีก่อนโจทย์ถัดไป
function success() {
  clearInterval(timer);
  state = 'between';              // กันไม่ให้ผลภาพเฟรมถัดไปนับซ้ำ
  const gain = 10 + timeLeft;
  score += gain;
  flash(`✅ +${gain}`);
  renderHud();
  setTimeout(nextChallenge, 1000);
}

// หมดเวลา: เสียชีวิต 1 ดวง ถ้ายังเหลือไปต่อ ถ้าหมดแล้วจบเกม
function fail() {
  clearInterval(timer);
  state = 'between';
  lives--;
  flash('⏰ หมดเวลา');
  renderHud();
  if (lives <= 0) return setTimeout(gameOver, 1000);
  setTimeout(nextChallenge, 1000);
}

// จบเกม: แสดงสรุป, โชว์ปุ่มเล่นอีกครั้ง, ส่งคะแนนไปบันทึกในฐานข้อมูล, แล้วโหลดตารางอันดับใหม่
async function gameOver() {
  state = 'over';
  $('chEmoji').textContent = '🏁';
  $('chText').textContent = `จบเกม! คุณได้ ${score} คะแนน`;
  $('startBtn').textContent = 'เล่นอีกครั้ง';
  $('startBtn').hidden = false;
  renderHud();
  try {
    await fetch('/api/score', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ score }) });
  } catch (e) { console.error(e); }
  loadBoard();
}

// โหลดตารางอันดับสูงสุด 10 คน + คะแนนสูงสุดของตัวเองจากเซิร์ฟเวอร์
async function loadBoard() {
  try {
    const d = await (await fetch('/api/leaderboard')).json();
    $('mine').textContent = d.mine;
    $('board').innerHTML = d.top.length
      // ตัดอักขระ < > & ออกจากชื่อผู้ใช้ กันแทรกโค้ด HTML ลงหน้าเว็บ
      ? d.top.map(r => `<li><span>${r.username.replace(/[<>&]/g, '')}</span><b>${r.best}</b></li>`).join('')
      : '<li>ยังไม่มีคะแนน เป็นคนแรกเลย</li>';
  } catch (e) { console.error(e); }
}

// ---- เริ่มเกม ---------------------------------------------------------------
// ปุ่ม "เริ่มเกม" / "เล่นอีกครั้ง": ครั้งแรกเปิดกล้อง + เริ่มส่งภาพ ครั้งต่อไปรีเซ็ตแค่คะแนนกับชีวิต
$('startBtn').addEventListener('click', async () => {
  try {
    if (!started) { await cam.start(); cam.draw(); cam.loop(); started = true; }
  } catch (e) {
    $('err').textContent = 'เปิดกล้องไม่ได้ - อนุญาตการใช้กล้องแล้วลองใหม่ (ต้องเปิดผ่าน localhost หรือ https)';
    return;
  }
  score = 0; lives = MAX_LIVES; current = null;
  $('startBtn').hidden = true;
  nextChallenge();
});

// ทำงานทันทีที่โหลดหน้า: วาดแถบสถานะเริ่มต้น, โหลดอันดับ, และขอรายชื่ออารมณ์ที่โมเดลรู้จัก
renderHud();
loadBoard();
fetch('/api/labels').then(r => r.json()).then(d => { if (d.labels && d.labels.length) availableEmotions = d.labels; }).catch(() => {});