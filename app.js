/* =====================================================
   YOLOv5 실시간 객체 탐지 - Web App
   (COCO-SSD + TensorFlow.js 기반)
   ===================================================== */

// ─── STATE ───────────────────────────────────────────
const state = {
  model: null,
  stream: null,
  isRunning: false,
  animFrameId: null,
  frameCount: 0,
  totalDetections: 0,
  classCounts: {},
  fpsHistory: [],
  lastFrameTime: 0,
  frameSkip: 1,
  frameIndex: 0,
  confidence: 0.5,
  boxStyle: 'filled',
  lastDetections: []
};

// ─── DOM REFS ─────────────────────────────────────────
const video      = document.getElementById('webcam');
const canvas     = document.getElementById('detection-canvas');
const ctx        = canvas.getContext('2d');
const overlay    = document.getElementById('overlay');
const overlayText= document.getElementById('overlay-text');
const startScreen= document.getElementById('start-screen');
const progressBar= document.getElementById('progress-bar');
const videoWrapper = document.getElementById('video-wrapper');

const startBtn   = document.getElementById('start-btn');
const stopBtn    = document.getElementById('stop-btn');
const snapBtn    = document.getElementById('snapshot-btn');
const fsBtn      = document.getElementById('fullscreen-btn');
const resetBtn   = document.getElementById('reset-stats');

const fpsDisplay      = document.getElementById('fps-display');
const statTotal       = document.getElementById('stat-total');
const statCurrent     = document.getElementById('stat-current');
const statClasses     = document.getElementById('stat-classes');
const statFrames      = document.getElementById('stat-frames');
const detectionList   = document.getElementById('detection-list');
const classChart      = document.getElementById('class-chart');
const tfBackend       = document.getElementById('tf-backend');
const modelStatus     = document.getElementById('model-status');

const confSlider  = document.getElementById('confidence-slider');
const confVal     = document.getElementById('confidence-value');
const skipSlider  = document.getElementById('skip-slider');
const skipVal     = document.getElementById('skip-value');
const boxStyleSel = document.getElementById('box-style');

const modalBackdrop = document.getElementById('modal-backdrop');
const snapshotCanvas= document.getElementById('snapshot-canvas');
const downloadLink  = document.getElementById('download-link');
const modalClose    = document.getElementById('modal-close');
const modalCloseBtn = document.getElementById('modal-close-btn');

// ─── COLOR PALETTE ────────────────────────────────────
const CLASS_COLORS = [
  '#6366f1','#22c55e','#f97316','#06b6d4','#ec4899',
  '#eab308','#14b8a6','#a78bfa','#f43f5e','#84cc16',
  '#0ea5e9','#d946ef','#fb923c','#4ade80','#38bdf8',
  '#c084fc','#f87171','#a3e635','#67e8f9','#fde68a'
];
const classColorMap = {};

function getClassColor(className) {
  if (!classColorMap[className]) {
    const keys = Object.keys(classColorMap).length;
    classColorMap[className] = CLASS_COLORS[keys % CLASS_COLORS.length];
  }
  return classColorMap[className];
}

// ─── MODEL LOAD ───────────────────────────────────────
async function loadModel() {
  setModelStatus('loading', '모델 로딩 중...');
  overlayText.textContent = 'COCO-SSD 모델 로딩 중...';
  progressBar.style.animation = 'loading 2s ease-in-out infinite';

  try {
    // TF 백엔드 감지
    await tf.ready();
    const backend = tf.getBackend();
    tfBackend.textContent = backend.toUpperCase();

    state.model = await cocoSsd.load({ base: 'mobilenet_v2' });
    overlayText.textContent = '모델 로드 완료!';
    progressBar.style.animation = 'none';
    progressBar.style.width = '100%';

    setTimeout(() => {
      overlay.classList.add('hidden');
    }, 600);

    setModelStatus('ready', '준비 완료');
    showToast('<i class="fa-solid fa-check-circle"></i> 모델 로드 완료!');
  } catch (err) {
    overlayText.textContent = '모델 로드 실패: ' + err.message;
    setModelStatus('idle', '로드 실패');
    showToast('<i class="fa-solid fa-circle-exclamation"></i> 모델 로드 실패', true);
  }
}

// ─── CAMERA ───────────────────────────────────────────
async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false
    });
    video.srcObject = state.stream;
    await new Promise(res => video.onloadedmetadata = res);
    video.play();

    // 캔버스 크기 동기화
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;

    return true;
  } catch (err) {
    showToast('<i class="fa-solid fa-circle-exclamation"></i> 카메라 접근 실패: ' + err.message, true);
    return false;
  }
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
    video.srcObject = null;
  }
}

// ─── DETECTION LOOP ───────────────────────────────────
async function detectionLoop() {
  if (!state.isRunning) return;

  state.frameIndex++;
  if (state.frameIndex % state.frameSkip !== 0) {
    // 스킵된 프레임에도 이전 결과 그리기
    if (state.lastDetections.length) drawDetections(state.lastDetections);
    state.animFrameId = requestAnimationFrame(detectionLoop);
    return;
  }

  const now = performance.now();
  const delta = now - state.lastFrameTime;
  state.lastFrameTime = now;

  // FPS 계산 (이동평균)
  if (delta > 0) {
    const fps = Math.round(1000 / delta);
    state.fpsHistory.push(fps);
    if (state.fpsHistory.length > 20) state.fpsHistory.shift();
    const avgFps = Math.round(state.fpsHistory.reduce((a, b) => a + b, 0) / state.fpsHistory.length);
    fpsDisplay.textContent = avgFps;
  }

  try {
    // 캔버스 크기 동기화
    if (canvas.width !== video.videoWidth) canvas.width  = video.videoWidth;
    if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;

    const predictions = await state.model.detect(video, 10, state.confidence);
    state.lastDetections = predictions;
    state.frameCount++;
    statFrames.textContent = state.frameCount;

    // 누적 탐지 수
    state.totalDetections += predictions.length;
    statTotal.textContent = state.totalDetections.toLocaleString();
    statCurrent.textContent = predictions.length;

    // 클래스별 카운트
    predictions.forEach(p => {
      const cls = p.class;
      state.classCounts[cls] = (state.classCounts[cls] || 0) + 1;
      getClassColor(cls);
    });
    statClasses.textContent = Object.keys(state.classCounts).length;

    // 그리기
    drawDetections(predictions);

    // UI 업데이트 (매 프레임 업데이트 시 성능 저하 방지: 3프레임마다)
    if (state.frameCount % 2 === 0) {
      updateDetectionList(predictions);
      updateClassChart();
    }

  } catch (err) {
    console.warn('Detection error:', err);
  }

  state.animFrameId = requestAnimationFrame(detectionLoop);
}

// ─── DRAW DETECTIONS ─────────────────────────────────
function drawDetections(predictions) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  predictions.forEach(pred => {
    const [x, y, w, h] = pred.bbox;
    const label = pred.class;
    const conf  = (pred.score * 100).toFixed(1);
    const color = getClassColor(label);

    if (state.boxStyle === 'filled') {
      // 반투명 채움
      ctx.fillStyle = color + '22';
      ctx.fillRect(x, y, w, h);
      // 테두리
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
    } else if (state.boxStyle === 'outline') {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x, y, w, h);
    } else if (state.boxStyle === 'corner') {
      drawCornerBox(x, y, w, h, color);
    }

    // 라벨 배경
    ctx.font = `bold ${Math.max(12, w * 0.07)}px Inter, sans-serif`;
    const labelText = `${label} ${conf}%`;
    const textW = ctx.measureText(labelText).width;
    const labelH = Math.max(20, w * 0.09);
    const labelY = y > labelH + 4 ? y - labelH - 4 : y + 2;

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(x - 1, labelY, textW + 14, labelH + 6, 5);
    ctx.fill();

    ctx.fillStyle = '#fff';
    ctx.fillText(labelText, x + 6, labelY + labelH);
  });
}

function drawCornerBox(x, y, w, h, color) {
  const len = Math.min(w, h) * 0.2;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';

  const corners = [
    [x, y, x + len, y, x, y + len],
    [x + w, y, x + w - len, y, x + w, y + len],
    [x, y + h, x + len, y + h, x, y + h - len],
    [x + w, y + h, x + w - len, y + h, x + w, y + h - len]
  ];

  corners.forEach(([sx, sy, ex1, ey1, ex2, ey2]) => {
    ctx.beginPath();
    ctx.moveTo(ex1, ey1);
    ctx.lineTo(sx, sy);
    ctx.lineTo(ex2, ey2);
    ctx.stroke();
  });
}

// ─── UI UPDATES ───────────────────────────────────────
function updateDetectionList(predictions) {
  if (!predictions.length) {
    detectionList.innerHTML = `
      <div class="empty-msg">
        <i class="fa-regular fa-face-meh"></i>
        <p>탐지된 객체가 없습니다</p>
      </div>`;
    return;
  }

  // 신뢰도 내림차순 정렬
  const sorted = [...predictions].sort((a, b) => b.score - a.score);

  detectionList.innerHTML = sorted.map(p => {
    const color = getClassColor(p.class);
    const confPct = (p.score * 100).toFixed(1);
    return `
      <div class="detection-item">
        <span class="det-color" style="background:${color}"></span>
        <span class="det-name">${p.class}</span>
        <span class="det-conf">${confPct}%</span>
        <div class="det-bar-wrap">
          <div class="det-bar" style="width:${confPct}%;background:${color}"></div>
        </div>
      </div>`;
  }).join('');
}

function updateClassChart() {
  const sorted = Object.entries(state.classCounts)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 12);

  if (!sorted.length) {
    classChart.innerHTML = '<p class="chart-empty">아직 탐지 데이터 없음</p>';
    return;
  }

  const maxVal = sorted[0][1];
  classChart.innerHTML = sorted.map(([cls, cnt]) => {
    const pct = (cnt / maxVal * 100).toFixed(1);
    const color = getClassColor(cls);
    return `
      <div class="chart-row">
        <span class="chart-label">${cls}</span>
        <div class="chart-bar-outer">
          <div class="chart-bar-inner" style="width:${pct}%;background:linear-gradient(90deg,${color},${color}cc)"></div>
        </div>
        <span class="chart-count">${cnt}</span>
      </div>`;
  }).join('');
}

function setModelStatus(type, text) {
  modelStatus.innerHTML = `<span class="status-dot ${type}"></span> ${text}`;
}

// ─── CONTROLS ────────────────────────────────────────
startBtn.addEventListener('click', async () => {
  if (!state.model) {
    showToast('<i class="fa-solid fa-circle-exclamation"></i> 모델 로딩 중입니다...', true);
    return;
  }

  const ok = await startCamera();
  if (!ok) return;

  state.isRunning = true;
  state.frameCount = 0;
  state.fpsHistory = [];
  state.lastFrameTime = performance.now();

  startScreen.classList.add('hidden');
  videoWrapper.classList.add('active');

  startBtn.disabled = true;
  stopBtn.disabled = false;
  snapBtn.disabled = false;
  fsBtn.disabled = false;

  setModelStatus('running', '탐지 중...');
  detectionLoop();
  showToast('<i class="fa-solid fa-play"></i> 탐지 시작!');
});

stopBtn.addEventListener('click', () => {
  state.isRunning = false;
  if (state.animFrameId) cancelAnimationFrame(state.animFrameId);
  stopCamera();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  startScreen.classList.remove('hidden');
  videoWrapper.classList.remove('active');

  startBtn.disabled = false;
  stopBtn.disabled = true;
  snapBtn.disabled = true;
  fsBtn.disabled = true;

  fpsDisplay.textContent = '0';
  statCurrent.textContent = '0';
  detectionList.innerHTML = `
    <div class="empty-msg">
      <i class="fa-regular fa-face-meh"></i>
      <p>탐지된 객체가 없습니다</p>
    </div>`;

  setModelStatus('ready', '준비 완료');
  showToast('<i class="fa-solid fa-stop"></i> 탐지 중지');
});

snapBtn.addEventListener('click', () => {
  const sc = snapshotCanvas;
  sc.width  = video.videoWidth;
  sc.height = video.videoHeight;
  const sctx = sc.getContext('2d');

  // 비디오 + 탐지 결과 합성
  sctx.drawImage(video, 0, 0);
  sctx.drawImage(canvas, 0, 0);

  // 다운로드 링크 설정
  const dataURL = sc.toDataURL('image/png');
  downloadLink.href = dataURL;
  downloadLink.download = `yolo_detection_${Date.now()}.png`;

  modalBackdrop.classList.add('visible');
  showToast('<i class="fa-solid fa-camera-retro"></i> 스냅샷 촬영!');
});

fsBtn.addEventListener('click', () => {
  if (!document.fullscreenElement) {
    videoWrapper.requestFullscreen?.();
    fsBtn.innerHTML = '<i class="fa-solid fa-compress"></i>';
  } else {
    document.exitFullscreen?.();
    fsBtn.innerHTML = '<i class="fa-solid fa-expand"></i>';
  }
});

resetBtn.addEventListener('click', () => {
  state.classCounts = {};
  state.totalDetections = 0;
  state.frameCount = 0;
  statTotal.textContent = '0';
  statClasses.textContent = '0';
  statFrames.textContent = '0';
  classChart.innerHTML = '<p class="chart-empty">아직 탐지 데이터 없음</p>';
  showToast('<i class="fa-solid fa-rotate"></i> 통계 초기화 완료');
});

// 모달 닫기
[modalClose, modalCloseBtn].forEach(btn => {
  btn.addEventListener('click', () => modalBackdrop.classList.remove('visible'));
});
modalBackdrop.addEventListener('click', (e) => {
  if (e.target === modalBackdrop) modalBackdrop.classList.remove('visible');
});

// 설정 슬라이더
confSlider.addEventListener('input', () => {
  state.confidence = parseFloat(confSlider.value);
  confVal.textContent = state.confidence.toFixed(2);
});
skipSlider.addEventListener('input', () => {
  state.frameSkip = parseInt(skipSlider.value);
  skipVal.textContent = state.frameSkip;
});
boxStyleSel.addEventListener('change', () => {
  state.boxStyle = boxStyleSel.value;
});

// 전체화면 변경 시 아이콘 업데이트
document.addEventListener('fullscreenchange', () => {
  if (!document.fullscreenElement) {
    fsBtn.innerHTML = '<i class="fa-solid fa-expand"></i>';
  }
});

// ─── TOAST ───────────────────────────────────────────
function showToast(html, isError = false) {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = 'toast' + (isError ? ' error' : '');
  toast.innerHTML = html;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

// ─── INIT ─────────────────────────────────────────────
(async () => {
  // 시작 화면 표시, 오버레이는 숨기기
  overlay.classList.remove('hidden');
  overlayText.textContent = 'TensorFlow.js 초기화 중...';
  await loadModel();
})();
