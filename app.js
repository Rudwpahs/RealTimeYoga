import {
  FilesetResolver,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35";

const MODEL_URL = "./assets/pose_landmarker_lite.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";

const checks = [
  ["Right knee", 24, 26, 28, 180],
  ["Left knee", 23, 25, 27, 270],
  ["Left side", 11, 23, 27, 180],
  ["Right side", 12, 24, 28, 180],
  ["Left elbow", 11, 13, 15, 315],
  ["Right elbow", 12, 14, 16, 45],
  ["Right wrist A", 16, 20, 15, 300],
  ["Right wrist B", 16, 19, 15, 300],
  ["Right shoulder", 24, 12, 14, 45],
  ["Left shoulder", 23, 11, 13, 315],
  ["Left hand", 13, 15, 19, 240],
  ["Right arm", 12, 14, 16, 45],
];

const state = {
  poseLandmarker: null,
  runningMode: "VIDEO",
  stream: null,
  activeSource: "idle",
  lastVideoTime: -1,
  holdStart: null,
  successPlayed: false,
  threshold: 20,
  demoImage: null,
  demoResult: null,
};

const els = {
  video: document.getElementById("video"),
  canvas: document.getElementById("output"),
  status: document.getElementById("statusText"),
  holdSeconds: document.getElementById("holdSeconds"),
  cameraButton: document.getElementById("cameraButton"),
  demoButton: document.getElementById("demoButton"),
  stopButton: document.getElementById("stopButton"),
  referenceSelect: document.getElementById("referenceSelect"),
  referenceImage: document.getElementById("referenceImage"),
  thresholdRange: document.getElementById("thresholdRange"),
  thresholdValue: document.getElementById("thresholdValue"),
  scoreText: document.getElementById("scoreText"),
  resultText: document.getElementById("resultText"),
  checkList: document.getElementById("checkList"),
};

const ctx = els.canvas.getContext("2d");
const successAudio = new Audio("./assets/audio1.wav");

function setStatus(text) {
  els.status.textContent = text;
}

function setupCheckList() {
  els.checkList.innerHTML = "";
  checks.forEach(([name], index) => {
    const row = document.createElement("div");
    row.className = "check";
    row.id = `check-${index}`;
    row.innerHTML = `
      <span class="dot" aria-hidden="true"></span>
      <span class="check-name">${name}</span>
      <span class="check-diff">--</span>
    `;
    els.checkList.appendChild(row);
  });
}

async function initPoseLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  state.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
    },
    runningMode: state.runningMode,
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  setStatus("Ready. Start camera or use the demo image.");
  els.cameraButton.disabled = false;
  els.demoButton.disabled = false;
}

async function setRunningMode(mode) {
  if (state.runningMode !== mode) {
    state.runningMode = mode;
    await state.poseLandmarker.setOptions({ runningMode: mode });
  }
}

function syncCanvas(width, height) {
  if (els.canvas.width !== width || els.canvas.height !== height) {
    els.canvas.width = width;
    els.canvas.height = height;
  }
}

function drawEmpty() {
  syncCanvas(640, 480);
  ctx.fillStyle = "#101820";
  ctx.fillRect(0, 0, els.canvas.width, els.canvas.height);
  ctx.fillStyle = "#d8dee5";
  ctx.font = "18px Arial";
  ctx.textAlign = "center";
  ctx.fillText("Start camera or run the demo image", els.canvas.width / 2, els.canvas.height / 2);
}

function landmarkVisible(landmark) {
  return !landmark || landmark.visibility === undefined || landmark.visibility > 0.35;
}

function drawPose(landmarks, width, height) {
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(255,255,255,0.88)";
  PoseLandmarker.POSE_CONNECTIONS.forEach((connection) => {
    const start = landmarks[connection.start];
    const end = landmarks[connection.end];
    if (!start || !end || !landmarkVisible(start) || !landmarkVisible(end)) {
      return;
    }
    ctx.beginPath();
    ctx.moveTo(start.x * width, start.y * height);
    ctx.lineTo(end.x * width, end.y * height);
    ctx.stroke();
  });

  landmarks.forEach((landmark) => {
    if (!landmarkVisible(landmark)) {
      return;
    }
    ctx.fillStyle = "#f9fbfb";
    ctx.beginPath();
    ctx.arc(landmark.x * width, landmark.y * height, 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

function calculateAngle(a, b, c) {
  const angle1 = Math.atan2(c.y - b.y, c.x - b.x);
  const angle2 = Math.atan2(a.y - b.y, a.x - b.x);
  let angle = (angle1 - angle2) * 180 / Math.PI;
  if (angle < 0) {
    angle += 360;
  }
  return angle;
}

function angleDiff(angle, reference) {
  let diff = angle - reference;
  if (diff < 0) {
    diff += 360;
  }
  if (diff > 180) {
    diff = 360 - diff;
  }
  return diff;
}

function updateHold(score, hasPose = false) {
  if (score > 10) {
    if (!state.holdStart) {
      state.holdStart = performance.now();
      state.successPlayed = false;
    }
    const seconds = Math.min(10, Math.floor((performance.now() - state.holdStart) / 1000));
    els.holdSeconds.textContent = String(seconds);
    if (seconds >= 10) {
      els.resultText.textContent = "Great job";
      if (!state.successPlayed) {
        state.successPlayed = true;
        successAudio.currentTime = 0;
        successAudio.play().catch(() => {});
      }
    } else {
      els.resultText.textContent = "Hold the pose";
    }
  } else {
    state.holdStart = null;
    state.successPlayed = false;
    els.holdSeconds.textContent = "0";
    els.resultText.textContent = hasPose ? "Adjust highlighted joints" : "No pose yet";
  }
}

function processResult(result, width, height) {
  const landmarks = result?.landmarks?.[0];
  if (!landmarks) {
    updateChecks([]);
    updateHold(0, false);
    els.scoreText.textContent = "0/12";
    return;
  }

  drawPose(landmarks, width, height);

  const outcomes = checks.map((check) => {
    const [, aIndex, bIndex, cIndex, reference] = check;
    const angle = calculateAngle(landmarks[aIndex], landmarks[bIndex], landmarks[cIndex]);
    const diff = angleDiff(angle, reference);
    return {
      success: diff < state.threshold,
      diff,
      landmark: landmarks[bIndex],
    };
  });

  outcomes.forEach((outcome) => {
    if (!outcome.landmark) {
      return;
    }
    ctx.strokeStyle = outcome.success ? "#2563eb" : "#dc2626";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(outcome.landmark.x * width, outcome.landmark.y * height, 12, 0, Math.PI * 2);
    ctx.stroke();
  });

  updateChecks(outcomes);
  const score = outcomes.filter((outcome) => outcome.success).length;
  els.scoreText.textContent = `${score}/12`;
  updateHold(score, true);
}

function updateChecks(outcomes) {
  checks.forEach((_, index) => {
    const row = document.getElementById(`check-${index}`);
    const diff = row.querySelector(".check-diff");
    const outcome = outcomes[index];
    row.classList.remove("good", "bad");
    if (!outcome) {
      diff.textContent = "--";
      return;
    }
    row.classList.add(outcome.success ? "good" : "bad");
    diff.textContent = `${Math.round(outcome.diff)} deg`;
  });
}

async function startCamera() {
  await stopActiveSource();
  await setRunningMode("VIDEO");
  setStatus("Requesting camera...");
  state.stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user",
    },
  });
  els.video.srcObject = state.stream;
  await els.video.play();
  state.activeSource = "camera";
  state.lastVideoTime = -1;
  setStatus("Camera running. Match the reference pose.");
  renderCamera();
}

function renderCamera() {
  if (state.activeSource !== "camera") {
    return;
  }
  const width = els.video.videoWidth || 640;
  const height = els.video.videoHeight || 480;
  syncCanvas(width, height);
  ctx.drawImage(els.video, 0, 0, width, height);

  if (els.video.currentTime !== state.lastVideoTime) {
    const result = state.poseLandmarker.detectForVideo(els.video, performance.now());
    processResult(result, width, height);
    state.lastVideoTime = els.video.currentTime;
  }
  requestAnimationFrame(renderCamera);
}

async function startDemo() {
  await stopActiveSource();
  await setRunningMode("IMAGE");
  setStatus("Demo image running.");
  state.demoImage = new Image();
  state.demoImage.src = "./assets/easy.png";
  await state.demoImage.decode();
  state.demoResult = state.poseLandmarker.detect(state.demoImage);
  state.activeSource = "demo";
  renderDemo();
}

function renderDemo() {
  if (state.activeSource !== "demo") {
    return;
  }
  const width = state.demoImage.naturalWidth || 480;
  const height = state.demoImage.naturalHeight || 480;
  syncCanvas(width, height);
  ctx.drawImage(state.demoImage, 0, 0, width, height);
  processResult(state.demoResult, width, height);
  requestAnimationFrame(renderDemo);
}

async function stopActiveSource() {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  state.activeSource = "idle";
  state.holdStart = null;
  state.successPlayed = false;
  els.holdSeconds.textContent = "0";
  els.scoreText.textContent = "0/12";
  els.resultText.textContent = "No pose yet";
  updateChecks([]);
  drawEmpty();
  setStatus("Ready. Start camera or use the demo image.");
}

els.cameraButton.disabled = true;
els.demoButton.disabled = true;
els.stopButton.addEventListener("click", stopActiveSource);
els.cameraButton.addEventListener("click", () => {
  startCamera().catch((error) => {
    console.error(error);
    setStatus("Camera unavailable. Use demo image or check browser permission.");
    stopActiveSource();
  });
});
els.demoButton.addEventListener("click", () => {
  startDemo().catch((error) => {
    console.error(error);
    setStatus("Demo image failed to load.");
  });
});
els.referenceSelect.addEventListener("change", (event) => {
  els.referenceImage.src = `./assets/${event.target.value}`;
});
els.thresholdRange.addEventListener("input", (event) => {
  state.threshold = Number(event.target.value);
  els.thresholdValue.textContent = `${state.threshold} deg`;
});

setupCheckList();
drawEmpty();
initPoseLandmarker().catch((error) => {
  console.error(error);
  setStatus("Pose model failed to load.");
});
