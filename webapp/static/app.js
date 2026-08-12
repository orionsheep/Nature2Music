/* Nature2Music 前端逻辑 — 无外部依赖，波形用 Web Audio + canvas 自绘 */

const $ = (sel) => document.querySelector(sel);

const GROUP_LABELS = {
  bird: "鸟类", insect: "昆虫", mammal: "哺乳",
  amphibian: "两栖", environment: "环境", unknown: "未知",
};

const GROUP_ICONS = {
  bird: '<svg viewBox="0 0 24 24"><path d="M3 15c3-4 6-4 9-1 2.5-4 6-5 9-2"/><path d="M12 14c-1 2-1 4 0 6"/></svg>',
  insect: '<svg viewBox="0 0 24 24"><ellipse cx="12" cy="13" rx="4" ry="6"/><path d="M12 7V4M8 10L4 7M16 10l4-3M8 15l-5 2M16 15l5 2M8 19l-4 3M16 19l4 3"/></svg>',
  mammal: '<svg viewBox="0 0 24 24"><circle cx="12" cy="14" r="5"/><circle cx="7" cy="7" r="1.6"/><circle cx="12" cy="5.5" r="1.6"/><circle cx="17" cy="7" r="1.6"/></svg>',
  amphibian: '<svg viewBox="0 0 24 24"><ellipse cx="12" cy="14" rx="7" ry="5"/><circle cx="8" cy="8" r="2"/><circle cx="16" cy="8" r="2"/><path d="M9 15c2 1.5 4 1.5 6 0"/></svg>',
  environment: '<svg viewBox="0 0 24 24"><path d="M2 12c2-5 4-5 6 0s4 5 6 0 4-5 6 0"/><path d="M2 18c2-3 4-3 6 0s4 3 6 0 4-3 6 0" opacity="0.5"/></svg>',
  unknown: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.8 2.1c-.8.5-1.3 1-1.3 2"/><circle cx="12" cy="16.5" r="0.4"/></svg>',
};

const STYLE_PRESETS = [
  { zh: "ambient 氛围", en: "ambient soundscape" },
  { zh: "中国风管弦", en: "ambient Chinese orchestral music" },
  { zh: "lo-fi 低保真", en: "lo-fi chill beats" },
  { zh: "电影配乐", en: "cinematic film score" },
  { zh: "电子 ambient", en: "electronic ambient" },
  { zh: "极简钢琴", en: "minimal piano ambient" },
];

const state = {
  identify: null,        // /api/identify 的返回
  sourcePeaks: null,     // 源音频波形 peaks
  sourceName: "",
  promptDirty: false,
  generating: false,
  taskId: null,
  pollTimer: null,
  elapsedTimer: null,
  resultAudioReady: false,
};

/* ---------- toast ---------- */

function toast(message, kind = "error", ms = 6000) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  $("#toast-root").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

/* ---------- 示例音频试听（顶层状态，识别完成后自动停止） ---------- */

let sampleAudio = null;
let samplePlayingChip = null;
function stopSamplePreview() {
  if (sampleAudio) { sampleAudio.pause(); sampleAudio = null; }
  if (samplePlayingChip) { samplePlayingChip.classList.remove("playing"); samplePlayingChip = null; }
}

async function api(path, options = {}) {
  const resp = await fetch(path, options);
  if (!resp.ok) {
    let detail = `${resp.status}`;
    try { detail = (await resp.json()).detail || detail; } catch { /* ignore */ }
    throw new Error(detail);
  }
  return resp.json();
}

/* ---------- 波形 ---------- */

let audioCtx = null;
async function decodePeaks(blob, buckets = 400) {
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  const buffer = await audioCtx.decodeAudioData(await blob.arrayBuffer());
  const data = buffer.getChannelData(0);
  const block = Math.floor(data.length / buckets) || 1;
  const peaks = [];
  for (let i = 0; i < buckets; i++) {
    let max = 0;
    const start = i * block;
    for (let j = start; j < Math.min(start + block, data.length); j += 8) {
      const v = Math.abs(data[j]);
      if (v > max) max = v;
    }
    peaks.push(max);
  }
  return { peaks, duration: buffer.duration };
}

const WAVE_INK = "rgba(247,247,239,0.82)";   // 源波形：浅暖白
const WAVE_FADED = "rgba(247,247,239,0.16)"; // 未播放部分：淡白
const WAVE_GRAD = ["#6fa287", "#b5c98b", "#eec27f"];  // 生成波形：雾绿→晨光金

function drawWave(canvas, peaks, { color = WAVE_INK, gradient = null, progress = 0, faded = null } = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  let fill = color;
  if (gradient) {
    fill = ctx.createLinearGradient(0, 0, w, 0);
    gradient.forEach((c, i) => fill.addColorStop(i / (gradient.length - 1), c));
  }
  const mid = h / 2;
  const n = peaks.length;
  const bw = w / n;
  for (let i = 0; i < n; i++) {
    const amp = Math.max(1.2, peaks[i] * (h - 6));
    ctx.fillStyle = faded && i / n > progress ? faded : fill;
    ctx.fillRect(i * bw, mid - amp / 2, Math.max(1, bw * 0.62), amp);
  }
}

/* 结果播放器波形（含进度与 seek） */
const player = {
  peaks: null,
  duration: 0,
  redraw() {
    if (!this.peaks) return;
    const el = $("#audio-el");
    const progress = this.duration ? el.currentTime / this.duration : 0;
    drawWave($("#result-wave"), this.peaks, { gradient: WAVE_GRAD, progress, faded: WAVE_FADED });
  },
};

/* ---------- 上传与识别 ---------- */

function setPhase(phase) {
  $("#panel-identifying").hidden = phase !== "identifying";
  $("#panel-recognition").hidden = !(phase === "recognized" || phase === "generating" || phase === "done");
  $("#panel-params").hidden = !(phase === "recognized");
  $("#panel-generating").hidden = phase !== "generating";
}

async function handleFile(file, displayName) {
  if (recorderState.active) { toast("正在录音，请先停止或完成录音", "info"); return; }
  const ext = (file.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
  if (![".wav", ".mp3", ".flac", ".ogg", ".m4a"].includes(ext)) {
    toast(`不支持的格式 ${ext || "(无扩展名)"}，请使用 WAV / MP3 / FLAC / OGG / M4A`);
    return;
  }
  if (file.size > 50 * 1024 * 1024) { toast("文件超过 50MB 限制"); return; }

  state.sourceName = displayName || file.name;
  // 本地先解码源波形（失败不阻塞识别）；解码完成时若面板已可见则直接补画
  decodePeaks(file).then(({ peaks }) => {
    state.sourcePeaks = peaks;
    if (!$("#panel-recognition").hidden) drawWave($("#source-wave"), peaks);
  }).catch(() => { state.sourcePeaks = null; });

  setPhase("identifying");
  const form = new FormData();
  form.append("file", file, file.name);
  try {
    const result = await api("/api/identify", { method: "POST", body: form });
    state.identify = result;
    state.promptDirty = false;
    renderRecognition(result);
    await refreshPrompt();
    stopSamplePreview();
    setPhase("recognized");
    // 面板从 hidden 变为可见后 canvas 才有尺寸，此时再画波形
    if (state.sourcePeaks) drawWave($("#source-wave"), state.sourcePeaks);
  } catch (err) {
    setPhase("empty");
    toast(`识别失败：${err.message}`);
  }
}

function renderRecognition({ recognition: r, audio_features: f }) {
  $("#species-icon").innerHTML = GROUP_ICONS[r.group] || GROUP_ICONS.unknown;
  $("#species-zh").textContent = r.common_name_zh || r.species;
  $("#species-en").textContent = r.species;
  $("#species-sci").textContent = r.scientific_name || "";
  const meta = [];
  meta.push(`类别 · ${GROUP_LABELS[r.group] || r.group}`);
  if (r.call_type) meta.push(`叫声类型 · ${r.call_type}`);
  if (r.background && r.background.length) meta.push(`背景声 · ${r.background.join("、")}`);
  $("#species-meta").textContent = meta.join("　");

  const pct = Math.round((r.confidence || 0) * 100);
  $("#confidence-value").textContent = `${pct}%`;
  $("#confidence-bar").style.width = `${pct}%`;
  $("#confidence-warn").hidden = (r.confidence || 0) >= 0.6;

  const loud = f.rms < 0.03 ? "偏弱" : f.rms < 0.12 ? "适中" : "充沛";
  const bright = f.spectral_centroid_hz < 1500 ? "偏暗沉" : f.spectral_centroid_hz < 4000 ? "均衡" : "明亮";
  const rows = [
    ["时长 / 采样率", `${f.duration_s} s · ${f.sample_rate} Hz`, ""],
    ["估计节奏", `${Math.round(f.estimated_bpm)} BPM`, "将作为生成音乐的速度基准"],
    ["主频", `${Math.round(f.dominant_frequency_hz)} Hz`, "主旋律轮廓将由此衍生"],
    ["响度", loud, `RMS ${f.rms}`],
    ["音色", bright, `频谱质心 ${Math.round(f.spectral_centroid_hz)} Hz`],
  ];
  $("#features-list").innerHTML = rows.map(([k, v, hint]) =>
    `<li><span class="k">${k}${hint ? `<span class="hint">${hint}</span>` : ""}</span><span class="v">${v}</span></li>`
  ).join("");

  if (state.sourcePeaks) drawWave($("#source-wave"), state.sourcePeaks);
}

/* ---------- 参数与提示词 ---------- */

function currentStyle() {
  return $("#style-input").value.trim() || "cinematic ambient world music";
}

async function refreshPrompt() {
  if (!state.identify) return;
  try {
    const p = await api("/api/preview-prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        recognition: state.identify.recognition,
        audio_features: state.identify.audio_features,
        style: currentStyle(),
      }),
    });
    $("#prompt-caption").textContent = p.caption;
    $("#prompt-cot").value = p.chain_of_thought;
    autoGrowCot();
    state.promptDirty = false;
    $("#prompt-edited-flag").hidden = true;
  } catch (err) {
    toast(`提示词构建失败：${err.message}`);
  }
}

/* ---------- CoT 提示词输入框自适应高度 ---------- */

function autoGrowCot() {
  const el = $("#prompt-cot");
  if (!el) return;
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight + 2, 320)}px`;
  el.style.overflowY = el.scrollHeight > 320 ? "auto" : "hidden";
}

/* ---------- 生成 ---------- */

async function startGenerate() {
  if (!state.identify || state.generating) return;
  state.generating = true;
  $("#btn-generate").disabled = true;
  const body = {
    recognition: state.identify.recognition,
    audio_features: state.identify.audio_features,
    style: currentStyle(),
    duration_s: Number($("#duration-slider").value),
    input_audio: state.sourceName,
  };
  if (state.promptDirty) {
    body.caption = $("#prompt-caption").textContent;
    body.chain_of_thought = $("#prompt-cot").value;
  }
  try {
    const { task_id } = await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.taskId = task_id;
    setPhase("generating");
    updateStages("pending");
    startElapsed();
    state.pollTimer = setInterval(pollTask, 1500);
    pollTask();
  } catch (err) {
    toast(err.message.includes("409") || err.message.includes("正在")
      ? "已有生成任务在进行中" : `提交生成失败：${err.message}`);
    state.generating = false;
    $("#btn-generate").disabled = false;
  }
}

function startElapsed() {
  const t0 = Date.now();
  const fmt = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  clearInterval(state.elapsedTimer);
  state.elapsedTimer = setInterval(() => { $("#gen-elapsed").textContent = fmt((Date.now() - t0) / 1000); }, 500);
}

const STAGE_ORDER = ["preparing", "extracting", "sampling", "finalizing"];
function updateStages(stage) {
  const idx = STAGE_ORDER.indexOf(stage);
  document.querySelectorAll("#stages-list li").forEach((li) => {
    const liIdx = STAGE_ORDER.indexOf(li.dataset.stage);
    li.classList.toggle("done", idx > liIdx || stage === "done");
    li.classList.toggle("active", idx === liIdx);
  });
}

async function pollTask() {
  if (!state.taskId) return;
  try {
    const task = await api(`/api/tasks/${state.taskId}`);
    updateStages(task.stage);
    if (task.stage === "done") {
      stopPolling();
      await showResult(task.id);
      setPhase("done");
      toast("生成完成", "info", 3000);
    } else if (task.stage === "failed") {
      stopPolling();
      setPhase("recognized");
      toast(`生成失败：${task.error || "未知错误"}`);
    } else if (task.stage === "cancelled") {
      stopPolling();
      setPhase("recognized");
      toast("已取消生成", "info");
    }
  } catch (err) {
    stopPolling();
    setPhase("recognized");
    toast(`任务查询失败：${err.message}`);
  }
}

function stopPolling() {
  clearInterval(state.pollTimer);
  clearInterval(state.elapsedTimer);
  state.pollTimer = null;
  state.generating = false;
  $("#btn-generate").disabled = false;
}

async function cancelTask() {
  if (!state.taskId) return;
  try {
    await api(`/api/tasks/${state.taskId}`, { method: "DELETE" });
  } catch (err) {
    toast(`取消失败：${err.message}`);
  }
}

/* ---------- 结果展示 ---------- */

function fmtTime(s) {
  if (!isFinite(s)) return "0:00";
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

async function showResult(taskId) {
  const audioUrl = `/api/results/${taskId}/audio`;
  const report = await api(`/api/results/${taskId}/report`);

  $("#result-empty").hidden = true;
  $("#result-body").hidden = false;
  const audioEl = $("#audio-el");
  audioEl.src = audioUrl;
  $("#dl-wav").href = audioUrl;
  $("#dl-json").href = `/api/results/${taskId}/report`;

  const blob = await (await fetch(audioUrl)).blob();
  try {
    const { peaks, duration } = await decodePeaks(blob);
    player.peaks = peaks;
    player.duration = duration;
    $("#time-total").textContent = fmtTime(duration);
    player.redraw();
  } catch {
    toast("波形解码失败，但仍可播放", "info");
  }

  // 源 / 生成波形对比
  if (state.sourcePeaks) {
    $("#compare-box").hidden = false;
    $("#compare-src-name").textContent = state.sourceName;
    drawWave($("#compare-src-wave"), state.sourcePeaks);
    if (player.peaks) drawWave($("#compare-gen-wave"), player.peaks, { gradient: WAVE_GRAD });
  } else {
    $("#compare-box").hidden = true;
  }

  // 创作回顾
  const r = report.recognition || {};
  const p = report.prompt || {};
  const rows = [
    ["源文件", state.sourceName || report.input_audio || "—"],
    ["识别", `${r.common_name_zh || r.species || "—"}（${r.species || ""}）· 置信度 ${Math.round((r.confidence || 0) * 100)}%`],
    ["风格", report.style || "—"],
    ["时长", `${report.duration_s || "—"} s · 44.1kHz 立体声`],
    ["caption", p.caption || "—"],
  ];
  $("#review-box").innerHTML = rows.map(([k, v]) =>
    `<div class="row"><span class="k">${k}</span><span class="v">${v}</span></div>`).join("");

  loadHistory();
}

/* ---------- 历史 ---------- */

async function loadHistory() {
  let list = [];
  try { list = await api("/api/history"); } catch { return; }
  const ul = $("#history-list");
  $("#history-empty").hidden = list.length > 0;
  ul.innerHTML = "";
  for (const item of list) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.className = "h-name";
    name.textContent = item.common_name_zh || item.species || "未知";
    const style = document.createElement("span");
    style.className = "h-style";
    style.textContent = item.style;
    const meta = document.createElement("span");
    meta.className = "h-meta";
    meta.textContent = `${item.duration_s}s · ${item.created_at}`;
    const del = document.createElement("button");
    del.className = "h-del";
    del.textContent = "×";
    del.title = "删除记录";
    del.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      try {
        await api(`/api/history/${item.id}`, { method: "DELETE" });
        loadHistory();
      } catch (err) { toast(`删除失败：${err.message}`); }
    });
    li.append(name, style, meta, del);
    li.addEventListener("click", async () => {
      if (state.generating) { toast("生成进行中，稍后再查看历史", "info"); return; }
      try { await showResult(item.id); setPhase("done"); }
      catch (err) { toast(`加载记录失败：${err.message}`); }
    });
    ul.appendChild(li);
  }
}

/* ---------- 事件绑定 ---------- */

function bindEvents() {
  const dz = $("#dropzone");
  const fi = $("#file-input");
  dz.addEventListener("click", () => fi.click());
  fi.addEventListener("change", () => { if (fi.files[0]) handleFile(fi.files[0]); fi.value = ""; });
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); }));
  dz.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });

  // 示例
  const SPK_ICON = '<svg class="spk" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M11 5 6 9H2v6h4l5 4V5Z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M18.5 5.5a9 9 0 0 1 0 13"/></svg>';
  api("/api/samples").then((samples) => {
    const row = $("#samples-row");
    for (const s of samples) {
      const chip = document.createElement("button");
      chip.className = "sample-chip";
      chip.innerHTML = SPK_ICON;
      chip.appendChild(document.createTextNode(s.name_zh));
      chip.addEventListener("click", async () => {
        if (samplePlayingChip === chip) { stopSamplePreview(); return; }
        stopSamplePreview();
        sampleAudio = new Audio(s.url);
        sampleAudio.addEventListener("ended", stopSamplePreview);
        sampleAudio.play().catch(() => {});
        chip.classList.add("playing");
        samplePlayingChip = chip;
        try {
          const blob = await (await fetch(s.url)).blob();
          handleFile(new File([blob], `${s.id}.wav`, { type: "audio/wav" }), `示例 · ${s.name_zh}`);
        } catch (err) { toast(`示例加载失败：${err.message}`); }
      });
      row.appendChild(chip);
    }
  }).catch(() => {});

  // 风格预设
  const chipsBox = $("#style-chips");
  for (const preset of STYLE_PRESETS) {
    const chip = document.createElement("button");
    chip.className = "style-chip";
    chip.textContent = preset.zh;
    chip.addEventListener("click", () => {
      $("#style-input").value = preset.en;
      chipsBox.querySelectorAll(".style-chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      refreshPrompt();
    });
    chipsBox.appendChild(chip);
  }
  let styleDebounce = null;
  $("#style-input").addEventListener("input", () => {
    chipsBox.querySelectorAll(".style-chip").forEach((c) => c.classList.remove("active"));
    clearTimeout(styleDebounce);
    styleDebounce = setTimeout(refreshPrompt, 500);
  });

  $("#duration-slider").addEventListener("input", () => {
    $("#duration-value").textContent = `${$("#duration-slider").value} 秒`;
  });

  $("#prompt-cot").addEventListener("input", () => {
    state.promptDirty = true;
    $("#prompt-edited-flag").hidden = false;
    autoGrowCot();
  });
  $("#prompt-cot").closest("details")?.addEventListener("toggle", (e) => {
    if (e.target.open) autoGrowCot();
  });
  $("#btn-refresh-prompt").addEventListener("click", refreshPrompt);

  $("#btn-generate").addEventListener("click", startGenerate);
  $("#btn-cancel").addEventListener("click", cancelTask);
  $("#btn-regenerate").addEventListener("click", () => {
    if (state.identify && !state.generating) {
      $("#audio-el").pause();
      setPhase("recognized");
      startGenerate();
    }
  });

  // 播放器
  const audioEl = $("#audio-el");
  $("#btn-play").addEventListener("click", () => {
    if (!audioEl.src) return;
    audioEl.paused ? audioEl.play() : audioEl.pause();
  });
  audioEl.addEventListener("play", () => { $("#icon-play").hidden = true; $("#icon-pause").hidden = false; });
  audioEl.addEventListener("pause", () => { $("#icon-play").hidden = false; $("#icon-pause").hidden = true; });
  audioEl.addEventListener("timeupdate", () => {
    $("#time-current").textContent = fmtTime(audioEl.currentTime);
    player.redraw();
  });
  audioEl.addEventListener("ended", () => player.redraw());
  $("#volume").addEventListener("input", () => { audioEl.volume = Number($("#volume").value); });
  $("#result-wave").addEventListener("click", (e) => {
    if (!player.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    audioEl.currentTime = ((e.clientX - rect.left) / rect.width) * player.duration;
  });

  // 弹窗
  $("#btn-help").addEventListener("click", () => { $("#help-modal").hidden = false; });
  $("#btn-help-close").addEventListener("click", () => { $("#help-modal").hidden = true; });
  $("#help-modal").addEventListener("click", (e) => { if (e.target.id === "help-modal") e.target.hidden = true; });

  // 窗口缩放重绘
  window.addEventListener("resize", () => {
    if (state.sourcePeaks && !$("#panel-recognition").hidden) {
      drawWave($("#source-wave"), state.sourcePeaks);
      if (!$("#compare-box").hidden) drawWave($("#compare-src-wave"), state.sourcePeaks);
    }
    player.redraw();
    if (player.peaks && !$("#compare-box").hidden) drawWave($("#compare-gen-wave"), player.peaks, { gradient: WAVE_GRAD });
  });
}

bindEvents();
loadHistory();

/* ---------- 实时录音 ---------- */

const RECORD_MAX_S = 30;
const RECORD_MIN_S = 1;

const recorderState = {
  active: false,
  stream: null,
  mediaRecorder: null,
  chunks: [],
  startedAt: 0,
  timerId: null,
  autoStopId: null,
  rafId: null,
  levelCtx: null,
  analyser: null,
  wavBlob: null,
  previewUrl: null,
  previewAudio: null,
};

/* 编码 16-bit PCM WAV（单声道，RIFF 头 + PCM16 小端） */
function encodeWAV(audioBuffer) {
  const samples = audioBuffer.getChannelData(0);
  const sampleRate = audioBuffer.sampleRate;
  const dataLen = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataLen);
  const view = new DataView(buffer);
  const writeStr = (offset, text) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataLen, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);          // fmt chunk size
  view.setUint16(20, 1, true);           // PCM
  view.setUint16(22, 1, true);           // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true);           // block align
  view.setUint16(34, 16, true);          // bits per sample
  writeStr(36, "data");
  view.setUint32(40, dataLen, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

/* 解码任意录制容器 → 重采样/混音到 44.1kHz 单声道 → WAV Blob */
async function recordingToWav(rawBlob) {
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await audioCtx.decodeAudioData(await rawBlob.arrayBuffer());
  const frames = Math.max(1, Math.ceil(decoded.duration * 44100));
  const offline = new OfflineAudioContext(1, frames, 44100);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return encodeWAV(rendered);
}

function setUploadLocked(locked) {
  $("#panel-upload").classList.toggle("locked", locked);
}

function fmtRecTime(seconds) {
  const s = Math.floor(seconds);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function drawRecordLevel() {
  const canvas = $("#record-level");
  const analyser = recorderState.analyser;
  if (!analyser) return;
  const data = new Uint8Array(analyser.fftSize);
  const bars = 64;
  const loop = () => {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, "#6fa287");
    grad.addColorStop(0.5, "#b5c98b");
    grad.addColorStop(1, "#eec27f");
    ctx.fillStyle = grad;
    analyser.getByteTimeDomainData(data);
    const bw = w / bars;
    const step = Math.floor(data.length / bars);
    const mid = h / 2;
    for (let i = 0; i < bars; i++) {
      let peak = 0;
      for (let j = i * step; j < (i + 1) * step && j < data.length; j += 4) {
        const v = Math.abs(data[j] - 128) / 128;
        if (v > peak) peak = v;
      }
      const amp = Math.max(1.6, peak * (h - 4));
      ctx.fillRect(i * bw, mid - amp / 2, Math.max(1, bw * 0.55), amp);
    }
    recorderState.rafId = requestAnimationFrame(loop);
  };
  loop();
}

function stopRecorderTimers() {
  clearInterval(recorderState.timerId);
  clearTimeout(recorderState.autoStopId);
  cancelAnimationFrame(recorderState.rafId);
  recorderState.timerId = null;
  recorderState.autoStopId = null;
  recorderState.rafId = null;
}

function releaseMic() {
  if (recorderState.stream) {
    recorderState.stream.getTracks().forEach((t) => t.stop());
    recorderState.stream = null;
  }
  if (recorderState.levelCtx) {
    recorderState.levelCtx.close().catch(() => {});
    recorderState.levelCtx = null;
    recorderState.analyser = null;
  }
}

function resetRecorder() {
  stopRecorderTimers();
  releaseMic();
  if (recorderState.previewAudio) {
    recorderState.previewAudio.pause();
    recorderState.previewAudio = null;
  }
  if (recorderState.previewUrl) {
    URL.revokeObjectURL(recorderState.previewUrl);
    recorderState.previewUrl = null;
  }
  recorderState.active = false;
  recorderState.mediaRecorder = null;
  recorderState.chunks = [];
  recorderState.wavBlob = null;
  $("#recorder-box").hidden = true;
  $("#record-preview").hidden = true;
  $("#upload-alt").hidden = false;
  setUploadLocked(false);
}

async function startRecording() {
  if (recorderState.active) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
    toast("当前浏览器不支持录音，请使用最新版 Chrome / Edge / Safari");
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    if (err && err.name === "NotAllowedError") {
      toast("麦克风权限被拒绝，请在浏览器地址栏允许麦克风后重试");
    } else if (err && (err.name === "NotFoundError" || err.name === "OverconstrainedError")) {
      toast("未检测到麦克风设备，请检查系统设置");
    } else {
      toast(`无法开启麦克风：${err.message || err}`);
    }
    return;
  }

  const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].find(
    (t) => MediaRecorder.isTypeSupported(t)
  );
  const mediaRecorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);

  recorderState.active = true;
  recorderState.stream = stream;
  recorderState.mediaRecorder = mediaRecorder;
  recorderState.chunks = [];
  recorderState.startedAt = Date.now();

  // 实时电平：AnalyserNode 画渐变声波条
  recorderState.levelCtx = new (window.AudioContext || window.webkitAudioContext)();
  const sourceNode = recorderState.levelCtx.createMediaStreamSource(stream);
  recorderState.analyser = recorderState.levelCtx.createAnalyser();
  recorderState.analyser.fftSize = 1024;
  sourceNode.connect(recorderState.analyser);

  mediaRecorder.ondataavailable = (e) => { if (e.data.size) recorderState.chunks.push(e.data); };
  mediaRecorder.onstop = onRecordingStopped;
  mediaRecorder.start(250);

  $("#upload-alt").hidden = true;
  $("#record-preview").hidden = true;
  $("#recorder-box").hidden = false;
  $("#record-time").textContent = "00:00";
  setUploadLocked(true);
  drawRecordLevel();

  recorderState.timerId = setInterval(() => {
    $("#record-time").textContent = fmtRecTime((Date.now() - recorderState.startedAt) / 1000);
  }, 250);
  recorderState.autoStopId = setTimeout(() => stopRecording(), RECORD_MAX_S * 1000);
}

function stopRecording() {
  const mr = recorderState.mediaRecorder;
  if (!mr || mr.state === "inactive") return;
  mr.stop();
}

async function onRecordingStopped() {
  const duration = (Date.now() - recorderState.startedAt) / 1000;
  const mimeType = recorderState.mediaRecorder.mimeType || "audio/webm";
  const rawBlob = new Blob(recorderState.chunks, { type: mimeType });
  stopRecorderTimers();
  releaseMic();
  $("#recorder-box").hidden = true;

  if (duration < RECORD_MIN_S) {
    toast(`录音太短（${duration.toFixed(1)} 秒），请至少录 ${RECORD_MIN_S} 秒`);
    resetRecorder();
    return;
  }

  try {
    recorderState.wavBlob = await recordingToWav(rawBlob);
  } catch (err) {
    toast(`录音处理失败：${err.message || err}`);
    resetRecorder();
    return;
  }

  // 预览：波形 + 试听
  try {
    const { peaks } = await decodePeaks(recorderState.wavBlob);
    drawWave($("#record-wave"), peaks);
  } catch { /* 波形失败不阻塞试听 */ }
  recorderState.previewUrl = URL.createObjectURL(recorderState.wavBlob);
  recorderState.previewAudio = new Audio(recorderState.previewUrl);
  recorderState.previewAudio.addEventListener("ended", () => { $("#btn-record-play").textContent = "试听"; });
  $("#record-preview").hidden = false;
  toast("录音完成，试听确认后开始识别", "info", 3000);
}

function bindRecorderEvents() {
  $("#btn-record").addEventListener("click", startRecording);
  $("#btn-record-stop").addEventListener("click", stopRecording);
  $("#btn-record-play").addEventListener("click", () => {
    const audio = recorderState.previewAudio;
    if (!audio) return;
    if (audio.paused) {
      audio.play();
      $("#btn-record-play").textContent = "暂停";
    } else {
      audio.pause();
      $("#btn-record-play").textContent = "试听";
    }
  });
  $("#btn-record-use").addEventListener("click", () => {
    if (!recorderState.wavBlob) return;
    const file = new File([recorderState.wavBlob], `recording-${Date.now()}.wav`, { type: "audio/wav" });
    resetRecorder();
    handleFile(file, "实时录音");
  });
  $("#btn-record-redo").addEventListener("click", () => {
    resetRecorder();
    startRecording();
  });
}

bindRecorderEvents();

/* ---------- 移动端菜单 ---------- */

(function initMobileMenu() {
  const toggle = $("#menu-toggle");
  const menu = $("#mobile-menu");
  if (!toggle || !menu) return;
  const setOpen = (open) => {
    toggle.classList.toggle("open", open);
    menu.classList.toggle("open", open);
    toggle.setAttribute("aria-expanded", String(open));
    menu.setAttribute("aria-hidden", String(!open));
  };
  toggle.addEventListener("click", () => setOpen(!menu.classList.contains("open")));
  $("#mobile-cta").addEventListener("click", () => setOpen(false));
  $("#btn-help-mobile").addEventListener("click", () => {
    setOpen(false);
    $("#help-modal").hidden = false;
  });
})();

/* ---------- 森林场景切换（Lumora） ---------- */

const SCENES = [
  { time: "06:18", note: "薄雾苏醒" },
  { time: "12:06", note: "林隙流光" },
  { time: "18:42", note: "暮色入林" },
  { time: "00:24", note: "万籁归静" },
];

(function initScenes() {
  const videos = document.querySelectorAll(".video-stack video");
  const buttons = document.querySelectorAll(".scene-control button");
  if (!videos.length || !buttons.length) return;
  let active = 0;
  let locked = false;
  const apply = (index) => {
    active = index;
    document.body.classList.remove("scene-0", "scene-1", "scene-2", "scene-3");
    document.body.classList.add(`scene-${index}`);
    videos.forEach((v, i) => v.classList.toggle("active", i === index));
    buttons.forEach((b, i) => {
      b.classList.toggle("active", i === index);
      b.setAttribute("aria-pressed", String(i === index));
    });
    $("#scene-time").textContent = SCENES[index].time;
    $("#scene-note").textContent = SCENES[index].note;
  };
  const choose = (index) => {
    if (index === active || locked) return;
    locked = true;
    apply(index);
    setTimeout(() => { locked = false; }, 1100);
  };
  buttons.forEach((b, i) => b.addEventListener("click", () => choose(i)));
  setInterval(() => { if (!locked) choose((active + 1) % SCENES.length); }, 9000);
  apply(0);
})();
