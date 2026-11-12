// Conversation page. Each page sets window.CHARACTER (name, avatar, texts) before loading this file.
//
// Flow: question (typed or spoken, ar-SA) -> POST /api/ask (RAG + chat model) -> chat bubble
//       -> POST /api/tts (Arabic speech + word timings; the vowelled answer gives each word its
//       vowels) -> TalkingHead plays it while lipsync-ar.mjs turns every Arabic word into mouth
//       shapes. If server speech fails, the browser's own Arabic voice is used.
import { TalkingHead } from "talkinghead";

const C = window.CHARACTER;
const API = "/api";
const DEBUG = new URLSearchParams(location.search).has("debug");
const LISTEN_TIMEOUT_MS = 10000; // hands-free mode: a round ends after 10 s of silence
const DIACRITICS = /[ً-ْٰ]/g;

const state = {
  head: null,
  avatarReady: false,
  busy: false,
  muted: false,
  recognition: null,
  dictating: false,
  live: false,
  liveTimer: null,
  browserVoice: null,
  speakingEl: null,
};
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const plain = (s) => (s || "").replace(DIACRITICS, "");

// ------------------------------------------------------------------ icons (inline SVG)
const ICON = {
  back: '<path d="M15 18l-6-6 6-6"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
  sound: '<path d="M11 5 6 9H3v6h3l5 4V5Z"/><path d="M15.5 8.5a5 5 0 0 1 0 7M18.5 5.5a9 9 0 0 1 0 13"/>',
  mute: '<path d="M11 5 6 9H3v6h3l5 4V5Z"/><path d="m22 9-6 6M16 9l6 6"/>',
  reset: '<path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/>',
  mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
  live: '<path d="M4 10v4M8 7v10M12 4v16M16 7v10M20 10v4"/>',
  send: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
  replay: '<path d="M11 5 6 9H3v6h3l5 4V5Z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/>',
  copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>',
  book: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5v14Z"/>',
  alert: '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>',
};
const icon = (name) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICON[name]}</svg>`;

// ------------------------------------------------------------------ page layout
function render() {
  document.title = `${C.name} · Arabic Historical Characters`;
  document.body.innerHTML = `
    <div class="app">
      <section class="stage" aria-label="3D avatar">
        <div id="avatar"></div>
        <div id="loader" class="loader">
          <img class="portrait" src="${C.portrait}" alt="" />
          <span id="loaderText">جاري تحميل الشخصية...</span>
          <div class="bar"><span id="loaderBar"></span></div>
        </div>
        <div class="stage-top">
          <a class="icon-btn" href="../index.html#characters" aria-label="All characters" title="كل الشخصيات">${icon("back")}</a>
          <div class="stage-title">
            <h1>${C.name}</h1>
            <p>${C.nameEn}</p>
          </div>
        </div>
        <div class="stage-bottom">
          <div id="status" class="status" data-state="loading" role="status"><span class="dot"></span><span id="statusText">جاري التحميل...</span></div>
          <button id="stopBtn" class="pill-btn" type="button" hidden>${icon("stop")}<span>إيقاف الكلام</span></button>
        </div>
      </section>

      <section class="panel" aria-label="المحادثة">
        <header class="panel-head">
          <img class="mini" src="${C.portrait}" alt="" />
          <div class="who"><strong>${C.name}</strong><span>${C.title}</span></div>
          <div class="head-actions">
            <button id="aboutBtn" class="icon-btn" type="button" aria-expanded="false" aria-controls="about" title="نبذة عن الشخصية">${icon("info")}</button>
            <button id="muteBtn" class="icon-btn" type="button" aria-pressed="false" title="كتم الصوت">${icon("sound")}</button>
            <button id="resetBtn" class="icon-btn" type="button" title="محادثة جديدة">${icon("reset")}</button>
          </div>
        </header>
        <p id="about" class="about" hidden></p>
        <div id="notice" class="banner" hidden>${icon("alert")}<span></span></div>
        <div id="chat" class="chat" role="log" aria-live="polite"></div>
        <div id="suggestions" class="chips" aria-label="أسئلة مقترحة"></div>
        <form id="composer" class="composer" autocomplete="off">
          <label for="text" class="visually-hidden">سؤالك</label>
          <textarea id="text" rows="1" maxlength="500" dir="rtl"></textarea>
          <button id="micBtn" class="icon-btn" type="button" title="تحدث لكتابة سؤالك">${icon("mic")}</button>
          <button id="liveBtn" class="icon-btn" type="button" aria-pressed="false" title="محادثة صوتية متواصلة">${icon("live")}</button>
          <button id="sendBtn" class="send-btn" type="submit" title="إرسال" disabled>${icon("send")}</button>
        </form>
        <p class="hint">Enter للإرسال · Shift+Enter لسطر جديد · Esc لإيقاف الكلام</p>
      </section>
    </div>`;
  $("about").textContent = C.about || "";
  $("text").placeholder = C.placeholder || "اكتب سؤالك...";
  for (const q of C.suggestions || []) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = q;
    b.addEventListener("click", () => ask(q));
    $("suggestions").appendChild(b);
  }
  if (DEBUG) {
    const panel = document.createElement("div");
    panel.id = "debugInfo";
    document.querySelector(".stage").appendChild(panel);
  }
}

// ------------------------------------------------------------------ helpers
const sessionId = (() => {
  const key = "hc_session_id";
  let id = null;
  try { id = localStorage.getItem(key); } catch (e) { /* storage blocked */ }
  if (!id || !/^[A-Za-z0-9_-]{1,64}$/.test(id)) {
    id = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2))
      .replace(/[^A-Za-z0-9_-]/g, "");
    try { localStorage.setItem(key, id); } catch (e) { /* ignore */ }
  }
  return id;
})();

function debugLog(message, data) {
  console.log("[character]", message, data ?? "");
  const panel = $("debugInfo");
  if (!panel) return;
  const line = document.createElement("div");
  let extra = "";
  if (data !== undefined) {
    try { extra = ": " + (typeof data === "object" ? JSON.stringify(data).slice(0, 200) : data); } catch (e) { /* ignore */ }
  }
  line.textContent = `[${new Date().toLocaleTimeString()}] ${message}${extra}`;
  panel.appendChild(line);
  panel.scrollTop = panel.scrollHeight;
}

function setStatus(stateName, text) {
  const el = $("status");
  el.dataset.state = stateName;
  $("statusText").textContent = text;
}

function idle() {
  if (state.live) setStatus("listening", "المحادثة الصوتية مفعلة...");
  else setStatus("ready", "جاهز للمحادثة");
}

function setBusy(value) {
  state.busy = value;
  for (const el of document.querySelectorAll(".chips button, #resetBtn, #micBtn")) el.disabled = value;
  updateSend();
}

function updateSend() {
  $("sendBtn").disabled = state.busy || !$("text").value.trim();
}

async function api(path, body) {
  const res = await fetch(API + path, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch (e) { /* not JSON */ }
  if (!res.ok) {
    const detail = data?.detail;
    throw new Error(typeof detail === "string" ? detail : `HTTP ${res.status}`);
  }
  return data;
}

function showNotice(text) {
  const n = $("notice");
  n.hidden = !text;
  n.querySelector("span").textContent = text || "";
}

// ------------------------------------------------------------------ chat
function scrollChat() {
  const chat = $("chat");
  chat.scrollTop = chat.scrollHeight;
}

function timeLabel(iso) {
  const d = iso ? new Date(iso) : new Date();
  return isNaN(d) ? "" : d.toLocaleTimeString("ar-EG", { hour: "2-digit", minute: "2-digit" });
}

/**
 * Add a message. role: "user" | "character" | "error".
 * extra: { vocalized, sources, mode, time }
 */
function addMessage(role, text, extra = {}) {
  const row = document.createElement("div");
  row.className = `msg ${role}`;
  if (role === "character") {
    const face = document.createElement("img");
    face.className = "face";
    face.src = C.portrait;
    face.alt = "";
    row.appendChild(face);
  }
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const body = document.createElement("div");
  body.textContent = text; // textContent: the answer is never inserted as HTML
  bubble.appendChild(body);

  if (role !== "error") {
    const meta = document.createElement("div");
    meta.className = "meta";
    const time = document.createElement("time");
    time.textContent = timeLabel(extra.time);
    meta.appendChild(time);

    if (role === "character") {
      if (extra.mode === "offline") {
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = "من ويكيبيديا";
        meta.appendChild(tag);
      }
      const replay = metaButton("replay", "استمع", () => speak(text, extra.vocalized, row));
      const copy = metaButton("copy", "نسخ", async () => {
        try {
          await navigator.clipboard.writeText(text);
          copy.querySelector("span").textContent = "تم النسخ";
          setTimeout(() => (copy.querySelector("span").textContent = "نسخ"), 1500);
        } catch (e) { /* clipboard blocked */ }
      });
      meta.append(replay, copy);

      if (extra.sources?.length) {
        const list = document.createElement("ul");
        list.className = "sources";
        list.hidden = true;
        for (const src of extra.sources) {
          const li = document.createElement("li");
          const a = document.createElement("a");
          a.href = src.url ? `${src.url}#${encodeURIComponent((src.section || "").split(" › ").pop().replace(/ /g, "_"))}` : "#";
          a.target = "_blank";
          a.rel = "noopener";
          a.textContent = `ويكيبيديا – ${src.section || src.title}`;
          li.appendChild(a);
          list.appendChild(li);
        }
        const toggle = metaButton("book", `المصادر (${extra.sources.length})`, () => {
          list.hidden = !list.hidden;
          toggle.setAttribute("aria-expanded", String(!list.hidden));
        });
        toggle.setAttribute("aria-expanded", "false");
        meta.appendChild(toggle);
        bubble.append(meta, list);
      } else {
        bubble.appendChild(meta);
      }
    } else {
      bubble.appendChild(meta);
    }
  }
  row.appendChild(bubble);
  $("chat").appendChild(row);
  scrollChat();
  return row;
}

function metaButton(iconName, label, onClick) {
  const b = document.createElement("button");
  b.type = "button";
  b.innerHTML = `${icon(iconName)}<span></span>`;
  b.querySelector("span").textContent = label;
  b.addEventListener("click", onClick);
  return b;
}

function addTyping() {
  const row = document.createElement("div");
  row.className = "msg character";
  row.innerHTML = `<img class="face" src="${C.portrait}" alt="" /><div class="bubble"><span class="typing" aria-label="${C.thinking}"><span></span><span></span><span></span></span></div>`;
  $("chat").appendChild(row);
  scrollChat();
  return row;
}

async function loadHistory() {
  try {
    const res = await api(`/history?character_name=${encodeURIComponent(C.id)}&session_id=${sessionId}`);
    const msgs = res.data.slice(-40);
    if (!msgs.length) return addMessage("character", C.greeting);
    for (const m of msgs) {
      if (m.role === "user") addMessage("user", m.content, { time: m.time });
      else addMessage("character", plain(m.content), { vocalized: m.content, mode: m.mode, time: m.time });
    }
  } catch (e) {
    debugLog("history unavailable", e.message);
    addMessage("character", C.greeting);
  }
}

// ------------------------------------------------------------------ speech output
function base64ToArrayBuffer(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

async function waitUntilSilent(maxMs) {
  const start = performance.now();
  await sleep(300);
  while (performance.now() - start < maxMs) {
    const talking = state.head && (state.head.isSpeaking || state.head.isAudioPlaying);
    if (!talking && !speechSynthesis.speaking) return;
    await sleep(150);
  }
}

function stopSpeaking() {
  try { state.head?.stopSpeaking(); } catch (e) { /* ignore */ }
  speechSynthesis.cancel();
  markSpeaking(null);
}

function markSpeaking(row) {
  state.speakingEl?.classList.remove("speaking");
  state.speakingEl = row;
  row?.classList.add("speaking");
  $("stopBtn").hidden = !row;
}

function pickBrowserVoice() {
  const voices = speechSynthesis.getVoices().filter((v) => v.lang?.toLowerCase().startsWith("ar"));
  state.browserVoice = voices.find((v) => /male|hamed|shakir|naayf|maged/i.test(v.name)) || voices[0] || null;
}

function speakWithBrowser(text, vocalized) {
  return new Promise((resolve) => {
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "ar-SA";
    if (state.browserVoice) u.voice = state.browserVoice;
    u.rate = 0.95;
    u.onend = u.onerror = () => resolve();
    speechSynthesis.speak(u);
    // No word timings from the browser voice: move the lips at an estimated pace
    if (state.avatarReady) {
      const words = (vocalized || text).split(/\s+/).filter(Boolean);
      const audio = { audio: null, words: [], wtimes: [], wdurations: [] };
      let t = 0;
      for (const w of words) {
        const d = 140 + plain(w).length * 65;
        audio.words.push(w); audio.wtimes.push(t); audio.wdurations.push(d);
        t += d + 60;
      }
      try {
        audio.audio = state.head.audioCtx.createBuffer(1, Math.max(1, Math.ceil(((t + 300) / 1000) * 22050)), 22050);
        state.head.speakAudio(audio, { lipsyncLang: "ar" });
      } catch (e) { debugLog("fallback lip sync failed", e.message); }
    }
  });
}

async function speak(text, vocalized, row = null) {
  stopSpeaking();
  if (state.muted) return idle();
  if (state.head?.audioCtx?.state === "suspended") await state.head.audioCtx.resume();
  setStatus("speaking", C.speaking);
  markSpeaking(row);
  try {
    const { data } = await api("/tts", { text, character_name: C.id, vocalized: vocalized || null });
    debugLog("TTS", { engine: data.engine, words: data.words.slice(0, 8) });
    if (!state.avatarReady) throw new Error("avatar not loaded");
    const buffer = await state.head.audioCtx.decodeAudioData(base64ToArrayBuffer(data.audio));
    state.head.speakAudio(
      { audio: buffer, words: data.words, wtimes: data.wtimes, wdurations: data.wdurations },
      { lipsyncLang: "ar" }
    );
    await waitUntilSilent(buffer.duration * 1000 + 3000);
  } catch (e) {
    debugLog("server TTS failed, using the browser voice", e.message);
    await speakWithBrowser(text, vocalized);
  }
  if (state.speakingEl === row) markSpeaking(null);
  idle();
}

// ------------------------------------------------------------------ asking
async function ask(question) {
  question = (question || "").trim();
  if (!question || state.busy) return;
  setBusy(true);
  stopSpeaking();
  addMessage("user", question);
  const typing = addTyping();
  setStatus("thinking", C.thinking);
  try {
    const { data } = await api("/ask", { character_name: C.id, question, session_id: sessionId });
    typing.remove();
    showNotice(data.notice);
    const row = addMessage("character", data.answer, { vocalized: data.answer_vocalized, sources: data.sources, mode: data.mode });
    setBusy(false);
    await speak(data.answer, data.answer_vocalized, row);
  } catch (e) {
    typing.remove();
    debugLog("ask failed", e.message);
    addMessage("error", `تعذر الحصول على إجابة: ${e.message}`);
    setStatus("error", "حدث خطأ، حاول مرة أخرى");
  } finally {
    setBusy(false);
  }
}

function submitQuestion() {
  const input = $("text");
  const q = input.value;
  if (!q.trim() || state.busy) return;
  input.value = "";
  autoGrow();
  updateSend();
  ask(q);
}

async function resetConversation() {
  if (state.busy) return;
  stopSpeaking();
  try {
    await api("/reset", { character_name: C.id, session_id: sessionId });
  } catch (e) {
    debugLog("reset failed", e.message);
  }
  $("chat").innerHTML = "";
  showNotice(null);
  addMessage("character", C.greeting);
  idle();
}

function autoGrow() {
  const t = $("text");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 140) + "px";
}

// ------------------------------------------------------------------ speech input
function initSpeechRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    $("micBtn").hidden = true;
    $("liveBtn").hidden = true;
    debugLog("speech recognition is not supported in this browser");
    return;
  }
  const r = new SR();
  r.lang = "ar-SA";
  r.continuous = false;
  r.interimResults = true;
  r.maxAlternatives = 1;
  state.recognition = r;
}

const SPEECH_ERRORS = {
  "audio-capture": "لا يمكن الوصول إلى الميكروفون",
  "not-allowed": "لم يتم السماح باستخدام الميكروفون",
  network: "خطأ في الشبكة أثناء التعرف على الصوت",
};

// One recognition round; resolves with the transcript ("" if nothing was said, null on a fatal error)
function listenOnce(onInterim) {
  const r = state.recognition;
  return new Promise((resolve) => {
    let finalText = "";
    let settled = false;
    const done = (v) => { if (!settled) { settled = true; resolve(v); } };
    r.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) finalText += event.results[i][0].transcript;
        else interim += event.results[i][0].transcript;
      }
      onInterim?.(finalText || interim);
    };
    r.onerror = (event) => {
      debugLog("speech error", event.error);
      if (SPEECH_ERRORS[event.error]) setStatus("error", SPEECH_ERRORS[event.error]);
      if (event.error === "not-allowed" || event.error === "audio-capture") done(null);
    };
    r.onend = () => done(finalText.trim());
    try {
      r.start();
    } catch (e) {
      debugLog("recognition start failed", e.message);
      done(null);
    }
  });
}

async function toggleDictation() {
  if (!state.recognition || state.live) return;
  if (state.dictating) {
    state.recognition.stop();
    return;
  }
  state.dictating = true;
  $("micBtn").classList.add("recording");
  setStatus("listening", "أستمع إليك... تحدث الآن");
  const text = await listenOnce((t) => { $("text").value = t; autoGrow(); updateSend(); });
  state.dictating = false;
  $("micBtn").classList.remove("recording");
  if (text) $("text").value = text;
  autoGrow();
  updateSend();
  $("text").focus();
  idle();
}

async function toggleLive() {
  if (!state.recognition) return;
  if (state.live) return stopLive();
  state.live = true;
  const btn = $("liveBtn");
  btn.setAttribute("aria-pressed", "true");
  btn.classList.add("recording");
  debugLog("hands-free conversation started");

  let silentRounds = 0;
  while (state.live) {
    setStatus("listening", "أستمع إليك... (اضغط زر المحادثة الصوتية للإنهاء)");
    state.liveTimer = setTimeout(() => state.recognition.stop(), LISTEN_TIMEOUT_MS);
    const text = await listenOnce((t) => { $("text").value = t; autoGrow(); });
    clearTimeout(state.liveTimer);
    $("text").value = "";
    autoGrow();
    if (!state.live || text === null) break;
    if (!text) {
      silentRounds += 1;
      if (silentRounds >= 2) {
        await speak(C.goodbye, null, addMessage("character", C.goodbye));
        break;
      }
      continue;
    }
    silentRounds = 0;
    await ask(text); // returns after the character has finished speaking
    await sleep(400);
  }
  stopLive();
}

function stopLive() {
  if (!state.live) return;
  state.live = false;
  clearTimeout(state.liveTimer);
  try { state.recognition.abort(); } catch (e) { /* ignore */ }
  const btn = $("liveBtn");
  btn.setAttribute("aria-pressed", "false");
  btn.classList.remove("recording");
  idle();
}

// ------------------------------------------------------------------ avatar
async function loadAvatar() {
  state.head = new TalkingHead($("avatar"), {
    ttsEndpoint: "N/A",           // speech comes from /api/tts, not from Google TTS
    lipsyncModules: ["ar"],       // assets/vendor/talkinghead/lipsync-ar.mjs: Arabic -> visemes
    lipsyncLang: "ar",
    cameraView: "upper",
    mixerGainSpeech: 1.5,
  });
  if (DEBUG) window.head = state.head; // inspect from the browser console with ?debug=1
  try {
    await state.head.showAvatar(
      { url: C.avatar, body: C.body || "M", avatarMood: "neutral", lipsyncLang: "ar" },
      (ev) => {
        if (ev.lengthComputable) {
          const p = Math.min(100, Math.round((ev.loaded / ev.total) * 100));
          $("loaderBar").style.width = p + "%";
          $("loaderText").textContent = `جاري تحميل الشخصية ${p}%`;
        }
      }
    );
    $("loader").hidden = true;
    state.avatarReady = true;
    debugLog("avatar loaded");
  } catch (e) {
    debugLog("avatar failed", e.message);
    $("loader").classList.add("failed");
    $("loaderText").textContent = "تعذر تحميل الشخصية ثلاثية الأبعاد، ويمكنك متابعة المحادثة بالنص والصوت";
  }
}

// ------------------------------------------------------------------ start
function bindEvents() {
  const input = $("text");
  $("composer").addEventListener("submit", (e) => { e.preventDefault(); submitQuestion(); });
  input.addEventListener("input", () => { autoGrow(); updateSend(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      submitQuestion();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { stopSpeaking(); idle(); }
  });
  $("micBtn").addEventListener("click", toggleDictation);
  $("liveBtn").addEventListener("click", toggleLive);
  $("stopBtn").addEventListener("click", () => { stopSpeaking(); idle(); });
  $("resetBtn").addEventListener("click", resetConversation);
  $("aboutBtn").addEventListener("click", () => {
    const about = $("about");
    about.hidden = !about.hidden;
    $("aboutBtn").setAttribute("aria-expanded", String(!about.hidden));
  });
  $("muteBtn").addEventListener("click", () => {
    state.muted = !state.muted;
    const b = $("muteBtn");
    b.setAttribute("aria-pressed", String(state.muted));
    b.innerHTML = icon(state.muted ? "mute" : "sound");
    b.title = state.muted ? "تشغيل الصوت" : "كتم الصوت";
    if (state.muted) { stopSpeaking(); idle(); }
  });
  // browsers start audio only after a user gesture
  document.body.addEventListener("pointerdown", () => {
    if (state.head?.audioCtx?.state === "suspended") state.head.audioCtx.resume();
  }, { once: true });
  document.addEventListener("visibilitychange", () => {
    if (!state.head || !state.avatarReady) return;
    document.visibilityState === "visible" ? state.head.start() : state.head.stop();
  });
}

async function start() {
  render();
  bindEvents();
  pickBrowserVoice();
  speechSynthesis.onvoiceschanged = pickBrowserVoice;
  initSpeechRecognition();
  setStatus("loading", "جاري التحميل...");
  loadHistory();
  await loadAvatar();
  idle();
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
else start();
