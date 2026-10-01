/*
 * Hands-free word-practice loop for CurioKidz Reading Buddy.
 *
 * HONEST NOTES ON HOW THIS ACTUALLY WORKS (same principle as every
 * Python module comment in this project):
 *
 * - One tap on "Enable microphone" is unavoidable -- no browser lets a
 *   page listen to the mic without an explicit user gesture first.
 *   After that single tap, nothing else needs to be clicked: the word
 *   is spoken aloud, the mic starts listening on its own, the result
 *   is spoken aloud, and the next word starts on its own.
 * - Two separate things listen to the mic for two different jobs:
 *     1. The browser's own SpeechRecognition (Web Speech API) listens
 *        continuously for short COMMAND words ("start", "repeat",
 *        "next", "stop"). It always listens in English ("en-IN"),
 *        regardless of lesson language -- these are the app's own
 *        fixed control words, not lesson content, and Kannada support
 *        in SpeechRecognition varies a lot by browser, so English
 *        commands are used everywhere for reliability. This is a
 *        deliberate, stated choice, not an oversight.
 *     2. MediaRecorder + a simple volume (RMS) check records the
 *        child's actual READING ATTEMPT and uploads it to this app's
 *        OWN server-side faster-whisper pipeline (modules/asr.py) to
 *        be scored -- the real, already-tuned accuracy work (VAD,
 *        beam-size retry, acoustic fallback) lives there, not in the
 *        browser, so scoring quality is exactly what the rest of this
 *        project already achieved.
 * - "Silence detection" here is a plain RMS energy threshold over the
 *   Web Audio API's AnalyserNode -- the same real technique (not a
 *   placebo) modules/voice_nav.py already used for the same purpose
 *   server-side, just running in the browser instead.
 */

const STATUS = document.getElementById('status');
const MIC_DOT = document.getElementById('mic-dot');
const WORD_CARD = document.getElementById('word-card');
const ENABLE_BTN = document.getElementById('enable-btn');
const BADGE = document.getElementById('badge');
const STOP_WRAP = document.getElementById('stop-wrap');
const player = new Audio();

let words = [];
let wordIdx = 0;
// EXTENSION: known immediately from the server-rendered page (not just
// 'en' until the first fetch resolves), so the very first screen a
// Kannada-lesson child sees -- the "Enable microphone" button, the
// voice-command hint, the Stop button -- is already in Kannada, not
// flashing English first and Kannada only after practice starts.
let language = window.LESSON_LANGUAGE || 'en';
let stanzaIdx = 0;
let stanzaCount = 1;
let allStanzaWords = []; // every stanza's word list, fetched once up front -- see renderReadingText()

// ---------------------------------------------------------------------
// Hearts (Duolingo-style lives). Deliberately SOFT, not a hard stop:
// this is a children's pronunciation-practice app, and a struggling
// reader hitting a wall mid-lesson is the opposite of what it's for.
// Losing all hearts pauses with an encouraging message and a "keep
// going anyway" option, rather than ending the session outright.
// ---------------------------------------------------------------------
const HEARTS_TOTAL = 5;
let hearts = HEARTS_TOTAL;

function renderHearts() {
  const el = document.getElementById('hearts-row');
  if (!el) return;
  el.innerHTML = Array.from({ length: HEARTS_TOTAL }, (_, i) =>
    `<span class="ck-heart ${i < hearts ? '' : 'lost'}">❤️</span>`).join(' ');
}

function loseHeart() {
  hearts = Math.max(0, hearts - 1);
  renderHearts();
  return hearts === 0;
}

// showXpFloat() itself now lives in static/js/ckheader.js (shared with
// Custom Practice's child read-back check) -- ckheader.js is loaded
// before this file on practice.html, so it's already available here.
let micStream = null;
let audioCtx = null;
let analyser = null;
let recognition = null;
let recognitionActive = false;
let running = false;
let awaitingCommandOnly = false; // true while idle, before "start" is heard

function setStatus(text) { STATUS.firstChild ? STATUS.firstChild.textContent = text + ' ' : STATUS.textContent = text; }

// EXTENSION: localize the static, always-visible labels (button text,
// the voice-command hint line) that templates/practice.html hardcodes
// in English -- run once at load, using the language already known
// from window.LESSON_LANGUAGE, so these never show English on a
// Kannada lesson's page.
(function localizeStaticLabels() {
  if (ENABLE_BTN) ENABLE_BTN.textContent = ckUiText(language, 'enable_btn');
  const stopBtn = document.getElementById('stop-btn');
  if (stopBtn) stopBtn.textContent = ckUiText(language, 'stop_btn');
  if (STATUS && STATUS.firstChild) STATUS.firstChild.textContent = ckUiText(language, 'start_hint') + ' ';
})();

function playClip(url) {
  return new Promise((resolve) => {
    player.src = url;
    player.onended = () => resolve();
    player.onerror = () => resolve(); // never hang the loop on a bad/missing clip
    player.play().catch(() => resolve());
  });
}

function say(text, lang, slow) {
  const url = `/api/audio/say?text=${encodeURIComponent(text)}&lang=${lang}&slow=${slow ? 1 : 0}`;
  return playClip(url);
}

// ---------------------------------------------------------------------
// Command recognition (Web Speech API) -- runs continuously in the
// background except during the brief window an attempt is actively
// being recorded (paused/resumed explicitly, so a read-aloud word is
// never mistaken for a command).
// ---------------------------------------------------------------------
function initRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    setStatus('Your browser doesn\'t support voice commands -- this needs Chrome or Edge.');
    return null;
  }
  const r = new SR();
  r.continuous = true;
  r.interimResults = false;
  r.lang = 'en-IN';
  r.onresult = (event) => {
    const last = event.results[event.results.length - 1];
    const heard = last[0].transcript.trim().toLowerCase();
    handleCommand(heard);
  };
  r.onend = () => {
    recognitionActive = false;
    // Chrome ends recognition sessions periodically on its own; restart
    // automatically as long as the practice loop is still running.
    if (running) startRecognition();
  };
  r.onerror = () => { recognitionActive = false; };
  return r;
}

function startRecognition() {
  if (!recognition || recognitionActive) return;
  try { recognition.start(); recognitionActive = true; } catch (e) { /* already started */ }
}

function stopRecognition() {
  if (!recognition || !recognitionActive) return;
  try { recognition.stop(); } catch (e) { /* ignore */ }
}

function handleCommand(heard) {
  const words_ = heard.split(/\s+/);
  const has = (list) => list.some(w => words_.includes(w) || heard === w);
  if (has(['stop', 'exit', 'quit', 'done'])) { stopPractice(); return; }
  if (awaitingCommandOnly && has(['start', 'begin', 'ready', 'go'])) { awaitingCommandOnly = false; runWord(); return; }
  if (!awaitingCommandOnly) {
    if (has(['repeat', 'again', 'replay'])) { runWord(true); return; }
    if (has(['next', 'continue', 'skip'])) { advanceWord(); return; }
  }
}

// ---------------------------------------------------------------------
// Attempt recording -- MediaRecorder + RMS-based auto-stop.
// ---------------------------------------------------------------------
function rms(dataArray) {
  let sum = 0;
  for (let i = 0; i < dataArray.length; i++) {
    const v = (dataArray[i] - 128) / 128;
    sum += v * v;
  }
  return Math.sqrt(sum / dataArray.length);
}

function recordAttempt({ maxWaitMs = 8000, silenceEndMs = 900, maxUtteranceMs = 8000, energyThreshold = 0.02 } = {}) {
  return new Promise((resolve) => {
    const chunks = [];
    const recorder = new MediaRecorder(micStream, { mimeType: pickMimeType() });
    recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };

    const data = new Uint8Array(analyser.fftSize);
    let speechStarted = null;
    let lastSpeechAt = null;
    const startedAt = performance.now();
    let stopped = false;

    function finish() {
      if (stopped) return;
      stopped = true;
      clearInterval(timer);
      if (recorder.state !== 'inactive') recorder.stop();
    }

    recorder.onstop = () => {
      if (!speechStarted) { resolve(null); return; }
      resolve(new Blob(chunks, { type: recorder.mimeType }));
    };

    const timer = setInterval(() => {
      analyser.getByteTimeDomainData(data);
      const level = rms(data);
      const now = performance.now();
      if (level >= energyThreshold) {
        if (!speechStarted) speechStarted = now;
        lastSpeechAt = now;
      }
      if (!speechStarted && (now - startedAt) > maxWaitMs) { finish(); return; }
      if (speechStarted && (now - speechStarted) > maxUtteranceMs) { finish(); return; }
      if (speechStarted && lastSpeechAt && (now - lastSpeechAt) > silenceEndMs) { finish(); return; }
    }, 100);

    recorder.start();
    MIC_DOT.classList.add('listening');
  }).finally(() => MIC_DOT.classList.remove('listening'));
}

function pickMimeType() {
  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg'];
  return candidates.find(t => MediaRecorder.isTypeSupported(t)) || '';
}

async function checkAttempt(blob, word) {
  const form = new FormData();
  const ext = (blob.type.split('/')[1] || 'webm').split(';')[0];
  form.append('audio', blob, `attempt.${ext}`);
  form.append('lesson_id', window.LESSON_ID);
  form.append('word', word);
  form.append('stanza', stanzaIdx);
  form.append('child', getChildName());
  const res = await fetch('/api/practice/word/check', { method: 'POST', body: form });
  return res.json();
}

// ---------------------------------------------------------------------
// EXTENSION: the WHOLE lesson/poem shown at once -- every stanza, not
// just the one currently being practiced -- with the word currently
// being practiced highlighted and every already-passed word (in the
// current stanza) or already-finished stanza (earlier stanzas) marked
// done, and stanzas not yet reached shown but dimmed. Built from
// `allStanzaWords`, fetched once up front in startHandsFree() via
// /api/lesson/<id>/all-words, plus a blank line between stanzas so the
// poem's stanza breaks are visible, matching how the reader sees the
// printed poem, not one flat run-on line.
// ---------------------------------------------------------------------
const READING_TEXT = document.getElementById('reading-text');

function renderReadingText() {
  if (!READING_TEXT) return;
  if (!allStanzaWords.length) {
    // Fallback for callers (or an old cached page) that haven't loaded
    // allStanzaWords yet -- behaves like the original current-stanza-only
    // rendering so nothing breaks.
    READING_TEXT.innerHTML = words.map((w, i) => {
      const cls = i === wordIdx ? 'current' : (i < wordIdx ? 'done' : '');
      return `<span class="word${cls ? ' ' + cls : ''}">${w}</span>`;
    }).join(' ');
    return;
  }
  READING_TEXT.innerHTML = allStanzaWords.map((stanzaWords, sIdx) => {
    const stanzaState = sIdx < stanzaIdx ? 'done' : (sIdx > stanzaIdx ? 'future' : 'active');
    const wordsHtml = stanzaWords.map((w, i) => {
      let cls;
      if (stanzaState === 'done') cls = 'done';
      else if (stanzaState === 'future') cls = '';
      else cls = i === wordIdx ? 'current' : (i < wordIdx ? 'done' : '');
      return `<span class="word${cls ? ' ' + cls : ''}">${w}</span>`;
    }).join(' ');
    return `<div class="stanza${stanzaState === 'future' ? ' stanza-future' : ''}">${wordsHtml}</div>`;
  }).join('');
}

// ---------------------------------------------------------------------
// EXTENSION: resume support. Position (stanza/word/phase) is saved to
// the server every time it changes, and read back once at the start of
// startHandsFree() -- so a child who stops mid-lesson (closes the tab,
// taps Stop, loses connection, whatever) picks up from exactly where
// they left off next time, instead of always restarting at stanza 0 /
// word 0. Fire-and-forget (not awaited by callers) since a save
// failing here should never block or slow down actual practice.
// ---------------------------------------------------------------------
function saveProgress(phase) {
  const form = new FormData();
  form.append('child', getChildName());
  form.append('stanza', stanzaIdx);
  form.append('word', wordIdx);
  form.append('phase', phase);
  fetch(`/api/lesson/${window.LESSON_ID}/save-position`, { method: 'POST', body: form }).catch(() => {});
}

// ---------------------------------------------------------------------
// Main per-word loop.
// ---------------------------------------------------------------------
async function runWord(isRepeat = false) {
  if (!running) return;
  renderReadingText();
  saveProgress('word');
  // EXTENSION FIX: a real, confirmed bug -- /api/lesson/<id>/words
  // returns a plain list of word STRINGS (e.g. ["I", "have", "a", ...]),
  // not a list of {word: "..."} objects. Treating each entry as an
  // object (the old `wordObj.word`) silently evaluated to `undefined`
  // for every word, which rendered as a BLANK word card (assigning
  // textContent = undefined shows as empty, not the text "undefined"),
  // and then `undefined.replace(...)` a few lines later threw and
  // silently killed the whole hands-free loop before the mic ever
  // started listening -- exactly the "stuck on Listen..., nothing
  // else happens" symptom. Each entry is used directly as the word
  // string now.
  const word = words[wordIdx];
  WORD_CARD.textContent = word;
  BADGE.textContent = '';
  setStatus(isRepeat ? 'Listen again...' : 'Listen...');
  await say(ckUiText(language, 'say_word', word.replace(/[.,!?]/g, '')), language, true);

  stopRecognition(); // don't let the command listener treat the attempt as a command
  setStatus('Your turn -- say it now.');
  const blob = await recordAttempt();
  startRecognition();

  if (!blob) {
    setStatus('I didn\'t hear anything.');
    await say(ckUiText(language, 'no_speech'), language, false);
    if (running) runWord();
    return;
  }

  setStatus('Checking...');
  const cancelReassurance = ckStartCheckingReassurance(language, setStatus);
  let result;
  try {
    result = await checkAttempt(blob, word);
  } catch (e) {
    cancelReassurance();
    setStatus('Couldn\'t check that -- try again.');
    await say(ckUiText(language, 'cant_check'), language, false);
    if (running) runWord();
    return;
  }
  cancelReassurance();

  BADGE.textContent = result.passed ? `🎉 ${result.feedback_text}` : `🔁 ${result.feedback_text}`;
  BADGE.className = 'ck-badge ' + (result.passed ? 'pass' : 'fail');
  if (result.passed) { showXpFloat(result.xp_awarded); }
  await say(result.feedback_text, language, false);

  if (!running) return;
  if (result.passed) { advanceWord(); return; }

  if (loseHeart()) {
    await handleOutOfHearts();
    if (!running) return;
  }
  runWord();
}

async function handleOutOfHearts() {
  setStatus('Out of hearts for now.');
  await say(ckUiText(language, 'out_of_hearts'), language, false);
  hearts = HEARTS_TOTAL; // soft reset -- see the hearts comment above: this never hard-stops a child's practice
  renderHearts();
}

function advanceWord() {
  wordIdx += 1;
  if (wordIdx >= words.length) {
    runUnitCheck('stanza', stanzaIdx);
    return;
  }
  runWord();
}

// ---------------------------------------------------------------------
// Whole-stanza / whole-poem hands-free check -- mirrors
// render_whole_unit_check's flow: after every word in a stanza has
// been practiced individually, the child listens to and reads back the
// WHOLE stanza; once every stanza is done, the same happens once more
// for the WHOLE poem, and only then is the lesson marked complete.
// ---------------------------------------------------------------------
async function runUnitCheck(phase, unitIdx, isRepeat = false) {
  if (!running) return;
  saveProgress(phase);
  const unitParam = phase === 'poem' ? 'poem' : unitIdx;
  const unit = await (await fetch(`/api/lesson/${window.LESSON_ID}/unit-text?stanza=${unitParam}`)).json();
  WORD_CARD.textContent = unit.text;
  if (READING_TEXT) READING_TEXT.innerHTML = `<span class="word current">${unit.text}</span>`;
  BADGE.textContent = '';
  setStatus(isRepeat ? 'Listen again...' : (phase === 'poem' ? unit.heading : unit.heading));
  await say(unit.text, language, false);

  stopRecognition();
  setStatus('Your turn -- read it back now.');
  const blob = await recordAttempt({ maxUtteranceMs: 15000 });
  startRecognition();

  if (!blob) {
    setStatus('I didn\'t hear anything.');
    await say(ckUiText(language, 'no_speech'), language, false);
    if (running) runUnitCheck(phase, unitIdx);
    return;
  }

  setStatus('Checking...');
  const cancelReassurance = ckStartCheckingReassurance(language, setStatus);
  let result;
  try {
    result = await checkUnitAttempt(blob, phase, unitIdx, unit.text);
  } catch (e) {
    cancelReassurance();
    setStatus('Couldn\'t check that -- try again.');
    await say(ckUiText(language, 'cant_check'), language, false);
    if (running) runUnitCheck(phase, unitIdx);
    return;
  }
  cancelReassurance();

  BADGE.textContent = result.passed ? `🌟 ${result.feedback_text}` : `🔁 ${result.feedback_text}`;
  BADGE.className = 'ck-badge ' + (result.passed ? 'pass' : 'fail');
  // EXTENSION: a whole-stanza/poem readback now earns proportional XP
  // even on a not-quite-passing attempt (e.g. 7 of 10 words), so the
  // floating "+N XP" should show whenever XP was actually awarded, not
  // only on a full pass -- partial credit a child can't see doesn't
  // feel like credit at all.
  if (result.xp_awarded > 0) { showXpFloat(result.xp_awarded); }
  await say(result.feedback_text, language, false);
  if (!running) return;

  if (!result.passed) {
    if (loseHeart()) { await handleOutOfHearts(); if (!running) return; }
    runUnitCheck(phase, unitIdx);
    return;
  }

  if (phase === 'stanza') {
    stanzaIdx += 1;
    if (stanzaIdx >= stanzaCount) {
      runUnitCheck('poem', stanzaCount);
    } else {
      // Already have every stanza's words from the single all-words
      // fetch in startHandsFree() -- no need to re-fetch per stanza.
      words = allStanzaWords[stanzaIdx] || [];
      wordIdx = 0;
      runWord();
    }
  } else {
    stopPractice(true);
  }
}

function checkUnitAttempt(blob, phase, unitIdx, text) {
  const form = new FormData();
  const ext = (blob.type.split('/')[1] || 'webm').split(';')[0];
  form.append('audio', blob, `attempt.${ext}`);
  form.append('lesson_id', window.LESSON_ID);
  form.append('phase', phase);
  form.append('stanza', unitIdx);
  form.append('text', text);
  form.append('child', getChildName());
  return fetch('/api/practice/unit/check', { method: 'POST', body: form }).then(r => r.json());
}

async function stopPractice(finished = false) {
  const wasRunning = running;
  running = false;
  stopRecognition();
  if (STOP_WRAP) STOP_WRAP.style.display = 'none';
  // A tap on the Stop button should feel instant, even if the app was
  // mid-way through saying a word or waiting for the child's voice --
  // pausing the player and closing the mic stream cuts both off right
  // away, rather than waiting for the current async step to notice
  // `running` went false on its own.
  if (!finished) { player.pause(); }
  if (micStream) micStream.getTracks().forEach(t => t.stop());
  if (!wasRunning && !finished) return; // already stopped -- avoid a duplicate "See you next time!"
  setStatus(finished ? ckUiText(language, 'complete_status') : ckUiText(language, 'stopped_status'));
  await say(finished ? ckUiText(language, 'finished') : ckUiText(language, 'see_you'), language, false);
  ENABLE_BTN.style.display = 'inline-block';
  ENABLE_BTN.textContent = ckUiText(language, 'practice_again_btn');
}

// ---------------------------------------------------------------------
// Entry point -- the ONE required tap.
// ---------------------------------------------------------------------
async function startHandsFree() {
  ENABLE_BTN.style.display = 'none';
  WORD_CARD.style.display = 'block';
  if (STOP_WRAP) STOP_WRAP.style.display = 'block';
  hearts = HEARTS_TOTAL;
  renderHearts();
  setStatus('Loading...');

  const allRes = await fetch(`/api/lesson/${window.LESSON_ID}/all-words`);
  const allData = await allRes.json();
  allStanzaWords = allData.stanzas;
  language = allData.language;
  stanzaCount = allData.stanza_count;

  // EXTENSION: resume from exactly where this child left off, instead
  // of always starting fresh at stanza 0 / word 0 -- see saveProgress()
  // above for where this position gets written during practice.
  let resumePhase = 'word';
  try {
    const resumeRes = await fetch(`/api/lesson/${window.LESSON_ID}/resume?child=${encodeURIComponent(getChildName())}`);
    const resumeData = await resumeRes.json();
    if (!resumeData.lesson_completed) {
      const savedStanza = resumeData.stanza_index || 0;
      if (savedStanza < allStanzaWords.length) {
        stanzaIdx = savedStanza;
        wordIdx = resumeData.word_index || 0;
        resumePhase = resumeData.phase || 'word';
      }
    }
  } catch (e) { /* resume is a convenience, never block starting practice over it */ }

  words = allStanzaWords[stanzaIdx] || [];
  if (wordIdx > words.length) wordIdx = 0; // guard against stale/out-of-range saved position

  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    setStatus('Microphone permission is needed for hands-free practice.');
    ENABLE_BTN.style.display = 'inline-block';
    if (STOP_WRAP) STOP_WRAP.style.display = 'none';
    return;
  }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(micStream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  source.connect(analyser);

  recognition = initRecognition();
  running = true;
  awaitingCommandOnly = false;
  startRecognition();

  if (resumePhase === 'stanza') {
    runUnitCheck('stanza', stanzaIdx);
  } else if (resumePhase === 'poem') {
    runUnitCheck('poem', stanzaCount);
  } else {
    runWord();
  }
}