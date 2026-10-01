/*
 * Custom Practice -- parent/teacher setup steps are plain click-driven
 * (an adult operating a recorder deliberately choosing when to start
 * and stop is normal and expected, same as any voice-memo app; the
 * "no buttons" requirement in this project is specifically about the
 * CHILD'S OWN practice turn). The final step -- the child listening
 * and reading the sentence back -- IS fully hands-free, reusing the
 * same RMS-based auto-stop recording approach as static/js/handsfree.js.
 */
let profileId = null;
let profileName = '';
let language = (window.LANGUAGES && window.LANGUAGES[0]) ? window.LANGUAGES[0][0] : 'en';
let customText = '';
let canClone = false; // whether a cloned voice can stand in for a NOT-yet-recorded sentence
const player = new Audio();

// Returns true once playback actually finished, false if the audio
// could not even be fetched (e.g. a 503 "no audio available" from a
// cloned-voice request that has nothing to clone from yet) -- callers
// that need to know whether the child actually heard anything (not
// just "did we try") check this return value instead of assuming
// silence always means "it played fine."
async function playClip(url) {
  let res;
  try {
    res = await fetch(url);
  } catch (e) {
    return false;
  }
  if (!res.ok) return false;
  const blobUrl = URL.createObjectURL(await res.blob());
  return new Promise((resolve) => {
    player.src = blobUrl;
    const cleanup = (ok) => { URL.revokeObjectURL(blobUrl); resolve(ok); };
    player.onended = () => cleanup(true);
    player.onerror = () => cleanup(false);
    player.play().catch(() => cleanup(false));
  });
}
function say(text, lang, slow) {
  return playClip(`/api/audio/say?text=${encodeURIComponent(text)}&lang=${lang}&slow=${slow ? 1 : 0}`);
}

// ---------------------------------------------------------------------
// Step 1: profile
// ---------------------------------------------------------------------
async function loadProfiles() {
  const res = await fetch('/api/voice/profiles');
  const profiles = await res.json();
  const sel = document.getElementById('profile-select');
  sel.innerHTML = '<option value="">— choose an existing voice —</option>' +
    profiles.map(p => `<option value="${p.id}" data-name="${p.name}">${p.name}</option>`).join('');
}

async function createProfile() {
  const sel = document.getElementById('profile-select');
  const newName = document.getElementById('new-profile-name').value.trim();
  if (newName) {
    const form = new FormData(); form.append('name', newName);
    const res = await fetch('/api/voice/profiles', { method: 'POST', body: form });
    const data = await res.json();
    profileId = data.id; profileName = data.name;
  } else if (sel.value) {
    profileId = parseInt(sel.value, 10);
    profileName = sel.selectedOptions[0].dataset.name;
  } else {
    alert('Choose an existing voice or type a new name.');
    return;
  }
  document.getElementById('lang-step').style.display = 'block';
  renderLangRadios();
}

function renderLangRadios() {
  const el = document.getElementById('lang-radios');
  el.innerHTML = (window.LANGUAGES || []).map(([code, label], i) => `
    <label style="margin-right:16px;"><input type="radio" name="lang" value="${code}" ${code===language?'checked':''} onchange="onLanguageChange('${code}')"> ${label}</label>
  `).join('');
  onLanguageChange(language);
}

async function onLanguageChange(code) {
  language = code;
  const info = await (await fetch(`/api/voice/language-info?lang=${language}`)).json();
  canClone = info.can_clone;
  document.getElementById('enroll-step').style.display = info.can_clone ? 'block' : 'none';
  if (info.can_clone) await refreshEnrollment();
  await loadSamples();
  document.getElementById('sentence-step').style.display = 'block';
}

// ---------------------------------------------------------------------
// Voice-cloning enrollment (click-to-record, parent-operated)
// ---------------------------------------------------------------------
let enrollStream = null, enrollRecorder = null, enrollChunks = [];

async function refreshEnrollment() {
  const data = await (await fetch(`/api/voice/enrollment?profile_id=${profileId}&lang=${language}`)).json();
  document.getElementById('enroll-progress').value = data.recorded;
  document.getElementById('enroll-progress').max = data.total;
  document.getElementById('enroll-status').textContent = `${data.recorded}/${data.total} phrases recorded`;
  const textEl = document.getElementById('enroll-text');
  const btn = document.getElementById('enroll-record-btn');
  if (data.next_idx === null) {
    textEl.textContent = '✅ All phrases recorded — cloning will use this voice for any typed sentence.';
    btn.style.display = 'none';
  } else {
    textEl.textContent = data.next_text;
    btn.style.display = 'inline-block';
    btn.dataset.idx = data.next_idx;
  }
}

async function toggleEnrollRecording() {
  const btn = document.getElementById('enroll-record-btn');
  if (!enrollRecorder || enrollRecorder.state === 'inactive') {
    enrollStream = enrollStream || await navigator.mediaDevices.getUserMedia({ audio: true });
    enrollChunks = [];
    enrollRecorder = new MediaRecorder(enrollStream, { mimeType: pickMimeType() });
    enrollRecorder.ondataavailable = (e) => { if (e.data.size > 0) enrollChunks.push(e.data); };
    enrollRecorder.onstop = async () => {
      const blob = new Blob(enrollChunks, { type: enrollRecorder.mimeType });
      const form = new FormData();
      form.append('audio', blob, 'clip.webm');
      form.append('profile_id', profileId); form.append('lang', language);
      form.append('sample_idx', btn.dataset.idx);
      await fetch('/api/voice/enrollment', { method: 'POST', body: form });
      await refreshEnrollment();
    };
    enrollRecorder.start();
    btn.textContent = '⏹ Stop recording'; btn.classList.add('recording');
  } else {
    enrollRecorder.stop();
    btn.textContent = '🔴 Record this phrase'; btn.classList.remove('recording');
  }
}

async function resetEnrollment() {
  const form = new FormData(); form.append('profile_id', profileId); form.append('lang', language);
  await fetch('/api/voice/enrollment/reset', { method: 'POST', body: form });
  await refreshEnrollment();
}

function pickMimeType() {
  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg'];
  return candidates.find(t => MediaRecorder.isTypeSupported(t)) || '';
}

// ---------------------------------------------------------------------
// Step 3: sentence choice
// ---------------------------------------------------------------------
async function loadSamples() {
  const samples = await (await fetch(`/api/voice/samples?lang=${language}`)).json();
  const sel = document.getElementById('sample-select');
  sel.innerHTML = '<option value="">Type my own...</option>' + samples.map(s => `<option value="${s}">${s}</option>`).join('');
}
function onSampleChange() {
  const sel = document.getElementById('sample-select');
  if (sel.value) document.getElementById('custom-text').value = sel.value;
}

async function useSentence() {
  const text = document.getElementById('custom-text').value.trim();
  if (!text) { alert('Type or pick a sentence first.'); return; }
  customText = text;
  const form = new FormData(); form.append('lang', language); form.append('text', text);
  await fetch('/api/voice/custom-text', { method: 'POST', body: form });
  document.getElementById('record-step').style.display = 'block';
  document.getElementById('record-text').textContent = customText;
  await checkSentenceRecorded();
}

// ---------------------------------------------------------------------
// Step 4: record the whole sentence (click-to-record, parent-operated)
// ---------------------------------------------------------------------
let sentenceStream = null, sentenceRecorder = null, sentenceChunks = [];

async function checkSentenceRecorded() {
  const data = await (await fetch(`/api/voice/sentence-recording/status?profile_id=${profileId}&lang=${language}&text=${encodeURIComponent(customText)}`)).json();
  document.getElementById('continue-btn').style.display = data.recorded ? 'inline-block' : 'none';
  const skipBtn = document.getElementById('skip-record-btn');
  const statusEl = document.getElementById('sentence-status');

  if (data.recorded) {
    statusEl.textContent = '✅ Recorded — your real voice will be used.';
    skipBtn.style.display = 'none';
    return;
  }
  if (!canClone) {
    statusEl.textContent = 'Not recorded yet -- recording is required (no voice cloning available for this language).';
    skipBtn.style.display = 'none';
    return;
  }
  const refCheck = await (await fetch(`/api/voice/has-reference?profile_id=${profileId}&lang=${language}`)).json();
  if (refCheck.available) {
    statusEl.textContent = 'Not recorded yet. Recording it yourself sounds most natural -- or skip and let the cloned voice handle it.';
    skipBtn.style.display = 'inline-block';
  } else {
    statusEl.textContent = 'Not recorded yet. Recording is needed at least once (or finish enrollment above) before cloning has a voice to work from.';
    skipBtn.style.display = 'none';
  }
}

async function toggleSentenceRecording() {
  const btn = document.getElementById('sentence-record-btn');
  if (!sentenceRecorder || sentenceRecorder.state === 'inactive') {
    sentenceStream = sentenceStream || await navigator.mediaDevices.getUserMedia({ audio: true });
    sentenceChunks = [];
    sentenceRecorder = new MediaRecorder(sentenceStream, { mimeType: pickMimeType() });
    sentenceRecorder.ondataavailable = (e) => { if (e.data.size > 0) sentenceChunks.push(e.data); };
    sentenceRecorder.onstop = async () => {
      const blob = new Blob(sentenceChunks, { type: sentenceRecorder.mimeType });
      const form = new FormData();
      form.append('audio', blob, 'sentence.webm');
      form.append('profile_id', profileId); form.append('lang', language); form.append('text', customText);
      await fetch('/api/voice/sentence-recording', { method: 'POST', body: form });
      await checkSentenceRecorded();
    };
    sentenceRecorder.start();
    btn.textContent = '⏹ Stop recording'; btn.classList.add('recording');
  } else {
    sentenceRecorder.stop();
    btn.textContent = '🔴 Record the sentence'; btn.classList.remove('recording');
  }
}

function showChildStep() {
  document.getElementById('child-step').style.display = 'block';
  document.getElementById('child-text').textContent = customText;
  document.getElementById('child-step').scrollIntoView({ behavior: 'smooth' });
}

// ---------------------------------------------------------------------
// Child's hands-free read-back turn.
// ---------------------------------------------------------------------
let childMicStream = null, childAudioCtx = null, childAnalyser = null;

function rms(dataArray) {
  let sum = 0;
  for (let i = 0; i < dataArray.length; i++) { const v = (dataArray[i] - 128) / 128; sum += v * v; }
  return Math.sqrt(sum / dataArray.length);
}

function recordAttempt({ maxWaitMs = 8000, silenceEndMs = 900, maxUtteranceMs = 10000, energyThreshold = 0.02 } = {}) {
  return new Promise((resolve) => {
    const chunks = [];
    const recorder = new MediaRecorder(childMicStream, { mimeType: pickMimeType() });
    recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    const data = new Uint8Array(childAnalyser.fftSize);
    let speechStarted = null, lastSpeechAt = null, stopped = false;
    const startedAt = performance.now();
    function finish() { if (stopped) return; stopped = true; clearInterval(timer); if (recorder.state !== 'inactive') recorder.stop(); }
    recorder.onstop = () => { resolve(speechStarted ? new Blob(chunks, { type: recorder.mimeType }) : null); };
    const timer = setInterval(() => {
      childAnalyser.getByteTimeDomainData(data);
      const level = rms(data); const now = performance.now();
      if (level >= energyThreshold) { if (!speechStarted) speechStarted = now; lastSpeechAt = now; }
      if (!speechStarted && (now - startedAt) > maxWaitMs) return finish();
      if (speechStarted && (now - speechStarted) > maxUtteranceMs) return finish();
      if (speechStarted && lastSpeechAt && (now - lastSpeechAt) > silenceEndMs) return finish();
    }, 100);
    recorder.start();
  });
}

async function startChildPractice() {
  document.getElementById('child-start-btn').style.display = 'none';
  try {
    childMicStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    document.getElementById('child-status').textContent = 'Microphone permission is needed.';
    document.getElementById('child-start-btn').style.display = 'inline-block';
    return;
  }
  childAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = childAudioCtx.createMediaStreamSource(childMicStream);
  childAnalyser = childAudioCtx.createAnalyser();
  childAnalyser.fftSize = 1024;
  source.connect(childAnalyser);
  runChildTurn();
}

async function runChildTurn() {
  const badge = document.getElementById('child-badge');
  badge.textContent = '';
  document.getElementById('child-status').textContent = 'Listen at a slower, clearer pace...';
  const played = await playClip(`/api/voice/sentence-audio?profile_id=${profileId}&lang=${language}&text=${encodeURIComponent(customText)}&mode=slow`);

  if (!played) {
    // No audio could be produced at all (e.g. the cloned voice failed --
    // check the Flask terminal's [voice_clone] log lines for why). Stop
    // here instead of silently asking the child to read back a sentence
    // they were never actually able to hear.
    badge.textContent = "⚠️ Couldn't play this sentence -- go back to Step 4 and record it yourself instead.";
    badge.className = 'ck-badge fail';
    document.getElementById('child-status').textContent = '';
    return;
  }

  document.getElementById('child-status').textContent = 'Your turn -- read it back now.';
  const blob = await recordAttempt();
  if (!blob) {
    document.getElementById('child-status').textContent = "I didn't hear anything.";
    await say(ckUiText(language, 'no_speech'), language, false);
    return runChildTurn();
  }

  document.getElementById('child-status').textContent = 'Checking...';
  const cancelReassurance = ckStartCheckingReassurance(language, (t) => {
    document.getElementById('child-status').textContent = t;
  });
  const form = new FormData();
  const ext = (blob.type.split('/')[1] || 'webm').split(';')[0];
  form.append('audio', blob, `attempt.${ext}`);
  form.append('profile_id', profileId); form.append('lang', language);
  form.append('text', customText); form.append('child', getChildName());

  let result;
  try {
    const res = await fetch('/api/voice/sentence-check', { method: 'POST', body: form });
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    result = await res.json();
  } catch (e) {
    cancelReassurance();
    document.getElementById('child-status').textContent = "Couldn't check that -- try again.";
    await say(ckUiText(language, 'cant_check'), language, false);
    return runChildTurn();
  }
  cancelReassurance();

  badge.textContent = result.passed ? `✅ ${result.feedback_text}` : `❌ ${result.feedback_text}`;
  badge.className = 'ck-badge ' + (result.passed ? 'pass' : 'fail');
  if (result.passed) showXpFloat(result.xp_awarded);
  await say(result.feedback_text, language, false);
  document.getElementById('child-status').textContent = result.passed
    ? 'Wonderful reading! 🎉' : 'Let\'s try that again.';
  if (!result.passed) return runChildTurn();

  // Passed: offer to keep going without a full page reload, instead of
  // just leaving the child stranded on a finished screen.
  document.getElementById('again-wrap').style.display = 'block';
}

// ---------------------------------------------------------------------
// EXTENSION: let a parent/teacher pick up a new sentence (same voice) or
// switch voices entirely (e.g. Mother -> Father) right after a child
// finishes, with no full browser reload. Both reuse the mic-cleanup
// helper below, then reset just the JS state + visible cards needed for
// whichever step they should land back on; everything else in this file
// (loadProfiles, the recording helpers, etc.) is untouched and still
// works exactly as before since these just call back into it.
// ---------------------------------------------------------------------
function stopChildMic() {
  if (childMicStream) {
    childMicStream.getTracks().forEach(t => t.stop());
    childMicStream = null;
  }
  if (childAudioCtx) {
    childAudioCtx.close().catch(() => {});
    childAudioCtx = null;
  }
  childAnalyser = null;
}

function newSentenceSameVoice() {
  stopChildMic();
  customText = '';
  document.getElementById('again-wrap').style.display = 'none';
  document.getElementById('child-step').style.display = 'none';
  document.getElementById('child-start-btn').style.display = 'inline-block';
  document.getElementById('child-badge').textContent = '';
  document.getElementById('child-badge').className = 'ck-badge';

  document.getElementById('record-step').style.display = 'none';
  document.getElementById('continue-btn').style.display = 'none';
  document.getElementById('skip-record-btn').style.display = 'none';
  document.getElementById('sentence-status').textContent = '';

  document.getElementById('custom-text').value = '';
  document.getElementById('sample-select').value = '';
  document.getElementById('sentence-step').style.display = 'block';
  document.getElementById('sentence-step').scrollIntoView({ behavior: 'smooth' });
}

function startOver() {
  stopChildMic();
  profileId = null;
  profileName = '';
  customText = '';

  document.getElementById('again-wrap').style.display = 'none';
  document.getElementById('child-step').style.display = 'none';
  document.getElementById('child-start-btn').style.display = 'inline-block';
  document.getElementById('child-badge').textContent = '';
  document.getElementById('child-badge').className = 'ck-badge';

  document.getElementById('record-step').style.display = 'none';
  document.getElementById('continue-btn').style.display = 'none';
  document.getElementById('skip-record-btn').style.display = 'none';
  document.getElementById('sentence-status').textContent = '';

  document.getElementById('custom-text').value = '';
  document.getElementById('sample-select').value = '';
  document.getElementById('sentence-step').style.display = 'none';

  document.getElementById('enroll-step').style.display = 'none';
  document.getElementById('lang-step').style.display = 'none';

  document.getElementById('new-profile-name').value = '';
  document.getElementById('profile-select').value = '';

  window.scrollTo({ top: 0, behavior: 'smooth' });
  loadProfiles();
}

loadProfiles();