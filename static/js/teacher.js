async function extract() {
  const fileInput = document.getElementById('pdf-file');
  const msg = document.getElementById('extract-msg');
  msg.style.color = '';
  msg.textContent = 'Reading PDF...';
  if (!fileInput.files.length) { msg.textContent = 'Choose a PDF first.'; return; }

  // EXTENSION: this whole function used to have no try/catch at all --
  // if the server ever returned something that wasn't valid JSON (its
  // own default HTML error page, for example, which happens whenever an
  // unexpected server-side exception isn't caught), `await res.json()`
  // threw, and because nothing here caught that, the function just
  // stopped silently: no error, no extracted text, nothing on the page
  // at all. Real symptom this caused directly: clicking "Extract text"
  // and seeing literally nothing happen. Now every failure path shows
  // SOMETHING concrete in red, so this never looks like a dead button
  // again -- check the browser's own console (F12) for the full
  // technical detail if this message alone isn't enough to diagnose it.
  try {
    const form = new FormData();
    form.append('pdf', fileInput.files[0]);
    form.append('lang', document.getElementById('lang-select').value);
    const res = await fetch('/api/teacher/extract', { method: 'POST', body: form });

    let data;
    try {
      data = await res.json();
    } catch (parseErr) {
      msg.style.color = 'var(--ck-red)';
      msg.textContent = `⚠️ Server returned an unexpected (non-JSON) response (HTTP ${res.status}). `
        + 'This usually means the Flask server hit an error it didn\'t handle -- check the '
        + 'terminal running flask_app.py for a traceback, and make sure every file was actually '
        + 'replaced (flask_app.py, modules/pdf_ingest.py, static/js/teacher.js all need to match).';
      console.error('extract(): non-JSON response', parseErr);
      return;
    }

    if (!res.ok) { msg.style.color = 'var(--ck-red)'; msg.textContent = data.error || 'Could not read that PDF.'; return; }

    document.getElementById('raw-text').value = data.text;
    if (data.used_ocr) {
      msg.style.color = '';
      msg.textContent = 'ℹ️ Used OCR to read this PDF (its embedded text looked garbled).';
    } else if (data.still_garbled) {
      // EXTENSION: this used to stay silent here -- the garbled text was
      // dropped into the box with no warning at all, looking exactly like
      // a successful extraction. Now it's a loud, specific warning naming
      // the real reason OCR couldn't fix it (see flask_app.py's
      // /api/teacher/extract and modules/pdf_ingest.diagnose_ocr_setup()),
      // so a teacher never saves unreadable text as a lesson by mistake.
      msg.style.color = 'var(--ck-red)';
      msg.textContent = '⚠️ This PDF\'s embedded text looks garbled (likely a legacy, '
        + 'non-Unicode font), and automatic OCR could not fix it'
        + (data.ocr_reason ? ' -- ' + data.ocr_reason + '.' : '.')
        + ' Review the text below carefully before saving -- it is probably not readable.';
    } else {
      msg.style.color = '';
      msg.textContent = '';
    }
    document.getElementById('edit-card').style.display = 'block';
    document.getElementById('edit-card').scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    msg.style.color = 'var(--ck-red)';
    msg.textContent = `⚠️ Could not reach the server (${err.message}). Is flask_app.py still running?`;
    console.error('extract(): request failed', err);
  }
}

async function split() {
  const text = document.getElementById('raw-text').value;
  const form = new FormData();
  form.append('text', text);
  const data = await (await fetch('/api/teacher/split', { method: 'POST', body: form })).json();
  const el = document.getElementById('chapters');
  el.innerHTML = data.chapters.map((c, i) => `
    <div class="chapter">
      <input type="text" placeholder="Chapter ${i + 1} title" data-idx="${i}" class="chapter-title">
      <textarea rows="8" class="chapter-text" data-idx="${i}">${c}</textarea>
    </div>
  `).join('');
  document.getElementById('chapters-card').style.display = 'block';
  document.getElementById('chapters-card').scrollIntoView({ behavior: 'smooth' });
}

async function saveAll() {
  const titles = [...document.querySelectorAll('.chapter-title')];
  const texts = [...document.querySelectorAll('.chapter-text')];
  const chapters = titles.map((t, i) => ({ title: t.value.trim(), text: texts[i].value }));
  const langSelect = document.getElementById('lang-select');
  const body = {
    lang: langSelect.value,
    lang_label: langSelect.selectedOptions[0].textContent,
    chapters,
    uploaded_by: 'teacher',
  };
  const res = await fetch('/api/teacher/save', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await res.json();
  const msg = document.getElementById('save-msg');
  if (!res.ok) { msg.textContent = data.error || 'Could not save.'; return; }
  msg.style.color = '#1BCFB4';
  msg.textContent = `🎉 Saved: ${data.saved.map(s => s.title).join(', ')}`;
  loadCloneLessons(); // a newly-saved lesson should show up in the clone-a-lesson dropdown right away
}

// ---------------------------------------------------------------------
// EXTENSION: clone a WHOLE saved lesson into one voice, right here in
// Teacher Upload -- separate code path from everything above (PDF
// extract/split/save) and from static/js/custom.js's single-sentence
// cloning flow. Just calls the existing backend cloning pipeline
// (/api/voice/clone-lesson etc., which itself reuses the same
// XTTS/Sarvam/OpenVoice tiers Custom Practice already uses) once per
// lesson instead of once per typed sentence.
// ---------------------------------------------------------------------
let cloneLessonId = null;
let cloneProfileId = null;
let clonedLines = [];
let cloningInFlight = false; // guards against double-clicking Clone, which used to restart the whole lesson from scratch mid-run

async function loadCloneLessons() {
  const langSelect = document.getElementById('lang-select');
  if (!langSelect) return;
  const lessons = await (await fetch(`/api/teacher/lessons?lang=${langSelect.value}`)).json();
  const sel = document.getElementById('clone-lesson-select');
  if (!sel) return;
  sel.innerHTML = lessons.length
    ? lessons.map(l => `<option value="${l.id}">${l.title}</option>`).join('')
    : '<option value="">No saved lessons in this language yet</option>';
  document.getElementById('clone-playback').style.display = 'none';
  loadCloneStanzas();
}

async function loadCloneProfiles() {
  const profiles = await (await fetch('/api/voice/profiles')).json();
  const sel = document.getElementById('clone-profile-select');
  if (!sel) return;
  sel.innerHTML = profiles.length
    ? profiles.map(p => `<option value="${p.id}">${p.name}</option>`).join('')
    : '<option value="">No voices recorded yet</option>';
}

// Populates "How much to clone" with one option per stanza of the
// currently-picked lesson (usually only 3-5 lines each -- much faster
// than the whole lesson) plus a "Whole lesson" option for convenience.
// Defaults to Stanza 1 so the fast path is what's selected out of the box.
async function loadCloneStanzas() {
  const lessonSel = document.getElementById('clone-lesson-select');
  const stanzaSel = document.getElementById('clone-stanza-select');
  if (!lessonSel || !stanzaSel || !lessonSel.value) {
    if (stanzaSel) stanzaSel.innerHTML = '';
    return;
  }
  const data = await (await fetch(`/api/lesson/${encodeURIComponent(lessonSel.value)}/all-words`)).json();
  const count = data.stanza_count || 1;
  let opts = '';
  for (let i = 0; i < count; i++) opts += `<option value="${i}">Stanza ${i + 1} only (fast)</option>`;
  opts += `<option value="all">Whole lesson (all ${count} stanza${count > 1 ? 's' : ''} -- slower)</option>`;
  stanzaSel.innerHTML = opts;
}

async function cloneLesson() {
  if (cloningInFlight) return; // ignore a second click while one clone is still running
  const lessonSel = document.getElementById('clone-lesson-select');
  const profileSel = document.getElementById('clone-profile-select');
  const stanzaSel = document.getElementById('clone-stanza-select');
  const btn = document.getElementById('clone-btn');
  const msg = document.getElementById('clone-msg');
  msg.style.color = ''; msg.textContent = '';
  if (!lessonSel.value) { msg.textContent = 'Pick a lesson first.'; return; }
  if (!profileSel.value) { msg.textContent = 'Pick a voice first (record one in Custom Practice if the list is empty).'; return; }

  const stanzaValue = stanzaSel && stanzaSel.value ? stanzaSel.value : 'all';
  const cloningWhole = stanzaValue === 'all';

  cloningInFlight = true;
  btn.disabled = true;
  btn.textContent = '⏳ Cloning...';
  document.getElementById('clone-progress-wrap').style.display = 'block';
  msg.textContent = cloningWhole
    ? 'Cloning every line of the whole lesson in this voice... on a CPU-only machine this can take a few MINUTES -- this is normal. Check the terminal window running the server for live line-by-line progress if you want to confirm it is not stuck.'
    : 'Cloning this stanza in this voice... usually well under a minute for a short stanza. Check the terminal for live line-by-line progress.';

  const form = new FormData();
  form.append('lesson_id', lessonSel.value);
  form.append('profile_id', profileSel.value);
  if (!cloningWhole) form.append('stanza', stanzaValue);
  let data;
  try {
    const res = await fetch('/api/voice/clone-lesson', { method: 'POST', body: form });
    data = await res.json();
    if (!res.ok) { msg.textContent = data.error || 'Could not clone this.'; return; }
  } catch (e) {
    msg.textContent = 'Something went wrong while cloning -- check the server terminal for details.';
    return;
  } finally {
    document.getElementById('clone-progress-wrap').style.display = 'none';
    btn.disabled = false;
    btn.textContent = '🎙️ Clone';
    cloningInFlight = false;
  }

  cloneLessonId = lessonSel.value;
  cloneProfileId = profileSel.value;

  // Refetch the FULL cached set for this lesson+voice (not just the
  // lines from this one call) -- so if stanza 1 was cloned earlier and
  // this call just added stanza 2, playback covers both, in order.
  const cacheRes = await fetch(`/api/voice/lesson-clone?lesson_id=${encodeURIComponent(cloneLessonId)}&profile_id=${cloneProfileId}`);
  const cacheData = await cacheRes.json();
  clonedLines = cacheData.lines || data.lines;

  msg.style.color = '#1BCFB4';
  msg.textContent = cloningWhole
    ? `🎉 Cloned ${data.cloned} of ${data.total} lines in ${profileSel.selectedOptions[0].textContent}'s voice.`
    : `🎉 Cloned ${data.cloned} of ${data.total} lines for this stanza. ${clonedLines.filter(l => l.ok).length} line(s) ready to play across everything cloned so far.`;
  if (clonedLines.some(l => l.ok)) document.getElementById('clone-playback').style.display = 'block';
}

// Plays every successfully-cloned line back to back, showing its text
// as it plays -- a simple, straight-through "listen to the whole lesson
// in this voice" experience, separate from the word-by-word hands-free
// Practice flow.
async function playClonedLesson() {
  const textEl = document.getElementById('clone-line-text');
  const player = new Audio();
  for (const line of clonedLines) {
    if (!line.ok) continue;
    textEl.textContent = line.text;
    const url = `/api/voice/lesson-clone-audio?lesson_id=${encodeURIComponent(cloneLessonId)}&profile_id=${cloneProfileId}&stanza=${line.stanza_idx}&line=${line.line_idx}`;
    await new Promise((resolve) => {
      player.src = url;
      player.onended = resolve;
      player.onerror = resolve;
      player.play().catch(resolve);
    });
  }
  textEl.textContent = '✅ Done!';
}

loadCloneLessons();
loadCloneProfiles();