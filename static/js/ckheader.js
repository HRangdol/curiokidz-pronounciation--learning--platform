/*
 * Shared header bar (mascot + child name + streak + XP), used on every
 * page. The child's name is remembered in this browser via
 * localStorage (a per-viewer convenience, not shared state) so it
 * doesn't have to be retyped on every screen -- click the name to
 * change it. Every page that includes this script gets a consistent
 * "who is this" identity without a real login system, matching how
 * the original Streamlit app's sidebar name field worked.
 */
function getChildName() {
  try {
    return localStorage.getItem('ck_child_name') || 'default';
  } catch (e) {
    return 'default';
  }
}

function setChildName(name) {
  try { localStorage.setItem('ck_child_name', name); } catch (e) { /* ignore */ }
}

function renameChild() {
  const current = getChildName();
  const next = prompt("What's your name?", current === 'default' ? '' : current);
  if (next && next.trim()) {
    setChildName(next.trim());
    location.reload();
  }
}

// ---------------------------------------------------------------------
// EXTENSION: spoken narration in the LESSON'S OWN language. The bug this
// fixes: handsfree.js and custom.js spoke fixed English sentences like
// "I didn't hear anything, let's try again" via say(text, language, ...)
// no matter what `language` actually was -- so a Kannada lesson handed
// its Kannada TTS voice an English sentence to read aloud, which came
// out mispronounced/garbled. (The actual pass/fail "Correct!"/score
// feedback text itself is now localized server-side in flask_app.py;
// this covers the rest of the spoken prompts that live in the browser.)
// Shared here (not duplicated in each JS file) since ckheader.js is
// already loaded on every page that needs it.
// ---------------------------------------------------------------------
const CK_UI_STRINGS = {
  en: {
    say_word: (w) => `Say the word: ${w}`,
    no_speech: "I did not hear anything. Let's try again.",
    cant_check: "Couldn't check that. Let's try again.",
    out_of_hearts: "You are out of hearts. Take a little breath, then let's keep going together.",
    finished: 'Great work! You finished the whole lesson.',
    see_you: 'See you next time!',
    // Visible (not spoken) UI labels -- these are plain text on
    // buttons/status lines that were always hardcoded English in the
    // HTML template regardless of lesson language. Covered here too so
    // a Kannada lesson's on-screen labels match its own language, not
    // just the spoken feedback.
    enable_btn: '🎙️ Enable microphone & start',
    practice_again_btn: '🎙️ Practice again',
    stop_btn: '⏹️ Stop for today',
    start_hint: 'Say "start" whenever you\'re ready, "repeat" to hear it again, or "stop" to end.',
    stopped_status: 'Stopped.',
    complete_status: 'Lesson complete! 🎉',
  },
  kn: {
    say_word: (w) => `ಈ ಪದವನ್ನು ಹೇಳು: ${w}`,
    no_speech: 'ನನಗೆ ಏನೂ ಕೇಳಿಸಲಿಲ್ಲ. ಮತ್ತೊಮ್ಮೆ ಪ್ರಯತ್ನಿಸೋಣ.',
    cant_check: 'ಅದನ್ನು ಪರಿಶೀಲಿಸಲು ಆಗಲಿಲ್ಲ. ಮತ್ತೊಮ್ಮೆ ಪ್ರಯತ್ನಿಸೋಣ.',
    out_of_hearts: 'ನಿನ್ನ ಹೃದಯಗಳು ಮುಗಿದಿವೆ. ಸ್ವಲ್ಪ ವಿಶ್ರಾಂತಿ ಪಡೆದು, ಮತ್ತೆ ಒಟ್ಟಿಗೆ ಮುಂದುವರಿಸೋಣ.',
    finished: 'ಅದ್ಭುತ ಕೆಲಸ! ನೀನು ಇಡೀ ಪಾಠವನ್ನು ಮುಗಿಸಿದೆ.',
    see_you: 'ಮತ್ತೆ ಸಿಗೋಣ!',
    enable_btn: '🎙️ ಮೈಕ್ರೋಫೋನ್ ಆನ್ ಮಾಡಿ ಪ್ರಾರಂಭಿಸಿ',
    practice_again_btn: '🎙️ ಮತ್ತೆ ಅಭ್ಯಾಸ ಮಾಡಿ',
    stop_btn: '⏹️ ಇಂದಿಗೆ ನಿಲ್ಲಿಸಿ',
    start_hint: '"start" ಎಂದು ಹೇಳಿ ಸಿದ್ಧವಾದಾಗ ಪ್ರಾರಂಭಿಸಿ, ಮತ್ತೆ ಕೇಳಲು "repeat" ಎಂದು ಹೇಳಿ, ಅಥವಾ ನಿಲ್ಲಿಸಲು "stop" ಎಂದು ಹೇಳಿ.',
    stopped_status: 'ನಿಲ್ಲಿಸಲಾಗಿದೆ.',
    complete_status: 'ಪಾಠ ಪೂರ್ಣಗೊಂಡಿದೆ! 🎉',
  },
};

// EXTENSION: reassurance for a real, inherent wait -- Kannada
// pronunciation checking uses a much larger, more accurate speech
// model than English does (see modules/asr.py's own comments: smaller
// models were confirmed to mis-recognize real Kannada speech), so a
// Kannada check can take genuinely longer on a CPU-only machine with
// no GPU. That's a deliberate accuracy trade-off, not a bug -- but
// with nothing on screen but a static "Checking..." for a long
// stretch, it looks exactly like the app has frozen. This starts a
// one-shot timer that, only if the check is still running a few
// seconds later, updates the status text to say so explicitly; call
// the returned function (with no arguments) as soon as the real result
// comes back, pass or fail, to cancel it if it hasn't fired yet.
function ckStartCheckingReassurance(language, setStatusFn) {
  if (language !== 'kn') return () => {};
  const timer = setTimeout(() => {
    setStatusFn('Still checking... Kannada checking uses a larger, more accurate '
      + 'model and can take longer, especially without a GPU.');
  }, 6000);
  return () => clearTimeout(timer);
}

function ckUiText(lang, key, ...args) {
  const table = CK_UI_STRINGS[lang] || CK_UI_STRINGS.en;
  const entry = table[key] !== undefined ? table[key] : CK_UI_STRINGS.en[key];
  return typeof entry === 'function' ? entry(...args) : entry;
}

// Shared "+N XP" floating animation + live XP-pill refresh -- used by
// both the hands-free lesson practice page and Custom Practice's
// child read-back check, any page that includes this script and has
// an element with id="xp-float" (see theme.css's .ck-xp-float).
function showXpFloat(amount) {
  if (!amount) return;
  const el = document.getElementById('xp-float');
  if (!el) return;
  el.textContent = `+${amount} XP`;
  el.classList.remove('show');
  void el.offsetWidth; // restart animation
  el.classList.add('show');
  const pill = document.getElementById('ck-xp-pill');
  if (pill) fetch(`/api/xp?child=${encodeURIComponent(getChildName())}`)
    .then(r => r.json()).then(d => { pill.textContent = `⭐ ${d.total_xp} XP`; });
}

async function renderCkHeader(targetId, title) {
  const el = document.getElementById(targetId);
  if (!el) return;
  const name = getChildName();
  el.innerHTML = `
    <div class="ck-brand">
      <span class="ck-mascot">🦜</span>
      <div>
        <h1>${title}</h1>
        <small onclick="renameChild()" style="cursor:pointer; color:var(--ck-muted);">👤 ${name} (change)</small>
      </div>
    </div>
    <div class="ck-pills">
      <span class="ck-pill streak" id="ck-streak-pill">🔥 …</span>
      <span class="ck-pill xp" id="ck-xp-pill">⭐ …</span>
    </div>
  `;
  try {
    const data = await (await fetch(`/api/dashboard?child=${encodeURIComponent(name)}`)).json();
    document.getElementById('ck-streak-pill').textContent = `🔥 ${data.streak}`;
    document.getElementById('ck-xp-pill').textContent = `⭐ ${data.xp} XP`;
  } catch (e) { /* header stats are decorative -- never block the page over them */ }
}