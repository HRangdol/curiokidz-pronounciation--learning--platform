"""
Persistent "voice cloning" for Custom Practice.

WHAT THIS ACTUALLY DOES, STATED HONESTLY (same principle as every other
module in this project -- audio.py and asr.py both say plainly what
their techniques do and don't do, and this module follows suit):

There are THREE tiers here, in the order this module actually tries
them, and they are NOT the same thing:

1. REAL RECORDED-VOICE REUSE (the reliable, always-available tier,
   both languages). A parent/teacher records themselves saying each
   word (and the full sentence) of whatever custom text they want the
   child to practice, once, using the exact same recording widget the
   child later uses. Those clips are saved to disk under
   data/voice_profiles/<profile>/ and re-used forever after -- the
   *actual* recording of that real person's voice, not a synthetic
   approximation. The next time ANY custom text reuses a word this
   profile already has on file (in the same language), it's instantly
   available with no new recording needed. Zero extra dependencies,
   works for both English and Kannada, and is what every OTHER tier
   below falls back to if it can't run.

2. OPTIONAL XTTS-v2 ZERO-SHOT TTS CLONING (bonus, English + ~16 other
   languages -- NOT Kannada). If the (large, optional) Coqui TTS
   package is installed, this can synthesize ARBITRARY new text in a
   voice cloned from a few seconds of the profile's recorded audio,
   directly. XTTS-v2's own documented language list genuinely does not
   include Kannada.

3. OPENVOICE V2 TONE-COLOUR CONVERSION (RECOMMENDED tier, ANY
   language, INCLUDING KANNADA -- see the long comment above
   _openvoice_clone below for the full, honest explanation). Uses the
   free, MIT-licensed, pip-installable `openvoice-cli` package (a
   packaging of MyShell/MIT's OpenVoice V2 -- confirmed to install and
   import cleanly). In short: this app's own existing TTS pipeline
   generates ordinary, correctly-pronounced speech in the target
   language first, and OpenVoice then re-voices that audio to sound
   like the profile's recorded reference -- real voice conversion, not
   a claim that any free model "speaks Kannada" in a cloned voice
   end-to-end. This is what makes true, arbitrary-text voice cloning
   (any random typed sentence, in the parent/teacher's own voice) work
   for BOTH languages, driven by the multi-phrase enrollment recording
   (10-15+ short phrases, see ENROLLMENT_SAMPLE_TEXTS below) as the
   speaker reference.

Tiers 2 and 3 are both skipped automatically, with no crash and no
silent lie about which tier actually produced a given clip, if their
(large, optional) libraries aren't installed -- exactly the graceful-
degradation pattern already used for edge-tts/gTTS/pyttsx3 in
modules/audio.py. Tier 1 always works regardless of what's installed.
"""
import os
import re
import json
import hashlib
import tempfile
import unicodedata

import numpy as np
import librosa
import soundfile as sf

from modules.db import get_connection

VOICE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "voice_profiles"
)
os.makedirs(VOICE_ROOT, exist_ok=True)


# ============================================================================
# XTTS-v2 availability (optional, heavy dependency -- see module docstring)
# ============================================================================
try:
    from TTS.api import TTS as _CoquiTTS
    XTTS_LIBRARY_AVAILABLE = True
except Exception:
    XTTS_LIBRARY_AVAILABLE = False

# XTTS-v2's own documented supported-language list. Kannada genuinely
# is not on it -- this is not a guess or a placeholder to fill in later.
_XTTS_SUPPORTED_LANGUAGES = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
    "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
}

_xtts_model = None
_xtts_load_failed = False


def xtts_available_for(language):
    """True only if the library is installed AND this specific language
    is one XTTS-v2 actually supports. Kannada always returns False."""
    return XTTS_LIBRARY_AVAILABLE and language in _XTTS_SUPPORTED_LANGUAGES


def any_cloning_available_for(language):
    """
    True if ANY true-cloning tier could apply to this language --
    XTTS-v2 (English and friends), Sarvam AI's cloud cloning (any
    language it supports, configured via SARVAM_API_KEY -- see
    modules/sarvam_voice.py), OR OpenVoice tone-colour conversion (any
    language, including Kannada, per _openvoice_clone below). Used by
    the UI to decide whether to even offer a "try voice-cloned
    synthesis" option, without caring which specific tier ends up
    doing the work.
    """
    from modules import sarvam_voice as _sarvam
    return xtts_available_for(language) or _sarvam.sarvam_available() or OPENVOICE_LIBRARY_AVAILABLE


def _get_xtts_model():
    global _xtts_model, _xtts_load_failed
    if _xtts_model is not None:
        return _xtts_model
    if _xtts_load_failed or not XTTS_LIBRARY_AVAILABLE:
        return None
    try:
        _xtts_model = _CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
        return _xtts_model
    except Exception:
        # First run downloads several GB of model weights; no internet,
        # no GPU driver quirks, or insufficient disk space all land
        # here -- fail once, remember it, and never crash the app for
        # this. Tier 1 (real recorded words) still works regardless.
        _xtts_load_failed = True
        return None


# ============================================================================
# Word/text normalization (kept local and simple -- deliberately not
# importing modules.asr here, to avoid loading Whisper just to hash a
# word key)
# ============================================================================
def _normalize(text):
    text = unicodedata.normalize("NFC", text or "")
    text = text.lower().strip()
    text = re.sub(r"[.,!?;:\"'‘’“”]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _word_key(word):
    return _normalize(word)


def _text_hash(text):
    return hashlib.md5(_normalize(text).encode("utf-8")).hexdigest()[:16]


def _safe_slug(name):
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
    return slug or "voice"


# ============================================================================
# Schema (mirrors modules/db.py's guarded-init convention for its own
# extension tables, e.g. uploaded_lessons)
# ============================================================================
def _init_tables():
    conn = get_connection()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS voice_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL COLLATE NOCASE,
            created_by TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS voice_words (
            profile_id INTEGER NOT NULL REFERENCES voice_profiles(id),
            language TEXT NOT NULL,
            word_key TEXT NOT NULL,
            display_word TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (profile_id, language, word_key)
        );

        CREATE TABLE IF NOT EXISTS voice_sentences (
            profile_id INTEGER NOT NULL REFERENCES voice_profiles(id),
            language TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            display_text TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (profile_id, language, text_hash)
        );

        CREATE TABLE IF NOT EXISTS custom_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            language TEXT NOT NULL,
            text TEXT NOT NULL,
            created_by TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        -- EXTENSION: dedicated voice-cloning ENROLLMENT samples, separate
        -- from ordinary practice-sentence recordings (voice_sentences
        -- above). A parent/teacher reads a fixed set of ~10-15 short,
        -- phonetically varied phrases ONCE per profile+language; these
        -- are combined into one longer reference clip (see
        -- _enrollment_reference_clip below) that XTTS-v2/OpenVoice use
        -- to clone that voice for ARBITRARY typed text afterward, not
        -- just text that happens to already have a real recording.
        -- A short single clip technically works for zero-shot cloning,
        -- but a longer, more varied reference is real, well-documented
        -- practice for a more stable, representative voice match.
        CREATE TABLE IF NOT EXISTS voice_enrollment_clips (
            profile_id INTEGER NOT NULL REFERENCES voice_profiles(id),
            language TEXT NOT NULL,
            sample_idx INTEGER NOT NULL,
            audio_path TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (profile_id, language, sample_idx)
        );

        -- EXTENSION: Sarvam AI cloud voice cloning (modules/sarvam_voice.py).
        -- Sarvam's own voice clones are cross-lingual (the same clone
        -- works for en-IN and kn-IN text), so ONE voice_id is cached per
        -- profile here, not per profile+language -- whichever language's
        -- reference clip happened to be available the first time this
        -- profile needed cloning is reused for every language after that.
        CREATE TABLE IF NOT EXISTS sarvam_voices (
            profile_id INTEGER PRIMARY KEY REFERENCES voice_profiles(id),
            voice_id TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        -- EXTENSION: whole-LESSON cloning, from the Teacher Upload tab --
        -- distinct from voice_sentences above (which is one sentence at a
        -- time from Custom Practice). Every line of an uploaded lesson is
        -- cloned once, in one chosen voice profile, and cached here so
        -- re-opening the same lesson+voice combo just replays it instead
        -- of re-cloning every line again (cloning can take a while per
        -- line, especially the OpenVoice fallback tier).
        CREATE TABLE IF NOT EXISTS lesson_clones (
            lesson_id TEXT NOT NULL,
            profile_id INTEGER NOT NULL REFERENCES voice_profiles(id),
            stanza_idx INTEGER NOT NULL,
            line_idx INTEGER NOT NULL,
            text TEXT NOT NULL,
            audio_path TEXT,
            tier TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (lesson_id, profile_id, stanza_idx, line_idx)
        );
        """
    )
    conn.commit()
    conn.close()


# ============================================================================
# EXTENSION: whole-lesson cloning storage + orchestration -- see the
# lesson_clones table comment above. save/get/has/delete mirror the
# simple per-row helper pattern already used everywhere else in this
# file (voice_words, voice_sentences, etc.).
# ============================================================================
_LESSON_CLONE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lesson_clone_audio"
)


def _save_lesson_clone_line(lesson_id, profile_id, stanza_idx, line_idx, text, audio_path, tier):
    _init_tables()
    conn = get_connection()
    conn.execute(
        """INSERT INTO lesson_clones (lesson_id, profile_id, stanza_idx, line_idx, text, audio_path, tier)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (lesson_id, profile_id, stanza_idx, line_idx)
           DO UPDATE SET text=excluded.text, audio_path=excluded.audio_path, tier=excluded.tier""",
        (lesson_id, profile_id, stanza_idx, line_idx, text, audio_path, tier),
    )
    conn.commit()
    conn.close()


def get_lesson_clone(lesson_id, profile_id):
    """Every cached cloned line for this lesson+voice, in reading order.
    Empty list if this lesson has never been cloned in this voice."""
    _init_tables()
    conn = get_connection()
    rows = conn.execute(
        """SELECT stanza_idx, line_idx, text, audio_path, tier FROM lesson_clones
           WHERE lesson_id = ? AND profile_id = ?
           ORDER BY stanza_idx, line_idx""",
        (lesson_id, profile_id),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_lesson_clone(lesson_id, profile_id):
    """Clears the cache for this lesson+voice so it can be re-cloned from
    scratch (e.g. after re-recording a cleaner enrollment)."""
    _init_tables()
    conn = get_connection()
    conn.execute("DELETE FROM lesson_clones WHERE lesson_id = ? AND profile_id = ?", (lesson_id, profile_id))
    conn.commit()
    conn.close()


def delete_lesson_clone_stanza(lesson_id, profile_id, stanza_idx):
    """Clears the cache for just ONE stanza of this lesson+voice --
    used by clone_lesson_stanza below so cloning stanza 2 doesn't wipe
    out stanza 1's already-cloned (and possibly slow-to-redo) lines."""
    _init_tables()
    conn = get_connection()
    conn.execute(
        "DELETE FROM lesson_clones WHERE lesson_id = ? AND profile_id = ? AND stanza_idx = ?",
        (lesson_id, profile_id, stanza_idx),
    )
    conn.commit()
    conn.close()


def _clone_lesson_lines(lesson_id, profile_id, language, stanza_items, log_label):
    """
    Shared per-line cloning loop used by both clone_whole_lesson and
    clone_lesson_stanza below -- `stanza_items` is a list of
    (stanza_idx, stanza_lines) pairs to clone (either every stanza, or
    just one). Always uses `allow_sarvam=False` (open-source tiers
    only: XTTS-v2, then OpenVoice V2 -- never the paid, closed-source
    Sarvam API), and prints per-line progress to the terminal so a slow
    clone is visibly still working, not stuck.
    """
    import time as _time
    os.makedirs(_LESSON_CLONE_DIR, exist_ok=True)
    results = []
    total_lines = sum(1 for _, lines in stanza_items for line in lines if line.strip())
    done_count = 0
    print(f"[voice_clone] {log_label}: starting -- {total_lines} lines to clone for "
          f"lesson={lesson_id!r} profile_id={profile_id} language={language!r}. Each line can take "
          f"anywhere from a few seconds to ~30s on CPU (OpenVoice tier especially) -- this is normal, "
          f"not stuck. Progress prints below as each line finishes.")
    for s_idx, stanza_lines in stanza_items:
        for l_idx, line_text in enumerate(stanza_lines):
            line_text = line_text.strip()
            if not line_text:
                continue
            done_count += 1
            out_path = os.path.join(
                _LESSON_CLONE_DIR, f"{lesson_id}_{profile_id}_{s_idx}_{l_idx}.wav"
            )
            tier = None
            _line_started = _time.monotonic()
            try:
                cloned = clone_synthesize(profile_id, line_text, language, out_path, allow_sarvam=False)
            except Exception as e:
                print(f"[voice_clone] {log_label}: line ({s_idx},{l_idx}) failed: {type(e).__name__}: {e}")
                cloned = None
            _line_secs = _time.monotonic() - _line_started
            print(f"[voice_clone] {log_label}: line {done_count}/{total_lines} "
                  f"(stanza {s_idx}, line {l_idx}) {'OK' if cloned else 'FAILED -- no reference/tier worked'} "
                  f"in {_line_secs:.1f}s -- {line_text[:60]!r}")
            if cloned:
                tier = "open-source"  # always XTTS or OpenVoice here -- allow_sarvam=False above
                _save_lesson_clone_line(lesson_id, profile_id, s_idx, l_idx, line_text, cloned, tier)
            else:
                _save_lesson_clone_line(lesson_id, profile_id, s_idx, l_idx, line_text, None, None)
            results.append({
                "stanza_idx": s_idx, "line_idx": l_idx, "text": line_text,
                "ok": bool(cloned),
            })
    return results


def clone_whole_lesson(lesson_id, lesson, profile_id, language):
    """
    Clones EVERY line of `lesson` (every stanza, in order) in the voice
    of `profile_id`. See _clone_lesson_lines above for how each line is
    actually cloned (open-source tiers only). Wipes and rebuilds the
    ENTIRE lesson's cache -- for a long lesson this can take several
    minutes on a CPU-only machine; clone_lesson_stanza below is the
    faster, incremental alternative (clone one stanza at a time).

    Returns a list of per-line result dicts:
        {"stanza_idx", "line_idx", "text", "ok"}
    so the caller (Teacher Upload UI) can show exactly which lines
    cloned successfully and which fell back to nothing (still readable
    via the app's normal, non-cloned voice -- see clone_synthesize's own
    docstring: None here just means "use the regular voice instead",
    never a hard failure of the whole feature).
    """
    delete_lesson_clone(lesson_id, profile_id)  # fresh clone each time this is invoked
    stanza_items = list(enumerate(lesson["stanzas"]))
    return _clone_lesson_lines(lesson_id, profile_id, language, stanza_items, "clone_whole_lesson")


def clone_lesson_stanza(lesson_id, lesson, profile_id, language, stanza_idx):
    """
    Clones just ONE stanza of `lesson` (the rest of the lesson's cache,
    if any, is left alone) -- much faster than clone_whole_lesson for a
    long lesson, since a stanza is usually only 3-5 lines instead of
    the whole thing. A teacher/parent can clone stanza 1 first to check
    the voice sounds right, then keep going stanza by stanza, and
    playback (get_lesson_clone) automatically picks up every stanza
    that's been cloned so far, in order.
    """
    if stanza_idx < 0 or stanza_idx >= len(lesson["stanzas"]):
        raise ValueError(f"stanza_idx {stanza_idx} out of range for lesson {lesson_id!r}")
    delete_lesson_clone_stanza(lesson_id, profile_id, stanza_idx)
    stanza_items = [(stanza_idx, lesson["stanzas"][stanza_idx])]
    return _clone_lesson_lines(lesson_id, profile_id, language, stanza_items, "clone_lesson_stanza")


# ============================================================================
# Voice profiles
# ============================================================================
def get_or_create_profile(name, created_by=None):
    _init_tables()
    name = name.strip()
    conn = get_connection()
    row = conn.execute(
        "SELECT id FROM voice_profiles WHERE name = ? COLLATE NOCASE", (name,)
    ).fetchone()
    if row:
        profile_id = row["id"]
    else:
        cur = conn.execute(
            "INSERT INTO voice_profiles (name, created_by) VALUES (?, ?)",
            (name, created_by),
        )
        conn.commit()
        profile_id = cur.lastrowid
    conn.close()
    return profile_id


def list_profiles():
    _init_tables()
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, name, created_by, created_at FROM voice_profiles ORDER BY name COLLATE NOCASE"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def profile_word_count(profile_id, language=None):
    _init_tables()
    conn = get_connection()
    if language:
        row = conn.execute(
            "SELECT COUNT(*) c FROM voice_words WHERE profile_id = ? AND language = ?",
            (profile_id, language),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) c FROM voice_words WHERE profile_id = ?", (profile_id,)
        ).fetchone()
    conn.close()
    return row["c"]


# ============================================================================
# Word-level recordings (the reliable tier)
# ============================================================================
def _profile_dir(profile_id, language):
    d = os.path.join(VOICE_ROOT, str(profile_id), language)
    os.makedirs(d, exist_ok=True)
    return d


def save_word_recording(profile_id, language, word, wav_bytes):
    _init_tables()
    key = _word_key(word)
    path = os.path.join(_profile_dir(profile_id, language), f"word_{key}.wav")
    with open(path, "wb") as f:
        f.write(wav_bytes)
    conn = get_connection()
    conn.execute(
        "INSERT INTO voice_words (profile_id, language, word_key, display_word, audio_path) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(profile_id, language, word_key) DO UPDATE SET "
        "audio_path = excluded.audio_path, display_word = excluded.display_word",
        (profile_id, language, key, word, path),
    )
    conn.commit()
    conn.close()
    return path


def get_word_recording(profile_id, language, word):
    _init_tables()
    key = _word_key(word)
    conn = get_connection()
    row = conn.execute(
        "SELECT audio_path FROM voice_words WHERE profile_id = ? AND language = ? AND word_key = ?",
        (profile_id, language, key),
    ).fetchone()
    conn.close()
    if row and os.path.exists(row["audio_path"]):
        return row["audio_path"]
    return None


def missing_words_for_text(profile_id, language, text):
    """Every word in `text` that this profile does NOT already have a
    real recorded clip for (in this language) -- persistent across
    every past custom-practice session, so a word recorded once for
    one sentence is never re-asked for in a later, different sentence."""
    words = text.split()
    return [w for w in words if get_word_recording(profile_id, language, w) is None]


# ============================================================================
# EXTENSION: auto-slicing a single sentence recording into per-word clips.
#
# Originally a parent/teacher had to record every individual word AND
# (optionally) the whole sentence -- real recorded audio, but real setup
# effort too. This slices the ONE sentence recording into one short clip
# per word automatically, so a parent/teacher never records word by word
# at all, but the child still gets a genuine per-word clip in that same
# real voice for word-by-word practice, not just the full sentence.
#
# Two tiers, tried in order, same honest-fallback spirit as every other
# module here:
#
# 1. REAL WORD-LEVEL ALIGNMENT, via asr.transcribe_word_timestamps
#    (faster-whisper's own word-timestamp feature). When it recognizes
#    the same NUMBER of words as the target sentence, each word's own
#    recognized start/end time in the recording is used to cut a real,
#    time-aligned clip of just that word.
# 2. EQUAL-DURATION FALLBACK. If ASR isn't installed, fails, or
#    recognizes a different word count than the target text (a paused
#    or mumbled recording, a short/child-unfriendly ASR model, etc.),
#    the whole recording's duration is divided evenly across the target
#    words in order instead. This is a real clip of the parent's actual
#    voice either way -- just not trimmed exactly to that one word's own
#    boundaries, which is stated here plainly rather than silently
#    passed off as a precise alignment.
# ============================================================================
def _wav_bytes_to_temp_file(wav_bytes):
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(wav_bytes)
    return path


def slice_sentence_into_words(language, text, wav_bytes):
    """
    Splits ONE recording of `text` (raw wav bytes) into one short clip
    per word of `text`. Returns {word_index: wav_bytes}, where
    word_index lines up 1:1 with text.split() (so the caller can map
    each slice back to its word, including repeated words, correctly).
    Returns {} if the recording can't be read at all.
    """
    target_words = text.split()
    if not target_words:
        return {}

    src_path = _wav_bytes_to_temp_file(wav_bytes)
    try:
        # EXTENSION FIX: a real report was that sliced words sounded
        # "muffled" even after the click/volume fixes -- and this is
        # exactly what forcing sr=16000 here would cause. 16kHz is
        # plenty for SPEECH RECOGNITION (that's why asr.py uses it), but
        # it throws away everything above 8kHz, which is real acoustic
        # information a human ear actually uses for clarity/crispness
        # (sibilants like "s"/"t"/"f" especially) -- a genuine, audible
        # quality loss for PLAYBACK, not a placebo. Loading at the
        # recording's own native rate (sr=None) instead keeps that
        # clarity in the saved word clips. This doesn't cost anything
        # for word-timestamp accuracy either: asr.transcribe_word_
        # timestamps is handed `src_path` (the original file) below, not
        # this array, so it does its own internal resampling regardless
        # of what rate this array is loaded at.
        y, sr = librosa.load(src_path, sr=None)
    except Exception as e:
        print(f"[voice_clone.slice_sentence_into_words] couldn't read recording: {e}")
        os.remove(src_path)
        return {}

    total_duration = len(y) / sr
    n = len(target_words)
    step = total_duration / n if n else 0
    # Always computed, even when ASR succeeds -- this is the safety net
    # every individual word falls back to below, not just a last resort
    # for the whole recording.
    equal_boundaries = [(i * step, (i + 1) * step) for i in range(n)]

    try:
        from modules import asr as _asr
        recognized = _asr.transcribe_word_timestamps(src_path, language, expected_text=text)
    except Exception as e:
        print(f"[voice_clone.slice_sentence_into_words] word-timestamp ASR failed, using equal-duration fallback: {e}")
        recognized = None

    # EXTENSION FIX: a real report was of a word (the FIRST word of a
    # long recording, specifically) coming back completely silent
    # (0:00 / 0:00 in the player) and staying that way even after
    # re-recording. Root cause, confirmed by reasoning through the
    # code: real word-timestamp ASR is well documented to be LEAST
    # reliable right at the very start of a clip (VAD warm-up, leading
    # silence before speech actually begins) -- so it's specifically
    # the first word or two whose recognized start/end can come back
    # zero-length, or even end <= start after padding. That used to
    # make THIS FUNCTION SKIP the word entirely -- no clip ever saved
    # for it, so any old file from a previous recording (possibly
    # itself broken) just sat there forever, silently, since nothing
    # ever overwrote it. Fixed by validating each word's own boundary
    # individually and falling back to that ONE word's equal-duration
    # slice whenever ASR's own answer for it isn't usable, rather than
    # trusting ASR wholesale (all words) or not at all.
    _MIN_WORD_SECONDS = 0.18
    pad = 0.12

    # EXTENSION FIX: a real report was that words in a LONG, multi-
    # sentence recording were unclear and "stopped in between" -- i.e.
    # a slice of dead silence in the middle, or parts of two different
    # words mashed together. Root cause: the old code demanded ASR's
    # recognized word count match text.split()'s count EXACTLY, or it
    # threw away ALL of ASR's timing and divided the WHOLE recording
    # evenly instead. For a short sentence that's rarely an issue, but
    # for a longer passage (several sentences, punctuation, numbers) a
    # count mismatch is common -- Whisper's own word segmentation
    # doesn't always land on the same count as a plain whitespace split
    # -- and a single global equal-division across a long recording is
    # a bad approximation: real words vary a lot in length, and real
    # multi-sentence speech has actual pauses (at commas/periods) that
    # don't correspond to any word at all, so by partway through, the
    # evenly-spaced guess is badly out of sync with where the words
    # actually are.
    #
    # Fixed by aligning target_words against recognized_words as
    # SEQUENCES (via difflib, the same technique used for text diffing)
    # instead of only comparing their counts. Words that align cleanly
    # keep their real ASR timing; only the words in between two aligned
    # anchors (or before the first / after the last) fall back to an
    # even split, and that split is done LOCALLY within the actual gap
    # between two known-good anchors -- not across the whole recording
    # -- which stays accurate even when just a few words in the middle
    # didn't match ASR's own segmentation.
    def _loose_norm(w):
        return re.sub(r"[^\w]", "", (w or "").lower())

    anchors = [None] * n  # (start, end) per target-word index, or None
    if recognized:
        import difflib
        target_norm = [_loose_norm(w) for w in target_words]
        recog_norm = [_loose_norm(w["word"]) for w in recognized]
        matcher = difflib.SequenceMatcher(a=recog_norm, b=target_norm, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                continue
            for k in range(i2 - i1):
                idx = i1 + k
                w = recognized[idx]
                start = max(0.0, w["start"] - pad)
                # EXTENSION FIX: a real report was that "lengthy words are
                # pronounced half" -- the trailing end getting cut off.
                # This is a well-documented real limitation of word-level
                # ASR timing generally, not specific to this app: per-word
                # end boundaries are derived from the model's internal
                # attention alignment, which tends to be LESS precise the
                # longer/more multi-syllable a word is -- the model can
                # "call it done" acoustically before the person has
                # actually finished saying the whole word. A flat pad
                # helps a little but isn't enough for a long word
                # specifically. Scaling the END-side pad by the word's own
                # recognized duration gives longer words proportionally
                # more trailing room, while short words (where this
                # problem doesn't occur) are barely affected. The
                # neighbour-boundary clamp right below still applies
                # after this, so a genuinely tight recording still can't
                # bleed into the next word -- this only claims MORE room
                # when there's actually room to give.
                word_len = w["end"] - w["start"]
                end_pad = pad + 0.25 * word_len
                end = min(total_duration, w["end"] + end_pad)
                # EXTENSION FIX: a real report was the flip side of the
                # earlier "cut off" fix -- widening the padding to stop
                # words being clipped can instead make the padding spill
                # INTO a neighbouring word's own speech when two words
                # are spoken close together with little gap (very common
                # in natural, connected reading), so the child then hears
                # part of the previous or next word too. Padding must
                # never be allowed to reach past the actual boundary of
                # the adjacent recognized word on either side -- clamping
                # to that (rather than to a fixed time) means a wide gap
                # still gets full breathing room, but two closely-spoken
                # words never bleed into each other regardless of pad
                # size.
                if idx > 0:
                    start = max(start, recognized[idx - 1]["end"])
                if idx < len(recognized) - 1:
                    end = min(end, recognized[idx + 1]["start"])
                if end - start > _MIN_WORD_SECONDS:
                    anchors[j1 + k] = (start, end)

    if all(a is None for a in anchors):
        # No usable ASR anchor at all (ASR unavailable, or nothing
        # aligned) -- the only remaining option is one even split
        # across the whole recording, same as before.
        boundaries = equal_boundaries
    else:
        boundaries = [None] * n
        i = 0
        prev_end = 0.0
        while i < n:
            if anchors[i] is not None:
                boundaries[i] = anchors[i]
                prev_end = anchors[i][1]
                i += 1
                continue
            j = i
            while j < n and anchors[j] is None:
                j += 1
            next_start = anchors[j][0] if j < n else total_duration
            gap_start, gap_end = prev_end, max(prev_end, next_start)
            run_len = j - i
            step = (gap_end - gap_start) / run_len if run_len else 0
            for k in range(run_len):
                boundaries[i + k] = (gap_start + k * step, gap_start + (k + 1) * step)
            prev_end = gap_end
            i = j

    # EXTENSION FIX: a real, confirmed bug -- found by directly inspecting
    # an actual reported-bad file (word_transport.wav) against the
    # sentence it came from. Cross-correlation showed the clip was NOT
    # random noise -- it was "Transport" + a genuine silent pause +
    # the START of the NEXT word ("means"), all inside one clip. That's
    # exactly what "broken in the middle" sounds like: a real word, a
    # gap, then another word beginning.
    #
    # Root cause: every fix so far (padding, the long-word tail stretch,
    # the neighbour clamp) has been a TIME-based heuristic, trusting
    # ASR's own reported start/end numbers. Real word-timestamp ASR can
    # be off in EITHER direction (too early OR too late) on either
    # boundary, and when the long-word tail-stretch adds extra room
    # AND the neighbour's own reported start is itself a bit late, the
    # two errors compound -- exactly what happened here.
    #
    # Fixed with a completely different, more reliable signal: after
    # computing each candidate boundary as before, SNAP it to the
    # nearest genuine silence in the ACTUAL recorded audio, found by
    # scanning the real energy (RMS) of the waveform itself -- not a
    # guessed time offset. This can only pull a boundary INWARD, toward
    # a real quiet gap already inside the current candidate window,
    # never push it further out, so it can only trim away wrongly-
    # captured extra content (a pause + next/previous word), never
    # re-introduce a cut-off. It only snaps when it finds a clearly
    # quieter spot (well below the loudness of the word itself) --
    # a soft consonant inside the real word won't trigger it.
    def _snap_to_silence(t, search_radius, prefer_earlier):
        # EXTENSION FIX: a real follow-up report -- SOME long words were
        # now getting cut in half. Root cause: a long, multi-syllable
        # word can have a genuine brief quiet moment INSIDE it (the
        # silent closure before a hard consonant like the "t" in
        # "transport", or a natural break between syllables), and the
        # first version of this snap-to-silence logic could mistake
        # that brief internal dip for the real inter-word pause,
        # cutting the word off right there. Real inter-word pauses are
        # both QUIETER and LONGER-LASTING than a within-word consonant
        # closure, so this now requires BOTH: a stricter loudness
        # threshold (was 35% of the window's peak, now 22%) AND a
        # sustained low-energy stretch of at least ~60ms, not just one
        # brief dip -- and picks the LONGEST such stretch in the search
        # window, which is the one most likely to be a genuine pause
        # rather than a consonant closure.
        win = max(1, int(0.02 * sr))  # 20ms energy window
        step = max(1, win // 4)
        center = int(t * sr)
        radius = int(search_radius * sr)
        if prefer_earlier:
            lo, hi = max(0, center - radius), center
        else:
            lo, hi = center, min(len(y), center + radius)
        if hi - lo < win:
            return t
        rms_vals, positions = [], []
        i = lo
        while i + win <= hi:
            seg = y[i:i + win]
            rms_vals.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))))
            positions.append(i + win // 2)
            i += step
        if not rms_vals:
            return t
        rms_vals = np.array(rms_vals)
        peak = rms_vals.max()
        if peak <= 1e-6:
            return t
        threshold = 0.22 * peak
        below = rms_vals < threshold
        min_run = max(1, int(round(0.06 * sr / step)))  # ~60ms minimum
        best_start, best_end, best_len = None, None, 0
        run_start = None
        for idx, is_low in enumerate(below):
            if is_low:
                if run_start is None:
                    run_start = idx
            elif run_start is not None:
                run_len = idx - run_start
                if run_len >= min_run and run_len > best_len:
                    best_len, best_start, best_end = run_len, run_start, idx
                run_start = None
        if run_start is not None:
            run_len = len(below) - run_start
            if run_len >= min_run and run_len > best_len:
                best_len, best_start, best_end = run_len, run_start, len(below)
        if best_start is None:
            # No sustained-enough quiet stretch found -- likely just a
            # within-word dip, or no real pause in range at all. Leave
            # the boundary exactly where it was rather than guessing.
            return t
        mid = (best_start + best_end) // 2
        return positions[mid] / sr

    refined_boundaries = []
    for i, (start, end) in enumerate(boundaries):
        # END can only move EARLIER (trims a trailing pause + next word
        # that shouldn't be there); START can only move LATER (trims a
        # leading pause + previous word). Never the reverse -- that
        # would risk re-introducing a cut-off word.
        new_end = _snap_to_silence(end, search_radius=0.2, prefer_earlier=True)
        new_start = _snap_to_silence(start, search_radius=0.2, prefer_earlier=False)
        if new_start >= start and new_end <= end and new_end - new_start > _MIN_WORD_SECONDS:
            refined_boundaries.append((new_start, new_end))
        else:
            refined_boundaries.append((start, end))
    boundaries = refined_boundaries

    os.remove(src_path)

    clips = {}
    for i, (start, end) in enumerate(boundaries):
        start_sample = max(0, int(start * sr))
        end_sample = min(len(y), int(end * sr))
        if end_sample <= start_sample:
            # Should be rare now that every word has a validated
            # boundary, but still don't silently save a broken/empty
            # clip if it somehow happens (e.g. total_duration itself
            # was 0 -- the whole recording failed to read properly).
            continue
        segment = y[start_sample:end_sample]
        # EXTENSION FIX: a real report was that sliced word clips "can't
        # be heard." The slicing boundaries (whether from ASR timestamps
        # or the equal-duration fallback) are an approximation, not a
        # lab-grade cut -- a slice can easily land mostly on a quiet
        # trailing/leading sound, or simply capture a softly-spoken
        # moment of a longer sentence, and a whole-sentence recording is
        # very often quieter per-word than a deliberately loud single-
        # word recording would have been. That produces a technically
        # valid but near-silent WAV, which plays with no crash and no
        # error -- just nothing audible, exactly what was reported.
        # Peak-normalizing every slice to a consistent, clearly audible
        # level fixes this regardless of why the original moment was
        # quiet, without needing to know the exact cause on any given
        # recording.
        peak = float(np.max(np.abs(segment))) if len(segment) else 0.0
        if peak > 1e-5:
            segment = segment * (0.92 / peak)
            segment = np.clip(segment, -1.0, 1.0)
        # Fade the first/last ~12ms in and out -- removes the audible
        # click/pop that comes from cutting a continuous waveform at an
        # arbitrary sample point (not a zero-crossing), which otherwise
        # makes an already-correct, audible clip still sound harsh or
        # "off" right at its start and end.
        fade_len = min(int(0.012 * sr), len(segment) // 2)
        if fade_len > 1:
            segment = segment.copy()
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            segment[:fade_len] *= fade_in
            segment[-fade_len:] *= fade_out
        out_path = _wav_bytes_to_temp_file(b"")
        try:
            sf.write(out_path, segment, sr, format="WAV")
            with open(out_path, "rb") as f:
                clips[i] = f.read()
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)
    return clips


def save_sentence_recording_and_autoslice(profile_id, language, text, wav_bytes):
    """
    Saves the parent/teacher's ONE whole-sentence recording (same as
    save_sentence_recording below) AND automatically slices it into a
    real per-word clip for every word in `text`, saving each one as a
    normal word recording too (via save_word_recording) -- so
    missing_words_for_text() comes back empty right away, and the
    parent/teacher never has to record word by word at all. If slicing
    fails for any reason, the sentence recording is still saved and
    kept -- only the automatic per-word split is skipped, never a crash.
    Returns the sentence clip's saved path.
    """
    sentence_path = save_sentence_recording(profile_id, language, text, wav_bytes)
    try:
        clips = slice_sentence_into_words(language, text, wav_bytes)
    except Exception as e:
        import traceback
        print(f"[voice_clone] auto-slicing into words failed, sentence-only recording kept: {e}")
        traceback.print_exc()
        clips = {}
    for i, word in enumerate(text.split()):
        clip_bytes = clips.get(i)
        if clip_bytes:
            save_word_recording(profile_id, language, word, clip_bytes)
    return sentence_path


# ============================================================================
# Whole-sentence recordings (for the final whole-unit "read it all" check)
# ============================================================================
def save_sentence_recording(profile_id, language, text, wav_bytes):
    _init_tables()
    h = _text_hash(text)
    path = os.path.join(_profile_dir(profile_id, language), f"sentence_{h}.wav")
    with open(path, "wb") as f:
        f.write(wav_bytes)
    conn = get_connection()
    conn.execute(
        "INSERT INTO voice_sentences (profile_id, language, text_hash, display_text, audio_path) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(profile_id, language, text_hash) DO UPDATE SET "
        "audio_path = excluded.audio_path, display_text = excluded.display_text",
        (profile_id, language, h, text, path),
    )
    conn.commit()
    conn.close()
    return path


def get_sentence_recording(profile_id, language, text):
    _init_tables()
    h = _text_hash(text)
    conn = get_connection()
    row = conn.execute(
        "SELECT audio_path FROM voice_sentences WHERE profile_id = ? AND language = ? AND text_hash = ?",
        (profile_id, language, h),
    ).fetchone()
    conn.close()
    if row and os.path.exists(row["audio_path"]):
        return row["audio_path"]
    return None


# EXTENSION: real feature-shape change, per explicit follow-up request --
# word-by-word splitting/practice was removed from Custom Practice
# entirely (see app.py's render_custom_practice), specifically because
# the automatic-speech-recognition slicing step it depended on
# (slice_sentence_into_words, used by save_sentence_recording_and_
# autoslice below) was reported as taking too long, especially for
# Kannada's large, most-accurate model. Removing that ASR step
# entirely -- rather than trying to speed it up further -- is both
# what was actually asked for AND removes the wait outright, since a
# sentence recording is now just saved directly (see app.py), no
# transcription involved at all.
#
# In its place, the parent/teacher's real recording is now the thing
# the child learns from directly: played back at a SLOWED pace so a
# young child can follow the words more easily than at natural
# conversational speed, while still being the real, familiar voice,
# not a synthetic one. This uses time-stretching (a phase-vocoder
# technique, via librosa), which changes DURATION only -- unlike
# simply playing the audio at a lower sample rate (which would also
# drop the pitch and make the voice sound artificially deep/slow, a
# "record player running down" effect) -- so the voice still sounds
# like the same person, just speaking more slowly.
_SLOW_PLAYBACK_RATE = 0.72  # 1.0 = normal speed; lower = slower. Chosen
# to be clearly, noticeably slower without stretching pauses so long
# that the sentence becomes hard to follow as a whole.


def get_slow_sentence_recording(profile_id, language, text, rate=_SLOW_PLAYBACK_RATE):
    """
    Returns a slowed-down (pitch-preserved) version of this profile's
    real sentence recording, generating and caching it on first use.
    Returns None if there is no real sentence recording for this
    profile/language/text yet (nothing to slow down), or if the
    slow-down itself fails for any reason -- callers should fall back
    to the normal-speed recording in that case, never treat this as
    required.
    """
    real_path = get_sentence_recording(profile_id, language, text)
    if real_path is None:
        return None
    h = _text_hash(text)
    rate_tag = str(rate).replace(".", "")
    slow_path = os.path.join(_profile_dir(profile_id, language), f"sentence_{h}_slow{rate_tag}.wav")
    # Regenerate only if missing, or older than the real recording it's
    # derived from (e.g. after a genuine re-record) -- cheap check,
    # avoids redoing the time-stretch on every single practice visit.
    if os.path.exists(slow_path) and os.path.getmtime(slow_path) >= os.path.getmtime(real_path):
        return slow_path
    try:
        y, sr = librosa.load(real_path, sr=None)
        y_slow = librosa.effects.time_stretch(y, rate=rate)
        # Same peak-normalization already used for sliced word clips,
        # for the same reason -- a real recording's own loudness varies,
        # and time-stretching itself does not renormalize it.
        peak = float(np.max(np.abs(y_slow))) if len(y_slow) else 0.0
        if peak > 0:
            y_slow = np.clip(y_slow * (0.92 / peak), -1.0, 1.0)
        sf.write(slow_path, y_slow, sr, format="WAV")
        return slow_path
    except Exception as e:
        print(f"[voice_clone.get_slow_sentence_recording] FAILED, falling back to normal speed: {e}")
        return None


# ============================================================================
# EXTENSION FIX: a real, confirmed bug -- there was previously NO way to
# re-record the same sentence for the same voice profile. render_custom_
# practice's "record" phase checks get_sentence_recording() and, the
# moment ANY recording already exists for this exact profile+language+
# text, skips the recording step entirely and jumps to "choose" -- every
# time, forever, for that exact sentence. That meant repeated slicing-
# quality fixes in this module could never actually be exercised on a
# fresh recording for a sentence already tried once; the child was
# always still hearing the OLD word clips, sliced by whatever code
# existed the first time that sentence was ever recorded. This deletes
# the saved sentence AND every per-word clip that came from THIS
# sentence specifically (so a fresh recording fully replaces them,
# rather than leaving a few stale old slices around for words whose
# NEW slice boundaries might differ) -- app.py's "Re-record this
# sentence" button calls this before resetting back to the "record"
# phase.
# ============================================================================
def delete_sentence_recording(profile_id, language, text):
    _init_tables()
    h = _text_hash(text)
    conn = get_connection()
    row = conn.execute(
        "SELECT audio_path FROM voice_sentences WHERE profile_id = ? AND language = ? AND text_hash = ?",
        (profile_id, language, h),
    ).fetchone()
    if row and row["audio_path"] and os.path.exists(row["audio_path"]):
        try:
            os.remove(row["audio_path"])
        except OSError:
            pass
    # EXTENSION: also remove any cached slow-playback version of this
    # sentence (see get_slow_sentence_recording) -- not strictly
    # required for correctness, since that function already regenerates
    # whenever the real recording is newer than its cached slow version,
    # but leaving the old one on disk untouched is needless clutter for
    # a recording that no longer exists.
    slow_path = os.path.join(
        _profile_dir(profile_id, language),
        f"sentence_{h}_slow{str(_SLOW_PLAYBACK_RATE).replace('.', '')}.wav",
    )
    if os.path.exists(slow_path):
        try:
            os.remove(slow_path)
        except OSError:
            pass
    conn.execute(
        "DELETE FROM voice_sentences WHERE profile_id = ? AND language = ? AND text_hash = ?",
        (profile_id, language, h),
    )
    for word in text.split():
        key = _word_key(word)
        wrow = conn.execute(
            "SELECT audio_path FROM voice_words WHERE profile_id = ? AND language = ? AND word_key = ?",
            (profile_id, language, key),
        ).fetchone()
        if wrow and wrow["audio_path"] and os.path.exists(wrow["audio_path"]):
            try:
                os.remove(wrow["audio_path"])
            except OSError:
                pass
        conn.execute(
            "DELETE FROM voice_words WHERE profile_id = ? AND language = ? AND word_key = ?",
            (profile_id, language, key),
        )
    conn.commit()
    conn.close()


# ============================================================================
# Optional true synthesis (tier 2 -- see module docstring)
# ============================================================================

# EXTENSION: real feature, per explicit request -- proper voice-cloning
# enrollment. A parent/teacher reads a fixed set of short, varied
# phrases ONCE per profile+language (real content already used
# elsewhere in this project: the built-in sample sentences, plus real
# lines from the actual KTBS Savi Kannada Grade-2 reader already
# ingested for this app -- not invented Kannada text), and every clip
# is combined into one longer, more representative reference recording
# for XTTS-v2/OpenVoice cloning. This is what lets ANY later typed
# sentence be heard in that real voice, not just sentences that happen
# to already have their own real recording.
ENROLLMENT_SAMPLE_TEXTS = {
    "en": [
        "The sun is shining bright today.",
        "I love to read my favourite story.",
        "My dog runs fast in the park.",
        "We are going to grandma's house.",
        "The little bird sings a sweet song.",
        "Can you see the blue sky above us?",
        "She is drawing a picture of a flower.",
        "Please close the door before you leave.",
        "The children are playing in the garden.",
        "He drinks a glass of milk every morning.",
        "Our school starts at nine o'clock.",
        "I am happy to see my friends today.",
    ],
    "kn": [
        "ಇಂದು ಸೂರ್ಯ ಪ್ರಕಾಶಿಸುತ್ತಿದ್ದಾನೆ.",
        "ನನಗೆ ಪುಸ್ತಕ ಓದಲು ಇಷ್ಟ.",
        "ನಮ್ಮ ನಾಯಿ ಉದ್ಯಾನದಲ್ಲಿ ಓಡುತ್ತದೆ.",
        "ನಾವು ಅಜ್ಜಿಯ ಮನೆಗೆ ಹೋಗುತ್ತಿದ್ದೇವೆ.",
        "ಪುಟ್ಟ ಹಕ್ಕಿ ಸುಂದರ ಹಾಡು ಹಾಡುತ್ತದೆ.",
        "ಬಾರಲೇ ಹಕ್ಕಿ ಬಣ್ಣದ ಹಕ್ಕಿ",
        "ಬಾರಲೇ ಹಕ್ಕಿ ಹಾರುವ ಹಕ್ಕಿ",
        "ಅವ್ವನು ನೀರಿಗೆ ಹೋಗಿಹಳು",
        "ಅಪ್ಪನು ಪೇಟೆಗೆ ಹೋಗಿಹನು",
        "ಅವ್ವನು ಬರಲಿ ಅಪ್ಪನು ಬರಲಿ",
        "ನಾನು ನೀನೂ ಇಬ್ಬರು ಕೂಡಿ",
        "ಹಾರುತ ಹಾಡುತ ನಲಿಯುವ ಬಾ.",
    ],
}


def enrollment_sample_texts(language):
    """The fixed list of short phrases a parent/teacher reads once to
    build a stronger voice-cloning reference. Real content already
    used elsewhere in this project (see ENROLLMENT_SAMPLE_TEXTS above),
    not invented on the fly."""
    return list(ENROLLMENT_SAMPLE_TEXTS.get(language, ENROLLMENT_SAMPLE_TEXTS["en"]))


def save_enrollment_clip(profile_id, language, sample_idx, wav_bytes):
    _init_tables()
    d = os.path.join(_profile_dir(profile_id, language), "enrollment")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"sample_{sample_idx}.wav")
    with open(path, "wb") as f:
        f.write(wav_bytes)
    conn = get_connection()
    conn.execute(
        "INSERT INTO voice_enrollment_clips (profile_id, language, sample_idx, audio_path) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(profile_id, language, sample_idx) DO UPDATE SET audio_path = excluded.audio_path",
        (profile_id, language, sample_idx, path),
    )
    conn.commit()
    conn.close()
    # A newly-recorded enrollment clip invalidates any previously
    # combined reference built from the old, smaller set.
    _invalidate_enrollment_reference(profile_id, language)
    return path


def enrollment_progress(profile_id, language):
    """(recorded_count, total_count) -- how many of the fixed sample
    phrases this profile has recorded for this language so far."""
    _init_tables()
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) c FROM voice_enrollment_clips WHERE profile_id = ? AND language = ?",
        (profile_id, language),
    ).fetchone()
    conn.close()
    total = len(ENROLLMENT_SAMPLE_TEXTS.get(language, ENROLLMENT_SAMPLE_TEXTS["en"]))
    return (row["c"] if row else 0), total


def recorded_enrollment_indices(profile_id, language):
    """Which sample indices (0-based) this profile has already recorded,
    as a set -- lets the caller find the next un-recorded phrase."""
    _init_tables()
    conn = get_connection()
    rows = conn.execute(
        "SELECT sample_idx FROM voice_enrollment_clips WHERE profile_id = ? AND language = ?",
        (profile_id, language),
    ).fetchall()
    conn.close()
    return {r["sample_idx"] for r in rows}


def reset_enrollment(profile_id, language):
    """Deletes every recorded enrollment clip (files and DB rows) and
    the combined reference built from them, so enrollment can be
    re-done from scratch. Never touches ordinary practice-sentence or
    word recordings -- only the dedicated enrollment set."""
    _init_tables()
    conn = get_connection()
    rows = conn.execute(
        "SELECT audio_path FROM voice_enrollment_clips WHERE profile_id = ? AND language = ?",
        (profile_id, language),
    ).fetchall()
    for r in rows:
        if r["audio_path"] and os.path.exists(r["audio_path"]):
            try:
                os.remove(r["audio_path"])
            except OSError:
                pass
    conn.execute(
        "DELETE FROM voice_enrollment_clips WHERE profile_id = ? AND language = ?",
        (profile_id, language),
    )
    conn.commit()
    conn.close()
    _invalidate_enrollment_reference(profile_id, language)


def _enrollment_reference_path(profile_id, language):
    return os.path.join(_profile_dir(profile_id, language), "enrollment", "_combined_reference.wav")


def _invalidate_enrollment_reference(profile_id, language):
    path = _enrollment_reference_path(profile_id, language)
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


# EXTENSION FIX: real reference-quality tuning -- OpenVoice's own
# (MyShell) documentation and community guidance recommend a
# REASONABLY SHORT reference clip (commonly cited as roughly 15-30
# seconds) rather than a very long one; a longer combined clip doesn't
# reliably improve tone-colour extraction and can dilute it. This
# project's full enrollment set is 12 phrases, which combined ran to
# ~40 seconds -- comfortably over that range. Capped here to ~18
# seconds (using as many COMPLETE phrases as fit under the cap, never
# cutting a phrase mid-word) so the reference used for cloning stays
# in the range OpenVoice is actually tuned for.
_MAX_REFERENCE_SECONDS = 18.0


def _enrollment_reference_clip(profile_id, language):
    """
    Combines recorded enrollment clips for this profile+language into
    one reference recording (capped at _MAX_REFERENCE_SECONDS -- see
    above), cached to disk and rebuilt only when a new enrollment clip
    is saved (see _invalidate_enrollment_reference above). Returns
    None if no enrollment clips have been recorded yet -- callers
    should fall back to _any_reference_clip (a single practice
    recording) in that case, the same graceful-degradation pattern
    used throughout this module.
    """
    combined_path = _enrollment_reference_path(profile_id, language)
    if os.path.exists(combined_path):
        return combined_path

    _init_tables()
    conn = get_connection()
    rows = conn.execute(
        "SELECT audio_path FROM voice_enrollment_clips WHERE profile_id = ? AND language = ? "
        "ORDER BY sample_idx ASC",
        (profile_id, language),
    ).fetchall()
    conn.close()
    paths = [r["audio_path"] for r in rows if r["audio_path"] and os.path.exists(r["audio_path"])]
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]

    try:
        chunks = []
        target_sr = None
        total_seconds = 0.0
        for p in paths:
            y, sr = librosa.load(p, sr=None)
            if target_sr is None:
                target_sr = sr
            elif sr != target_sr:
                # EXTENSION: real recordings can genuinely come in at
                # different native sample rates across separate browser
                # recording sessions -- resample to the first clip's
                # rate so concatenation doesn't corrupt or mis-speed
                # later clips.
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            clip_seconds = len(y) / target_sr
            if total_seconds > 0 and total_seconds + clip_seconds > _MAX_REFERENCE_SECONDS:
                # Stop BEFORE adding this phrase, rather than truncating
                # it mid-word -- always keep whole phrases, and always
                # keep at least the first one even if it alone exceeds
                # the cap (never return an empty reference).
                break
            chunks.append(y)
            # A short silence gap between phrases, matching the pause
            # a real speaker naturally leaves between separate
            # sentences -- avoids two clips' waveforms running directly
            # into each other with no boundary.
            chunks.append(np.zeros(int(0.35 * target_sr), dtype=y.dtype))
            total_seconds += clip_seconds + 0.35
        combined = np.concatenate(chunks)
        os.makedirs(os.path.dirname(combined_path), exist_ok=True)
        sf.write(combined_path, combined, target_sr, format="WAV")
        print(f"[voice_clone] Built a {total_seconds:.1f}s combined enrollment reference "
              f"for profile {profile_id} ({language}) from {len(chunks)//2} of {len(paths)} "
              f"recorded phrases (capped at {_MAX_REFERENCE_SECONDS}s).")
        return combined_path
    except Exception as e:
        print(f"[voice_clone._enrollment_reference_clip] FAILED to combine clips, falling back to single clip: {e}")
        return paths[-1]


def _any_reference_clip(profile_id, language):
    """
    Picks the best available real recorded reference audio for this
    profile+language to use as the speaker-reference for XTTS-v2/
    OpenVoice cloning: the combined, multi-phrase ENROLLMENT reference
    if the parent/teacher has recorded one (stronger, more
    representative of their real voice -- see _enrollment_reference_clip
    above), otherwise falls back to whatever single real recording
    happens to exist already (a sentence clip if one exists, else any
    single word clip) -- a few seconds is still enough for zero-shot
    cloning to work at all, just less robust than a proper enrollment.
    """
    enrolled = _enrollment_reference_clip(profile_id, language)
    if enrolled:
        return enrolled
    conn = get_connection()
    row = conn.execute(
        "SELECT audio_path FROM voice_sentences WHERE profile_id = ? AND language = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (profile_id, language),
    ).fetchone()
    if not row:
        row = conn.execute(
            "SELECT audio_path FROM voice_words WHERE profile_id = ? AND language = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (profile_id, language),
        ).fetchone()
    conn.close()
    if row and os.path.exists(row["audio_path"]):
        return row["audio_path"]
    return None


_SARVAM_LANGUAGE_CODES = {"en": "en-IN", "kn": "kn-IN"}


def _sarvam_language_code(language):
    return _SARVAM_LANGUAGE_CODES.get(language, "en-IN")


def _get_sarvam_voice_id(profile_id, language):
    """
    Returns this profile's cached Sarvam voice_id, creating it (once,
    ever, per profile -- see the sarvam_voices table above) from
    whichever real reference clip is available. Prefers a reference in
    the REQUESTED language, but falls back to any other language's
    reference clip this profile has, since Sarvam's clones work
    cross-lingually anyway (a clone made from an English recording can
    still speak Kannada text, and vice versa -- per Sarvam's own docs).
    Returns None if no reference clip exists yet for this profile in
    ANY language, or if creating the clone fails (see
    sarvam_voice.create_voice_clone's own logging for why).
    """
    _init_tables()
    conn = get_connection()
    row = conn.execute(
        "SELECT voice_id FROM sarvam_voices WHERE profile_id = ?", (profile_id,)
    ).fetchone()
    conn.close()
    if row:
        return row["voice_id"]

    reference = _any_reference_clip(profile_id, language)
    ref_language = language
    if reference is None:
        for other_lang in ENROLLMENT_SAMPLE_TEXTS.keys():
            if other_lang == language:
                continue
            reference = _any_reference_clip(profile_id, other_lang)
            if reference:
                ref_language = other_lang
                break
    if reference is None:
        return None

    from modules import sarvam_voice as _sarvam
    voice_id = _sarvam.create_voice_clone(
        reference,
        name=f"CurioKidz profile {profile_id}",
        language=_sarvam_language_code(ref_language),
    )
    if not voice_id:
        return None

    conn = get_connection()
    conn.execute(
        "INSERT INTO sarvam_voices (profile_id, voice_id) VALUES (?, ?) "
        "ON CONFLICT(profile_id) DO UPDATE SET voice_id = excluded.voice_id",
        (profile_id, voice_id),
    )
    conn.commit()
    conn.close()
    return voice_id


def _sarvam_clone(profile_id, text, language, out_path):
    from modules import sarvam_voice as _sarvam
    if not _sarvam.SARVAM_LIBRARY_AVAILABLE:
        print("[voice_clone] Sarvam cloning requested, but the `sarvamai` package "
              "isn't installed -- run `pip install sarvamai`.")
        return None
    if not _sarvam.api_key_configured():
        print("[voice_clone] Sarvam cloning requested, but SARVAM_API_KEY isn't set -- "
              "see requirements.txt for how to set it (falling back to the next tier).")
        return None
    voice_id = _get_sarvam_voice_id(profile_id, language)
    if voice_id is None:
        print(f"[voice_clone] Sarvam cloning failed: no real reference recording exists "
              f"yet for profile {profile_id} in any language (record at least one "
              f"enrollment phrase or sentence first) -- falling back to the next tier.")
        return None
    result_path = _sarvam.text_to_speech(text, voice_id, _sarvam_language_code(language), out_path)
    if result_path:
        return result_path
    print(f"[voice_clone] Sarvam text-to-speech failed for profile {profile_id}, "
          f"text={text!r}, language={language!r} -- see the [sarvam_voice] error above "
          f"-- falling back to the next tier.")
    return None


def clone_synthesize(profile_id, text, language, out_path, allow_sarvam=True):
    """
    Attempts real, arbitrary-text voice cloning of `text` in this
    profile's voice, trying whichever tier actually applies, in order:

      1. XTTS-v2 zero-shot TTS cloning (English, and the ~16 other
         languages it supports -- see xtts_available_for above). Open
         source (Coqui TTS).
      2. Sarvam AI cloud cloning (see modules/sarvam_voice.py) -- a
         PAID, CLOSED-SOURCE hosted API, tried before OpenVoice because
         it's simpler and more reliable once an API key is configured.
         Skipped automatically (no error) if SARVAM_API_KEY isn't set,
         and can also be skipped deliberately via `allow_sarvam=False`
         (see below) for a caller that specifically wants an
         all-open-source pipeline.
      3. OpenVoice tone-colour conversion (see _openvoice_clone below)
         -- works for ANY language including Kannada, because it does
         not synthesize speech itself: it takes speech this app's own
         existing, already-correct TTS pipeline (modules/audio.py)
         generates in the right language, and converts THAT audio's
         voice timbre to match the profile's recorded voice, leaving
         the actual Kannada (or English) words untouched. Open source
         (MyShell OpenVoice V2).

    `allow_sarvam` (default True, unchanged for every existing caller
    e.g. Custom Practice) lets a caller opt OUT of the one non-open-
    source tier and use ONLY XTTS + OpenVoice -- both open-source
    models -- e.g. modules/voice_clone.clone_whole_lesson(), used by
    the Teacher Upload tab's whole-lesson cloning, which was
    specifically asked to stick to open-source models only.

    Returns out_path on success, or None if no tier is available/works
    for any reason. Callers must treat None as "use tier 1 [real
    recorded words] instead", never as an error to surface to the
    child.
    """
    reference = _any_reference_clip(profile_id, language)
    if reference is None:
        return None

    result = None
    tier = None

    if xtts_available_for(language):
        model = _get_xtts_model()
        if model is not None:
            try:
                model.tts_to_file(text=text, speaker_wav=reference, language=language, file_path=out_path)
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    result, tier = out_path, "xtts"
            except Exception:
                pass

    if result is None and allow_sarvam:
        sarvam_result = _sarvam_clone(profile_id, text, language, out_path)
        if sarvam_result:
            result, tier = sarvam_result, "sarvam"

    if result is None:
        openvoice_result = _openvoice_clone(text, language, reference, out_path)
        if openvoice_result:
            result, tier = openvoice_result, "openvoice"

    if result:
        _save_debug_copy(result, profile_id, language, text, tier)
        return result
    return None


# ============================================================================
# DEBUGGING AID: every successfully cloned clip is also saved to a
# permanent, easy-to-find folder (instead of only the OS temp directory,
# which gets cleaned up and is awkward to browse to) -- specifically
# added because manually hunting for a temp-folder filename by hand
# (e.g. via %TEMP% in File Explorer) turned out to be unreliable in
# practice. Safe to delete data/cloned_audio_debug/ at any time; it's
# purely for listening back to/inspecting past cloning attempts, not
# used by the app itself for anything.
# ============================================================================
_DEBUG_CLONE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cloned_audio_debug"
)


def _save_debug_copy(src_path, profile_id, language, text, tier):
    try:
        os.makedirs(_DEBUG_CLONE_DIR, exist_ok=True)
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_text = re.sub(r"[^a-zA-Z0-9]+", "_", text)[:40].strip("_") or "text"
        dest_name = f"{timestamp}_profile{profile_id}_{language}_{tier}_{safe_text}.wav"
        dest_path = os.path.join(_DEBUG_CLONE_DIR, dest_name)
        import shutil
        shutil.copyfile(src_path, dest_path)
        print(f"[voice_clone] Saved a permanent debug copy of this cloned clip to: {dest_path}")
    except Exception as e:
        # Never let debug-copy failures break the actual feature.
        print(f"[voice_clone] (non-fatal) couldn't save debug copy: {type(e).__name__}: {e}")


# ============================================================================
# OpenVoice tone-colour conversion -- the tier that makes true,
# arbitrary-text voice cloning possible for KANNADA too, not just
# English.
#
# HOW, STATED HONESTLY: this does NOT teach a TTS model to speak
# Kannada in a cloned voice end-to-end (no free, easily-installable
# model does that for Kannada today, and this project isn't going to
# pretend otherwise). Instead it does two separate, both genuinely
# real, steps:
#   (a) generate ordinary Kannada speech with THIS APP'S OWN existing,
#       already-correct TTS pipeline (modules/audio.py's edge-tts/gTTS
#       Kannada voice) -- normal, generic-voice, correctly-pronounced
#       Kannada audio.
#   (b) run OpenVoice's tone-colour converter (a real, open-source,
#       language-agnostic voice-conversion model -- it operates on
#       acoustic tone colour, not text/phonemes, so it doesn't need to
#       have ever seen Kannada during its own training) to re-voice
#       that audio so it sounds like the parent/teacher's recorded
#       reference clip, while the actual Kannada words are untouched.
#
# This is real voice conversion, not a text-to-speech model that
# "knows" Kannada -- worth stating plainly, the same way this project
# states plainly what the acoustic DTW fallback in audio.py does and
# doesn't measure.
#
# PACKAGE USED, CONFIRMED WORKING: `openvoice-cli` (PyPI), a
# pip-installable, MIT-licensed repackaging of MyShell/MIT's OpenVoice
# V2 tone-colour converter (the same model at
# https://docs.myshell.ai/technology/openvoice /
# https://github.com/myshell-ai/OpenVoice -- the official repo isn't
# itself on PyPI, so `openvoice-cli` is what actually gets `pip
# install`-able OpenVoice V2 code, importable as `openvoice_cli`).
#
# HONEST DISCLOSURE ON TESTING: `pip install openvoice-cli` succeeds
# in this project's sandbox and the imports below (`ToneColorConverter`,
# `se_extractor`) work -- confirmed directly, this is real, working
# code. What could NOT be verified end-to-end here is the actual
# converter checkpoint download: this sandbox's network allowlist
# blocks direct requests to huggingface.co (the same restriction
# already documented for other model downloads in this project), so
# `_ensure_openvoice_checkpoint()` below could not be exercised past
# that point in this environment. On your machine (ordinary home
# internet, no sandbox proxy), that download is a plain HTTPS GET to a
# public Hugging Face file and should just work. If it doesn't, the
# checkpoint files can also be downloaded by hand from
# https://huggingface.co/daswer123/openvoice-tunner-v2 and placed at
# data/openvoice_checkpoints/{config.json,checkpoint.pth}. This never
# blocks tier 1 (real recorded words), which needs none of this and
# always works.
# ============================================================================
try:
    from openvoice_cli.api import ToneColorConverter as _OpenVoiceConverter
    from openvoice_cli import se_extractor as _ov_se_extractor
    from openvoice_cli.downloader import download_checkpoint as _ov_download_checkpoint
    OPENVOICE_LIBRARY_AVAILABLE = True
except Exception:
    OPENVOICE_LIBRARY_AVAILABLE = False

_OPENVOICE_CKPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "openvoice_checkpoints"
)
_openvoice_converter = None
_openvoice_load_failed = False


def openvoice_available():
    return OPENVOICE_LIBRARY_AVAILABLE


def _ensure_openvoice_checkpoint():
    """
    Downloads OpenVoice V2's converter checkpoint the first time it's
    needed, via the `openvoice_cli` package's own downloader (a public,
    unauthenticated Hugging Face file -- no API token needed), and
    caches it under data/openvoice_checkpoints/ from then on -- the
    same first-run-download-then-cache pattern already used for
    faster-whisper's models (see modules/asr.py) and for this app's
    own TTS audio cache. Returns the local checkpoint directory, or
    None if the download itself fails (no internet, or this specific
    sandbox's network restriction -- see this module's honest
    disclosure above).
    """
    config_path = os.path.join(_OPENVOICE_CKPT_DIR, "config.json")
    ckpt_path = os.path.join(_OPENVOICE_CKPT_DIR, "checkpoint.pth")
    if os.path.exists(config_path) and os.path.exists(ckpt_path):
        return _OPENVOICE_CKPT_DIR
    print("[voice_clone] OpenVoice checkpoint not found locally -- downloading it now "
          "(~200MB from Hugging Face, one-time only, needs a normal internet connection)...")
    try:
        _ov_download_checkpoint(_OPENVOICE_CKPT_DIR)
        ok = os.path.exists(config_path) and os.path.exists(ckpt_path)
        if ok:
            print("[voice_clone] OpenVoice checkpoint downloaded successfully.")
        else:
            print("[voice_clone] OpenVoice checkpoint download finished but the expected files "
                  f"are missing under {_OPENVOICE_CKPT_DIR} -- treating this as a failed download.")
        return _OPENVOICE_CKPT_DIR if ok else None
    except Exception as e:
        print(f"[voice_clone] OpenVoice checkpoint download FAILED: {type(e).__name__}: {e}")
        return None


def _get_openvoice_converter():
    global _openvoice_converter, _openvoice_load_failed
    if _openvoice_converter is not None:
        return _openvoice_converter
    if _openvoice_load_failed or not OPENVOICE_LIBRARY_AVAILABLE:
        return None
    ckpt_dir = _ensure_openvoice_checkpoint()
    if ckpt_dir is None:
        _openvoice_load_failed = True
        return None
    try:
        converter = _OpenVoiceConverter(f"{ckpt_dir}/config.json", device="cpu")
        converter.load_ckpt(f"{ckpt_dir}/checkpoint.pth")
        _openvoice_converter = converter
        return converter
    except Exception as e:
        print(f"[voice_clone] OpenVoice converter FAILED to load: {type(e).__name__}: {e}")
        _openvoice_load_failed = True
        return None


_openvoice_source_se_cache = {}  # language -> cached speaker-embedding tensor


def _get_openvoice_source_se(language, converter):
    """
    Returns a cached "source speaker embedding" describing the generic
    tone-colour of THIS APP'S OWN TTS voice in `language`, computing it
    once (ever, per language, per server run) rather than re-extracting
    it from every individual practice sentence.

    REAL BUG THIS FIXES: OpenVoice's own embedding extraction requires a
    minimum amount of actual (non-silent) speech to work reliably, and
    raises `AssertionError: input audio is too short` when it doesn't
    get enough -- confirmed directly from a real failure log, where a
    short practice sentence ("The weather is nice today.") produced
    only ~2.7 seconds of voiced audio after silence-trimming, under
    OpenVoice's own minimum. Short practice sentences are exactly what
    this app generates constantly, so extracting the source embedding
    per-sentence was guaranteed to fail often.

    The actual fix: OpenVoice's source embedding only needs to capture
    what the GENERIC TTS voice sounds like -- it does not need to be
    re-derived from the specific sentence being converted (the sentence
    audio itself is still passed to converter.convert() below at its
    own natural length; only the EMBEDDING EXTRACTION step needs more
    material than one short sentence provides). So a longer, reusable
    calibration clip is built once per language (by concatenating a few
    of this project's own existing ENROLLMENT_SAMPLE_TEXTS through this
    app's own TTS pipeline -- real content already used elsewhere, not
    invented for this), safely exceeding OpenVoice's minimum, and its
    embedding is cached both in memory and on disk so this only ever
    runs once per language per machine, not once per sentence.
    """
    if language in _openvoice_source_se_cache:
        return _openvoice_source_se_cache[language]

    from modules import audio as _audio_module
    calib_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "openvoice_checkpoints"
    )
    os.makedirs(calib_dir, exist_ok=True)
    calib_path = os.path.join(calib_dir, f"_source_calibration_{language}.wav")

    if not os.path.exists(calib_path):
        sample_texts = ENROLLMENT_SAMPLE_TEXTS.get(language, ENROLLMENT_SAMPLE_TEXTS["en"])[:4]
        combined_text = " ".join(sample_texts)
        base_path = _audio_module.generate_reference_audio(combined_text, language, slow=False)
        if not base_path:
            print(f"[voice_clone] Couldn't build a reusable OpenVoice source-voice calibration "
                  f"clip for language {language!r}: this app's own TTS failed to generate audio "
                  f"for the calibration text (see audio.py's own TTS fallback chain).")
            return None
        import shutil
        shutil.copyfile(base_path, calib_path)

    try:
        source_se, _ = _ov_se_extractor.get_se(calib_path, converter, vad=True)
    except Exception as e:
        print(f"[voice_clone] Failed to compute a reusable OpenVoice source-voice embedding for "
              f"language {language!r}: {type(e).__name__}: {e}")
        return None

    _openvoice_source_se_cache[language] = source_se
    return source_se


def _openvoice_clone(text, language, reference_clip, out_path):
    if not OPENVOICE_LIBRARY_AVAILABLE:
        print("[voice_clone] Cloned-voice playback requested, but the `openvoice-cli` package "
              "isn't installed in this environment -- run `pip install -r requirements.txt`.")
        return None
    converter = _get_openvoice_converter()
    if converter is None:
        print("[voice_clone] Cloned-voice playback requested, but the OpenVoice converter isn't "
              "available (see the error above, if any) -- falling back to no audio for this clip.")
        return None
    try:
        # Step (a): ordinary, correctly-pronounced speech in the right
        # language, from this app's own existing TTS pipeline.
        from modules import audio as _audio_module
        base_audio_path = _audio_module.generate_reference_audio(text, language, slow=False)
        if not base_audio_path:
            print(f"[voice_clone] Cloned-voice playback failed: this app's own TTS could not "
                  f"generate base audio for {text!r} in language {language!r} (see audio.py's "
                  f"own TTS fallback chain -- likely no internet reached edge-tts/gTTS, and "
                  f"pyttsx3 also isn't available here).")
            return None

        # Step (b): re-voice that audio to match the recorded profile,
        # via OpenVoice's tone-colour extraction + conversion. The
        # SOURCE embedding uses a cached, longer calibration clip (see
        # _get_openvoice_source_se above) rather than this specific
        # short sentence, since OpenVoice's own embedding extraction
        # needs more speech than one short practice sentence reliably
        # provides -- the sentence audio itself (base_audio_path) is
        # still what actually gets converted below, at its own length.
        source_se = _get_openvoice_source_se(language, converter)
        if source_se is None:
            return None
        target_se, _ = _ov_se_extractor.get_se(reference_clip, converter, vad=True)
        converter.convert(
            audio_src_path=base_audio_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=out_path,
        )
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
        print(f"[voice_clone] OpenVoice's convert() ran without raising, but produced no output "
              f"file at {out_path!r} -- treating this as a failed clone.")
    except Exception as e:
        print(f"[voice_clone] Cloned-voice playback FAILED during OpenVoice conversion: "
              f"{type(e).__name__}: {e}")
    return None


# ============================================================================
# Saved custom texts (so a parent/teacher doesn't have to retype a
# sentence they've already used before -- persisted, not just cached
# for this session)
# ============================================================================
_SAMPLE_SENTENCES = {
    "en": [
        "The sun is shining bright today.",
        "I love to read my favourite story.",
        "My dog runs fast in the park.",
        "We are going to grandma's house.",
        "The little bird sings a sweet song.",
    ],
    "kn": [
        "ಇಂದು ಸೂರ್ಯ ಪ್ರಕಾಶಿಸುತ್ತಿದ್ದಾನೆ.",
        "ನನಗೆ ಪುಸ್ತಕ ಓದಲು ಇಷ್ಟ.",
        "ನಮ್ಮ ನಾಯಿ ಉದ್ಯಾನದಲ್ಲಿ ಓಡುತ್ತದೆ.",
        "ನಾವು ಅಜ್ಜಿಯ ಮನೆಗೆ ಹೋಗುತ್ತಿದ್ದೇವೆ.",
        "ಪುಟ್ಟ ಹಕ್ಕಿ ಸುಂದರ ಹಾಡು ಹಾಡುತ್ತದೆ.",
    ],
}


def sample_sentences(language):
    saved = [c["text"] for c in list_custom_texts(language)]
    return saved + [s for s in _SAMPLE_SENTENCES.get(language, []) if s not in saved]


def save_custom_text(language, text, created_by=None):
    _init_tables()
    text = text.strip()
    if not text:
        return
    conn = get_connection()
    exists = conn.execute(
        "SELECT 1 FROM custom_texts WHERE language = ? AND text = ?", (language, text)
    ).fetchone()
    if not exists:
        conn.execute(
            "INSERT INTO custom_texts (language, text, created_by) VALUES (?, ?, ?)",
            (language, text, created_by),
        )
        conn.commit()
    conn.close()


def list_custom_texts(language):
    _init_tables()
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, text, created_by, created_at FROM custom_texts WHERE language = ? "
        "ORDER BY created_at DESC",
        (language,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]