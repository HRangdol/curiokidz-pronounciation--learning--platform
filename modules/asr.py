"""
Real speech recognition (ASR) for word/line matching -- transcribes
what the child actually said and compares the recognized TEXT against
the target word or line, rather than only measuring acoustic
similarity (see audio.py). This is what actually answers "did the
child say the right word," not just "does this sound similar to the
reference voice" -- a meaningfully more honest check, especially
against a robotic TTS reference voice that never sounds quite like a
real child.

Uses faster-whisper (CTranslate2-based) rather than openai-whisper
specifically because it does NOT require a full PyTorch install --
confirmed directly in this project's own sandbox that openai-whisper's
default PyPI install pulls in a GPU-oriented PyTorch build with several
gigabytes of CUDA dependencies that then fail to import at all on a
CPU-only machine unless the exact right combination of packages is
present. faster-whisper avoids that dependency chain entirely.

GRACEFUL FALLBACK, same principle used throughout this whole project:
if the ASR model genuinely isn't available (not installed, or the
model files haven't been downloaded yet and there's no internet this
moment), callers get a clear, honest signal (transcribe() returns
None) rather than a crash or a silently wrong result. app.py falls back
to the acoustic DTW/MFCC scoring in audio.py when this happens, instead
of the practice loop breaking entirely.
"""
import os
import re
import json
import unicodedata

try:
    from faster_whisper import WhisperModel
    ASR_LIBRARY_AVAILABLE = True
except ImportError:
    ASR_LIBRARY_AVAILABLE = False

_MODEL_CACHE = {}

# EXTENSION FIX: a real report was that pronunciation-checking got
# SLOWER after the tiered Kannada fallback below was added (large-v3 ->
# medium -> small). Root cause: _MODEL_LOAD_FAILED (below) only lives in
# memory for the current process, so it resets on every Streamlit
# restart -- and this app was restarted very often today during
# testing. Every restart re-paid the FULL cascade from scratch: a slow,
# doomed attempt to load large-v3 again, then likely a first-time
# DOWNLOAD of "medium" (never used before today's fix, so not cached
# locally yet) before finally settling on a working size -- real,
# avoidable overhead on every single restart, not imagined. Persisting
# which sizes are known to fail to a small file on disk means that
# cascade only ever has to happen ONCE per machine, not once per
# restart: a later run reads this file first and skips straight past
# any size already confirmed to fail, going directly to the size that's
# known to actually work.
_RESOLUTION_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "asr_model_resolution.json"
)


def _load_persisted_failed_sizes():
    try:
        with open(_RESOLUTION_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("failed_sizes", []))
    except Exception:
        return set()


def _persist_failed_size(size):
    try:
        os.makedirs(os.path.dirname(_RESOLUTION_CACHE_PATH), exist_ok=True)
        try:
            with open(_RESOLUTION_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        failed = set(data.get("failed_sizes", []))
        failed.add(size)
        data["failed_sizes"] = sorted(failed)
        with open(_RESOLUTION_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        # Never let a caching failure break actual model loading --
        # worst case, the cascade just isn't remembered this time.
        pass

# EXTENSION FIX: a real failure report showed the Kannada model
# ("large-v3", ~3GB) failing to load with `mkl_malloc: failed to
# allocate memory` -- not enough free RAM on that machine. Before this
# fix, that failure was NOT remembered: _get_model raises before
# _MODEL_CACHE gets set, so every single stanza's word-timing attempt
# (and every practice-tab transcription attempt) retried loading the
# same 3GB model from scratch, failing every time -- a real, confirmed
# slowdown on low-memory machines, not a hypothetical one. Tracking
# failed sizes here means a size that has already failed to load is
# never retried again for the rest of this process, so callers get an
# immediate None (their existing, already-correct fallback path)
# instead of one more slow, doomed load attempt per call. Seeded from
# disk (see _RESOLUTION_CACHE_PATH above) so this protection survives
# an app restart too, not just repeated calls within one run.
_MODEL_LOAD_FAILED = _load_persisted_failed_sizes()

# ============================================================================
# EXTENSION: real report -- Kannada pronunciation checks take a very
# long time to come back after the child speaks. This is NOT a new bug
# introduced by anything else fixed today; it's the direct, inherent
# cost of _MODEL_SIZE_CANDIDATES_BY_LANGUAGE below loading large-v3 (a
# ~1.5B-parameter model) for Kannada specifically, run entirely on CPU.
# That size was a deliberate, documented accuracy trade-off (see the
# comment above it: smaller sizes were confirmed to mis-recognize real
# Kannada speech, even script-swapping it into Devanagari), so the fix
# here is not to shrink the model -- that would undo the accuracy fix
# -- it's to actually use a GPU for it when one is genuinely present.
# faster-whisper/CTranslate2 supports CUDA directly and needs no torch
# install to do it; this was never wired up before, so every machine,
# GPU or not, was always running the heaviest Kannada model on CPU
# alone. ctranslate2.get_cuda_device_count() is a real, direct check
# (not guessed from the OS) for whether a usable CUDA GPU is actually
# present right now, so this never claims a GPU that isn't really
# there. Checked once per process and cached, since a GPU does not
# appear or disappear mid-run.
# ============================================================================
def _detect_cuda_device_count():
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


_CUDA_DEVICE_COUNT = _detect_cuda_device_count()

# "small" for English -- upgraded from "tiny" after a real report that
# it wasn't recognizing a child's speech accurately even when correctly
# pronounced. This is a well-documented, genuine weak spot for speech
# recognition generally, not specific to this app: children's speech
# differs acoustically from the mostly-adult speech these models are
# trained on (shorter vocal tract, different formant frequencies, less
# consistent enunciation), and "tiny" -- the smallest, least accurate
# Whisper size -- is exactly the size most likely to struggle with
# that gap. "small" is a meaningful accuracy step up while still being
# far lighter than what Kannada needs below.
#
# Kannada needs a much larger model still -- confirmed necessary the
# hard way in the sibling capstone2_extension project earlier in this
# same overall effort, and confirmed again here: a real test with
# "small" transcribed genuine Kannada speech ("ಬಾರಲೇ") back as
# "बार अले" -- correct-sounding phonemes, but rendered in Devanagari
# (Hindi) script entirely, not Kannada script. Smaller Whisper-family
# models are well documented to struggle with lower-resource languages
# like Kannada specifically in this way -- not just less accurate, but
# liable to pick the wrong script/language outright. large-v3 is a
# large download (~3GB, once) but is the size that was actually proven
# to work correctly for Kannada in the sibling project.
# EXTENSION FIX: a real report showed large-v3 failing to load
# outright on one machine (mkl_malloc: failed to allocate memory --
# not enough free RAM for a ~3GB model). Before this fix, Kannada was
# "large-v3 or nothing" -- when it failed, EVERY Kannada check silently
# fell all the way back to raw acoustic similarity (audio.py's DTW/MFCC
# scoring), which is a much weaker check specifically for Custom
# Practice, where it compares a CHILD's voice against a DIFFERENT
# PERSON's (parent/teacher's) recorded reference -- two structurally
# different voices (age, pitch, timbre) that don't line up acoustically
# even when the word is said correctly. Real transcription-based
# checking (does the recognized TEXT match, not does the waveform look
# similar) is far more reliable there, so it's worth trying harder to
# get SOME real Kannada ASR running before giving up to the acoustic
# fallback. This tries progressively smaller multilingual Whisper sizes
# that still include Kannada, on the chance one of them fits in the RAM
# actually available -- each is real speech recognition, just with
# lower accuracy at smaller sizes (see the standalone comment above on
# why "small" is a real, documented risk for Kannada specifically:
# script confusion, not just lower accuracy). A size that fails to
# load is remembered (_MODEL_LOAD_FAILED) and skipped on every later
# call, so this never repeatedly retries a doomed larger model -- it
# settles on the largest one that actually works, once, then reuses it.
_MODEL_SIZE_CANDIDATES_BY_LANGUAGE = {
    "en": ["small"],
    "kn": ["large-v3", "medium", "small"],
}

# EXTENSION: opt-in speed override for a genuinely slow, CPU-only
# machine. large-v3 is the right DEFAULT for Kannada -- smaller sizes
# were confirmed (see comment above) to mis-recognize real Kannada
# speech, sometimes into the wrong script entirely -- so this is never
# switched automatically; it only takes effect if CURIOKIDZ_FAST_ASR=1
# is set in the environment before starting the app, a conscious choice
# to trade some of that accuracy for a real, substantial speed-up on a
# machine where large-v3 alone (even on CPU with every core + int8, and
# even without a GPU to fall back to) is simply too slow to feel usable
# for a child waiting on an answer. "medium" is still a real multi-
# lingual Whisper size with Kannada support, just lighter than large-v3.
if os.environ.get("CURIOKIDZ_FAST_ASR") == "1":
    _MODEL_SIZE_CANDIDATES_BY_LANGUAGE["kn"] = ["medium", "small"]
    print("[asr] CURIOKIDZ_FAST_ASR=1 -- Kannada will use 'medium' instead of "
          "'large-v3' for faster (but slightly less accurate) checking.")

_WHISPER_LANGUAGE_CODE = {"en": "en", "kn": "kn"}

# EXTENSION FIX: self-heal any stale blacklist entry from BEFORE the
# "never permanently blacklist the last candidate" fix above. If a
# machine had already hit a transient failure loading "small" (e.g.
# English's only candidate) under an earlier version of this file,
# asr_model_resolution.json could already contain "small" as
# permanently failed on disk -- which would otherwise keep disabling
# real English ASR forever even after this fix, since that old entry
# would just get loaded right back in by _load_persisted_failed_sizes()
# above. Dropping every language's last/safety-net candidate from the
# in-memory failed-set right here (and rewriting the file without it)
# means a stale blacklist from before this fix clears itself on the
# very next run, with no manual file edit needed.
_last_candidates = {c[-1] for c in _MODEL_SIZE_CANDIDATES_BY_LANGUAGE.values() if c}
if _MODEL_LOAD_FAILED & _last_candidates:
    _MODEL_LOAD_FAILED -= _last_candidates
    try:
        os.makedirs(os.path.dirname(_RESOLUTION_CACHE_PATH), exist_ok=True)
        with open(_RESOLUTION_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"failed_sizes": sorted(_MODEL_LOAD_FAILED)}, f)
    except Exception:
        pass


def asr_available():
    return ASR_LIBRARY_AVAILABLE


def _get_model(language):
    candidates = _MODEL_SIZE_CANDIDATES_BY_LANGUAGE.get(language, ["small"])
    # EXTENSION FIX: a real report was that BOTH English and Kannada got
    # slower to check -- not just Kannada, which the large-v3 cascade
    # above can't explain on its own, since English only ever has ONE
    # candidate ("small"). Root cause found on inspection: the very last
    # (smallest, lightest) candidate in a language's list is its safety
    # net -- for English it's the ONLY candidate. The old code treated
    # every load failure the same way: permanently blacklist that size
    # to disk, forever, for that machine. That's the right call for a
    # genuinely-too-big model (large-v3 hitting a real mkl_malloc OOM,
    # confirmed in this project's own terminal log earlier) -- it's
    # never going to fit, no point retrying. But a failure loading the
    # SMALLEST/last candidate is far more likely to be TRANSIENT: a
    # first-time download hiccup, or (concretely, in this app) a moment
    # of real memory pressure caused by Kannada's own large-v3 attempt
    # running around the same time. Permanently blacklisting "small" from
    # one bad moment silently disables real ASR for that language for
    # good (every future call raises here, gets caught in transcribe()'s
    # except block, and falls back to the much slower, less reliable
    # acoustic DTW/MFCC comparison in audio.py -- comparing a CHILD's
    # voice against a DIFFERENT adult's reference recording, which is
    # both slower AND the "still not able to recognize word by word"
    # kind of unreliable). So: only the non-last candidates (the
    # optional, larger upgrades) get permanently remembered as failed.
    # The last candidate is always retried on every call instead of
    # being written to the permanent blacklist -- it's the one size
    # every language absolutely depends on, so a bad-luck failure there
    # should never be permanent.
    last_candidate = candidates[-1] if candidates else None
    last_error = None
    for size in candidates:
        is_last = (size == last_candidate)
        if size in _MODEL_LOAD_FAILED and not is_last:
            continue
        if size in _MODEL_CACHE:
            return _MODEL_CACHE[size]
        # EXTENSION: try a real GPU first when one is actually present --
        # by far the biggest speed lever for large-v3 specifically (often
        # 5-10x over CPU), with zero change to recognition quality since
        # it's the exact same model weights, just run on different
        # hardware. If this machine has no CUDA GPU, or loading onto it
        # fails for any reason (wrong CUDA/cuDNN version, out-of-memory
        # on a small GPU, etc.), this falls straight through to the
        # proven CPU/int8 path below -- never a hard failure, only a
        # speed difference.
        if _CUDA_DEVICE_COUNT > 0:
            try:
                _MODEL_CACHE[size] = WhisperModel(size, device="cuda", compute_type="float16")
                _MODEL_LOAD_FAILED.discard(size)
                print(f"[asr._get_model] loaded '{size}' on GPU (CUDA) -- fast path")
                return _MODEL_CACHE[size]
            except Exception as e:
                print(f"[asr._get_model] GPU load of '{size}' failed ({e}); falling back to CPU")
        try:
            # EXTENSION FIX (speed, no accuracy cost): faster-whisper/
            # CTranslate2 defaults to just 4 CPU threads (cpu_threads=0
            # means "4 by default" per its own docs) regardless of how
            # many real cores the machine has. On a machine with more
            # cores than that -- common on most modern laptops/desktops
            # -- large-v3 (the slow, heavy model Kannada uses first for
            # accuracy) was leaving real available CPU parallelism on
            # the table. Explicitly using every logical core the OS
            # reports speeds up inference proportionally with zero
            # change to the model itself, so recognition quality is
            # unaffected -- this is pure throughput, not a smaller/
            # faster/less-accurate model swap like the size cascade
            # above is.
            _MODEL_CACHE[size] = WhisperModel(
                size, device="cpu", compute_type="int8",
                cpu_threads=os.cpu_count() or 4,
            )
            _MODEL_LOAD_FAILED.discard(size)
            return _MODEL_CACHE[size]
        except Exception as e:
            if not is_last:
                _MODEL_LOAD_FAILED.add(size)
                _persist_failed_size(size)
            last_error = e
            continue
    raise RuntimeError(
        f"No Whisper model size for language '{language}' could be loaded "
        f"(tried {candidates}); last error: {last_error}"
    )


def _transcribe_once(model, audio_path, language, expected_text, beam_size):
    segments, _ = model.transcribe(
        audio_path,
        language=_WHISPER_LANGUAGE_CODE.get(language, language),
        initial_prompt=expected_text,
        # Filters out silence/non-speech before transcribing, using
        # the library's own voice-activity detection. Directly
        # relevant here: real microphone recordings have silence
        # before/after the actual speech (confirmed and handled
        # separately for the acoustic-fallback path in audio.py),
        # and Whisper-family models are known to sometimes
        # hallucinate spurious words on pure-silence audio rather
        # than correctly returning nothing.
        vad_filter=True,
        # Default VAD thresholds are tuned for continuous speech; a
        # short, possibly quiet or hesitant single-word child
        # utterance is more likely to have its actual speech
        # mistaken for silence under the default settings.
        # threshold: lowered from the library's default 0.5 to
        # 0.35 -- audio needs less confidence to be classified as
        # speech, directly addressing a quieter child voice being
        # misread as silence and discarded before transcription.
        # min_silence_duration_ms: lowered from the library's
        # default 2000ms to 300ms -- per faster_whisper's own
        # documentation, this is how long the VAD waits after
        # detected speech stops before finalizing that chunk; for
        # one short word (no internal pauses to worry about
        # cutting), there's no benefit to the default's full
        # 2-second wait, only unnecessary latency.
        vad_parameters={"threshold": 0.35, "min_silence_duration_ms": 300},
        beam_size=beam_size,
    )
    return " ".join(seg.text for seg in segments).strip()


def transcribe(audio_path, language, expected_text=None):
    """
    Returns the recognized text (str) for the given audio file, or None
    if ASR isn't available at all (library not installed, or model
    couldn't be loaded/downloaded -- e.g. no internet the first time a
    given model size is needed). Callers should treat None as "fall
    back to acoustic scoring," not as "the child said nothing."

    expected_text, if given, is passed to Whisper as an initial_prompt
    hint. This is a real, documented technique specifically suited to
    a controlled-vocabulary task like this one: unlike open-ended
    transcription, this app always knows in advance which specific
    word or line the child is *supposed* to be saying, and biasing the
    model toward that expected vocabulary is a legitimate way to
    improve recognition of less clear, less consistent child speech --
    not a way to cheat the check, since a genuinely different word
    still won't match it (word_matches/line_match_ratio still compare
    the recognized text against the target afterward, unchanged).

    EXTENSION FIX: a real, confirmed-by-terminal-log report -- a child
    correctly said "the" and it was recognized as "dough." / "Dove.",
    every single time. This is a genuine, well-documented Whisper
    weakness, not a bug in this app's matching logic: a short function
    word spoken ALONE, with no surrounding sentence to give the
    language model context, is acoustically ambiguous (that quick,
    reduced "th-" sound), and beam_size=1's greedy decoding (the speed
    fix above) picks its single best guess and never reconsiders it --
    unlike the default beam_size=5, which explores several candidate
    decodings and can self-correct exactly this kind of ambiguous case.
    Rather than giving up the real speed win everywhere (going back to
    beam_size=5 for every single check would reintroduce the slowdown
    that was a real, separate complaint), this now only PAYS for the
    wider, slower search on the cases that actually need it: try the
    fast beam_size=1 decode first as before, and only when there's a
    specific expected word/line to check against AND the fast result
    doesn't match it, retry ONCE with beam_size=5. This is still a
    real second transcription of the actual audio, not a shortcut that
    assumes the child was right -- a genuinely different word still
    won't match on the retry either, so this can't be used to cheat
    the check, only to give an ambiguous-but-correct utterance a fair
    second look the same way a human listening carefully would.
    """
    if not ASR_LIBRARY_AVAILABLE:
        return None
    try:
        model = _get_model(language)
        text = _transcribe_once(model, audio_path, language, expected_text, beam_size=1)
        if expected_text and text and not word_matches(expected_text, text):
            retry_text = _transcribe_once(model, audio_path, language, expected_text, beam_size=5)
            print(
                f"[asr.transcribe] beam_size=1 result {text!r} didn't match "
                f"expected {expected_text!r}; retried with beam_size=5 -> {retry_text!r}"
            )
            if word_matches(expected_text, retry_text):
                return retry_text
        return text
    except Exception as e:
        # EXTENSION FIX: this used to swallow the exception completely,
        # which meant the app fell back to acoustic scoring silently --
        # with no way for anyone (including the person developing this)
        # to see WHY transcription failed on a specific real recording,
        # even after confirming the model itself loads and works fine
        # in isolation. Printing here surfaces the real error in the
        # terminal running `streamlit run app.py`, which is exactly
        # what's needed to diagnose a failure that only happens on
        # real recorded audio, not on a standalone model-load test.
        import traceback
        print(f"[asr.transcribe] FAILED for {audio_path} (language={language}): {e}")
        traceback.print_exc()
        return None


# ============================================================================
# EXTENSION: word-level timestamps, added specifically to support
# auto-slicing ONE parent/teacher sentence recording into per-word clips
# (see modules/voice_clone.py's slice_sentence_into_words) -- so a
# parent/teacher only ever records a sentence once, never word by word,
# and the child still gets a real per-word clip in that same voice.
# ============================================================================
def transcribe_word_timestamps(audio_path, language, expected_text=None):
    """
    Returns a list of {"word": str, "start": float, "end": float}
    (seconds, relative to the start of audio_path) using faster-
    whisper's own word-level timestamp feature, or None if ASR isn't
    available or genuinely fails to produce any words -- same honest
    "None means fall back" contract as transcribe() above. Callers
    should NOT treat a returned list as perfectly accurate down to the
    millisecond -- Whisper-family word timestamps are a real, useful
    signal but an approximation, not a lab-grade forced aligner -- see
    slice_sentence_into_words' own comments for how the caller compensates
    (a small padding around each boundary, and a length-mismatch fallback
    to equal-duration slicing when the word count doesn't line up).
    """
    if not ASR_LIBRARY_AVAILABLE:
        return None
    try:
        model = _get_model(language)
        segments, _ = model.transcribe(
            audio_path,
            language=_WHISPER_LANGUAGE_CODE.get(language, language),
            initial_prompt=expected_text,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"threshold": 0.35, "min_silence_duration_ms": 300},
            # Same speedup as transcribe() above, same reasoning: this
            # is only ever used to time-align words within ONE already-
            # known sentence (see slice_sentence_into_words), not open
            # dictation, so the default beam_size=5's extra exploration
            # isn't earning its cost here either.
            beam_size=1,
        )
        words = []
        for seg in segments:
            seg_words = getattr(seg, "words", None) or []
            for w in seg_words:
                token = (w.word or "").strip()
                if token:
                    words.append({"word": token, "start": float(w.start), "end": float(w.end)})
        return words or None
    except Exception as e:
        import traceback
        print(f"[asr.transcribe_word_timestamps] FAILED for {audio_path} (language={language}): {e}")
        traceback.print_exc()
        return None


def normalize_text(s):
    """Lowercase, strip punctuation/diacritics-sensitive whitespace
    quirks, collapse multiple spaces -- so 'Bicycle.' and 'bicycle'
    (or minor ASR punctuation differences) compare equal."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.lower().strip()
    s = re.sub(r"[.,!?;:\"'\u2018\u2019\u201c\u201d]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def word_matches(target_word, recognized_text):
    """
    True if the target word appears to have actually been said.
    Checks for an exact normalized match OR the target word appearing
    as a whole token inside the recognized text -- the latter case
    covers Whisper occasionally picking up a small extra leading/
    trailing sound ("the bicycle" when only "bicycle" was the target).
    """
    target_norm = normalize_text(target_word)
    recognized_norm = normalize_text(recognized_text)
    if not target_norm:
        return False
    if target_norm == recognized_norm:
        return True
    recognized_tokens = recognized_norm.split()
    return target_norm in recognized_tokens


def line_match_ratio(target_text, recognized_text):
    """
    For a full stanza/poem (not a single word): what fraction of the
    target's words were actually found, in order-independent fashion,
    in what was recognized. Returns (ratio_0_to_1, matched_words,
    total_words) so the UI can show real, specific feedback (e.g.
    "8 of 9 words") rather than only a bare percentage.
    """
    target_words = normalize_text(target_text).split()
    recognized_words = set(normalize_text(recognized_text).split())
    if not target_words:
        return 0.0, 0, 0
    matched = sum(1 for w in target_words if w in recognized_words)
    return matched / len(target_words), matched, len(target_words)