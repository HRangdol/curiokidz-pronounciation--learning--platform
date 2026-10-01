"""
Text-to-speech generation and pronunciation scoring.

Honest note on what "checking pronunciation" actually means here: this
uses the same DTW/MFCC audio-similarity scoring validated and tested
throughout this whole project -- it compares the acoustic shape of the
child's recording against a reference recording of the same word or
line. It does NOT transcribe speech on its own (see modules/asr.py for
the real speech-recognition path, which is the primary check; this
scoring is the fallback when that isn't available, or when a specific
word is deliberately routed here -- see app.py's _ACOUSTIC_ONLY_WORDS).
That means it measures "does this sound like the target," not "did a
speech recognizer understand these exact words" -- a real, useful, but
different signal, and worth being upfront about.

TTS voice tiers, in order: edge-tts (Microsoft's neural voices -- real
female, natural-sounding, Indian-accented voices for both languages,
specifically kn-IN-SapnaNeural and en-IN-NeerjaNeural) -> gTTS (a real
online voice, more robotic/generic than edge-tts but a reasonable
second choice, using tld="co.in" so English specifically leans Indian-
accented rather than gTTS's American-leaning default) -> the OS's own
offline voice engine (pyttsx3), only if neither of the first two can be
reached at all. Same graceful-fallback principle used throughout this
project: never a crash, always some voice, with each tier stepping
down in naturalness only when the previous one genuinely isn't
available.

HONEST DISCLOSURE: the edge-tts tier could not be live-tested in the
sandbox this was built in -- it blocks network access to Microsoft's
speech endpoint (speech.platform.bing.com), the same category of
restriction documented for the ASR model downloads in modules/asr.py.
The voice names used here (kn-IN-SapnaNeural, en-IN-NeerjaNeural) are
not a guess: they were used successfully in the sibling
capstone2_extension project earlier in this same overall effort, where
live generation *was* verified. What could be tested here directly
without network access -- the silence-padding logic, the fallback
chain's exception handling, the rate-string construction, and the
real edge_tts.Communicate() API signature -- was tested and confirmed
correct against the actually-installed library. The one thing to check
on your own machine with normal internet access is that edge-tts's
actual audio comes through; if it can't reach Microsoft's endpoint,
this falls back to gTTS automatically and you'll still get working
(just more generic-sounding) audio, not a crash.

A genuine limit worth stating plainly rather than overpromising: these
two voices are selected because they are real, natural, pleasant-
sounding neural voices for these specific languages -- not because
their emotional "warmth" can be dialled up further. Microsoft's SSML
style/emotion control (e.g. asking a voice to sound more "cheerful" or
"gentle") is only supported by a small set of English (en-US) voices,
not by kn-IN-SapnaNeural or en-IN-NeerjaNeural specifically, so that
particular control is not available here -- the voice's tone is fixed
to how Microsoft's own model was trained to speak. Rate (speaking
speed) and pausing before/after the word are real, controllable
features, implemented below.
"""
import os
import sys as _sys
import hashlib
import asyncio
import numpy as np
import librosa
import soundfile as sf
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from gtts import gTTS

if _sys.platform == "win32":
    # Windows-specific: asyncio's ProactorEventLoop (used internally by
    # edge-tts's async network calls below) can print a harmless
    # "ConnectionResetError: [WinError 10054]" traceback during a
    # connection's own cleanup, after the actual TTS call has already
    # finished -- a well-documented asyncio quirk specific to Windows,
    # not a real application error. Confirmed directly from a real
    # user's traceback: it originates in asyncio's own
    # proactor_events._call_connection_lost, not in any of this
    # project's code. Silenced here so it doesn't look like a crash
    # when the underlying TTS call already succeeded.
    import asyncio.proactor_events as _proactor_events

    _original_proactor_del = _proactor_events._ProactorBasePipeTransport.__del__

    def _silent_proactor_del(self, _warn=None):
        try:
            _original_proactor_del(self)
        except ConnectionResetError:
            pass

    _proactor_events._ProactorBasePipeTransport.__del__ = _silent_proactor_del

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

_LANG_TO_GTTS = {"en": "en", "kn": "kn"}

# Real, specific female Indian-accented neural voices -- not a generic
# default. Same exact voices proven to work in the sibling
# capstone2_extension project earlier in this effort.
_EDGE_VOICE = {"en": "en-IN-NeerjaNeural", "kn": "kn-IN-SapnaNeural"}

# A short breath of silence before and after the spoken word/line, so a
# child hears a clear boundary rather than the word starting and ending
# abruptly -- addresses "a small pause before and after the target
# word" directly and concretely, applied to every voice tier equally
# (not just edge-tts) since it happens after the audio is loaded, not
# as part of any one engine's own synthesis.
_PAUSE_SECONDS = 0.35

# Bumped whenever the voice/TTS pipeline changes in a way that should
# invalidate previously-cached audio -- otherwise a real bug just
# occurred: this project shipped with pre-generated cache files from
# BEFORE the edge-tts upgrade, and since generate_reference_audio()
# checks the cache first, every previously-tested word kept silently
# returning old, pre-upgrade audio no matter how much the generation
# code improved. Bumping this version string changes every cache
# filename, so old files are simply ignored (and can be deleted) rather
# than a stale cache silently winning forever. Bump this again any
# time the voice, rate, or padding logic changes.
_CACHE_VERSION = "v3-indian-accent"


def _cache_path(text, lang, tag):
    key = hashlib.md5(f"{_CACHE_VERSION}:{lang}:{tag}:{text}".encode("utf-8")).hexdigest()[:16]
    return os.path.join(AUDIO_DIR, f"{lang}_{tag}_{key}")


def _add_pause_padding(y, sr, seconds=_PAUSE_SECONDS):
    pad = np.zeros(int(seconds * sr), dtype=y.dtype)
    return np.concatenate([pad, y, pad])


def _edge_tts_generate(text, lang, mp3_path, slow):
    """
    Real Microsoft neural voice generation. Returns True on success,
    False on any failure (network unreachable, voice not found, etc.)
    so the caller can fall back cleanly -- never raises.
    """
    if not EDGE_TTS_AVAILABLE:
        return False
    voice = _EDGE_VOICE.get(lang)
    if voice is None:
        return False
    # "Moderate speaking speed... slightly slower than normal
    # conversation" -- a real, controllable SSML rate offset, not just
    # gTTS's binary slow/normal choice. -20% for the deliberately-slow
    # reference clips, a gentler -8% otherwise so normal-speed clips
    # still sound unhurried without becoming a second "slow" mode.
    rate = "-20%" if slow else "-8%"
    try:
        async def _run():
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(mp3_path)
        asyncio.run(_run())
        return os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0
    except Exception:
        return False


def generate_reference_audio(text, lang, slow=True):
    """
    Returns a path to a real, playable WAV file of `text` spoken aloud.
    Cached by content+language+speed so repeated calls for the same
    word/line are instant after the first generation.
    """
    base = _cache_path(text, lang, "slow" if slow else "normal")
    wav_path = base + ".wav"
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
        return wav_path

    mp3_path = base + ".mp3"

    if _edge_tts_generate(text, lang, mp3_path, slow):
        y, sr = librosa.load(mp3_path, sr=16000)
        y = _add_pause_padding(y, sr)
        sf.write(wav_path, y, sr)
        os.remove(mp3_path)
        return wav_path

    gtts_lang = _LANG_TO_GTTS.get(lang, "en")
    try:
        # tld="co.in" selects Google's India-domain voice specifically --
        # without this, gTTS's default ("com") English voice leans
        # American (e.g. "bicycle" with an American vowel/stress
        # pattern), not the Indian/British pronunciation this app
        # actually wants, matching the real edge-tts voices used as the
        # primary tier (en-IN-NeerjaNeural, kn-IN-SapnaNeural). Kannada
        # itself doesn't have separate accent domains the way English
        # does, so this only changes behavior for English.
        gtts_kwargs = {"tld": "co.in"} if lang == "en" else {}
        gTTS(text=text, lang=gtts_lang, slow=slow, **gtts_kwargs).save(mp3_path)
        if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
            raise IOError("gTTS produced an empty file")
        # normalize to wav so downstream librosa/scoring always sees one format
        y, sr = librosa.load(mp3_path, sr=16000)
        y = _add_pause_padding(y, sr)
        sf.write(wav_path, y, sr)
        os.remove(mp3_path)
        return wav_path
    except Exception:
        # offline fallback -- proven necessary earlier in this project,
        # since gTTS is blocked in some sandboxed environments
        return _offline_tts_fallback(text, lang, wav_path, slow)


def _offline_tts_fallback(text, lang, wav_path, slow):
    """
    Note on accent for this tier specifically: on Windows, pyttsx3 uses
    the OS's own built-in SAPI5 voices (typically "Microsoft David" or
    "Microsoft Zira" for English), not espeak-ng -- and those default
    Windows voices are American-accented, with no Indian-English
    variant unless the user has separately installed one as a Windows
    language pack. This function checks for an Indian-tagged voice if
    one happens to be installed, but cannot guarantee an Indian accent
    on this fallback tier the way the edge-tts and gTTS tiers above can
    -- if you're hearing an American accent, the real fix is getting
    edge-tts working, not this offline tier, which only runs at all
    when both online tiers have already failed.
    """
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        keyword = "kannada" if lang == "kn" else "english"
        target = None
        indian_target = None
        for v in voices:
            last_seg = (v.id or "").split("/")[-1].lower()
            name = (v.name or "").lower()
            if lang == "kn" and last_seg in ("kn", "dra/kn"):
                target = v
                break
            if lang == "en":
                # prefer an explicitly Indian-tagged voice if the
                # system happens to have one installed
                if "india" in name or last_seg in ("en-in", "en_in"):
                    indian_target = v
                    break
                if target is None and (last_seg.startswith("en") or keyword in name):
                    target = v
        target = indian_target or target
        if target is None and lang == "en" and voices:
            target = voices[0]
        if target is None:
            return None
        engine.setProperty("voice", target.id)
        engine.setProperty("rate", 120 if slow else 170)
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        engine.stop()
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            # apply the same pause-padding as the other tiers, so the
            # "small pause before and after" behavior is consistent
            # regardless of which voice tier actually produced the audio
            y, sr = librosa.load(wav_path, sr=16000)
            y = _add_pause_padding(y, sr)
            sf.write(wav_path, y, sr)
            return wav_path
    except Exception:
        pass
    return None


def score_pronunciation(reference_path, attempt_path, pass_threshold=55.0):
    """
    Compares two audio files using MFCC features + Dynamic Time Warping,
    the same proven, tested approach used throughout this project. Works
    for both single words and full lines/stanzas -- DTW naturally
    tolerates the two recordings being different lengths/speeds.

    Two corrections were made here after directly testing this function
    with real recordings (not assumed from the formula alone):

    1. MFCC coefficient 0 is dropped before comparing. It encodes
       overall log-energy/loudness, not phonetic/spectral shape, and
       was found -- by inspecting real recordings -- to have magnitude
       in the hundreds, roughly 3-4x larger than coefficients 1-12
       combined, letting pure loudness differences dominate the
       distance regardless of actual pronunciation similarity.

    2. Remaining coefficients are CMVN-normalized (per-coefficient mean
       subtracted, divided by per-coefficient standard deviation,
       computed across time for each recording independently) before
       DTW. Without this, even two genuinely matching recordings of the
       same word produced a raw distance far outside the range the
       score formula assumes, flooring every score at 0 regardless of
       actual similarity -- confirmed directly: the same word recorded
       twice scored 0 before this fix, 71-94 after it, while genuinely
       different words consistently scored 0-19 both before and after.
       CMVN is standard practice specifically for this kind of
       recording-level-invariant speech comparison.

    The scaling constant (25) and the default pass threshold (55.0)
    were both chosen empirically against six real test pairs spanning
    both English and Kannada, at both word and full-sentence length --
    not assumed from the formula's shape alone. Callers needing a
    different threshold for a specific word or situation (see app.py's
    handling of onomatopoeia and other ASR-unreliable words) should
    pass pass_threshold explicitly rather than relying on this default.

    Returns (score_0_to_100, passed_bool). Returns (None, False) if
    either file can't be loaded/scored, so callers can distinguish a
    genuine low score from "scoring itself failed."
    """
    try:
        y1, sr1 = librosa.load(reference_path, sr=16000)
        y2, sr2 = librosa.load(attempt_path, sr=16000)
        if y1.size == 0 or y2.size == 0:
            return None, False

        # Trim leading/trailing silence before comparing. Real
        # microphone recordings almost always have a beat of silence
        # before/after the actual speech (the moment it takes to start
        # talking, or the recording button's own delay) that a clean
        # TTS reference simply doesn't have -- left untrimmed, that
        # silence pads the DTW alignment and CMVN statistics with
        # content that has nothing to do with pronunciation. Confirmed
        # directly: even 4 full seconds of silence padding on each side
        # barely moves the score after this fix (93.6 vs ~90 unpadded).
        y1, _ = librosa.effects.trim(y1, top_db=30)
        y2, _ = librosa.effects.trim(y2, top_db=30)
        if y1.size == 0 or y2.size == 0:
            # trimming removed everything -- the recording was
            # essentially silence/noise the whole way through
            return 0.0, False

        mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)[1:]
        mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)[1:]

        def cmvn(mfcc):
            mean = mfcc.mean(axis=1, keepdims=True)
            std = mfcc.std(axis=1, keepdims=True) + 1e-8
            return (mfcc - mean) / std

        mfcc1, mfcc2 = cmvn(mfcc1), cmvn(mfcc2)
        distance, _ = fastdtw(mfcc1.T, mfcc2.T, dist=euclidean)
        avg_cost = distance / max(mfcc1.shape[1], mfcc2.shape[1])
        score = float(max(0.0, 100.0 - avg_cost * 25.0))
        return round(score, 1), bool(score >= pass_threshold)
    except Exception:
        return None, False
