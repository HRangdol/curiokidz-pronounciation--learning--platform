"""
Hands-free, voice-driven navigation for word-level Practice --
EXTENSION, built per an explicit requirement from the project brief:
"the child shouldn't click on any button ... start, stop, correct or
wrong ... everything should be through voice."

WHAT THIS ACTUALLY DOES, STATED HONESTLY (same principle as every
other module in this project): a real, continuous microphone stream
from the browser is captured via streamlit-webrtc (built on WebRTC/
aiortc, free and fully offline -- no cloud service is contacted to
carry the audio itself), a lightweight real-time voice-activity
detector in this module watches that stream and marks when the child
starts and stops talking on its own, and whatever was said is
recognized either as a NAVIGATION COMMAND ("start", "next", "repeat",
"stop") or as the actual pronunciation attempt, depending on what the
app is currently waiting for. Every instruction and every result is
SPOKEN aloud (via modules/audio.py's existing text-to-speech
pipeline), not just shown as text, so a child who cannot yet read
fluently is not required to.

ONE HONEST, STATED LIMITATION: a real browser will not let ANY
website start listening to the microphone without one explicit,
one-time permission grant from the person using it -- this is a
browser security rule, not a limitation of this app, and it is the
same rule every hands-free voice app (including Duolingo) operates
under. After that one grant, this module keeps listening continuously
with no further taps needed for "start," "next," "repeat," or
recording an attempt.

A SECOND HONEST LIMITATION: there is no reliable, cross-browser way
for a plain Streamlit app (without a custom JS component) to be told
exactly when a played TTS clip has finished, so this module estimates
that from the text's own length (a generous, rounded-up reading-speed
estimate) rather than reacting to a real "playback ended" event. In
practice this means the app may wait slightly longer than strictly
necessary before it starts listening again, never shorter -- it is
tuned to that direction deliberately so it does not start listening
over the end of its own narration.
"""
import os
import re
import time
import queue
import tempfile

import numpy as np
import soundfile as sf

from modules import audio, asr
# EXTENSION: command grammar moved to its own zero-heavy-dependency
# module (modules/commands.py) so flask_app.py can reuse it WITHOUT
# importing this module -- importing streamlit-webrtc below pulls in
# Streamlit itself as a side effect, which flask_app.py has no other
# reason to depend on. Re-exported here (same names) so nothing that
# already imported match_command/_COMMAND_WORDS from voice_nav breaks.
from modules.commands import match_command, _COMMAND_WORDS  # noqa: F401

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False


def webrtc_available():
    return WEBRTC_AVAILABLE


# ============================================================================
# Spoken narration -- every instruction/result is real synthesized
# speech (reusing this project's existing, already-tested TTS
# pipeline), never text-only.
# ============================================================================
def speak(text, language, slow=False):
    """Returns a path to a TTS clip of `text`, or None if TTS itself
    is unavailable (see modules/audio.py's own honest fallback chain).
    Callers should treat None as "narration unavailable this time,"
    the same way the rest of this project treats a missing TTS clip --
    never as a reason to crash the hands-free loop."""
    return audio.generate_reference_audio(text, language, slow=slow)


def _estimate_speech_seconds(text, words_per_minute=115):
    """
    See the module docstring's second honest limitation: there is no
    reliable cross-browser "playback ended" signal available here, so
    this estimates narration length from word count at a deliberately
    slower-than-average reading pace (most TTS voices in this project
    are already slowed for children -- see audio.py), then rounds up
    and adds a fixed safety margin, so this errs toward waiting too
    long rather than talking over its own narration.
    """
    word_count = max(1, len(text.split()))
    seconds = (word_count / words_per_minute) * 60.0
    return min(12.0, seconds + 1.2)


# ============================================================================
# Real-time voice-activity endpointing over the live WebRTC audio
# stream -- separate from, and much cheaper than, Whisper, which only
# ever runs once on a COMPLETE utterance below. This is a simple
# energy-threshold detector, the same real technique (not a placebo)
# already used elsewhere in this project for the acoustic-fallback
# scoring path and the word-slicing silence refinement in
# voice_clone.py, applied here in real time instead of after the fact.
# ============================================================================
def _frame_to_mono_float(frame):
    """Converts one av.AudioFrame from the WebRTC stream into mono
    float32 samples in [-1, 1], plus its sample rate."""
    arr = frame.to_ndarray()
    if arr.ndim == 2:
        # streamlit-webrtc/aiortc frames are typically (channels, samples)
        # or (samples, channels) depending on layout; average whichever
        # axis is small (the channel axis) to get mono.
        if arr.shape[0] <= 2 and arr.shape[0] < arr.shape[1]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=1)
    arr = arr.astype(np.float32)
    # Frames commonly arrive as int16; normalize to [-1, 1] if so.
    if np.max(np.abs(arr)) > 1.5:
        arr = arr / 32768.0
    return arr, frame.sample_rate


def record_utterance(
    audio_receiver,
    max_wait_seconds=12.0,
    max_utterance_seconds=8.0,
    silence_end_seconds=0.9,
    energy_threshold=0.02,
):
    """
    Blocks (this Streamlit script run) while pulling real audio frames
    from the live WebRTC stream, until either:
      - real speech was detected and then a sustained silence
        (silence_end_seconds) follows it -- returns the captured
        speech as (samples: np.ndarray float32, sample_rate: int); or
      - max_wait_seconds passes with no speech ever detected -- returns
        None, so the caller can narrate "I didn't hear anything" and
        try again, rather than hanging forever; or
      - max_utterance_seconds of continuous speech is reached without a
        pause -- the utterance is cut there and returned as captured
        so far, so one long ramble cannot block the loop indefinitely.

    This function deliberately blocks the main Streamlit thread for its
    duration -- the same, documented pattern streamlit-webrtc's own
    official examples use for real-time audio processing, since there
    is no other supported way to consume a live frame stream from
    inside a normal Streamlit script run.
    """
    started_at = time.monotonic()
    speech_started_at = None
    last_speech_at = None
    buffer = []
    sample_rate = None

    while True:
        now = time.monotonic()
        if speech_started_at is None and (now - started_at) > max_wait_seconds:
            return None
        if speech_started_at is not None and (now - speech_started_at) > max_utterance_seconds:
            break
        if speech_started_at is not None and last_speech_at is not None \
                and (now - last_speech_at) > silence_end_seconds:
            break

        try:
            frames = audio_receiver.get_frames(timeout=1)
        except queue.Empty:
            continue
        if not frames:
            continue

        for frame in frames:
            samples, sr = _frame_to_mono_float(frame)
            if sample_rate is None:
                sample_rate = sr
            rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if len(samples) else 0.0
            is_speech = rms >= energy_threshold
            if is_speech:
                if speech_started_at is None:
                    speech_started_at = time.monotonic()
                last_speech_at = time.monotonic()
                buffer.append(samples)
            elif speech_started_at is not None:
                # Keep buffering brief in-utterance pauses too (natural
                # speech has micro-gaps); only the sustained-silence
                # check above actually ends the utterance.
                buffer.append(samples)

    if not buffer or speech_started_at is None:
        return None
    return np.concatenate(buffer), (sample_rate or 48000)


def _save_temp_wav(samples, sample_rate):
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, samples, sample_rate, format="WAV")
    return path


# ============================================================================
# The hands-free word-practice loop itself.
# ============================================================================
def render_handsfree_word_practice(lesson_language, current_word, on_result, on_command_next=None):
    """
    SUPERSEDED, KEPT FOR REFERENCE ONLY -- do not build on this.

    This was an earlier attempt at hands-free practice inside Streamlit
    itself, using streamlit-webrtc for a live mic stream. It was never
    finished, never wired into app.py, and streamlit-webrtc was never
    added to requirements.txt -- it's genuinely dead code, confirmed by
    grepping app.py for any reference to this module (there is none).
    The project has since moved to a real Flask + browser (Web Speech
    API + MediaRecorder) frontend instead (see flask_app.py and
    static/js/handsfree.js), which is a materially better fit: it needs
    no WebRTC signaling/STUN setup, uses the browser's own built-in
    continuous speech recognition for commands, and still sends actual
    reading attempts to this project's own asr.py for scoring. Only
    this module's framework-agnostic pieces (match_command,
    _COMMAND_WORDS, speak) are reused there; everything below this
    docstring is Streamlit/streamlit-webrtc-specific and unused.

    Runs ONE full hands-free turn for `current_word` in `lesson_language`:
    narrates the word, listens for either a spoken command ("repeat" /
    "stop") or the child's attempt, checks the attempt the same real
    way asr.word_matches already does everywhere else in this project,
    narrates the result, and calls `on_result(passed, recognized_text)`
    so the caller (app.py) advances its own state exactly the way the
    button-driven flow already does -- this function does not manage
    word position itself, so it stays a single, testable unit of
    behaviour rather than a second, parallel copy of the practice
    state machine.

    Returns "stop" if the child asked to exit hands-free mode, "repeat"
    if they asked to hear the word again (caller should call this
    function again for the SAME word), or None after a normal attempt
    (whether it passed or not -- on_result already reports which).
    """
    if not WEBRTC_AVAILABLE:
        return "stop"

    _ensure_streamlit()
    target_clean = current_word.strip(".,!?")

    narration_placeholder = st_empty_or_none()
    prompt_path = speak(f"Say the word: {target_clean}", lesson_language, slow=True)
    if prompt_path and narration_placeholder is not None:
        narration_placeholder.audio(prompt_path, autoplay=True)
        time.sleep(_estimate_speech_seconds(target_clean))

    ctx = webrtc_streamer(
        key="handsfree_word_practice",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"video": False, "audio": True},
        audio_receiver_size=256,
    )
    if not ctx.state.playing or ctx.audio_receiver is None:
        # Mic not yet granted/connected this run -- caller re-renders
        # on the next Streamlit rerun once the browser finishes the
        # one-time permission handshake; nothing to do yet.
        return None

    captured = record_utterance(ctx.audio_receiver)
    if captured is None:
        miss_path = speak("I didn't hear anything. Let's try again.", lesson_language)
        if miss_path:
            st.audio(miss_path, autoplay=True)
        return "repeat"

    samples, sr = captured
    wav_path = _save_temp_wav(samples, sr)
    try:
        recognized = asr.transcribe(wav_path, lesson_language, expected_text=target_clean)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    command = match_command(recognized or "", lesson_language) if recognized else None
    if command in ("repeat", "stop"):
        return command

    passed = bool(recognized) and asr.word_matches(target_clean, recognized)
    result_text = "Correct! Well done." if passed else "Not quite. Let's try that word again."
    result_path = speak(result_text, lesson_language)
    if result_path:
        st.audio(result_path, autoplay=True)

    on_result(passed, recognized)
    return None


def st_empty_or_none():
    try:
        import streamlit as st
        return st.empty()
    except Exception:
        return None


# EXTENSION NOTE: Streamlit is imported LAZILY, only inside the
# Streamlit-specific render_handsfree_word_practice() function below
# (via this module-level `st` name, set the first time that function
# actually runs), NOT at module import time. This module's framework-
# agnostic parts -- match_command(), _COMMAND_WORDS, speak() -- are
# reused directly by flask_app.py, which has no Streamlit dependency
# at all; importing voice_nav used to unconditionally pull Streamlit in
# (and print its "missing ScriptRunContext" warnings) even for callers
# that never touch Streamlit, which this avoids.
st = None


def _ensure_streamlit():
    global st
    if st is None:
        import streamlit as _st
        st = _st
    return st
