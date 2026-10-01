"""
Sarvam AI voice cloning -- a hosted, CLOUD tier for Custom Practice's
voice cloning, added alongside (not instead of, in the code itself --
see voice_clone.py's tier order) the free/local OpenVoice tier.

STATED HONESTLY, same as every other module in this project:

- This is a PAID, third-party cloud API (Sarvam AI), not a free/local
  model. It needs an internet connection every time a sentence is
  generated, and a SARVAM_API_KEY you get from your own Sarvam
  dashboard (https://www.sarvam.ai) -- never commit that key to source
  control or paste it into code; it's read from an environment
  variable only (see _get_client below).
- The parent/teacher's recorded reference clip is uploaded to Sarvam's
  servers to create the voice clone, and every generated sentence's
  audio is produced server-side. Worth knowing before recording real
  family members' voices for a real deployment, not just a local demo.
- In exchange, it's genuinely simpler and more reliable than the local
  OpenVoice pipeline: one clone -> one voice_id -> reused for ANY
  future sentence, in EITHER language (Sarvam's voice clones work
  cross-lingually between en-IN and kn-IN), with no local checkpoint
  download, no PyTorch model to load, nothing that can be killed by a
  dev-server restart mid-request.

Graceful degradation, same pattern as every other tier in this
project: if the `sarvamai` package isn't installed, or SARVAM_API_KEY
isn't set, or any call fails, functions here return None and print a
clear reason -- callers (voice_clone.py) treat that as "try the next
tier" or "fall back to real recorded playback", never as a crash.
"""
import os
import base64

try:
    from sarvamai import SarvamAI
    SARVAM_LIBRARY_AVAILABLE = True
except Exception:
    SARVAM_LIBRARY_AVAILABLE = False

_client = None
_client_init_failed = False


def api_key_configured():
    """True only if SARVAM_API_KEY is actually set in the environment.
    Doesn't validate the key itself (that only happens on first real
    call to Sarvam) -- just whether there's anything to try at all."""
    return bool(os.getenv("SARVAM_API_KEY"))


def sarvam_available():
    """True only if BOTH the `sarvamai` package is installed AND an API
    key is configured -- the two independent things that can each make
    this whole tier a no-op. Used by voice_clone.py to decide whether
    to even attempt Sarvam before falling through to OpenVoice."""
    return SARVAM_LIBRARY_AVAILABLE and api_key_configured()


def _get_client():
    global _client, _client_init_failed
    if _client is not None:
        return _client
    if _client_init_failed or not SARVAM_LIBRARY_AVAILABLE:
        return None
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        # Not an error -- most setups won't have this configured, and
        # that's fine; other cloning tiers (OpenVoice, real recorded
        # playback) still work regardless. Printed once so a run where
        # Sarvam cloning was actually expected shows why it's silently
        # skipped.
        print("[sarvam_voice] SARVAM_API_KEY environment variable is not set -- "
              "Sarvam cloning is skipped for this run (see requirements.txt for "
              "how to set it). This is not an error; the other cloning tiers "
              "still work.")
        _client_init_failed = True
        return None
    try:
        _client = SarvamAI(api_subscription_key=api_key)
        return _client
    except Exception as e:
        print(f"[sarvam_voice] Failed to initialize the Sarvam client: {type(e).__name__}: {e}")
        _client_init_failed = True
        return None


def create_voice_clone(audio_path, name="CurioKidz Voice", language="en-IN"):
    """
    Creates a cloned voice from a real recorded reference clip
    (`audio_path`, an existing WAV file on disk -- e.g. one of this
    profile's enrollment or sentence recordings). Returns a `voice_id`
    string to reuse for every future sentence in ANY supported
    language, or None if cloning couldn't be created for any reason
    (library missing, no API key, upload/API failure) -- the printed
    message says which.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        with open(audio_path, "rb") as f:
            result = client.voice_cloning.create_voice(file=f, name=name, language=language)
        voice_id = result.data.voice_id
        print(f"[sarvam_voice] Created cloned voice {voice_id!r} ({name!r}) from {audio_path}")
        return voice_id
    except Exception as e:
        print(f"[sarvam_voice] create_voice_clone FAILED for {audio_path} "
              f"(name={name!r}, language={language!r}): {type(e).__name__}: {e}")
        return None


def text_to_speech(text, voice_id, language_code, out_path):
    """
    Generates `text` in the already-cloned voice identified by
    `voice_id`, writing WAV bytes to `out_path`. Returns out_path on
    success, or None on any failure (bad/expired voice_id, network
    error, API error) -- the printed message says which.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        result = client.voice_cloning.text_to_speech(
            voice_id=voice_id, text=text, language_code=language_code
        )
        audio_bytes = base64.b64decode(result.audio)
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
        print(f"[sarvam_voice] text_to_speech ran without raising, but produced no audio "
              f"for text={text!r} voice_id={voice_id!r} language_code={language_code!r}.")
        return None
    except Exception as e:
        print(f"[sarvam_voice] text_to_speech FAILED for text={text!r} voice_id={voice_id!r} "
              f"language_code={language_code!r}: {type(e).__name__}: {e}")
        return None