"""
Spoken navigation-command grammar -- deliberately its own tiny module
with ZERO optional/heavy dependencies (no Streamlit, no streamlit-webrtc,
nothing that could accidentally pull in something unrelated at import
time). Used by both flask_app.py (the real, active hands-free frontend)
and modules/voice_nav.py (its Streamlit/streamlit-webrtc predecessor,
now superseded -- see that module's docstring). Kept intentionally
small and phonetically distinct within each language, since matching
is done with the same real speech-recognition/normalization already
used everywhere else in this project, not a special wake-word engine.
"""
from modules import asr

_COMMAND_WORDS = {
    "en": {
        "start": ["start", "begin", "ready", "go"],
        "next": ["next", "continue", "skip"],
        "repeat": ["repeat", "again", "replay"],
        "stop": ["stop", "exit", "quit", "done"],
    },
    "kn": {
        "start": ["ಶುರು", "ಪ್ರಾರಂಭ"],
        "next": ["ಮುಂದೆ"],
        "repeat": ["ಮತ್ತೆ"],
        "stop": ["ನಿಲ್ಲಿಸು", "ನಿಲ್ಲಿಸಿ"],
    },
}


def match_command(recognized_text, language):
    """
    Returns the canonical command ("start"/"next"/"repeat"/"stop") if
    the recognized text loosely matches one of that command's known
    spoken forms in this language, else None. Uses the same normalize/
    contains-a-token approach as asr.word_matches, deliberately --
    consistency with how every other word-match in this project works,
    not a new, separate matching rule to reason about.
    """
    grammar = _COMMAND_WORDS.get(language, _COMMAND_WORDS["en"])
    norm = asr.normalize_text(recognized_text)
    if not norm:
        return None
    tokens = set(norm.split())
    for command, forms in grammar.items():
        for form in forms:
            form_norm = asr.normalize_text(form)
            if form_norm == norm or form_norm in tokens:
                return command
    return None
