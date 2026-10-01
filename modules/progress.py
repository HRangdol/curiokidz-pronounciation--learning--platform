"""Small helpers for turning a stanza's lines into practiceable words,
and for computing where a child currently is in a lesson."""


def stanza_words(stanza_lines):
    """Flattens a stanza's lines into an ordered list of words, keeping
    simple trailing punctuation attached (so what's shown/read matches
    how the line actually reads) but usable as individual TTS/scoring
    targets."""
    words = []
    for line in stanza_lines:
        words.extend(line.split())
    return words


def full_text(stanza_lines):
    return " ".join(stanza_lines)


def whole_poem_text(lesson):
    return " ".join(" ".join(stanza) for stanza in lesson["stanzas"])


# ============================================================================
# EXTENSION: flattened word list across the WHOLE lesson (every stanza),
# each tagged with which stanza it belongs to and its position within
# that stanza -- exactly what the synced, word-highlighted reading pane
# needs to know "is this the word currently being practiced" while
# showing the full lesson/poem text on one side, not just the current
# stanza.
# ============================================================================
def flatten_lesson_words(lesson):
    """
    Returns a flat, ordered list of
        {"stanza_idx": int, "word_in_stanza": int, "word": str}
    covering every word in every stanza of `lesson`, in reading order.
    """
    flat = []
    for s_idx, stanza_lines in enumerate(lesson["stanzas"]):
        for w_idx, word in enumerate(stanza_words(stanza_lines)):
            flat.append({"stanza_idx": s_idx, "word_in_stanza": w_idx, "word": word})
    return flat
