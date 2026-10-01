"""
Real lesson content for the guided reading-practice flow.

Every stanza below is real, verified text -- extracted directly from the
official source PDF and, for Kannada, additionally checked against a
rendered image of the actual page (the KTBS PDF uses a legacy, non-
Unicode font; raw text extraction produces mojibake, so this content was
OCR'd and then manually verified line-by-line against the page image
before being used here -- not trusted from OCR alone).

Sources:
  - "My Bicycle" and "Come Back Soon": NCERT Mridang, Grade 2 English
    reader (bemr101.pdf, bemr105.pdf), Reprint 2026-27.
  - "ಬಣ್ಣದ ಹಕ್ಕಿ" (Bannada Hakki): official KTBS ಸವಿ ಕನ್ನಡ (Savi Kannada)
    Grade 2 reader, Part 1, page 8 (ಪಾಠ 1), 2026-27 session.

Lessons are ordered deliberately -- English, Kannada, English -- and are
meant to unlock strictly in sequence (see modules/progress.py): a lesson
is not shown at all until the previous one has been fully completed.
"""

LESSONS = [
    {
        "id": "en-my-bicycle",
        "order": 1,
        "language": "en",
        "language_label": "English",
        "title": "My Bicycle",
        "subtitle": "NCERT Mridang, Grade 2 \u2014 Unit 1: Fun with Friends",
        "image": "assets/images/en_my_bicycle.jpeg",
        "stanzas": [
            [
                "I have a red bicycle.",
                "I pedal and I ride.",
                "Trin, trin, trin, goes the bell.",
                "Makes my friends move aside.",
            ],
            [
                "The seat is blue in colour,",
                "The tyres and pedals are black.",
                "Sometimes fast, sometimes slow,",
                "But my bicycle is always on track.",
            ],
            [
                "To keep my bicycle moving,",
                "All I need is some air.",
                "I put on my little helmet,",
                "And go happily everywhere.",
            ],
        ],
    },
    {
        "id": "kn-bannada-hakki",
        "order": 2,
        "language": "kn",
        "language_label": "ಕನ್ನಡ (Kannada)",
        "title": "ಬಣ್ಣದ ಹಕ್ಕಿ",
        "subtitle": "KTBS ಸವಿ ಕನ್ನಡ, Grade 2 \u2014 ಪಾಠ 1",
        "image": "assets/images/kn_bannada_hakki_full.png",
        "stanzas": [
            [
                "ಬಾರಲೇ ಹಕ್ಕಿ ಬಣ್ಣದ ಹಕ್ಕಿ",
                "ಬಾರಲೇ ಹಕ್ಕಿ ಹಾರುವ ಹಕ್ಕಿ",
            ],
            [
                "ಗೆಳೆಯರು ಆಡಲು ಯಾರೂ ಇಲ್ಲ",
                "ಕಳೆಯುವುದೆಂತು ವೇಳೆಯನೆಲ್ಲ",
                "ಬಾ ಬಾ ನನಗೂ ಹಾಡಲು ಕಲಿಸು",
                "ಬಾ ಬಾ ನನಗೂ ಹಾರಲು ಕಲಿಸು",
            ],
            [
                "ಅವ್ವನು ನೀರಿಗೆ ಹೋಗಿಹಳು",
                "ಅಪ್ಪನು ಪೇಟೆಗೆ ಹೋಗಿಹನು",
                "ಅವ್ವನು ಬರಲಿ ಅಪ್ಪನು ಬರಲಿ",
                "ತಿನ್ನಲು ಹಣ್ಣನು ಕೊಡಿಸುವೆನು",
            ],
            [
                "ನನಗೂ ಎರಡು ರೆಕ್ಕೆಯ ಹಚ್ಚು",
                "ಹಿಂಗಡೆ ಬಣ್ಣದ ಪುಚ್ಚವ ಚುಚ್ಚು",
                "ನೆತ್ತಿಯ ಮೇಲೆ ಜುಟ್ಟನು ಹಚ್ಚು",
                "ಮೆಲ್ಲನೆ ಮೇಲೆ ಹಾರಲು ಹಚ್ಚು",
            ],
            [
                "ನಾನು ನೀನೂ ಇಬ್ಬರು ಕೂಡಿ",
                "ಮುಗಿಲಿನ ಕಡೆಗೆ ಹಾರುವ ಬಾ",
                "ಅಲ್ಲಿಂದಿತ್ತ ಇಲ್ಲಿಂದತ್ತ",
                "ಹಾರುತ ಹಾಡುತ ನಲಿಯುವ ಬಾ.",
            ],
        ],
    },
    {
        "id": "en-come-back-soon",
        "order": 3,
        "language": "en",
        "language_label": "English",
        "title": "Come Back Soon",
        "subtitle": "NCERT Mridang, Grade 2 \u2014 Unit 3: Going Places",
        "image": "assets/images/en_come_back_soon.jpeg",
        "stanzas": [
            [
                "Take a bus",
                "Or take a train,",
                "Take a boat",
                "Or take a plane,",
                "Take a taxi,",
                "Take a car,",
                "Maybe near",
                "Or maybe far,",
            ],
            [
                "Take a rocket",
                "To the moon,",
                "But be sure",
                "To come back soon.",
            ],
        ],
    },
]


def get_lesson(lesson_id):
    for lesson in LESSONS:
        if lesson["id"] == lesson_id:
            return lesson
    # EXTENSION: also check teacher-uploaded lessons (see modules/db.py)
    # so every existing caller of get_lesson() -- the whole practice
    # flow, resume, gating -- works on an uploaded lesson exactly the
    # same way it already works on a built-in one, with no change to
    # any of that calling code.
    from modules import db
    for lesson in db.get_uploaded_lessons():
        if lesson["id"] == lesson_id:
            return lesson
    return None


def get_lesson_by_order(order):
    for lesson in LESSONS:
        if lesson["order"] == order:
            return lesson
    return None


def total_lessons():
    # EXTENSION: includes uploaded lessons, so dashboard counts like
    # "3/5 lessons completed" are correct once a teacher has added
    # content, not silently stuck at the original built-in count.
    from modules import db
    return len(LESSONS) + len(db.get_uploaded_lessons())


def lessons_for_language(lang):
    """
    Real, ordered lesson list for ONE language track (e.g. just the
    English lessons, in the order they should be taken). Used so a
    child can choose to practice English or Kannada specifically,
    with each language progressing and gating independently, rather
    than one single mixed-language sequence.

    EXTENSION: teacher-uploaded lessons for this language are appended
    AFTER the built-in ones, in upload order. Because is_lesson_unlocked()
    in modules/db.py gates purely by POSITION within whatever list this
    function returns, an uploaded lesson naturally becomes the next
    lesson in the existing unlock sequence -- it only unlocks once every
    built-in lesson before it in this same language is completed, using
    the exact same gating logic already in place, unchanged.
    """
    from modules import db
    built_in = [l for l in LESSONS if l["language"] == lang]
    uploaded = [l for l in db.get_uploaded_lessons() if l["language"] == lang]
    return built_in + uploaded


def available_languages():
    """(code, label) pairs for every language actually present in
    LESSONS, in the order they first appear -- avoids hardcoding
    language names separately from the real content.

    EXTENSION: also considers uploaded lessons, so if a teacher's
    upload workflow ever introduces a language not already in the
    built-in set, it still shows up as a real subject choice -- though
    in practice a teacher is only offered the two already-supported
    languages when uploading (see render_teacher_upload() in app.py),
    since a new language would need its own speech-recognition and
    text-to-speech support this app doesn't have.
    """
    from modules import db
    seen = []
    for l in LESSONS:
        pair = (l["language"], l["language_label"])
        if pair not in seen:
            seen.append(pair)
    for l in db.get_uploaded_lessons():
        pair = (l["language"], l["language_label"])
        if pair not in seen:
            seen.append(pair)
    return seen
