"""
Extracts readable text from a teacher-uploaded PDF and turns it into
the same stanza/line structure the rest of this app already expects
(see modules/content.py's LESSONS -- a list of stanzas, each stanza a
list of line strings).

Uses PyPDF2 specifically -- the same library already used for the
original NCERT/KTBS content ingestion for this project (see
modules/content.py's module docstring) -- rather than introducing a
different PDF library just for this new feature.

HONEST LIMITATION, stated plainly: raw automated PDF text extraction is
inherently less reliable than the hand-verified built-in lessons (which
were checked line-by-line against the source page image before being
used). A real PDF can have multi-column layouts, headers/footers mixed
into the body text, or unusual spacing that breaks clean extraction.
For exactly this reason, extraction here is NOT the final step -- the
teacher always sees and can edit the extracted text before it's saved
as a lesson (see render_teacher_upload() in app.py), the same principle
used for the original content: never trust raw extraction alone.

A SECOND, MORE SEVERE limitation, confirmed directly with a real
teacher-uploaded PDF: some Kannada (and other Indic-language) PDFs --
including, per modules/content.py's own docstring, the original KTBS
source PDF this project already had to work around once -- use legacy,
pre-Unicode fonts. These fonts render correctly on screen, but the
PDF's internal text layer maps each glyph to an arbitrary Latin-range
byte, not the real Kannada Unicode codepoint. Direct text extraction
on such a PDF doesn't produce slightly-wrong text -- it produces
complete mojibake ("§UÉÎ vÀªÀÄä..." instead of "ಈಗ ಅವರ...") that no
amount of stanza/line cleanup can fix, because the extracted characters
never corresponded to the real ones in the first place.

The real, working fix -- confirmed directly here with genuine Kannada
text through the full pipeline -- is OCR: render each PDF page as an
actual image and read the visual glyphs with Tesseract (using its
trained Kannada model, package tesseract-ocr-kan), rather than trusting
the PDF's internal text layer at all. is_likely_mojibake() below
detects this situation automatically by checking whether extracted
"Kannada" text actually contains real Kannada Unicode characters; the
upload UI uses this to offer OCR automatically rather than silently
saving garbage or requiring the teacher to already know their PDF uses
a legacy font.
"""
import io
import re
import PyPDF2

try:
    import pymupdf as fitz  # PyMuPDF -- renders PDF pages as real images for OCR
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

CHAPTER_MARKER = "---CHAPTER---"

# App language code -> Tesseract's own language code for its trained
# model files (installed via the tesseract-ocr-kan/tesseract-ocr-eng
# system packages, not a Python dependency).
_TESSERACT_LANG = {"en": "eng", "kn": "kan"}

# The real Kannada Unicode block (U+0C80-U+0CFF) -- used to detect
# whether extracted "Kannada" text is genuine or mojibake, per the
# module docstring above.
_KANNADA_UNICODE_RANGE = (0x0C80, 0x0CFF)


def extract_text_from_pdf(file_obj):
    """
    Returns the raw extracted text of every page in the PDF, joined
    with a blank line between pages (a reasonable default stanza/
    paragraph boundary, since a new page very often starts a new
    thought) so the teacher has a sensible starting point to edit from
    rather than one giant unbroken block of text.
    """
    reader = PyPDF2.PdfReader(file_obj)
    page_texts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            page_texts.append(text)
    return "\n\n".join(page_texts)


def is_likely_mojibake(text, language):
    """
    For Kannada specifically: real Kannada text should be made up
    substantially of actual Kannada Unicode characters. A PDF using a
    legacy, non-Unicode font instead produces text that LOOKS like it
    might contain Kannada-adjacent shapes but is actually just Latin-
    range bytes and symbols -- confirmed directly against a real
    teacher-uploaded PDF, where extracted "Kannada" text contained
    zero real Kannada Unicode characters. Checked only for Kannada,
    since this specific legacy-font problem is specific to some Indic-
    script PDFs, not English ones.
    """
    if language != "kn":
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    kannada_count = sum(1 for c in letters if _KANNADA_UNICODE_RANGE[0] <= ord(c) <= _KANNADA_UNICODE_RANGE[1])
    return (kannada_count / len(letters)) < 0.3


def ocr_available():
    return OCR_AVAILABLE


# ============================================================================
# EXTENSION: real diagnosis of WHY OCR didn't run, instead of a single
# silent None. Two genuinely different failure modes were found by
# testing this directly, and each needs a different fix from whoever is
# running the app:
#   1. The Python packages (pymupdf/pytesseract/Pillow) are not
#      installed -- OCR_AVAILABLE is False before OCR is ever attempted.
#   2. The packages ARE installed, but pytesseract is only a thin
#      wrapper around the real, separate Tesseract OCR program -- if
#      that program (and, for Kannada, its "kan" trained-language-data
#      file) isn't installed at the OS level and on PATH, every OCR
#      call raises pytesseract.TesseractNotFoundError (or a decode
#      error for a missing language file), which the previous version
#      of this function caught silently and returned None for -- same
#      as case 1 to any caller, even though the fix is completely
#      different (an OS-level installer, not a pip install).
# diagnose_ocr_setup() lets callers (flask_app.py) tell these apart and
# show the teacher the real, actionable reason, rather than silently
# handing back unreadable text as if nothing had gone wrong.
# ============================================================================
def diagnose_ocr_setup(language):
    """Returns (ready: bool, reason: str | None). reason is None when
    ready is True; otherwise a short, specific explanation of exactly
    what is missing."""
    if not OCR_AVAILABLE:
        return False, (
            "the pymupdf/pytesseract/Pillow Python packages are not installed "
            "(run: pip install pymupdf pytesseract Pillow)"
        )
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return False, (
            "the Tesseract OCR program itself is not installed, or not on your "
            "system PATH (this is separate from the Python packages above -- "
            "on Windows, install it from the UB-Mannheim Tesseract build)"
        )
    tess_lang = _TESSERACT_LANG.get(language, "eng")
    try:
        langs = pytesseract.get_languages()
    except Exception:
        langs = None
    if langs is not None and tess_lang not in langs:
        return False, (
            f"Tesseract is installed, but its '{tess_lang}' language data file is "
            f"missing (install the tesseract-ocr-{tess_lang} language pack/traineddata)"
        )
    return True, None


def extract_text_via_ocr(file_obj, language):
    """
    The real fix for legacy-font PDFs (see module docstring): renders
    each page as an actual image at 300 DPI (high enough for reliable
    character recognition, confirmed directly) and reads the visual
    glyphs with Tesseract, completely bypassing the PDF's own
    unreliable internal text layer. Requires OCR_AVAILABLE (PyMuPDF +
    pytesseract installed) and the matching Tesseract language model
    installed at the OS level -- returns None if either isn't
    available, so callers can fall back to a clear error message
    rather than crash. Use diagnose_ocr_setup() first to tell a caller
    WHY, rather than just that it returned None.
    """
    if not OCR_AVAILABLE:
        return None
    tess_lang = _TESSERACT_LANG.get(language, "eng")
    try:
        file_obj.seek(0)
        pdf_bytes = file_obj.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_texts = []
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang=tess_lang).strip()
            if text:
                page_texts.append(text)
        doc.close()
        return "\n\n".join(page_texts)
    except Exception:
        return None


def split_into_chapters(raw_text):
    """
    A teacher can mark separate chapters within one uploaded PDF's
    extracted text by inserting a line containing exactly
    "---CHAPTER---" between them (shown as an instruction in the
    upload UI) -- this lets one PDF covering several chapters become
    several separate, independently-gated lessons in one upload,
    matching "even that chapters have to be read... those lessons
    also should work like this" rather than forcing one lesson per
    upload. If no marker is present, the whole text is one chapter.
    """
    parts = [p.strip() for p in raw_text.split(CHAPTER_MARKER)]
    return [p for p in parts if p]


def parse_stanzas(chapter_text):
    """
    Splits one chapter's text into stanzas (paragraphs, separated by
    one or more blank lines) and each stanza into lines (separated by
    single newlines), matching exactly the nested-list shape every
    built-in lesson in modules/content.py already uses -- so a parsed
    upload is structurally indistinguishable from hand-authored
    content by the time it reaches the practice flow. Blank/whitespace-
    only lines are dropped; a stanza that ends up with zero real lines
    (e.g. from stray blank lines) is dropped entirely rather than
    saved as an empty, unpracticeable stanza.
    """
    raw_stanzas = re.split(r"\n\s*\n", chapter_text.strip())
    stanzas = []
    for raw_stanza in raw_stanzas:
        lines = [line.strip() for line in raw_stanza.split("\n")]
        lines = [line for line in lines if line]
        if lines:
            stanzas.append(lines)
    return stanzas


def make_lesson_id(title, existing_id_checker):
    """
    Builds a URL/key-safe lesson id from the teacher's title (e.g.
    "The Forest Story" -> "up-the-forest-story"), and appends a
    numeric suffix if that id is already taken -- existing_id_checker
    is a function(id) -> bool, checked against BOTH built-in lesson
    ids and already-uploaded ones (see app.py's call site), so a
    teacher naming two lessons identically doesn't silently overwrite
    or collide with anything.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if not slug:
        slug = "lesson"
    base_id = f"up-{slug}"
    candidate = base_id
    n = 2
    while existing_id_checker(candidate):
        candidate = f"{base_id}-{n}"
        n += 1
    return candidate