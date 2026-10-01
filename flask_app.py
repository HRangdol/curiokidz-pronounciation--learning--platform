"""
Flask backend + hands-free, voice-navigated frontend for CurioKidz
Reading Buddy -- built to replace Streamlit specifically because
Streamlit's `st.audio_input` cannot listen continuously: every browser
requires an explicit click/tap before a page may use the microphone at
all (a browser security rule, not a Streamlit limitation), but nothing
in Streamlit lets a page keep listening afterwards without the user
clicking again for the next turn. A real HTML/JS frontend can: grant
the microphone once, then use the browser's own Web Speech API
(SpeechRecognition) to listen continuously for spoken commands
("start", "next", "repeat", "stop") with no further taps, while actual
pronunciation attempts are still recorded and checked by THIS
PROJECT'S OWN existing, already-tuned faster-whisper pipeline
(modules/asr.py) -- Web Speech API is used only for short navigation
commands, never for scoring the child's reading, since that accuracy
work (VAD, beam-size retry, acoustic fallback) is what actually makes
scoring reliable and a generic browser transcript would throw it away.

THIS FILE REUSES EVERY EXISTING MODULE AS-IS (db, content, progress,
audio, asr, voice_clone) -- none of that logic is duplicated or
rewritten, only exposed over HTTP instead of rendered by Streamlit.

STATUS, STATED HONESTLY: this is the working foundation -- lesson
listing, word-by-word hands-free practice (English + Kannada), and
spoken feedback all work end-to-end against the real modules. Whole-
stanza/poem practice, Custom Practice (including the voice-cloning
enrollment UI), and the teacher-upload/dashboard screens from the
Streamlit app are NOT YET ported here -- see the bottom of this file
for exactly what's left.
"""
import os
import tempfile
import mimetypes

from flask import Flask, request, jsonify, send_file, render_template, send_from_directory

from modules import db, content, progress, audio, asr, voice_clone, pdf_ingest
from modules import commands  # framework-agnostic voice-command grammar (no Streamlit dependency)

db.init_db()

app = Flask(__name__)


def _save_uploaded_audio(file_storage):
    """
    Saves a browser-recorded clip (MediaRecorder typically produces
    webm/opus or ogg, NOT wav) to a temp file with the RIGHT extension
    for its real content -- guessed from the reported mimetype, falling
    back to the filename's own extension, then plain ".webm" as a last
    resort. This matters: several format-sniffing libraries downstream
    (ffmpeg via faster-whisper, librosa/soundfile) go by file extension,
    and silently forcing everything to ".wav" regardless of its actual
    encoding is exactly the kind of bug that fails quietly instead of
    loudly. Returns the temp file path; caller is responsible for
    deleting it.
    """
    ext = mimetypes.guess_extension(file_storage.mimetype or "") or ""
    if not ext:
        _, ext = os.path.splitext(file_storage.filename or "")
    if not ext:
        ext = ".webm"
    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    file_storage.save(path)
    return path


# ============================================================================
# EXTENSION: serve the real lesson cover images (assets/images/*.jpeg,
# referenced by each lesson's "image" field in modules/content.py but
# never actually exposed over HTTP before -- Flask's built-in static
# serving only covers the static/ folder, not assets/). Lets the home
# page show the child the actual book-cover artwork for each lesson
# instead of a plain emoji icon, which is far more inviting to a kid
# deciding what to read next.
# ============================================================================
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


@app.route("/assets/<path:filename>")
def serve_asset(filename):
    return send_from_directory(_ASSETS_DIR, filename)


# ============================================================================
# Pages
# ============================================================================
@app.route("/")
def home():
    return render_template("index.html", languages=content.available_languages())


@app.route("/practice/<lesson_id>")
def practice_page(lesson_id):
    lesson = content.get_lesson(lesson_id)
    if lesson is None:
        return "Lesson not found", 404
    return render_template("practice.html", lesson=lesson)


@app.route("/custom")
def custom_page():
    return render_template("custom.html", languages=content.available_languages())


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@app.route("/teacher")
def teacher_page():
    return render_template("teacher.html", languages=content.available_languages())


# ============================================================================
# Lesson data
# ============================================================================
@app.route("/api/lessons")
def api_lessons():
    """
    `all=1` includes LOCKED lessons too (with locked:true, no progress
    lookup needed for them) -- used by the winding lesson-path UI on
    the home page, which shows locked-but-not-yet-reachable lessons as
    greyed-out nodes further down the path, the same way Duolingo shows
    the whole unit path with future lessons visible but un-tappable,
    rather than only ever showing what's already unlocked.
    """
    lang = request.args.get("lang")
    child_name = request.args.get("child", "default")
    include_locked = request.args.get("all") == "1"
    child_id = db.get_or_create_child(child_name)
    lang_lessons = content.lessons_for_language(lang) if lang else []
    out = []
    for lesson in lang_lessons:
        unlocked = db.is_lesson_unlocked(child_id, lesson["id"], lang_lessons)
        if not unlocked and not include_locked:
            continue
        prog = db.get_lesson_progress(child_id, lesson["id"]) if unlocked else {"lesson_completed": False}
        out.append({
            "id": lesson["id"], "title": lesson["title"], "subtitle": lesson["subtitle"],
            "language": lesson["language"], "completed": bool(prog["lesson_completed"]),
            "locked": not unlocked,
            # EXTENSION: the lesson's real cover image (served via the new
            # /assets/<path> route above), when it has one -- lets the home
            # page show actual book-cover artwork instead of a plain emoji,
            # which is far more inviting for a child choosing what to read.
            # A teacher-uploaded lesson has no "image" field, so this is
            # None for those and the frontend falls back to the emoji icon.
            "image": lesson.get("image"),
            # EXTENSION: where this child left off in this lesson, so the
            # Practice tab can RESUME here instead of always restarting
            # from stanza 0 / word 0 -- this data was already being stored
            # (lesson_progress table) but nothing ever read or wrote it
            # during actual hands-free practice until now.
            "resume_stanza": prog.get("stanza_index", 0) if unlocked else 0,
            "resume_word": prog.get("word_index", 0) if unlocked else 0,
            "resume_phase": prog.get("phase", "word") if unlocked else "word",
        })
    return jsonify(out)


# ============================================================================
# EXTENSION: resume support -- read/save exactly where a child left off
# in a lesson (stanza, word, and phase), so stopping mid-practice and
# coming back later continues from there instead of restarting at
# stanza 0 / word 0 every time. The storage (db.get_lesson_progress /
# db.set_progress_position) already existed for this; it just wasn't
# being called by the hands-free practice flow until now.
# ============================================================================
@app.route("/api/lesson/<lesson_id>/resume")
def api_lesson_resume(lesson_id):
    child_name = request.args.get("child", "default")
    child_id = db.get_or_create_child(child_name)
    prog = db.get_lesson_progress(child_id, lesson_id)
    return jsonify({
        "stanza_index": prog.get("stanza_index", 0),
        "word_index": prog.get("word_index", 0),
        "phase": prog.get("phase", "word"),
        "lesson_completed": bool(prog.get("lesson_completed")),
    })


@app.route("/api/lesson/<lesson_id>/save-position", methods=["POST"])
def api_lesson_save_position(lesson_id):
    child_name = request.form.get("child", "default")
    stanza_index = int(request.form.get("stanza", 0))
    word_index = int(request.form.get("word", 0))
    phase = request.form.get("phase", "word")
    child_id = db.get_or_create_child(child_name)
    db.set_progress_position(child_id, lesson_id, stanza_index, word_index, phase)
    return jsonify({"ok": True})


@app.route("/api/lesson/<lesson_id>/words")
def api_lesson_words(lesson_id):
    lesson = content.get_lesson(lesson_id)
    if lesson is None:
        return jsonify({"error": "not found"}), 404
    stanza_idx = int(request.args.get("stanza", 0))
    if stanza_idx >= len(lesson["stanzas"]):
        return jsonify({"done": True})
    words = progress.stanza_words(lesson["stanzas"][stanza_idx])
    return jsonify({"language": lesson["language"], "words": words, "stanza_count": len(lesson["stanzas"])})


# ============================================================================
# EXTENSION: the FULL lesson/poem, every stanza's words at once, so the
# Practice tab can show the whole thing on screen (not just the one
# stanza currently being practiced) while still highlighting progress
# word-by-word and stanza-by-stanza as the child works through it.
# Reuses the exact same progress.stanza_words() helper the per-stanza
# route above already uses -- just called once per stanza instead of
# once per request.
# ============================================================================
@app.route("/api/lesson/<lesson_id>/all-words")
def api_lesson_all_words(lesson_id):
    lesson = content.get_lesson(lesson_id)
    if lesson is None:
        return jsonify({"error": "not found"}), 404
    stanzas = [progress.stanza_words(s) for s in lesson["stanzas"]]
    return jsonify({
        "language": lesson["language"],
        "stanzas": stanzas,
        "stanza_count": len(stanzas),
    })


@app.route("/api/lesson/<lesson_id>/unit-text")
def api_lesson_unit_text(lesson_id):
    """
    Full text for a whole-stanza or whole-poem hands-free check, mirroring
    render_whole_unit_check's two call sites in the Streamlit app exactly:
    `stanza=<n>` -> that one stanza's full text; `stanza=poem` -> the
    entire lesson's text (every stanza, in order).
    """
    lesson = content.get_lesson(lesson_id)
    if lesson is None:
        return jsonify({"error": "not found"}), 404
    raw = request.args.get("stanza", "0")
    if raw == "poem":
        text = progress.whole_poem_text(lesson)
        heading = "Read the WHOLE poem, start to finish!"
    else:
        stanza_idx = int(raw)
        if stanza_idx >= len(lesson["stanzas"]):
            return jsonify({"error": "stanza out of range"}), 400
        text = progress.full_text(lesson["stanzas"][stanza_idx])
        heading = "Now read the WHOLE stanza together!"
    return jsonify({"text": text, "heading": heading, "language": lesson["language"]})


# ============================================================================
# EXTENSION: spoken + written feedback in the LESSON'S OWN language, not
# always English. The bug this fixes: every "Correct!"/"Try again"/tier
# message below was a hardcoded English string, no matter what language
# the child was practicing. For an English lesson that's merely
# incomplete; for a KANNADA lesson it was actively broken, because that
# English string was then handed to generate_reference_audio(text, lang)
# with lang="kn" -- i.e. the Kannada voice was being asked to read
# English words aloud, which is exactly the "AI should speak/tell
# correct or wrong in Kannada" bug the user reported. Every feedback
# string the child hears or reads now comes from this table instead of
# being typed inline at each call site, keyed by the LESSON's language
# (lesson["language"] / the practice's own `lang`), so English lessons
# keep sounding exactly as before and Kannada lessons now get real
# Kannada feedback that the Kannada TTS voice can actually read aloud
# correctly. Add more languages here later the same way, without
# touching any of the call sites below.
# ============================================================================
_FEEDBACK_STRINGS = {
    "en": {
        "tier_95": "Perfect! You're a reading superstar!",
        "tier_80": "Excellent reading!",
        "tier_70": "Great job!",
        "tier_50": "Good try — you're getting there!",
        "tier_25": "Nice effort! Keep practicing!",
        "tier_0": "Let's try that again together!",
        "wonderful": "Wonderful reading!",
        "almost": "Almost! Try once more.",
        "words_right": "You got {matched} of {total} words right ({score:.0f}%).",
        "personal_best": " New personal best!",
        "correct": "Correct! Well done!",
        "try_again": "Try again.",
    },
    "kn": {
        "tier_95": "ಅದ್ಭುತ! ನೀನು ಓದುವಿಕೆಯ ಸೂಪರ್‌ಸ್ಟಾರ್!",
        "tier_80": "ಅತ್ಯುತ್ತಮ ಓದುಗ!",
        "tier_70": "ಚೆನ್ನಾಗಿ ಮಾಡಿದೆ!",
        "tier_50": "ಒಳ್ಳೆಯ ಪ್ರಯತ್ನ — ಇನ್ನೂ ಸ್ವಲ್ಪ ಅಭ್ಯಾಸ ಮಾಡು!",
        "tier_25": "ಚೆನ್ನಾಗಿದೆ! ಅಭ್ಯಾಸ ಮುಂದುವರಿಸು!",
        "tier_0": "ಮತ್ತೊಮ್ಮೆ ಜೊತೆಯಾಗಿ ಪ್ರಯತ್ನಿಸೋಣ!",
        "wonderful": "ಅದ್ಭುತವಾಗಿ ಓದಿದೆ!",
        "almost": "ಬಹುತೇಕ ಸರಿ! ಮತ್ತೊಮ್ಮೆ ಪ್ರಯತ್ನಿಸು.",
        "words_right": "{total} ಪದಗಳಲ್ಲಿ {matched} ಪದಗಳನ್ನು ಸರಿಯಾಗಿ ಹೇಳಿದೆ ({score:.0f}%).",
        "personal_best": " ಇದು ನಿನ್ನ ಹೊಸ ಅತ್ಯುತ್ತಮ ಸಾಧನೆ!",
        "correct": "ಸರಿಯಾಗಿದೆ! ಚೆನ್ನಾಗಿ ಮಾಡಿದೆ!",
        "try_again": "ಮತ್ತೊಮ್ಮೆ ಪ್ರಯತ್ನಿಸು.",
    },
}


def _fb(language, key, **kwargs):
    """Look up one feedback string in the lesson's own language, falling
    back to English for any language/key this table doesn't have yet
    (so an unrecognized language code never crashes or returns blank --
    it just reads in English, same as before this fix existed)."""
    strings = _FEEDBACK_STRINGS.get(language, _FEEDBACK_STRINGS["en"])
    template = strings.get(key) or _FEEDBACK_STRINGS["en"].get(key, "")
    return template.format(**kwargs) if kwargs else template


# Kept as a thin wrapper (same name/signature as before, language-aware
# now) so nothing else calling _motivation_tier(score) needs to change
# beyond passing the language it already has in scope.
def _motivation_tier(score, language="en"):
    """One short, warm phrase for a 0-100 score -- Duolingo-style
    encouragement that scales with how well the attempt actually went,
    not just pass/fail -- in the lesson's own language."""
    if score >= 95:
        return _fb(language, "tier_95")
    if score >= 80:
        return _fb(language, "tier_80")
    if score >= 70:
        return _fb(language, "tier_70")
    if score >= 50:
        return _fb(language, "tier_50")
    if score >= 25:
        return _fb(language, "tier_25")
    return _fb(language, "tier_0")


# ============================================================================
# Whole-stanza / whole-poem check -- mirrors render_whole_unit_check +
# advance_stanza/finish_lesson from the Streamlit app, with one real
# extension: matched/total word counts and a 0-100 score were already
# being computed here but never shown to the child or rewarded -- an
# attempt that got 8 of 10 words right used to earn EXACTLY the same
# (zero) XP as one that got 0 of 10, as long as neither cleared the 0.7
# pass threshold needed to advance. Now every attempt with a real score
# earns XP proportional to that score (passing still earns the full,
# larger amount and is still required to move on to the next stanza/
# poem -- this doesn't make the lesson easier to finish, it just gives
# honest credit for partial effort along the way), the feedback names
# the actual word count and percentage, and a child beating their own
# previous-best score on this exact stanza/poem is told so explicitly.
# ============================================================================
@app.route("/api/practice/unit/check", methods=["POST"])
def api_check_unit():
    lesson_id = request.form.get("lesson_id")
    phase = request.form.get("phase")  # "stanza" or "poem"
    stanza_idx = int(request.form.get("stanza", 0))
    text = request.form.get("text", "")
    child_name = request.form.get("child", "default")
    lesson = content.get_lesson(lesson_id)
    if lesson is None or phase not in ("stanza", "poem") or "audio" not in request.files:
        return jsonify({"error": "bad request"}), 400
    child_id = db.get_or_create_child(child_name)
    language = lesson["language"]

    previous_best = db.get_best_score(child_id, lesson_id, stanza_idx, phase)

    attempt_path = _save_uploaded_audio(request.files["audio"])
    ref_path = audio.generate_reference_audio(text, language, slow=False)
    try:
        recognized = asr.transcribe(attempt_path, language, expected_text=text)
        if recognized:
            ratio, matched, total = asr.line_match_ratio(text, recognized)
            score = round(ratio * 100, 1)
            passed = ratio >= 0.7
        else:
            score, passed = audio.score_pronunciation(ref_path, attempt_path, pass_threshold=55.0) \
                if ref_path else (None, False)
            matched, total = None, None
    finally:
        try:
            os.remove(attempt_path)
        except OSError:
            pass

    xp_awarded = 0
    is_personal_best = False
    if score is not None:
        db.log_attempt(child_id, lesson_id, stanza_idx, phase, text[:80], score, bool(passed))
        if passed:
            xp_awarded = db.award_xp(child_id, phase)
        elif score > 0:
            # Partial credit: proportional to the score, out of the
            # same full amount a pass would earn -- real effort that
            # fell short of the pass line is still worth something,
            # never zero just because it wasn't perfect.
            xp_awarded = db.award_xp_amount(child_id, phase, db.XP_VALUES.get(phase, 0) * (score / 100))
        is_personal_best = previous_best is None or score > previous_best

    if passed and phase == "poem":
        db.mark_lesson_completed(child_id, lesson_id)

    tier_msg = _motivation_tier(score, language) if score is not None \
        else (_fb(language, "wonderful") if passed else _fb(language, "almost"))
    if matched is not None:
        feedback_text = f"{tier_msg} " + _fb(language, "words_right", matched=matched, total=total, score=score)
    else:
        feedback_text = tier_msg
    if is_personal_best and score is not None and previous_best is not None:
        feedback_text += " 🏆" + _fb(language, "personal_best")

    return jsonify({
        "passed": bool(passed), "score": score, "matched": matched, "total": total,
        "feedback_text": feedback_text, "lesson_completed": bool(passed and phase == "poem"),
        "xp_awarded": xp_awarded, "total_xp": db.get_total_xp(child_id),
        "is_personal_best": bool(is_personal_best),
    })


# ============================================================================
# Audio: reference (TTS) clips and spoken command/feedback narration --
# both reuse modules/audio.py's real, already-tested pipeline directly.
# ============================================================================
@app.route("/api/audio/say")
def api_audio_say():
    text = request.args.get("text", "")
    lang = request.args.get("lang", "en")
    slow = request.args.get("slow", "1") == "1"
    if not text:
        return jsonify({"error": "text required"}), 400
    path = audio.generate_reference_audio(text, lang, slow=slow)
    if not path:
        return jsonify({"error": "tts unavailable"}), 503
    return send_file(path, mimetype="audio/wav")


# ============================================================================
# Pronunciation check -- child's recorded attempt is uploaded here and
# checked with the SAME asr.transcribe()/word_matches()/acoustic-
# fallback logic the Streamlit app used, completely unmodified.
# ============================================================================
@app.route("/api/practice/word/check", methods=["POST"])
def api_check_word():
    lesson_id = request.form.get("lesson_id")
    word = request.form.get("word", "")
    child_name = request.form.get("child", "default")
    stanza_idx = int(request.form.get("stanza", 0))
    lesson = content.get_lesson(lesson_id)
    if lesson is None or "audio" not in request.files:
        return jsonify({"error": "bad request"}), 400
    child_id = db.get_or_create_child(child_name)
    language = lesson["language"]
    target_clean = word.strip(".,!?")

    attempt_path = _save_uploaded_audio(request.files["audio"])

    ref_path = audio.generate_reference_audio(target_clean, language, slow=True)
    try:
        recognized = asr.transcribe(attempt_path, language, expected_text=target_clean)
        if recognized:
            passed = asr.word_matches(target_clean, recognized)
            score = 100.0 if passed else 0.0
        else:
            score, passed = audio.score_pronunciation(ref_path, attempt_path, pass_threshold=55.0) \
                if ref_path else (None, False)
            recognized = None
    finally:
        try:
            os.remove(attempt_path)
        except OSError:
            pass

    xp_awarded = 0
    if score is not None:
        db.log_attempt(child_id, lesson_id, stanza_idx, "word", word, score, bool(passed))
        if passed:
            xp_awarded = db.award_xp(child_id, "word")

    return jsonify({
        "passed": bool(passed), "score": score, "recognized": recognized,
        "feedback_text": _fb(language, "correct") if passed else _fb(language, "try_again"),
        "xp_awarded": xp_awarded, "total_xp": db.get_total_xp(child_id),
    })


# ============================================================================
# Voice-command recognition -- a short recorded clip believed to be a
# NAVIGATION word ("start"/"next"/"repeat"/"stop"), NOT a pronunciation
# attempt. In the browser this path is normally short-circuited by the
# Web Speech API directly (see static/js/handsfree.js) since that's
# faster and needs no upload; this endpoint exists as the same real
# check server-side, for any browser without SpeechRecognition support.
# ============================================================================
@app.route("/api/practice/command", methods=["POST"])
def api_check_command():
    lang = request.form.get("lang", "en")
    if "audio" not in request.files:
        return jsonify({"error": "bad request"}), 400
    attempt_path = _save_uploaded_audio(request.files["audio"])
    try:
        recognized = asr.transcribe(attempt_path, lang)
    finally:
        try:
            os.remove(attempt_path)
        except OSError:
            pass
    command = commands.match_command(recognized or "", lang)
    return jsonify({"command": command, "recognized": recognized})


# ============================================================================
# Custom Practice -- voice profiles, cloning enrollment, sentence
# recording/playback, and the hands-free child read-back check. Every
# call here is a thin HTTP wrapper around modules/voice_clone.py,
# completely unchanged from what the Streamlit app already used.
# ============================================================================
@app.route("/api/voice/profiles")
def api_voice_profiles():
    return jsonify(voice_clone.list_profiles())


@app.route("/api/voice/profiles", methods=["POST"])
def api_voice_create_profile():
    name = (request.form.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    profile_id = voice_clone.get_or_create_profile(name, created_by=request.form.get("created_by"))
    return jsonify({"id": profile_id, "name": name})


@app.route("/api/voice/language-info")
def api_voice_language_info():
    lang = request.args.get("lang", "en")
    return jsonify({
        "can_clone": voice_clone.any_cloning_available_for(lang),
        "xtts_available": voice_clone.xtts_available_for(lang),
        "openvoice_available": voice_clone.openvoice_available(),
    })


@app.route("/api/voice/has-reference")
def api_voice_has_reference():
    """
    Whether cloning has ANY real audio of this profile's voice to work
    from yet (an enrollment clip, or any past word/sentence recording)
    -- used by Custom Practice's UI to decide whether "skip recording,
    use the cloned voice instead" is actually possible right now, or
    would just fail with no reference to clone from.
    """
    profile_id = int(request.args.get("profile_id"))
    lang = request.args.get("lang", "en")
    return jsonify({"available": voice_clone._any_reference_clip(profile_id, lang) is not None})


@app.route("/api/voice/enrollment")
def api_voice_enrollment_status():
    profile_id = int(request.args.get("profile_id"))
    lang = request.args.get("lang", "en")
    recorded, total = voice_clone.enrollment_progress(profile_id, lang)
    texts = voice_clone.enrollment_sample_texts(lang)
    recorded_idxs = sorted(voice_clone.recorded_enrollment_indices(profile_id, lang))
    next_idx = next((i for i in range(total) if i not in recorded_idxs), None)
    return jsonify({
        "recorded": recorded, "total": total, "recorded_indices": recorded_idxs,
        "next_idx": next_idx, "next_text": texts[next_idx] if next_idx is not None else None,
    })


@app.route("/api/voice/enrollment", methods=["POST"])
def api_voice_enrollment_save():
    profile_id = int(request.form.get("profile_id"))
    lang = request.form.get("lang", "en")
    sample_idx = int(request.form.get("sample_idx"))
    if "audio" not in request.files:
        return jsonify({"error": "bad request"}), 400
    voice_clone.save_enrollment_clip(profile_id, lang, sample_idx, request.files["audio"].read())
    return jsonify({"ok": True})


@app.route("/api/voice/enrollment/reset", methods=["POST"])
def api_voice_enrollment_reset():
    profile_id = int(request.form.get("profile_id"))
    lang = request.form.get("lang", "en")
    voice_clone.reset_enrollment(profile_id, lang)
    return jsonify({"ok": True})


@app.route("/api/voice/samples")
def api_voice_samples():
    lang = request.args.get("lang", "en")
    return jsonify(voice_clone.sample_sentences(lang))


@app.route("/api/voice/custom-text", methods=["POST"])
def api_voice_custom_text():
    lang = request.form.get("lang", "en")
    text = (request.form.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text required"}), 400
    voice_clone.save_custom_text(lang, text, created_by=request.form.get("created_by"))
    return jsonify({"ok": True})


@app.route("/api/voice/sentence-recording", methods=["POST"])
def api_voice_sentence_recording_save():
    profile_id = int(request.form.get("profile_id"))
    lang = request.form.get("lang", "en")
    text = request.form.get("text", "")
    if "audio" not in request.files:
        return jsonify({"error": "bad request"}), 400
    voice_clone.save_sentence_recording(profile_id, lang, text, request.files["audio"].read())
    return jsonify({"ok": True})


@app.route("/api/voice/sentence-recording/status")
def api_voice_sentence_recording_status():
    profile_id = int(request.args.get("profile_id"))
    lang = request.args.get("lang", "en")
    text = request.args.get("text", "")
    exists = voice_clone.get_sentence_recording(profile_id, lang, text) is not None
    return jsonify({"recorded": exists})


@app.route("/api/voice/sentence-recording/delete", methods=["POST"])
def api_voice_sentence_recording_delete():
    profile_id = int(request.form.get("profile_id"))
    lang = request.form.get("lang", "en")
    text = request.form.get("text", "")
    voice_clone.delete_sentence_recording(profile_id, lang, text)
    return jsonify({"ok": True})


def _custom_reference_audio_path(profile_id, lang, text):
    """Mirrors app.py's _custom_reference_audio: the real recorded
    sentence clip first (most faithful, once one exists), falling back
    to cloned synthesis only when no real recording is on file yet."""
    real_path = voice_clone.get_sentence_recording(profile_id, lang, text)
    if real_path:
        return real_path
    if voice_clone.any_cloning_available_for(lang):
        out_path = os.path.join(tempfile.gettempdir(), f"clone_{profile_id}_{voice_clone._text_hash(text)}.wav")
        return voice_clone.clone_synthesize(profile_id, text, lang, out_path)
    return None


@app.route("/api/voice/sentence-audio")
def api_voice_sentence_audio():
    profile_id = int(request.args.get("profile_id"))
    lang = request.args.get("lang", "en")
    text = request.args.get("text", "")
    mode = request.args.get("mode", "slow")  # "slow" or "normal"
    if mode == "slow":
        path = voice_clone.get_slow_sentence_recording(profile_id, lang, text)
        if path is None:
            path = _custom_reference_audio_path(profile_id, lang, text)
    else:
        path = _custom_reference_audio_path(profile_id, lang, text)
    if not path:
        print(f"[flask_app] /api/voice/sentence-audio: no audio could be produced for "
              f"profile_id={profile_id} lang={lang!r} text={text!r} mode={mode!r} -- see the "
              f"[voice_clone] log lines just above this for the actual reason.")
        return jsonify({"error": "no audio available"}), 503
    return send_file(path, mimetype="audio/wav")


@app.route("/api/voice/sentence-check", methods=["POST"])
def api_voice_sentence_check():
    """Mirrors _render_custom_sentence_check's scoring exactly."""
    profile_id = int(request.form.get("profile_id"))
    lang = request.form.get("lang", "en")
    text = request.form.get("text", "")
    child_name = request.form.get("child", "default")
    if "audio" not in request.files:
        return jsonify({"error": "bad request"}), 400
    child_id = db.get_or_create_child(child_name)

    attempt_path = _save_uploaded_audio(request.files["audio"])
    # EXTENSION FIX: real bug -- this used to compute the (possibly
    # cloned-voice) reference clip UNCONDITIONALLY before scoring even
    # started, even though it's only actually used as a fallback when
    # ASR fails. When the parent skipped recording and relies on
    # OpenVoice cloning, that meant every single attempt synchronously
    # triggered a fresh clone (and, the first time, a ~200MB checkpoint
    # download) before "Checking..." could ever resolve -- exactly the
    # stuck-forever symptom this was causing. Now it's computed lazily,
    # only in the branch that actually needs it.
    try:
        recognized = asr.transcribe(attempt_path, lang, expected_text=text)
        if recognized:
            ratio, matched, total = asr.line_match_ratio(text, recognized)
            score = round(ratio * 100, 1)
            passed = ratio >= 0.7
        else:
            ref_path = _custom_reference_audio_path(profile_id, lang, text)
            score, passed = audio.score_pronunciation(ref_path, attempt_path, pass_threshold=55.0) \
                if ref_path else (None, False)
    finally:
        try:
            os.remove(attempt_path)
        except OSError:
            pass

    xp_awarded = 0
    if score is not None:
        db.log_attempt(child_id, f"custom-{profile_id}", 0, "sentence", text[:80], score, bool(passed))
        if passed:
            xp_awarded = db.award_xp(child_id, "sentence")

    return jsonify({
        "passed": bool(passed), "score": score,
        "feedback_text": _fb(lang, "correct") if passed else _fb(lang, "try_again"),
        "xp_awarded": xp_awarded, "total_xp": db.get_total_xp(child_id),
    })


# ============================================================================
# Parent/teacher dashboard -- read-only, so this stays plain click-driven
# HTML (the "no buttons" requirement is specifically about the CHILD'S
# own practice loop, not an adult reviewing progress). Reuses every
# db.py stat function unchanged.
# ============================================================================
@app.route("/api/dashboard")
def api_dashboard():
    child_name = request.args.get("child", "default")
    child_id = db.get_or_create_child(child_name)
    lang_map = _lesson_language_map()
    language_stats = {
        code: db.get_language_stats(child_id, lessons)
        for code, lessons in _lessons_by_language().items()
    }
    return jsonify({
        "streak": db.get_streak(child_id),
        "xp": db.get_total_xp(child_id),
        "stats": db.get_dashboard_stats(child_id),
        "language_stats": language_stats,
        "tricky_words": db.get_tricky_words(child_id, limit=8),
        "score_series": db.get_language_score_series(child_id, lang_map, limit=20),
    })


@app.route("/api/xp")
def api_xp():
    child_id = db.get_or_create_child(request.args.get("child", "default"))
    return jsonify({"total_xp": db.get_total_xp(child_id)})


# ============================================================================
# EXTENSION: teacher-facing "All Students" view -- every child who has
# ever practiced, with an at-a-glance summary (streak, XP, lessons
# completed, average score), so a teacher/parent doesn't have to know
# and type each child's name individually to check on them. Reuses the
# exact same per-child stat functions the single-child dashboard
# already uses (db.get_streak, db.get_total_xp, db.get_dashboard_stats)
# -- nothing new computed, just aggregated across every child at once.
# ============================================================================
@app.route("/api/teacher/students")
def api_teacher_students():
    out = []
    for child in db.list_children():
        stats = db.get_dashboard_stats(child["id"])
        out.append({
            "name": child["name"],
            "streak": db.get_streak(child["id"]),
            "xp": db.get_total_xp(child["id"]),
            "lessons_completed": stats["lessons_completed"],
            "avg_score": stats["avg_score"],
        })
    out.sort(key=lambda c: c["name"].lower())
    return jsonify({"students": out})


@app.route("/students")
def students_page():
    return render_template("students.html")


def _lesson_language_map():
    out = {}
    for code, _ in content.available_languages():
        for lesson in content.lessons_for_language(code):
            out[lesson["id"]] = lesson["language"]
    return out


def _lessons_by_language():
    out = {}
    for code, _ in content.available_languages():
        out[code] = content.lessons_for_language(code)
    return out


# ============================================================================
# Teacher PDF upload -- extract -> (parent/teacher edits raw text
# client-side) -> split into chapters -> save each as its own lesson.
# Reuses modules/pdf_ingest.py exactly as the Streamlit app did.
# ============================================================================
@app.route("/api/teacher/extract", methods=["POST"])
def api_teacher_extract():
    # EXTENSION: this whole route is now wrapped in one try/except so it
    # can NEVER return Flask's default HTML error page. Real bug found
    # directly: when a file replacement is partial (e.g. flask_app.py
    # updated but modules/pdf_ingest.py left on an older version without
    # diagnose_ocr_setup()), the resulting AttributeError produced
    # Flask's default HTML 500 page; the frontend's `await res.json()`
    # then threw on that HTML (not valid JSON) and -- since that fetch
    # had no try/catch of its own -- failed completely silently, with
    # nothing shown on the page at all and only an error in the
    # browser's own console. Every path below now returns JSON, and
    # teacher.js's extract() now also catches and displays this kind of
    # failure instead of going silent.
    try:
        lang = request.form.get("lang", "en")
        if "pdf" not in request.files:
            return jsonify({"error": "pdf required"}), 400
        pdf_file = request.files["pdf"]
        try:
            raw_text = pdf_ingest.extract_text_from_pdf(pdf_file)
        except Exception as e:
            return jsonify({"error": f"Couldn't read that PDF: {e}"}), 400

        is_empty = not raw_text or not raw_text.strip()
        is_garbled = (not is_empty) and pdf_ingest.is_likely_mojibake(raw_text, lang)
        used_ocr = False
        # Real diagnosis of why OCR did or didn't run, instead of
        # silently handing back mojibake with no explanation -- see
        # pdf_ingest.diagnose_ocr_setup()'s docstring for the two
        # genuinely different failure modes this distinguishes (Python
        # packages never installed vs. the separate, OS-level Tesseract
        # program/language pack missing).
        ocr_ready, ocr_reason = pdf_ingest.diagnose_ocr_setup(lang)
        if (is_empty or is_garbled) and ocr_ready:
            pdf_file.seek(0)
            ocr_text = pdf_ingest.extract_text_via_ocr(pdf_file, lang)
            if ocr_text and ocr_text.strip():
                raw_text = ocr_text
                used_ocr = True

        if not raw_text or not raw_text.strip():
            reason = f" ({ocr_reason})" if ocr_reason else ""
            return jsonify({
                "error": "This PDF's text can't be read directly, and this app's OCR fallback "
                         f"isn't available or couldn't read it either{reason}. Try a different PDF, "
                         "or a clearer scan."
            }), 400

        return jsonify({
            "text": raw_text, "used_ocr": used_ocr, "was_garbled": is_garbled,
            # True exactly when the text shown is STILL garbled -- was_garbled
            # but OCR either wasn't ready or didn't fix it -- so the frontend
            # can warn loudly instead of silently accepting unreadable text.
            "still_garbled": bool(is_garbled and not used_ocr),
            "ocr_reason": None if used_ocr or not is_garbled else ocr_reason,
        })
    except Exception as e:
        # Last-resort catch-all: whatever broke, the teacher sees a real
        # JSON error message naming it, never a blank screen.
        import traceback
        print("[api_teacher_extract] unexpected error:", traceback.format_exc())
        return jsonify({"error": f"Unexpected server error in PDF extraction: {e}"}), 500


@app.route("/api/teacher/split", methods=["POST"])
def api_teacher_split():
    text = request.form.get("text", "")
    chapters = pdf_ingest.split_into_chapters(text)
    return jsonify({"chapters": chapters})


@app.route("/api/teacher/save", methods=["POST"])
def api_teacher_save():
    """
    Body (JSON): {lang, lang_label, chapters: [{title, text}], uploaded_by}
    Each chapter becomes its own new lesson, appended after existing
    lessons in that language -- same as add_uploaded_lesson already did.
    """
    data = request.get_json(force=True, silent=True) or {}
    lang = data.get("lang", "en")
    lang_label = data.get("lang_label", "English")
    chapters = data.get("chapters", [])
    uploaded_by = data.get("uploaded_by")
    if not chapters:
        return jsonify({"error": "no chapters to save"}), 400

    saved = []
    for c in chapters:
        title = (c.get("title") or "").strip()
        text = c.get("text", "")
        if not title or not text.strip():
            continue
        stanzas = pdf_ingest.parse_stanzas(text)
        lesson_id = pdf_ingest.make_lesson_id(title, db.lesson_id_exists)
        db.add_uploaded_lesson(
            lesson_id, lang, lang_label, title, subtitle="Uploaded by teacher",
            stanzas=stanzas, uploaded_by=uploaded_by,
        )
        saved.append({"id": lesson_id, "title": title})

    if not saved:
        return jsonify({"error": "Please give a title and some text for at least one chapter."}), 400
    return jsonify({"saved": saved})


# ============================================================================
# EXTENSION: whole-LESSON voice cloning, right from the Teacher Upload
# tab -- a teacher/parent picks one of their saved lessons (any lesson,
# not just ones just uploaded this session) and an existing voice
# profile, and every line of the lesson gets cloned into that voice in
# one go, so a child can then listen to the whole lesson read aloud in
# that familiar voice. Reuses voice_clone.clone_whole_lesson(), which
# calls the same clone_synthesize() function Custom Practice uses, but
# with allow_sarvam=False -- this path uses ONLY the two open-source
# cloning models (XTTS-v2, OpenVoice V2), never the paid/closed-source
# Sarvam API, as requested.
# ============================================================================
@app.route("/api/teacher/lessons")
def api_teacher_lessons():
    """Every lesson for one language, id+title only -- populates the
    "which lesson to clone" dropdown. Deliberately ignores per-child
    lock/progress state (irrelevant for a teacher picking a lesson to
    clone) unlike /api/lessons, which is child-progress-aware."""
    lang = request.args.get("lang", "en")
    lessons = content.lessons_for_language(lang)
    return jsonify([{"id": l["id"], "title": l["title"]} for l in lessons])


@app.route("/api/voice/clone-lesson", methods=["POST"])
def api_voice_clone_lesson():
    """
    `stanza` is optional (form field): when given (an integer index),
    only that ONE stanza is (re-)cloned -- much faster than the whole
    lesson, and leaves any other already-cloned stanzas in the cache
    alone, so a teacher can clone stanza by stanza and each call just
    adds to what's already playable. Omitted (or "all") clones the
    entire lesson, wiping and rebuilding its whole cache as before.
    """
    lesson_id = request.form.get("lesson_id", "")
    profile_id = request.form.get("profile_id")
    stanza_raw = request.form.get("stanza", "").strip()
    lesson = content.get_lesson(lesson_id)
    if lesson is None:
        return jsonify({"error": "lesson not found"}), 404
    if not profile_id:
        return jsonify({"error": "profile_id required"}), 400
    profile_id = int(profile_id)
    if not voice_clone._any_reference_clip(profile_id, lesson["language"]):
        return jsonify({
            "error": "This voice has no recordings yet for this language. "
                     "Record an enrollment or at least one sentence for it in Custom Practice first."
        }), 400
    if stanza_raw and stanza_raw != "all":
        try:
            stanza_idx = int(stanza_raw)
        except ValueError:
            return jsonify({"error": "invalid stanza index"}), 400
        try:
            results = voice_clone.clone_lesson_stanza(lesson_id, lesson, profile_id, lesson["language"], stanza_idx)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    else:
        results = voice_clone.clone_whole_lesson(lesson_id, lesson, profile_id, lesson["language"])
    ok_count = sum(1 for r in results if r["ok"])
    return jsonify({"lines": results, "cloned": ok_count, "total": len(results)})


@app.route("/api/voice/lesson-clone")
def api_voice_lesson_clone():
    """Whatever's already cached for this lesson+voice (possibly
    nothing) -- lets the Teacher Upload page show existing cloned-lesson
    playback immediately on load, without re-cloning every visit."""
    lesson_id = request.args.get("lesson_id", "")
    profile_id = int(request.args.get("profile_id", 0))
    rows = voice_clone.get_lesson_clone(lesson_id, profile_id)
    return jsonify({"lines": [
        {"stanza_idx": r["stanza_idx"], "line_idx": r["line_idx"], "text": r["text"], "ok": bool(r["audio_path"])}
        for r in rows
    ]})


@app.route("/api/voice/lesson-clone-audio")
def api_voice_lesson_clone_audio():
    lesson_id = request.args.get("lesson_id", "")
    profile_id = int(request.args.get("profile_id", 0))
    stanza_idx = int(request.args.get("stanza", 0))
    line_idx = int(request.args.get("line", 0))
    rows = voice_clone.get_lesson_clone(lesson_id, profile_id)
    match = next((r for r in rows if r["stanza_idx"] == stanza_idx and r["line_idx"] == line_idx), None)
    if not match or not match["audio_path"] or not os.path.exists(match["audio_path"]):
        return jsonify({"error": "no cloned audio for this line"}), 404
    return send_file(match["audio_path"], mimetype="audio/wav")


if __name__ == "__main__":
    # use_reloader=False: Flask's debug auto-reloader watches every file
    # that's ever been imported, including deep inside torch's and
    # librosa's site-packages trees. Those libraries touch/recompile their
    # own internal files while running (e.g. torch.jit warnings, weight_norm
    # deprecation shims), and the watchdog mistakes that for "your code
    # changed" and restarts the whole server mid-request -- which is what
    # was silently killing the OpenVoice cloning requests (the browser saw
    # a dropped connection, not a real error). debug=True is kept so you
    # still get Flask's in-browser error pages/traceback on a real crash;
    # only the file-watching restart is disabled. If you do edit this
    # project's own .py files while the server is running, stop it with
    # CTRL+C and re-run `python flask_app.py` to pick up the change.
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)


# ============================================================================
# WHAT'S LEFT, STATED HONESTLY (not built yet):
#  - Whole-stanza/whole-poem hands-free check (mirrors render_whole_unit_check)
#  - Custom Practice: recording your own voice, the slow-playback feature,
#    and the voice-cloning enrollment UI (all real in modules/voice_clone.py
#    already -- only the Streamlit UI for them hasn't been ported to a page
#    here yet)
#  - Teacher PDF upload and the parent/teacher dashboard screens
#  - Duolingo-style visual design (streaks, hearts, mascot, animations) --
#    the current templates are plain, functional HTML, not yet styled to
#    match the reference repo
# Ported first because it's the actual crux of "voice navigation, no
# buttons" -- word-by-word practice is where a child spends most of
# their time, so proving hands-free works there first, end-to-end
# (Web Speech commands + real ASR-scored attempts + spoken feedback),
# is the right thing to get right before copying the same pattern to
# every other screen.
# ============================================================================