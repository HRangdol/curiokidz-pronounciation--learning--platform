"""
Animated, word-highlighted lesson video -- generated automatically and
entirely offline/free, built out of pieces this project already has.

STATED PLAINLY, same honest-disclosure pattern as every other module
in this project (audio.py, asr.py, voice_clone.py all do this too):

WHAT THIS ACTUALLY DOES: for each stanza of a lesson, this generates
(and caches) a normal TTS narration clip using the exact same
modules/audio.py pipeline the rest of the app already uses, works out
each word's real timing in that narration (real ASR word-level
timestamps when available -- see asr.transcribe_word_timestamps --
with the same equal-duration fallback used elsewhere in this project
when ASR isn't available), and renders a karaoke-style caption using
the SAME green/blue/grey "done / current / still to come" colour
language as the on-screen word-highlight pane, over the lesson's own
picture with a gentle zoom. Every stanza's clip is then joined into
one video per lesson.

WHAT THIS IS NOT: no generative video model draws any new imagery, and
nothing here calls a paid third-party video-generation API. Every
visual element is either the lesson's own existing picture (or a
plain solid-colour card when a lesson has none, e.g. uploaded content)
or plain, code-drawn text and shapes -- an animated, narrated,
highlighted READING of the lesson, not an AI reimagining of it.

DEPENDENCIES: moviepy + imageio-ffmpeg. imageio-ffmpeg specifically
because it ships its own portable ffmpeg binary via pip -- no separate
system-level ffmpeg install needed. That matters a lot on Windows,
where this project has already gone out of its way once before to
avoid a manual ffmpeg install (see the React+Flask README's WAV-
recording design note). Text is drawn with Pillow's own bundled
scalable default font (ImageFont.load_default(size=...), Pillow
>=10.1), not a font file this project would have to ship -- so nothing
here depends on a particular font being installed on the machine
running it, either.

GRACEFUL FALLBACK: if moviepy isn't installed, video_available()
returns False and the UI simply doesn't offer the video feature --
same pattern as every other optional dependency in this project.
"""
import os
import hashlib

from PIL import Image, ImageDraw, ImageFont

from modules import audio, asr

try:
    import numpy as np
    import librosa
    from moviepy import ImageClip, VideoClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips
    MOVIEPY_AVAILABLE = True
except Exception:
    MOVIEPY_AVAILABLE = False

VIDEO_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lesson_videos"
)
os.makedirs(VIDEO_ROOT, exist_ok=True)

_SIZE = (960, 540)
_FONT_SIZE = 34
_LINE_HEIGHT = 50
_PADDING = 26
_FPS = 20
# EXTENSION FIX: the original 0.06 (6%) zoom was reported as "no
# movement" -- confirmed real: on a short stanza (a few seconds) a 6%
# zoom is imperceptible, especially with no panning to go with it and
# the zoom resetting back to 1.0 at the start of every new stanza.
# Raised to a clearly-visible zoom AND added a diagonal pan, alternating
# direction/zoom-in-vs-out per stanza (see the `variant` argument below)
# so consecutive stanzas don't all move the same way -- a real, obvious
# Ken Burns effect a child will actually notice, not a static picture
# with narration playing over it.
_ZOOM_AMOUNT = 0.28

# Same colours as the .lword-done / .lword-ai / .lword-todo CSS classes
# in app.py's stylesheet, kept in sync deliberately.
_DONE_BG, _DONE_FG = (183, 242, 204, 235), (27, 138, 74, 255)
_ACTIVE_BG, _ACTIVE_FG = (75, 203, 235, 235), (255, 255, 255, 255)
_TODO_FG = (170, 170, 180, 255)
_BANNER_BG = (30, 22, 55, 200)
_PLACEHOLDER_BG = (108, 92, 231)


def video_available():
    return MOVIEPY_AVAILABLE


def _font(size=_FONT_SIZE):
    return ImageFont.load_default(size=size)


def _content_hash(lesson):
    """Changes if the lesson's own text/image changes, so an edited
    uploaded lesson gets a fresh video instead of serving a stale
    cached one under the same lesson id."""
    raw = lesson["id"] + "|" + (lesson.get("image") or "") + "|" + "|".join(
        line for stanza in lesson["stanzas"] for line in stanza
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def lesson_video_path(lesson):
    return os.path.join(VIDEO_ROOT, f"{lesson['id']}_{_content_hash(lesson)}.mp4")


def existing_video(lesson):
    path = lesson_video_path(lesson)
    return path if os.path.exists(path) else None


# ============================================================================
# Word timing (real ASR alignment on the CLEAN TTS narration, with an
# equal-duration fallback) -- same two-tier approach already used for
# voice-cloning in modules/voice_clone.py's slice_sentence_into_words.
# ============================================================================
def _word_timings(narration_path, language, words):
    y, sr = librosa.load(narration_path, sr=16000)
    total = len(y) / sr
    if total <= 0 or not words:
        return []

    recognized = None
    try:
        recognized = asr.transcribe_word_timestamps(narration_path, language, expected_text=" ".join(words))
    except Exception:
        recognized = None

    if recognized and len(recognized) == len(words):
        return [(w["start"], w["end"]) for w in recognized]

    n = len(words)
    step = total / n
    return [(i * step, (i + 1) * step) for i in range(n)]


# ============================================================================
# Karaoke caption rendering
# ============================================================================
def _layout_caption(words, font, max_width):
    probe = Image.new("RGBA", (10, 10))
    draw = ImageDraw.Draw(probe)
    positions = []
    x, y = 0, 0
    space_w = draw.textlength(" ", font=font)
    for w in words:
        bbox = draw.textbbox((0, 0), w, font=font)
        ww, hh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if x > 0 and x + ww > max_width:
            x = 0
            y += _LINE_HEIGHT
        positions.append({"word": w, "x": x, "y": y, "w": ww, "h": hh})
        x += ww + space_w
    return positions, y + _LINE_HEIGHT


_POP_SECONDS = 0.22  # how long the "just became active" word stays enlarged
_POP_MAX_SCALE = 1.35


def _caption_frame(font, positions, total_height, active_idx, canvas_width, pop_scale=1.0):
    """
    Renders the karaoke caption banner. `pop_scale` briefly enlarges
    just the ACTIVE word's text+highlight right as it becomes current
    (see _pop_scale_at below) -- a short bounce, not a static colour
    swap, so each new word visibly "pops" instead of the caption just
    silently changing state between frames.
    """
    img_h = total_height + _PADDING * 2
    img = Image.new("RGBA", (canvas_width, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, canvas_width, img_h], radius=22, fill=_BANNER_BG)
    line_w = max((p["x"] + p["w"] for p in positions), default=0)
    offset_x = max(0, (canvas_width - _PADDING * 2 - line_w) // 2)

    big_font = _font(int(_FONT_SIZE * _POP_MAX_SCALE)) if pop_scale > 1.01 else None

    for i, p in enumerate(positions):
        if i < active_idx:
            bg, fg = _DONE_BG, _DONE_FG
        elif i == active_idx:
            bg, fg = _ACTIVE_BG, _ACTIVE_FG
        else:
            bg, fg = None, _TODO_FG
        tx = p["x"] + offset_x + _PADDING
        ty = p["y"] + _PADDING

        if i == active_idx and pop_scale > 1.01:
            use_font = _font(int(_FONT_SIZE * pop_scale))
            bbox = draw.textbbox((0, 0), p["word"], font=use_font)
            ww, hh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            # keep the word centred on its normal position while it
            # temporarily grows, rather than growing from its top-left
            # corner (which would look like it's sliding, not popping)
            cx, cy = tx + p["w"] / 2, ty + p["h"] / 2
            ptx, pty = cx - ww / 2, cy - hh / 2
            draw.rounded_rectangle([ptx - 9, pty - 6, ptx + ww + 9, pty + hh + 10], radius=12, fill=bg)
            draw.text((ptx, pty), p["word"], font=use_font, fill=fg)
        else:
            if bg:
                draw.rounded_rectangle([tx - 7, ty - 4, tx + p["w"] + 7, ty + p["h"] + 8], radius=10, fill=bg)
            draw.text((tx, ty), p["word"], font=font, fill=fg)
    return np.array(img)


def _pop_scale_at(t_since_word_start):
    """Eases from _POP_MAX_SCALE down to 1.0 over _POP_SECONDS -- a
    quick bounce-in rather than an abrupt size change."""
    if t_since_word_start >= _POP_SECONDS or t_since_word_start < 0:
        return 1.0
    progress = t_since_word_start / _POP_SECONDS
    eased = 1 - (1 - progress) ** 2  # ease-out
    return _POP_MAX_SCALE - (_POP_MAX_SCALE - 1.0) * eased


# ============================================================================
# EXTENSION: simple floating decorative particles (stars/dots), drifting
# slowly across the frame -- pure code-drawn shapes, no new art assets --
# added specifically because a static picture with only a slow zoom/pan
# still read as "not much is happening" (real feedback). This gives the
# frame constant, gentle motion of its own on top of the Ken Burns
# background, varied per stanza via `variant` so it doesn't repeat
# identically every time.
# ============================================================================
_PARTICLE_COLORS = [
    (255, 255, 255, 140), (249, 189, 77, 150), (75, 203, 235, 150),
    (255, 107, 74, 140), (27, 207, 180, 150),
]


def _particles_clip(duration, variant=0):
    import random
    rng = random.Random(1000 + variant)
    n = 9
    particles = []
    for _ in range(n):
        particles.append({
            "x": rng.uniform(0, _SIZE[0]),
            "y": rng.uniform(0, _SIZE[1] * 0.75),
            "r": rng.uniform(5, 13),
            "speed": rng.uniform(14, 34),
            "drift": rng.uniform(-8, 8),
            "color": rng.choice(_PARTICLE_COLORS),
            "shape": rng.choice(["circle", "star"]),
        })

    def _draw_star(draw, cx, cy, r, color):
        pts = []
        for k in range(10):
            ang = k * 36 * 3.14159265 / 180 - 3.14159265 / 2
            rad = r if k % 2 == 0 else r * 0.45
            pts.append((cx + rad * __import__("math").cos(ang), cy + rad * __import__("math").sin(ang)))
        draw.polygon(pts, fill=color)

    def make_frame(t):
        img = Image.new("RGBA", _SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        for p in particles:
            y = (p["y"] - p["speed"] * t) % (_SIZE[1] + 40) - 20
            x = (p["x"] + p["drift"] * t) % _SIZE[0]
            if p["shape"] == "circle":
                draw.ellipse([x - p["r"], y - p["r"], x + p["r"], y + p["r"]], fill=p["color"])
            else:
                _draw_star(draw, x, y, p["r"], p["color"])
        return np.array(img)

    return VideoClip(make_frame, duration=duration)


_PAN_DIRECTIONS = [(0, 0), (1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1)]


def _background_clip(image_path, duration, variant=0):
    """
    A clearly-visible Ken-Burns zoom + pan over the lesson's own
    picture (or a plain colour card when there is none). `variant`
    (the stanza's index) picks a zoom direction (in for even stanzas,
    out for odd) and a pan direction from _PAN_DIRECTIONS, so
    consecutive stanzas in the same video visibly move differently
    from each other instead of repeating the exact same drift.

    Built with a hand-written make_frame callback that resizes/crops
    each frame directly with Pillow, rather than moviepy's own built-in
    Resize/Crop effects -- confirmed directly, in this project's own
    testing, that moviepy 2.1.2's Resize effect washes out a real
    photo's colours badly (a real, reproducible bug in that specific
    version/pipeline, not a theoretical concern) whereas the same
    resize done with Pillow itself, frame by frame, renders correctly.
    This is slower per frame than the built-in effect would be, but
    lesson videos are generated once and cached (see
    generate_lesson_video), not regenerated on every view, so the
    extra cost is a one-time thing.
    """
    if image_path and os.path.exists(image_path):
        source = Image.open(image_path).convert("RGB")
    else:
        source = Image.new("RGB", _SIZE, _PLACEHOLDER_BG)

    src_w, src_h = source.size
    target_ratio = _SIZE[0] / _SIZE[1]
    src_ratio = src_w / src_h
    safe_duration = max(duration, 0.01)
    zoom_in = (variant % 2 == 0)
    pan_dx, pan_dy = _PAN_DIRECTIONS[variant % len(_PAN_DIRECTIONS)]

    def make_frame(t):
        progress = min(1.0, t / safe_duration)
        move = progress if zoom_in else (1 - progress)
        scale = 1 + _ZOOM_AMOUNT * move
        if src_ratio > target_ratio:
            new_h = max(_SIZE[1], int(_SIZE[1] * scale))
            new_w = max(_SIZE[0], int(new_h * src_ratio))
        else:
            new_w = max(_SIZE[0], int(_SIZE[0] * scale))
            new_h = max(_SIZE[1], int(new_w / src_ratio))
        resized = source.resize((new_w, new_h), Image.LANCZOS)

        max_x = new_w - _SIZE[0]
        max_y = new_h - _SIZE[1]
        # pan from one edge of the available room towards the other as
        # the stanza plays, instead of staying dead-centre the whole time
        x1 = int((max_x / 2) * (1 + pan_dx * move)) if pan_dx else max_x // 2
        y1 = int((max_y / 2) * (1 + pan_dy * move)) if pan_dy else max_y // 2
        x1 = max(0, min(max_x, x1))
        y1 = max(0, min(max_y, y1))

        cropped = resized.crop((x1, y1, x1 + _SIZE[0], y1 + _SIZE[1]))
        return np.array(cropped)

    return VideoClip(make_frame, duration=duration)


def _stanza_clip(stanza_lines, language, image_path, variant=0):
    """One CompositeVideoClip: the stanza's own TTS narration, a Ken-
    Burns background (the lesson's own picture, visibly zooming and
    panning, or a plain colour card for lessons with none -- e.g.
    uploaded content), and karaoke captions timed to the narration.
    `variant` (the stanza's own index) varies the zoom/pan direction so
    consecutive stanzas don't all move identically -- see
    _background_clip."""
    text = " ".join(stanza_lines)
    words = text.split()
    narration_path = audio.generate_reference_audio(text, language, slow=False)
    if not narration_path or not os.path.exists(narration_path):
        return None  # TTS unavailable -- caller skips this stanza rather than crashing

    timings = _word_timings(narration_path, language, words)
    if not timings:
        return None
    duration = timings[-1][1]

    font = _font()
    positions, total_height = _layout_caption(words, font, _SIZE[0] - _PADDING * 2 - 20)

    bg = _background_clip(image_path, duration, variant=variant)
    particles = _particles_clip(duration, variant=variant)

    # EXTENSION: captions are now one continuous, time-based clip (not
    # a separate static image per word) specifically so the active
    # word can BOUNCE -- briefly pop larger right as it becomes current
    # (see _pop_scale_at) -- rather than just silently swapping colour
    # between frames, which read as too static on its own.
    def caption_make_frame(t):
        active_idx = 0
        t_since_start = 0.0
        for i, (start, end) in enumerate(timings):
            if start <= t < end or (i == len(timings) - 1 and t >= start):
                active_idx = i
                t_since_start = t - start
                break
        pop_scale = _pop_scale_at(t_since_start)
        frame = _caption_frame(font, positions, total_height, active_idx, _SIZE[0], pop_scale=pop_scale)
        return frame[:, :, :3]

    def caption_make_mask(t):
        active_idx = 0
        t_since_start = 0.0
        for i, (start, end) in enumerate(timings):
            if start <= t < end or (i == len(timings) - 1 and t >= start):
                active_idx = i
                t_since_start = t - start
                break
        pop_scale = _pop_scale_at(t_since_start)
        frame = _caption_frame(font, positions, total_height, active_idx, _SIZE[0], pop_scale=pop_scale)
        return frame[:, :, 3] / 255.0

    caption_h = total_height + _PADDING * 2
    caption_clip = VideoClip(caption_make_frame, duration=duration)
    caption_mask = VideoClip(caption_make_mask, duration=duration, is_mask=True)
    caption_clip = caption_clip.with_mask(caption_mask).with_position(("center", "bottom"))

    composite = CompositeVideoClip([bg, particles, caption_clip], size=_SIZE).with_duration(duration)
    narration = AudioFileClip(narration_path)
    return composite.with_audio(narration)


def _title_card_clip(lesson, duration=2.5):
    """A short opening card showing the lesson's own picture and title,
    before the narrated/captioned stanzas begin."""
    img = Image.new("RGB", _SIZE, _PLACEHOLDER_BG)
    image_path = lesson.get("image")
    if image_path and os.path.exists(image_path):
        pic = Image.open(image_path).convert("RGB")
        pic.thumbnail((_SIZE[0] - 80, _SIZE[1] - 160))
        img.paste(pic, ((_SIZE[0] - pic.width) // 2, 30))
    draw = ImageDraw.Draw(img)
    title_font = ImageFont.load_default(size=44)
    bbox = draw.textbbox((0, 0), lesson["title"], font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((_SIZE[0] - tw) // 2, _SIZE[1] - 90), lesson["title"], font=title_font, fill=(255, 255, 255, 255))
    clip = ImageClip(np.array(img)).with_duration(duration)
    return clip


def generate_lesson_video(lesson, force=False):
    """
    Builds (or returns the cached path to) a full lesson video: an
    opening title/picture card, then one narrated, word-highlighted
    clip per stanza, joined together. Returns the video file's path,
    or None if generation isn't possible right now (moviepy not
    installed, or TTS narration itself unavailable -- e.g. no internet
    for gTTS/edge-tts, same real-world condition modules/audio.py
    already handles everywhere else in this app).
    """
    if not MOVIEPY_AVAILABLE:
        return None

    out_path = lesson_video_path(lesson)
    if os.path.exists(out_path) and not force:
        return out_path

    clips = [_title_card_clip(lesson)]
    for i, stanza_lines in enumerate(lesson["stanzas"]):
        clip = _stanza_clip(stanza_lines, lesson["language"], lesson.get("image"), variant=i)
        if clip is not None:
            clips.append(clip)

    if len(clips) <= 1:
        return None  # every stanza failed (no TTS available) -- nothing worth saving

    final = concatenate_videoclips(clips, method="compose")
    tmp_path = out_path + ".tmp.mp4"
    final.write_videofile(tmp_path, fps=_FPS, codec="libx264", audio_codec="aac", logger=None)
    os.replace(tmp_path, out_path)
    return out_path
