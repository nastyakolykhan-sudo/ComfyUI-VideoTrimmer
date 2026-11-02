import io
import json
import math
import os
import re
import uuid
from fractions import Fraction

import av
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import folder_paths
from comfy_execution.graph_utils import ExecutionBlocker

from .video_trimmer import VideoFromFile, _probe, _window

try:
    from comfy_api.latest import InputImpl, Types
    VideoFromComponents = InputImpl.VideoFromComponents
    VideoComponents = Types.VideoComponents
except ImportError:  # older ComfyUI
    from comfy_api.input_impl import VideoFromComponents
    from comfy_api.util import VideoComponents

# Frames are numbered from 1 in the source file, as in Load Video (Trim + Keyframe).
# A scene is a window into the source (VideoFromFile with start_time/duration),
# so splitting decodes nothing at full size and re-encodes nothing.

DETECTORS = ["all", "content", "adaptive", "fade"]
ADAPTIVE_WINDOW = 2        # frames on each side the adaptive detector averages over
ADAPTIVE_MIN_CONTENT = 15  # adaptive cuts also need this much raw content change
MERGE_WINDOW = 3           # cuts from different detectors this close are one cut


# ---------------------------------------------------------------------------
# Source: where the frames of a VIDEO input live
# ---------------------------------------------------------------------------

def _file_source(video):
    """(path, info, first, last, crop) for a file-backed VIDEO, else None.

    first/last are 0-based frames of the file that the VIDEO's own trim window keeps,
    picked the way VideoFromFile picks them (pts in [start, start + duration)).
    """
    if not isinstance(video, VideoFromFile):
        return None
    path = video.get_stream_source()
    if not isinstance(path, str) or not os.path.isfile(path):
        return None
    info = _probe(path)
    pts, tb = info["pts"], info["time_base"]
    start_time, duration = video.get_active_trim_window()
    start_pts = int(start_time / tb)
    end_pts = int((start_time + duration) / tb) if duration else None
    kept = [i for i, p in enumerate(pts) if p >= start_pts and (end_pts is None or p < end_pts)]
    if not kept:
        raise ValueError("The input video's trim window holds no frames.")
    crop = getattr(video, "_VideoFromFile__crop", None)
    return path, info, kept[0], kept[-1], crop


def _scene_video(path, info, first, last, crop):
    start_time, duration = _window(info, first, last)
    kwargs = {"start_time": start_time, "duration": duration}
    if crop is not None:
        kwargs["crop"] = crop
    if first == 0 and last == len(info["pts"]) - 1 and crop is None:
        return VideoFromFile(path)
    return VideoFromFile(path, **kwargs)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _analysis_frames(path, info, first, last, size):
    """Small HSV frames (N, h, w, 3) float32, 0-255, for frames first..last of the file."""
    pts = info["pts"]
    scale = min(1.0, size / max(info["width"], info["height"]))
    w = max(8, round(info["width"] * scale))
    h = max(8, round(info["height"] * scale))
    out = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        start, end = pts[first], pts[last]
        if first > 0:
            container.seek(start, stream=stream)
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < start:
                continue
            if frame.pts > end:
                break
            out.append(frame.to_ndarray(width=w, height=h, format="rgb24"))
    if len(out) != last - first + 1:
        raise RuntimeError(f"Decoded {len(out)} frames for analysis, expected {last - first + 1}.")
    return _rgb_to_hsv(np.stack(out).astype(np.float32))


def _tensor_analysis_frames(images, size):
    """Same as _analysis_frames for an in-memory IMAGE batch (N, H, W, 3) 0-1."""
    n, H, W, _ = images.shape
    scale = min(1.0, size / max(H, W))
    h, w = max(8, round(H * scale)), max(8, round(W * scale))
    small = torch.nn.functional.interpolate(images.movedim(-1, 1).float(), size=(h, w), mode="area")
    return _rgb_to_hsv(small.movedim(1, -1).clamp(0, 1).mul(255).cpu().numpy())


def _rgb_to_hsv(rgb):
    """RGB 0-255 → HSV with every channel on 0-255 (hue wraps)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    v = rgb.max(-1)
    c = v - rgb.min(-1)
    s = np.where(v > 0, c / np.maximum(v, 1e-6) * 255, 0)
    safe = np.maximum(c, 1e-6)
    hue = np.where(v == r, (g - b) / safe % 6, np.where(v == g, (b - r) / safe + 2, (r - g) / safe + 4))
    hue = np.where(c > 0, hue / 6 * 255, 0)
    return np.stack([hue, s, v], -1).astype(np.float32)


def _content_scores(hsv):
    """Mean HSV change from each frame to the next (score[i] = change into frame i+1)."""
    d = np.abs(np.diff(hsv, axis=0))
    d[..., 0] = np.minimum(d[..., 0], 255 - d[..., 0])  # hue is circular
    return d.mean(axis=(1, 2, 3))


def _content_cuts(scores, threshold):
    return [(i + 1, float(s / threshold), "content") for i, s in enumerate(scores) if s >= threshold]


def _adaptive_cuts(scores, threshold):
    cuts = []
    k = ADAPTIVE_WINDOW
    for i, s in enumerate(scores):
        around = np.concatenate([scores[max(0, i - k):i], scores[i + 1:i + 1 + k]])
        if len(around) == 0 or s < ADAPTIVE_MIN_CONTENT:
            continue
        ratio = s / max(float(around.mean()), 1e-3)
        if ratio >= threshold:
            cuts.append((i + 1, float(ratio / threshold), "adaptive"))
    return cuts


def _fade_cuts(hsv, threshold, min_scene_length):
    """A run of dark frames between two scenes is one cut, placed at its middle."""
    dark = hsv[..., 2].mean(axis=(1, 2)) < threshold
    cuts = []
    i, n = 0, len(dark)
    while i < n:
        if not dark[i]:
            i += 1
            continue
        j = i
        while j < n and dark[j]:
            j += 1
        if i > 0 and j < n:  # dark at the very start or end is not a cut
            cuts.append(((i + j) // 2, 1.0, "fade", (i, j)))
        i = j
    return cuts


def _merge(candidates, n, min_scene_length):
    """One cut per spot, strongest first, no scene shorter than min_scene_length."""
    kept = []
    for frame, strength, source in sorted(candidates, key=lambda c: -c[1]):
        near = next((c for c in kept if abs(c["frame"] - frame) <= MERGE_WINDOW), None)
        if near is not None:
            if source not in near["detectors"]:
                near["detectors"].append(source)
            continue
        if frame < min_scene_length or n - frame < min_scene_length:
            continue
        if any(abs(c["frame"] - frame) < min_scene_length for c in kept):
            continue
        kept.append({"frame": frame, "strength": round(strength, 3), "detectors": [source]})
    return sorted(kept, key=lambda c: c["frame"])


def _parse_overrides(text):
    """'+120, -340' → ({120}, {340}); numbers are 1-based frames of the source file."""
    add, remove = set(), set()
    for token in re.split(r"[,\s;]+", text or ""):
        if not token:
            continue
        m = re.fullmatch(r"([+-]?)(\d+)", token)
        if not m:
            raise ValueError(f"cut_overrides: can't read '{token}'. Use +FRAME to add a cut, -FRAME to remove one.")
        (remove if m.group(1) == "-" else add).add(int(m.group(2)))
    return add, remove


def detect_cuts(hsv, detector, content_threshold, adaptive_threshold, fade_threshold, min_scene_length):
    """Cuts as 0-based positions inside the analysed frames: a cut at k starts a new scene at frame k."""
    scores = _content_scores(hsv) if len(hsv) > 1 else np.zeros(0, np.float32)
    candidates = []
    if detector in ("all", "content"):
        candidates += _content_cuts(scores, content_threshold)
    if detector in ("all", "adaptive"):
        candidates += _adaptive_cuts(scores, adaptive_threshold)
    candidates = [c for c in candidates if not _is_flash(hsv, c[0], content_threshold, min_scene_length)]
    if detector in ("all", "fade"):
        # A fade through black is one cut in the middle of the dark run: the jumps into
        # and out of the run are the same transition, not scenes of their own.
        fades = _fade_cuts(hsv, fade_threshold, min_scene_length)
        runs = [f[3] for f in fades]
        candidates = [c for c in candidates
                      if not any(a - MERGE_WINDOW <= c[0] <= b + MERGE_WINDOW for a, b in runs)]
        candidates += [f[:3] for f in fades]
    return _merge(candidates, len(hsv), max(1, min_scene_length)), scores


def _is_flash(hsv, k, threshold, min_scene_length):
    """A jump at k is a flash (not a cut) when the picture before it returns within min_scene_length frames,
    or when it jumps back to the picture before a jump less than min_scene_length frames earlier."""
    def change(a, b):
        d = np.abs(hsv[a] - hsv[b])
        d[..., 0] = np.minimum(d[..., 0], 255 - d[..., 0])
        return float(d.mean())
    span = max(1, min_scene_length)
    for j in range(k + 1, min(len(hsv), k + span)):
        if change(k - 1, j) < threshold * 0.5:
            return True
    for j in range(max(0, k - span), k - 1):
        if j >= 1 and change(j - 1, k) < threshold * 0.5:
            return True
    return False


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _timecode(frame0, fps):
    """0-based frame → HH:MM:SS:FF (non-drop)."""
    rate = max(1, round(float(fps)))
    s, f = divmod(frame0, rate)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def _contact_sheet(images, max_tile=320, columns=0):
    """Grid of the scenes' first frames, each tile numbered from 0 (the scene index)."""
    n = len(images)
    cols = columns or min(n, max(1, math.ceil(math.sqrt(n * 16 / 9 / (images[0].shape[1] / images[0].shape[0])))))
    rows = math.ceil(n / cols)
    H, W = images[0].shape[0], images[0].shape[1]
    scale = min(1.0, max_tile / max(H, W))
    th, tw = max(1, round(H * scale)), max(1, round(W * scale))
    pad = 4
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + pad) + pad), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=max(12, th // 8))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for i, img in enumerate(images):
        tile = Image.fromarray((img.clamp(0, 1).mul(255).byte().cpu().numpy())).resize((tw, th), Image.LANCZOS)
        x, y = pad + (i % cols) * (tw + pad), pad + (i // cols) * (th + pad)
        sheet.paste(tile, (x, y))
        label = str(i)
        box = draw.textbbox((0, 0), label, font=font)
        draw.rectangle([x, y, x + box[2] - box[0] + 10, y + box[3] - box[1] + 8], fill=(0, 0, 0))
        draw.text((x + 5 - box[0], y + 4 - box[1]), label, fill=(255, 255, 255), font=font)
    return torch.from_numpy(np.asarray(sheet).astype(np.float32) / 255.0)[None]


def _first_frame(video):
    images = video.get_components().images
    if images.shape[0] == 0:
        raise RuntimeError("Could not decode a scene's first frame.")
    return images[:1].contiguous()


def _parse_verdict(text, min_confidence):
    """(has_product, confidence, note) from an LLM answer: JSON {product, confidence} or a yes/no word."""
    if isinstance(text, bool):
        return text, 1.0, ""
    raw = str(text or "").strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            value = next((data[k] for k in ("product", "has_product", "present", "answer") if k in data), None)
            conf = float(data.get("confidence", 1.0))
            if isinstance(value, str):
                value = value.strip().lower() in ("yes", "true", "y", "1")
            if value is not None:
                return bool(value) and conf >= min_confidence, conf, str(data.get("why", ""))[:200]
        except (ValueError, TypeError):
            pass
    word = re.match(r"[\W_]*([a-z]+)", raw.lower())
    word = word.group(1) if word else ""
    if word in ("yes", "true", "y"):
        return True, 1.0, ""
    if word in ("no", "false", "n", "none"):
        return False, 1.0, ""
    return None, 0.0, raw[:200]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class SplitVideoByScene:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "Video to split. From Load Video (Trim + Keyframe), only the trimmed "
                                               "range is split."}),
                "detector": (DETECTORS, {"default": "all",
                                         "tooltip": "all: every detector, any one can call a cut (near-duplicates merge). "
                                                    "content: hard cuts (colour/brightness change). "
                                                    "adaptive: changes much bigger than the frames around them, so fast "
                                                    "pans don't cut. fade: dark frames between scenes."}),
                "content_threshold": ("FLOAT", {"default": 27.0, "min": 1.0, "max": 255.0, "step": 0.5,
                                                "tooltip": "content: mean HSV change (0-255) that counts as a cut. Lower = more cuts."}),
                "adaptive_threshold": ("FLOAT", {"default": 3.0, "min": 1.1, "max": 20.0, "step": 0.1,
                                                 "tooltip": "adaptive: how many times bigger than the surrounding frames' change "
                                                            "a change must be. Lower = more cuts."}),
                "fade_threshold": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 128.0, "step": 0.5,
                                             "tooltip": "fade: mean brightness (0-255) below which a frame counts as black."}),
                "min_scene_length": ("INT", {"default": 12, "min": 1, "max": 10_000,
                                             "tooltip": "Shortest scene in frames. Shorter ones merge into a neighbour "
                                                        "(stops flash frames becoming scenes)."}),
                "trim_edges": ("INT", {"default": 0, "min": 0, "max": 100,
                                       "tooltip": "Frames dropped on each side of every cut (dissolve leftovers). "
                                                  "Those frames are left out of every scene."}),
                "analysis_size": ("INT", {"default": 256, "min": 32, "max": 2048, "step": 16,
                                          "tooltip": "Long side in px of the frames detection looks at. Outputs stay full size."}),
                "cut_overrides": ("STRING", {"default": "", "multiline": False,
                                             "tooltip": "Manual fixes, frames counted from 1 in the source file: "
                                                        "+120 adds a cut (frame 120 starts a scene), -340 removes the cut "
                                                        "detected nearest 340 (within 3 frames). Comma separated."}),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "INT", "INT", "INT", "STRING", "IMAGE")
    RETURN_NAMES = ("scenes", "first_frames", "start_frames", "end_frames", "scene_count", "cut_list", "contact_sheet")
    OUTPUT_IS_LIST = (True, True, True, True, False, False, False)
    OUTPUT_TOOLTIPS = (
        "One video per scene, with its audio. A list: the nodes after it run once per scene.",
        "First frame of each scene (list).",
        "First frame of each scene, counted from 1 in the source file (list).",
        "Last frame of each scene, inclusive, counted from 1 in the source file (list).",
        "Number of scenes.",
        "JSON: every scene's frames, timecodes and length, and the score and detectors of every cut.",
        "Every scene's first frame in one grid, numbered from 0 = the scene index.",
    )
    FUNCTION = "split"
    CATEGORY = "video"
    DESCRIPTION = ("Find the cuts in a video and split it into one video per scene. Scenes are frame-exact windows "
                   "into the source: nothing is re-encoded. Frame numbers start at 1.")

    def split(self, video, detector, content_threshold, adaptive_threshold, fade_threshold,
              min_scene_length, trim_edges, analysis_size, cut_overrides):
        source = _file_source(video)
        if source is not None:
            path, info, first, last, crop = source
            hsv = _analysis_frames(path, info, first, last, analysis_size)
            fps = info["fps"]
            components = None
        else:
            components = video.get_components()
            first, last = 0, components.images.shape[0] - 1
            if last < 0:
                raise ValueError("The input video has no frames.")
            hsv = _tensor_analysis_frames(components.images, analysis_size)
            fps = components.frame_rate

        n = last - first + 1
        cuts, scores = detect_cuts(hsv, detector, content_threshold, adaptive_threshold, fade_threshold,
                                   min_scene_length)

        # Overrides are file frames (1-based); cuts are positions inside the analysed range.
        add, remove = _parse_overrides(cut_overrides)
        for f in remove:
            k = f - 1 - first
            near = min(cuts, key=lambda c: abs(c["frame"] - k), default=None)
            if near is not None and abs(near["frame"] - k) <= MERGE_WINDOW:
                cuts.remove(near)
        for f in add:
            k = f - 1 - first
            if 0 < k < n and all(c["frame"] != k for c in cuts):
                score = float(scores[k - 1]) if k - 1 < len(scores) else 0.0
                cuts.append({"frame": k, "strength": round(score / content_threshold, 3), "detectors": ["manual"]})
        cuts.sort(key=lambda c: c["frame"])

        bounds = [0] + [c["frame"] for c in cuts] + [n]
        ranges = []
        for i in range(len(bounds) - 1):
            a = bounds[i] + (trim_edges if i > 0 else 0)
            b = bounds[i + 1] - 1 - (trim_edges if i < len(bounds) - 2 else 0)
            if b >= a:
                ranges.append((a, b))
        if not ranges:
            raise ValueError("trim_edges removed every frame; lower it.")

        scenes = []
        for a, b in ranges:
            if components is not None:
                audio = components.audio
                if audio is not None:
                    sr = audio["sample_rate"]
                    s0, s1 = round(a / float(fps) * sr), round((b + 1) / float(fps) * sr)
                    audio = {"waveform": audio["waveform"][..., s0:s1], "sample_rate": sr}
                scenes.append(VideoFromComponents(VideoComponents(
                    images=components.images[a:b + 1], audio=audio, frame_rate=fps)))
            else:
                scenes.append(_scene_video(path, info, first + a, first + b, crop))

        first_frames = []
        for (a, _), scene in zip(ranges, scenes):
            if components is not None:
                first_frames.append(components.images[a:a + 1].contiguous())
            else:
                first_frames.append(_first_frame(_scene_video(path, info, first + a, first + a, crop)))

        starts = [first + a + 1 for a, _ in ranges]
        ends = [first + b + 1 for _, b in ranges]
        cut_list = {
            "fps": float(fps),
            "source_frames": [first + 1, last + 1],
            "detector": detector,
            "scenes": [{"index": i, "start_frame": s, "end_frame": e, "frames": e - s + 1,
                        "start_tc": _timecode(s - 1, fps), "end_tc": _timecode(e - 1, fps),
                        "seconds": round((e - s + 1) / float(fps), 3)}
                       for i, (s, e) in enumerate(zip(starts, ends))],
            "cuts": [{"frame": first + c["frame"] + 1, "tc": _timecode(first + c["frame"], fps),
                      "strength": c["strength"], "detectors": c["detectors"]} for c in cuts],
        }
        sheet = _contact_sheet([f[0] for f in first_frames])
        return (scenes, first_frames, starts, ends, len(scenes), json.dumps(cut_list, indent=2), sheet)


class SelectScene:
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "videos": ("VIDEO", {"tooltip": "Any list of videos: the scenes from Split Video by Scene, the "
                                                "product scenes from Scene Gate, a model's per-scene outputs..."}),
                "index": ("INT", {"default": 0, "min": -10_000, "max": 10_000,
                                  "tooltip": "Video to pick, counted from 0 (the contact sheet's numbers). "
                                             "Negative counts from the end (-1 = last)."}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "select"
    CATEGORY = "video"
    DESCRIPTION = "Pick one video out of a list by its index."

    def select(self, videos, index):
        i = index[0]
        if not -len(videos) <= i < len(videos):
            raise ValueError(f"Video {i} doesn't exist: the list holds {len(videos)} (0-{len(videos) - 1}).")
        return (videos[i],)


class SceneGate:
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scenes": ("VIDEO", {"tooltip": "The scenes list from Split Video by Scene."}),
                "verdicts": ("STRING", {"forceInput": True,
                                        "tooltip": "One answer per scene, in scene order (an LLM node run on the scenes "
                                                   "list). Reads JSON like {\"product\": true, \"confidence\": 0.9} or an "
                                                   "answer starting with yes/no."}),
                "min_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                             "tooltip": "JSON answers below this confidence count as no product."}),
                "if_unclear": (["send", "skip", "error"], {"default": "send",
                                                           "tooltip": "An answer that is neither yes nor no: send the scene "
                                                                      "on (safer: a missed product costs a wrong film), skip "
                                                                      "it, or stop the run."}),
            },
        }

    RETURN_TYPES = ("VIDEO", "INT", "INT", "STRING")
    RETURN_NAMES = ("product_scenes", "product_indices", "product_count", "report")
    OUTPUT_IS_LIST = (True, True, False, False)
    OUTPUT_TOOLTIPS = (
        "The scenes with the product, in order (list). Nothing downstream runs when there are none.",
        "Their positions in the scenes list, counted from 0 (list). Wire into Join Scenes.",
        "How many scenes have the product.",
        "JSON: every scene's verdict, confidence and length, and any unclear answers.",
    )
    FUNCTION = "gate"
    CATEGORY = "video"
    DESCRIPTION = ("Route the scenes an LLM said show the product on to the next model, "
                   "and keep their positions so Join Scenes can put the results back.")

    def gate(self, scenes, verdicts, min_confidence, if_unclear):
        min_confidence, if_unclear = min_confidence[0], if_unclear[0]
        if len(verdicts) != len(scenes):
            raise ValueError(f"{len(verdicts)} verdicts for {len(scenes)} scenes: run the LLM node on the scenes list "
                             f"so it answers once per scene.")
        picked, rows, warnings = [], [], []
        for i, (scene, text) in enumerate(zip(scenes, verdicts)):
            has, conf, note = _parse_verdict(text, min_confidence)
            seconds = round(scene.get_duration(), 3)
            if has is None:
                if if_unclear == "error":
                    raise ValueError(f"Scene {i}: can't read the answer as yes/no: {note!r}")
                has = if_unclear == "send"
                warnings.append(f"scene {i}: unclear answer, {'sent' if has else 'skipped'}")
            if has:
                picked.append(i)
            rows.append({"index": i, "product": has, "confidence": conf, "seconds": seconds, "answer": str(text)[:300]})
        report = json.dumps({"product_indices": picked, "scenes": rows, "warnings": warnings}, indent=2)
        product_scenes = [scenes[i] for i in picked] or [ExecutionBlocker(None)]
        return (product_scenes, picked, len(picked), report)


class PadVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "A video, or a list of them (each is padded on its own)."}),
                "min_frames": ("INT", {"default": 0, "min": 0, "max": 100_000,
                                       "tooltip": "Shortest length in frames. Shorter videos get their last frame "
                                                  "repeated up to it. 0 = off."}),
                "min_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1,
                                          "tooltip": "Shortest length in seconds, at the video's own frame rate "
                                                     "(rounded up to whole frames). 0 = off. With both set, the "
                                                     "longer one wins."}),
            },
        }

    RETURN_TYPES = ("VIDEO", "INT", "INT")
    RETURN_NAMES = ("video", "original_frames", "added_frames")
    OUTPUT_TOOLTIPS = (
        "The video, padded when it was shorter; otherwise the same video untouched.",
        "Its frame count before padding.",
        "Frames added (0 when it was long enough).",
    )
    FUNCTION = "pad"
    CATEGORY = "video"
    DESCRIPTION = ("Make a video at least a given length by repeating its last frame (audio padded with "
                   "silence). For models that need a minimum clip length; Join Scenes keeps only the "
                   "original frames afterwards.")

    def pad(self, video, min_frames, min_seconds):
        fps = Fraction(video.get_frame_rate())
        n = video.get_frame_count()
        want = max(min_frames, math.ceil(round(min_seconds * float(fps), 6)) if min_seconds else 0)
        if n >= want:
            return (video, n, 0)
        comp = video.get_components()
        images = comp.images
        n = images.shape[0]
        if n == 0:
            raise ValueError("Can't pad a video with no frames.")
        images = torch.cat([images, images[-1:].expand(want - n, -1, -1, -1)])
        audio = comp.audio
        if audio is not None:
            sr = int(audio["sample_rate"])
            wave = audio["waveform"]
            extra = max(0, round(want / float(fps) * sr) - wave.shape[-1])
            audio = {"waveform": torch.nn.functional.pad(wave, (0, extra)), "sample_rate": sr}
        return (VideoFromComponents(VideoComponents(images=images.contiguous(), audio=audio, frame_rate=fps)), n,
                want - n)


class VideoDuration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "A video, or a list of them (one answer per video)."}),
                "rounding": (["up", "nearest", "down"], {"default": "up",
                                                       "tooltip": "How seconds becomes whole_seconds. up: never "
                                                                  "shorter than the video (a model asked for this "
                                                                  "many seconds covers every frame)."}),
                "min_whole_seconds": ("INT", {"default": 0, "min": 0, "max": 3600,
                                              "tooltip": "Clamp whole_seconds to at least this. 0 = no clamp."}),
                "max_whole_seconds": ("INT", {"default": 0, "min": 0, "max": 3600,
                                              "tooltip": "Clamp whole_seconds to at most this. 0 = no clamp."}),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "INT", "COMBO", "FLOAT")
    RETURN_NAMES = ("frames", "seconds", "whole_seconds", "whole_seconds_choice", "fps")
    OUTPUT_TOOLTIPS = (
        "Frame count.",
        "Length in seconds (frames / fps).",
        "Seconds as a whole number, rounded and clamped as set.",
        "whole_seconds as text, for a node's duration dropdown (a model node that takes \"2\"..\"30\").",
        "Frame rate.",
    )
    FUNCTION = "measure"
    CATEGORY = "video"
    DESCRIPTION = ("Measure a video: frames, seconds, whole seconds (also as a dropdown value, to set a model "
                   "node's duration per video) and frame rate.")

    def measure(self, video, rounding, min_whole_seconds, max_whole_seconds):
        fps = Fraction(video.get_frame_rate())
        frames = video.get_frame_count()
        seconds = frames / float(fps)
        exact = round(seconds, 6)
        whole = {"up": math.ceil(exact), "down": math.floor(exact), "nearest": round(exact)}[rounding]
        if min_whole_seconds:
            whole = max(whole, min_whole_seconds)
        if max_whole_seconds:
            whole = min(whole, max_whole_seconds)
        return (frames, seconds, whole, str(whole), float(fps))


class JoinScenes:
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scenes": ("VIDEO", {"tooltip": "The original scenes list from Split Video by Scene."}),
                "product_indices": ("INT", {"forceInput": True,
                                            "tooltip": "From Scene Gate: where each replacement goes."}),
                "audio": (["source", "none"], {"default": "source",
                                               "tooltip": "source: the original audio under every scene, replacements "
                                                          "included. none: no audio."}),
                "fit": (["time", "stretch"], {"default": "time",
                                              "tooltip": "How a replacement fills its scene. time: the scene's own frame "
                                                         "count taken from the replacement's start, matched by time when "
                                                         "the frame rates differ, so padding (Pad Video) and any extra "
                                                         "length are dropped. stretch: the replacement's whole length is "
                                                         "squeezed or stretched onto the scene."}),
            },
            "optional": {
                "replacements": ("VIDEO", {"lazy": True,
                                           "tooltip": "New versions of the product scenes, in Scene Gate's order "
                                                      "(e.g. a video model's output). Only run when there are product scenes. "
                                                      "Each is fitted to its scene's frame count, size and frame rate."}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "report")
    FUNCTION = "join"
    CATEGORY = "video"
    DESCRIPTION = ("Put the replaced scenes back in their slots and join every scene into one video, "
                   "keeping each scene's exact frame count and the source audio.")

    def check_lazy_status(self, scenes, product_indices, audio, fit, replacements=None):
        if product_indices and (replacements is None or all(r is None for r in replacements)):
            return ["replacements"]
        return []

    def join(self, scenes, product_indices, audio, fit, replacements=None):
        replacements = [r for r in (replacements or []) if r is not None]
        if len(replacements) != len(product_indices):
            raise ValueError(f"{len(replacements)} replacements for {len(product_indices)} product scenes.")
        swap = dict(zip(product_indices, replacements))
        bad = [i for i in swap if not 0 <= i < len(scenes)]
        if bad:
            raise ValueError(f"Product indices {bad} are outside the {len(scenes)} scenes.")

        fps = Fraction(scenes[0].get_frame_rate()).limit_denominator(100_000)
        width, height = scenes[0].get_dimensions()
        counts = [s.get_frame_count() for s in scenes]
        windows = [_file_window(s) for s in scenes]

        waves, sample_rate = [], None
        for scene, n in zip(scenes, counts):
            wave = _scene_audio(scene, n, fps) if audio[0] == "source" else None
            if wave is not None:
                sample_rate = sample_rate or wave[1]
            waves.append(wave)

        # Lossless intermediate in the temp folder; Save Video encodes it once for delivery.
        # Source scenes go in as decoded frames, untouched; replacements are converted with
        # the same default matrix ComfyUI decoded the source with, so their colours line up.
        out_path = os.path.join(folder_paths.get_temp_directory(), f"join_scenes_{uuid.uuid4().hex[:12]}.mkv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        notes = []
        with av.open(out_path, mode="w") as out:
            pix_fmt, tags, options = "bgr0", {}, {}  # no source file: lossless RGB for the tensors
            if windows[0] is not None:
                pix_fmt, tags = _stream_format(windows[0][0])
                options = {"chroma_sample_location": _chroma_location(*windows[0])}
            vs = out.add_stream("ffv1", rate=fps, options=options)
            vs.width, vs.height = width, height
            vs.codec_context.time_base = 1 / fps
            vs.pix_fmt = pix_fmt if pix_fmt in FFV1_PIX_FMTS else "yuv444p"
            for k, v in tags.items():
                setattr(vs.codec_context, k, v)
            astream = None
            if sample_rate:
                channels = 2 if max(w[0].shape[0] for w in waves if w is not None) > 1 else 1
                astream = out.add_stream("pcm_s16le", rate=sample_rate, layout="stereo" if channels > 1 else "mono")

            pts = 0

            def put(frame):
                nonlocal pts
                if frame.width != width or frame.height != height or frame.format.name != vs.pix_fmt:
                    frame = frame.reformat(width=width, height=height, format=vs.pix_fmt)
                frame.pts = pts
                frame.time_base = vs.codec_context.time_base
                pts += 1
                for packet in vs.encode(frame):
                    out.mux(packet)

            for i, scene in enumerate(scenes):
                n = counts[i]
                if i in swap or windows[i] is None:
                    comp = swap[i].get_components() if i in swap else scene.get_components()
                    images, note = _fit(comp.images, n, height, width, i, fps, comp.frame_rate,
                                        fit[0] if i in swap else "time")
                    if note and i in swap:
                        notes.append(note)
                    for img in images:
                        rgb = img.clamp(0, 1).mul(255).round().byte().cpu().numpy()
                        put(av.VideoFrame.from_ndarray(rgb, format="rgb24"))
                else:
                    written = 0
                    for frame in _window_frames(*windows[i]):
                        put(frame)
                        written += 1
                    if written != n:
                        raise RuntimeError(f"Scene {i}: decoded {written} frames, expected {n}.")
            for packet in vs.encode():
                out.mux(packet)

            if astream is not None:
                apts = 0
                for w, n in zip(waves, counts):
                    want = round(n / float(fps) * sample_rate)
                    if w is None or w[1] != sample_rate:
                        x = torch.zeros(channels, want)  # no audio, or another rate: silence
                    else:
                        x = w[0][:channels] if w[0].shape[0] >= channels else w[0].repeat(channels, 1)
                        x = torch.nn.functional.pad(x[:, :want], (0, max(0, want - x.shape[1])))
                    pcm = (x.clamp(-1, 1) * 32767).round().to(torch.int16).numpy()
                    for c0 in range(0, want, 4096):
                        chunk = np.ascontiguousarray(pcm[:, c0:c0 + 4096].T).reshape(1, -1)
                        af = av.AudioFrame.from_ndarray(chunk, format="s16", layout=astream.layout.name)
                        af.sample_rate = sample_rate
                        af.pts = apts
                        af.time_base = Fraction(1, sample_rate)
                        apts += chunk.shape[1] // channels
                        for packet in astream.encode(af):
                            out.mux(packet)
                for packet in astream.encode():
                    out.mux(packet)

        report = json.dumps({"scenes": len(scenes), "frames": sum(counts), "replaced": sorted(swap),
                             "notes": notes}, indent=2)
        return (VideoFromFile(out_path), report)


FFV1_PIX_FMTS = ("yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le", "yuv444p10le", "gbrp", "bgr0")


def _file_window(scene):
    """(path, start_pts, end_pts or None) for a plain, unrotated file window, else None."""
    if not isinstance(scene, VideoFromFile) or getattr(scene, "_VideoFromFile__crop", None) is not None:
        return None
    path = scene.get_stream_source()
    if not isinstance(path, str):
        return None
    with av.open(path) as container:
        stream = container.streams.video[0]
        tb = stream.time_base
        for frame in container.decode(stream):
            if frame.rotation:
                return None  # rotated sources go through ComfyUI's own decode
            break
    start_time, duration = scene.get_active_trim_window()
    return path, int(start_time / tb), (int((start_time + duration) / tb) if duration else None)


def _window_frames(path, start_pts, end_pts):
    """The decoded frames of a window, picked the way VideoFromFile picks them."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if start_pts:
            container.seek(start_pts, stream=stream)
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < start_pts:
                continue
            if end_pts is not None and frame.pts >= end_pts:
                break
            yield frame


CHROMA_LOCATIONS = ("left", "center", "topleft", "top", "bottomleft", "bottom", "unspecified")


def _chroma_location(path, start_pts, end_pts):
    """The source's chroma siting, which PyAV can't read: re-encode its first frame with each
    siting and keep the one that decodes to the same RGB. Without it, colour edges decode
    differently from the source even though the YUV bytes are identical."""
    frame = next(_window_frames(path, start_pts, end_pts), None)
    if frame is None or frame.format.name not in FFV1_PIX_FMTS or not frame.format.name.startswith("yuv4"):
        return "unspecified"
    ref = frame.to_ndarray(format="gbrpf32le")
    for loc in CHROMA_LOCATIONS:
        buf = io.BytesIO()
        with av.open(buf, mode="w", format="matroska") as out:
            st = out.add_stream("ffv1", rate=24, options={"chroma_sample_location": loc})
            st.width, st.height, st.pix_fmt = frame.width, frame.height, frame.format.name
            f = frame.reformat()  # copy; keep the source frame's pts untouched
            f.pts, f.time_base = 0, Fraction(1, 24)
            for packet in st.encode(f):
                out.mux(packet)
            for packet in st.encode():
                out.mux(packet)
        buf.seek(0)
        with av.open(buf) as back:
            if np.array_equal(next(back.decode(video=0)).to_ndarray(format="gbrpf32le"), ref):
                return loc
    return "left"


def _stream_format(path):
    """Pixel format and the colour tags that are actually set on a file's video stream."""
    with av.open(path) as container:
        cc = container.streams.video[0].codec_context
        tags = {}
        for k in ("color_primaries", "color_trc", "colorspace", "color_range"):
            v = getattr(cc, k, None)
            if v not in (None, 0, 2):  # 2 = unspecified
                tags[k] = v
        return cc.pix_fmt, tags


def _scene_audio(scene, n, fps):
    """(waveform (C, S), sample_rate) under a scene, or None. File windows are sliced from the source audio."""
    if isinstance(scene, VideoFromFile):
        path = scene.get_stream_source()
        if isinstance(path, str):
            full = _source_audio(path)
            if full is None:
                return None
            wave, sr = full
            start_time, _ = scene.get_active_trim_window()
            s0 = max(0, round(start_time * sr))
            return wave[:, s0:s0 + round(n / float(fps) * sr)], sr
    comp = scene.get_components()
    if comp.audio is None:
        return None
    return comp.audio["waveform"][0], int(comp.audio["sample_rate"])


_audio_cache = {}


def _source_audio(path):
    """The whole audio track of a file as float (C, S), decoded once per file version."""
    key = (path, os.path.getmtime(path), os.path.getsize(path))
    if key in _audio_cache:
        return _audio_cache[key]
    result = None
    with av.open(path) as container:
        stream = next((s for s in container.streams.audio), None)
        if stream is not None:
            resampler = av.audio.resampler.AudioResampler(format="fltp")
            chunks = [f.to_ndarray() for frame in container.decode(stream) for f in resampler.resample(frame)]
            if chunks:
                result = (torch.from_numpy(np.concatenate(chunks, axis=1)), int(stream.sample_rate))
    _audio_cache.clear()
    _audio_cache[key] = result
    return result


def _fit(images, n, height, width, index, fps, src_fps, mode):
    """A replacement as n frames of height x width at fps (nearest frame in time, area/bicubic resize).

    time: frame k of the scene takes the replacement's frame at the same time from its start.
    stretch: the replacement's whole length is mapped onto the scene.
    """
    note = []
    m = images.shape[0]
    if m == 0:
        raise ValueError(f"Replacement for scene {index} has no frames.")
    if mode == "time":
        ratio = float(src_fps) / float(fps)
        idx = torch.clamp(torch.round(torch.arange(n) * ratio).long(), max=m - 1)
        if m < round((n - 1) * ratio) + 1:
            note.append(f"{m} frames for {n}: last frame held")
        elif m != n or src_fps != fps:
            note.append(f"{m} frames at {float(src_fps):g} fps -> {n} at {float(fps):g} fps")
    else:
        idx = torch.clamp(((torch.arange(n) + 0.5) * m / n).long(), max=m - 1)
        if m != n:
            note.append(f"{m}->{n} frames stretched")
    images = images[idx]
    h, w = images.shape[1], images.shape[2]
    if (h, w) != (height, width):
        if abs(w / h - width / height) > 0.01:
            note.append(f"aspect {w}x{h} stretched to {width}x{height}")
        else:
            note.append(f"{w}x{h}->{width}x{height}")
        mode = "area" if h > height else "bicubic"
        images = torch.nn.functional.interpolate(images.movedim(-1, 1).float(), size=(height, width),
                                                 mode=mode).movedim(1, -1).clamp(0, 1)
    return images.contiguous(), (f"scene {index}: " + ", ".join(note)) if note else ""


NODE_CLASS_MAPPINGS = {
    "SplitVideoByScene": SplitVideoByScene,
    "SelectScene": SelectScene,
    "SceneGate": SceneGate,
    "PadVideo": PadVideo,
    "VideoDuration": VideoDuration,
    "JoinScenes": JoinScenes,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplitVideoByScene": "Split Video by Scene",
    "SelectScene": "Select Video from List",
    "SceneGate": "Scene Gate (product scenes)",
    "PadVideo": "Pad Video (repeat last frame)",
    "VideoDuration": "Video Duration",
    "JoinScenes": "Join Scenes",
}
