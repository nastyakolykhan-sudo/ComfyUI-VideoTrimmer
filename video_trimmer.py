import io
import os
from collections import OrderedDict
from fractions import Fraction

import av
import numpy as np
from aiohttp import web
from PIL import Image

import folder_paths
from server import PromptServer

try:
    from comfy_api.latest import InputImpl
    VideoFromFile = InputImpl.VideoFromFile
except ImportError:  # older ComfyUI
    from comfy_api.input_impl import VideoFromFile

# Frames are numbered from 1, as shown in the trimmer.
# A trim window is turned into the (start_time, duration) that ComfyUI's own
# VideoFromFile uses, chosen from the stream's real timestamps so the decode
# keeps exactly frames start..end: start_time sits a tenth of a tick after the
# start frame's pts, and the window ends on the next frame's pts.
PTS_NUDGE = 0.1
PREVIEW_MAX_SIDE = 1024
STRIP_MAX_TILES = 40


# ---------------------------------------------------------------------------
# Probing: exact frame timestamps without decoding
# ---------------------------------------------------------------------------

_probe_cache = OrderedDict()


def _probe(path):
    """Return the video stream's presented frames as sorted pts, plus timing info.

    Packets are demuxed, not decoded. Frames before pts 0 are left out because
    VideoFromFile drops them too, so the count matches what it decodes.
    """
    key = (path, os.path.getmtime(path), os.path.getsize(path))
    if key in _probe_cache:
        _probe_cache.move_to_end(key)
        return _probe_cache[key]

    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in '{os.path.basename(path)}'")
        stream = container.streams.video[0]
        time_base = stream.time_base
        pts = []
        last_duration = 0
        for packet in container.demux(stream):
            if packet.pts is None or packet.size == 0 or getattr(packet, "is_discard", False):
                continue
            if packet.pts >= 0:
                pts.append(packet.pts)
                last_duration = max(last_duration, packet.duration or 0)
        pts.sort()
        if not pts:
            raise ValueError(f"No video frames in '{os.path.basename(path)}'")
        fps = Fraction(stream.average_rate) if stream.average_rate else Fraction(1)
        if not last_duration:
            last_duration = pts[-1] - pts[-2] if len(pts) > 1 else max(1, round(1 / (fps * time_base)))
        width, height = stream.codec_context.width, stream.codec_context.height

    info = {
        "pts": pts,
        "end_pts": pts[-1] + last_duration,
        "time_base": time_base,
        "fps": fps,
        "width": width,
        "height": height,
    }
    _probe_cache[key] = info
    while len(_probe_cache) > 16:
        _probe_cache.popitem(last=False)
    return info


def _window(info, first, last):
    """(start_time, duration) in seconds that keeps 0-based frames first..last."""
    pts, tb = info["pts"], info["time_base"]
    start_time = float((pts[first] + PTS_NUDGE) * tb) if first > 0 else 0.0
    if last >= len(pts) - 1:
        return start_time, 0.0  # to the end of the file
    return start_time, float((pts[last + 1] - pts[first]) * tb)


def _clamp_range(total, start_frame, end_frame, keyframe):
    start = min(max(1, start_frame), total)
    end = total if end_frame <= 0 else min(max(start, end_frame), total)
    key = start if keyframe <= 0 else min(max(1, keyframe), total)
    return start, end, key


def _resolve_input(name):
    """Path of a file in ComfyUI's input folder; refuses anything outside it."""
    if not name or not folder_paths.exists_annotated_filepath(name):
        raise FileNotFoundError(f"Invalid video file: {name}")
    path = os.path.realpath(folder_paths.get_annotated_filepath(name))
    roots = [folder_paths.get_input_directory(), folder_paths.get_output_directory(),
             folder_paths.get_temp_directory()]
    if not any(os.path.commonpath([path, os.path.realpath(r)]) == os.path.realpath(r) for r in roots):
        raise FileNotFoundError(f"Invalid video file: {name}")
    return path


# ---------------------------------------------------------------------------
# Thumbnails for the trimmer (exact frames, any codec the server can decode)
# ---------------------------------------------------------------------------

def _decode_frames(path, info, indices, max_side):
    """PIL images of the given 0-based frames, decoded at reduced size."""
    pts = info["pts"]
    wanted = sorted(set(indices))
    out = {}
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        scale = min(1.0, max_side / max(info["width"], info["height"]))
        w = max(2, round(info["width"] * scale))
        h = max(2, round(info["height"] * scale))
        frame_ticks = (info["end_pts"] - pts[0]) / len(pts)
        i = 0
        while i < len(wanted):
            target = pts[wanted[i]]
            container.seek(target, stream=stream)
            for frame in container.decode(stream):
                if frame.pts is None or frame.pts < target:
                    continue
                img = frame.to_image(width=w, height=h)
                if frame.rotation:
                    img = img.rotate(frame.rotation, expand=True)
                out[wanted[i]] = img
                i += 1
                if i >= len(wanted):
                    break
                target = pts[wanted[i]]
                # Far ahead: seeking is cheaper than decoding up to it.
                if target - frame.pts > 60 * frame_ticks:
                    break
            else:
                # Ran off the end: fill the rest with the last picture.
                while i < len(wanted):
                    out[wanted[i]] = out.get(wanted[i - 1]) or Image.new("RGB", (w, h))
                    i += 1
    return [out[k] for k in indices]


def _jpeg(img, quality=85):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return web.Response(body=buf.getvalue(), content_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})


routes = PromptServer.instance.routes


@routes.get("/video_trimmer/info")
async def video_trimmer_info(request):
    try:
        path = _resolve_input(request.query.get("filename", ""))
        info = _probe(path)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)
    pts, tb = info["pts"], info["time_base"]
    return web.json_response({
        "frames": len(pts),
        "fps": float(info["fps"]),
        "width": info["width"],
        "height": info["height"],
        # Frame start times relative to the first frame (the browser's time 0).
        "times": [round(float((p - pts[0]) * tb), 6) for p in pts],
        "duration": float((info["end_pts"] - pts[0]) * tb),
    })


@routes.get("/video_trimmer/strip")
async def video_trimmer_strip(request):
    try:
        path = _resolve_input(request.query.get("filename", ""))
        info = _probe(path)
        total = len(info["pts"])
        count = min(max(1, int(request.query.get("count", 10))), STRIP_MAX_TILES, total)
        tile_h = min(max(16, int(request.query.get("height", 96))), 256)
        indices = [min(total - 1, int((j + 0.5) * total / count)) for j in range(count)]
        frames = _decode_frames(path, info, indices, max_side=tile_h * 4)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)
    tile_w = max(1, round(tile_h * frames[0].width / frames[0].height))
    sheet = Image.new("RGB", (tile_w * count, tile_h))
    for j, img in enumerate(frames):
        sheet.paste(img.resize((tile_w, tile_h), Image.BILINEAR), (j * tile_w, 0))
    return _jpeg(sheet, quality=80)


@routes.get("/video_trimmer/frame")
async def video_trimmer_frame(request):
    try:
        path = _resolve_input(request.query.get("filename", ""))
        info = _probe(path)
        index = min(max(1, int(request.query.get("frame", 1))), len(info["pts"])) - 1
        (img,) = _decode_frames(path, info, [index], max_side=PREVIEW_MAX_SIDE)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)
    return _jpeg(img)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class LoadVideoTrim:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {
            "required": {
                "file": (sorted(files), {"tooltip": "Video in ComfyUI's input folder. Use the node's Upload button or drop a file on the node."}),
                "start_frame": ("INT", {"default": 1, "min": 1, "max": 10_000_000,
                                        "tooltip": "First frame kept (1 = first frame of the file)."}),
                "end_frame": ("INT", {"default": 0, "min": 0, "max": 10_000_000,
                                      "tooltip": "Last frame kept, inclusive. 0 = last frame of the file."}),
                "keyframe": ("INT", {"default": 0, "min": 0, "max": 10_000_000,
                                     "tooltip": "Frame sent to the keyframe output, counted from 1 in the whole file; "
                                                "it must lie inside the trim. 0 = no keyframe set "
                                                "(the start frame is used)."}),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("video", "keyframe", "start_frame", "end_frame", "duration_trimmed", "duration_total",
                    "keyframe_index")
    OUTPUT_TOOLTIPS = (
        "The trimmed video, frames start_frame..end_frame.",
        "The frame under the keyframe marker (the start frame when no keyframe is set).",
        "First frame kept, counted from 1 in the whole file.",
        "Last frame kept, counted from 1 in the whole file.",
        "Number of frames in the trimmed video.",
        "Number of frames in the whole file.",
        "Position of the keyframe inside the trimmed video, counted from 0 "
        "(keyframe - start_frame). Plugs into a frame index of nodes that take the trimmed frames.",
    )
    FUNCTION = "load"
    CATEGORY = "video"
    DESCRIPTION = ("Load a video, trim it to a frame range on a timeline, and pick a keyframe. "
                   "Frame numbers start at 1; durations are in frames.")

    def load(self, file, start_frame, end_frame, keyframe):
        path = _resolve_input(file)
        info = _probe(path)
        total = len(info["pts"])
        start, end, key = _clamp_range(total, start_frame, end_frame, keyframe)
        if not start <= key <= end:
            raise ValueError(f"Keyframe {key} is outside the trimmed range {start}-{end}. "
                             f"Move the keyframe inside the trim, or the trim around it.")

        start_time, duration = _window(info, start - 1, end - 1)
        if start == 1 and end == total:
            video = VideoFromFile(path)
        else:
            video = VideoFromFile(path, start_time=start_time, duration=duration)

        key_start, key_duration = _window(info, key - 1, key - 1)
        key_images = VideoFromFile(path, start_time=key_start, duration=key_duration).get_components().images
        if key_images.shape[0] == 0:
            raise RuntimeError(f"Could not decode frame {key} of '{file}'")

        return (video, key_images[:1].contiguous(), start, end, end - start + 1, total, key - start)

    @classmethod
    def IS_CHANGED(cls, file, **kwargs):
        path = folder_paths.get_annotated_filepath(file)
        return os.path.getmtime(path) if os.path.exists(path) else ""

    @classmethod
    def VALIDATE_INPUTS(cls, file, **kwargs):
        if not folder_paths.exists_annotated_filepath(file):
            return f"Invalid video file: {file}"
        return True


NODE_CLASS_MAPPINGS = {"LoadVideoTrim": LoadVideoTrim}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadVideoTrim": "Load Video (Trim + Keyframe)"}
