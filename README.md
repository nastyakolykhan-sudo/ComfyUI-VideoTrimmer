# ComfyUI-VideoTrimmer

**Load Video (Trim + Keyframe)**: ComfyUI's Load Video with a timeline trimmer
and a keyframe picker built into the node.

- **Playhead** (white): click or drag on the strip to scrub through the video.
- **Trim handles** (yellow): drag either side. Within 10 px of the playhead or the
  keyframe they snap onto it, so you can park the playhead on the exact frame first
  and then pull the handle to it.
- **Add keyframe**: drops a second marker (cyan) at the playhead. Drag it by its
  diamond under the strip; it snaps to the playhead and the trim edges the same way.
  The button then becomes **Remove keyframe**. The keyframe always stays inside the
  trim: a trim handle dragged past it pushes it along.
- **Arrow keys** (click the timeline first): move the selected marker by 1 frame,
  Shift for 10. **Space** plays and pauses. **Play** loops the trimmed range.
- **Upload video**, or drop a video file on the node.
- The `start_frame`, `end_frame` and `keyframe` fields stay editable: type a number
  and the timeline follows.

Frame numbers start at **1**, as in the timeline. The range is inclusive.

## Outputs

| Output | Type | Value |
|---|---|---|
| `video` | VIDEO | The trimmed video (with its audio), frames `start_frame`..`end_frame` |
| `keyframe` | IMAGE | The frame under the keyframe marker. No keyframe set: the start frame |
| `start_frame` | INT | First frame kept |
| `end_frame` | INT | Last frame kept |
| `duration_trimmed` | INT | Frames in the trimmed video (`end_frame - start_frame + 1`) |
| `duration_total` | INT | Frames in the whole file |
| `keyframe_index` | INT | Position of the keyframe **inside the trimmed video**, counted from 0 (`keyframe - start_frame`; 0 when no keyframe is set) |

`keyframe_index` is what nodes working on the trimmed frames expect, e.g. SeC Video
Segmentation's `annotation_frame_idx`: with the trim at 35-204 and the keyframe on 120,
it is 85, the 86th frame of the trimmed video.

`end_frame = 0` means "to the last frame" and `keyframe = 0` means "no keyframe",
so the node also works from the API without the editor.

## How the trim stays frame-exact

The node reads the timestamp of every frame from the file (packets only, no decoding)
and hands ComfyUI's own `VideoFromFile` a time window that starts just after the start
frame's timestamp and ends on the next frame's timestamp. Decoding is ComfyUI's, so
the trimmed frames are identical to the same frames of the untrimmed video. Checked on
H.264 with B-frames, VP9 WebM, ProRes MOV and 29.97 fps.

The preview plays the file in the browser. When the browser can't decode the codec
(ProRes, for example), the preview switches to frames rendered by the server and
playback is turned off; scrubbing and trimming still work.

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this repo>
```

No extra Python dependencies. Needs a ComfyUI recent enough to have the core
**Trim Video** node (`VideoFromFile` with `start_time`/`duration`). Restart ComfyUI.
