# ComfyUI-VideoTrimmer

Frame-exact video nodes: a timeline trimmer with a keyframe picker, and scene nodes that
split a video at its cuts, route some scenes through a model and join them back.

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

## Scene nodes

Split a multi-scene video into one video per scene, choose which scenes to edit, edit
them with any video model and join everything back in order. Scene counts vary from
video to video; the graph doesn't have to: the scenes come out as a **list**, and every
ordinary node fed a list runs once per scene (async API nodes run the scenes
concurrently).

```
Load Video → Split Video by Scene ─┬─ scenes ──────────────────────────────────────────────┐
                                   ├→ filter LLM (per scene) ─→ Scene Gate (filter on/off)  │
                                   │                              │ selected scenes        │
                                   │      Scene Prompt Context ←──┘ + report               │
                                   │               ↓ one context per scene                 │
                                   │      prompt-writer LLM (per scene) → Scene Gate         │
                                   │                 (edit yes/no)        │ scenes to edit  │
                                   │      Get JSON Field (prompt) → video model (per scene) │
                                   │                                      ↓ replacements    │
                                   └───────────────────────────────→ Join Scenes → Save Video
```

`example_workflows/scene_edit_wan.json` is the general version. Its inputs and outputs are
[Workflow IO](https://github.com/aw93silverside/comfy-workflow-io-nodes) nodes, so it
doubles as an API: each input has a `param_name`, description, example, group and order.

| Input (`param_name`) | Type | |
|---|---|---|
| `video` | video | The video to edit |
| `filter` | bool | On: only scenes matching `filter_prompt` are edited. Off: every scene goes to the prompt writer and the filter LLM doesn't run |
| `filter_prompt` | text | What to look for: a product or brand, people, nature, furniture, anything visible |
| `edit_prompt` | text | What to change, once for the whole video. Can name scenes ("in scene 3...") or describe them |
| `reference_image` | image | What the edit refers to (e.g. the new product). Required: the IO image input can't be left empty |
| `detector`, `min_scene_length`, `resolution` | advanced | Scene detection and Wan's output resolution |

Outputs: `final_film` (mp4), the edited scenes (`scene_N`, one file per scene, named by
Scene Gate's `labels`) and the `contact_sheet`.

How it runs: the filter LLM (Gemini) watches each scene against `filter_prompt`. A second
Gemini node runs once per remaining scene. Scene Prompt Context gives it the request,
every scene's number, timing and description and which scene it's looking at; it also
sees the contact sheet and the reference image. It writes a self-contained Wan 3.0 prompt
(what changes, and everything that stays the same), or decides that this scene needs no
edit.

Wan 3.0 Reference to Video then edits each chosen scene (`@Video1` = the scene, `@Image1`
= the reference), with its duration set per scene by Video Duration, and Join Scenes puts
the results back. The **Review sheet** at the bottom shows every decision, its reason and
every Wan prompt in full, and saves the same page as Markdown in `output/scene_edit/`.

The filter is told to lean towards yes: a scene it isn't sure about passes with a lower
confidence, since a missed scene can't be recovered while an extra one only costs a writer
call (and the writer can still keep it as it is). A product doesn't need a readable label.

`example_workflows/scene_product_swap.json` is the earlier, simpler version: one LLM yes/no
per scene, then Seedance 2.5 edit or Wan 3.0 (`use_wan`) behind ComfyUI's If/Else Switch
(only the selected model runs; Pad Video brings scenes up to Seedance's 4 s minimum).
Inputs `video`, `product_image`, `use_wan`; outputs `final_film` and the swapped scenes.

Both need the Workflow IO pack. Any LLM node that returns text, and any video node that
takes a VIDEO and returns one, can stand in for Gemini and Wan.

### Split Video by Scene

| Input | Default | |
|---|---|---|
| `video` | | Any VIDEO. From Load Video (Trim), only the trimmed range is split |
| `detector` | `all` | `content`: hard cuts (HSV change). `adaptive`: changes much bigger than the frames around them, so fast pans don't cut. `fade`: one cut in the middle of a run of black frames. `all`: every detector; any one can call a cut, near-duplicates merge, a fade replaces the jumps at its edges |
| `content_threshold` | 27 | Mean HSV change (0-255) for a cut. Lower = more cuts |
| `adaptive_threshold` | 3.0 | How many times the surrounding change a cut must be |
| `fade_threshold` | 12 | Mean brightness (0-255) under which a frame is black |
| `min_scene_length` | 12 | Frames. Shorter scenes merge away; a jump that comes back to the previous picture within this many frames is a flash, not a cut |
| `trim_edges` | 0 | Frames dropped on each side of every cut (dissolve leftovers) |
| `analysis_size` | 256 | Long side in px that detection looks at; outputs stay full size |
| `cut_overrides` | | `+120, -340`: add a cut at frame 120, remove the one nearest 340 (source frames, from 1) |

| Output | |
|---|---|
| `scenes` | VIDEO list. Each scene is a frame-exact window into the source (nothing decoded at full size, nothing re-encoded), with its audio |
| `first_frames` | IMAGE list, each scene's first frame |
| `start_frames`, `end_frames` | INT lists, source frames counted from 1, inclusive |
| `scene_count` | INT |
| `cut_list` | JSON: scenes with frames, timecodes, seconds; cuts with strength and the detectors that found them |
| `contact_sheet` | IMAGE: every first frame in a grid, numbered from 1 (Scene 1 = list index 0) |

### Select Video from List

Takes any VIDEO list and an `index` (from 0, so index 0 = Scene 1; negative counts from the
end) and outputs that one video.

### Scene Gate

Selects scenes. With `filter` off it passes every scene on and never asks for its
`verdicts`, so the LLM behind them doesn't run. With `filter` on it keeps the scenes
whose verdict says yes: JSON like `{"match": true, "confidence": 0.9, "description": "..."}`
(the keys `selected`, `edit`, `product`... work too; LLM answers wrapped in prose or code
fences are fine) or an answer starting with yes/no. `min_confidence` turns weak yeses into
no; `if_unclear` sends, skips or stops on answers it can't read.

Outputs the `selected_scenes` (list), their `selected_indices` (positions in `scenes`,
list), `count`, a JSON `report` (every scene's number, length, verdict, confidence and
description), the selected scenes' `answers` (list) and `labels` ("scene_3"... per selected
scene: wire into a save node's filename so per-scene files don't overwrite each other).
With nothing selected, nothing after `selected_scenes` / `answers` / `labels` runs.

**Chaining:** a second gate takes the same `scenes`, the first gate's `selected_indices` as
`indices`, and verdicts for just those scenes. Its outputs are still positions in the full
scenes list, so they go straight into Join Scenes; an empty `indices` means nothing to judge.

### Scene Prompt Context

Builds the user message for an LLM that runs once per scene: the `request` (written once for
the whole video), every scene's number, start, length and description (from a gate's
`report`), which scenes are selected, and which scene this one is. One text per selected
scene, in the gate's order, so it pairs with that gate's `selected_scenes`.

### Scene Review Sheet

One readable page per run, for reviewing results and refining the LLM instructions: a
summary (what the first gate kept, what the second chose to edit, the instructions they
were given), then every scene with its timing, the first gate's yes/no, confidence,
description and reason, the second gate's decision and reason, and the generated prompt
(the `prompt` field of its answer) in full. It always runs, even when nothing was
selected, and saves the page as Markdown in the output folder (`filename_prefix`).

### Get JSON Field

Reads one field (`prompt` by default; `a.b` for nested keys) of the JSON object inside a
text, such as an LLM answer, and outputs it as text. Works per item on a list.

### Pad Video (repeat last frame)

For models that need a minimum clip length. A video shorter than `min_frames` (or
`min_seconds`, at its own frame rate; the longer of the two wins) gets its last frame
repeated up to that length, audio padded with silence. Longer videos pass through
untouched. Both 0 = off. Works on a list (each video padded on its own). Outputs the
video, `original_frames` and `added_frames`. Example: Seedance 2.5 edit takes clips of
4-30 s, so `min_seconds` 4 in front of it.

### Video Duration

Measures a video (or each video in a list): `frames`, `seconds`, `whole_seconds`
(rounded `up`, `nearest` or `down`, clamped by `min_whole_seconds` / `max_whole_seconds`,
0 = no clamp), the same value as text in `whole_seconds_choice` (a COMBO output, to drive a
model node's duration dropdown per scene) and `fps`. Rounding up means a model asked for
that many seconds covers every frame of the scene; Join Scenes drops the extra.

### Join Scenes

Takes the original `scenes`, the gate's `selected_indices` (input `product_indices`) and
the model's `replacements` (in the gate's order). `replacements` is a lazy input: it is
only evaluated when scenes were selected, so when nothing is selected the model never
runs and the film comes back untouched.

Each replacement fills exactly its scene's frame count, so the cuts stay where they
were. `fit` = `time` (default) takes the frames from the replacement's start at the
scene's timing: padding from Pad Video and any extra length are dropped, a different
frame rate is matched by time, and a replacement that comes back short holds its last
frame. `stretch` maps the replacement's whole length onto the scene. Replacements are
resized to the source's size (a different aspect ratio is stretched and noted in the
`report`). The source audio plays under every scene.

The film is written as lossless FFV1 + PCM to ComfyUI's temp folder; Save Video encodes
it once for delivery. Scenes that weren't replaced decode bit-identical to the source.

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this repo>
```

No extra Python dependencies. Needs a ComfyUI recent enough to have the core
**Trim Video** node (`VideoFromFile` with `start_time`/`duration`). Restart ComfyUI.
The scene nodes were tested on ComfyUI 0.41.
