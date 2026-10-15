import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "LoadVideoTrim";
const TRIM_COLOR = "#e8b923";
const PLAYHEAD_COLOR = "#ffffff";
const KEYFRAME_COLOR = "#22d3ee";
const SNAP_PX = 10;
const HANDLE_W = 9;
const PAD_X = HANDLE_W + 2;
const STRIP_TOP = 12;
const STRIP_H = 52;
const MARKER_ZONE = 12;
const RULER_H = 18;
const CANVAS_H = STRIP_TOP + STRIP_H + MARKER_ZONE + RULER_H;

const widget = (node, name) => node.widgets?.find((w) => w.name === name);
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), Math.max(lo, hi));

function el(tag, style = {}, props = {}) {
    const e = document.createElement(tag);
    Object.assign(e.style, style);
    return Object.assign(e, props);
}

function splitName(value) {
    const i = (value || "").lastIndexOf("/");
    return i < 0 ? { subfolder: "", filename: value || "" } : { subfolder: value.slice(0, i), filename: value.slice(i + 1) };
}

const BUTTON_STYLE = {
    background: "#2a2a2a", color: "#ddd", border: "1px solid #444", borderRadius: "6px",
    font: "12px sans-serif", padding: "4px 10px", cursor: "pointer", whiteSpace: "nowrap",
};

// Frames are numbered from 1, matching the Python node. The timeline places frame f
// at a point: first frame at the left edge, last frame at the right edge. The trim
// handles, the playhead and the keyframe all sit on those points, so a handle dragged
// within SNAP_PX of another marker lands on exactly the same frame.
class VideoTrimmer {
    constructor(node) {
        this.node = node;
        this.info = null;         // {frames, fps, width, height, times, duration}
        this.loadedFile = null;
        this.resetOnLoad = false;
        this.playhead = node.properties?.trim_playhead || 1;
        this.focus = "playhead";  // marker shown in the preview and moved by arrow keys
        this.drag = null;
        this.snapTarget = null;
        this.strip = null;
        this.stripKey = "";
        this.useFrameImages = false;
        this.playing = false;

        this.element = el("div", {
            display: "flex", flexDirection: "column", width: "100%", height: "100%", gap: "6px",
            font: "12px sans-serif", color: "#ccc", userSelect: "none",
        });

        this.header = el("div", { display: "flex", alignItems: "center", gap: "8px", minHeight: "24px" });
        this.title = el("div", { flex: "1 1 auto", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" });
        this.uploadButton = el("button", BUTTON_STYLE, { textContent: "Upload video" });
        this.fileInput = el("input", { display: "none" }, { type: "file", accept: "video/*,.mkv,.mov,.webm,.mp4" });
        this.header.append(this.title, this.uploadButton, this.fileInput);

        this.previewBox = el("div", {
            position: "relative", flex: "1 1 auto", minHeight: "80px", background: "#0b0b0b",
            borderRadius: "8px", overflow: "hidden",
        });
        this.video = el("video", { width: "100%", height: "100%", objectFit: "contain", display: "block" },
            { playsInline: true, preload: "auto" });
        this.frameImage = el("img", {
            position: "absolute", inset: "0", width: "100%", height: "100%", objectFit: "contain", display: "none",
        });
        this.frameLabel = el("div", {
            position: "absolute", left: "8px", bottom: "8px", padding: "3px 8px", borderRadius: "4px",
            background: "rgba(0,0,0,0.7)", font: "12px monospace", color: "#fff",
        });
        this.previewBox.append(this.video, this.frameImage, this.frameLabel);

        this.timelineRow = el("div", { display: "flex", alignItems: "flex-start", gap: "6px" });
        this.playButton = el("button", {
            ...BUTTON_STYLE, width: "40px", height: `${STRIP_H}px`, marginTop: `${STRIP_TOP}px`,
            padding: "0", fontSize: "16px",
        }, { textContent: "▶", title: "Play the trimmed range (loops)" });
        this.canvas = el("canvas", {
            flex: "1 1 auto", minWidth: "0", height: `${CANVAS_H}px`, touchAction: "none", outline: "none",
        }, { tabIndex: 0 });
        this.timelineRow.append(this.playButton, this.canvas);

        this.footer = el("div", { display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" });
        this.readout = el("div", { flex: "1 1 auto", font: "12px monospace", lineHeight: "1.6" });
        this.keyButton = el("button", BUTTON_STYLE, { textContent: "Add keyframe" });
        this.footer.append(this.readout, this.keyButton);

        this.element.append(this.header, this.previewBox, this.timelineRow, this.footer);

        this.canvas.addEventListener("pointerdown", (e) => this.onPointerDown(e));
        this.canvas.addEventListener("pointermove", (e) => this.onPointerMove(e));
        this.canvas.addEventListener("pointerup", (e) => this.onPointerUp(e));
        this.canvas.addEventListener("pointercancel", (e) => this.onPointerUp(e));
        this.canvas.addEventListener("keydown", (e) => this.onKeyDown(e));
        this.playButton.addEventListener("click", () => this.togglePlay());
        this.keyButton.addEventListener("click", () => this.toggleKeyframe());
        this.uploadButton.addEventListener("click", () => this.fileInput.click());
        this.fileInput.addEventListener("change", () => {
            if (this.fileInput.files?.[0]) this.upload(this.fileInput.files[0]);
            this.fileInput.value = "";
        });
        this.video.addEventListener("error", () => {
            if (!this.video.getAttribute("src")) return;
            this.useFrameImages = true;  // the browser can't play this codec: show server frames
            this.setPlayable(false);
            this.showPreview();
        });
        this.video.addEventListener("loadeddata", () => this.showPreview());
        new ResizeObserver(() => this.draw()).observe(this.canvas);
        this.updateReadout();
    }

    // --- values ---------------------------------------------------------------

    get total() { return this.info?.frames || 0; }
    get start() { return clamp(widget(this.node, "start_frame")?.value || 1, 1, this.total); }
    get end() {
        const v = widget(this.node, "end_frame")?.value || 0;
        return v <= 0 ? this.total : clamp(v, this.start, this.total);
    }
    get keyframe() {
        const v = widget(this.node, "keyframe")?.value || 0;
        return v <= 0 ? 0 : clamp(v, 1, this.total);
    }

    setWidget(name, value) {
        const w = widget(this.node, name);
        if (w && w.value !== value) w.value = value;
    }

    markerFrame(name) {
        return { playhead: this.playhead, start: this.start, end: this.end, keyframe: this.keyframe }[name];
    }

    setMarker(name, frame) {
        const f = clamp(Math.round(frame), 1, this.total);
        if (name === "playhead") {
            this.playhead = f;
            this.node.properties.trim_playhead = f;
        } else if (name === "start") {
            this.setWidget("start_frame", Math.min(f, this.end));
        } else if (name === "end") {
            this.setWidget("end_frame", Math.max(f, this.start));
        } else if (name === "keyframe") {
            this.setWidget("keyframe", f);
        }
        this.changed();
    }

    changed() {
        this.updateReadout();
        this.draw();
        this.showPreview();
        this.node.setDirtyCanvas(true, true);
    }

    // Typed values in the number widgets: keep them inside the video.
    normalizeWidgets() {
        if (!this.total) return;
        const start = clamp(widget(this.node, "start_frame").value, 1, this.total);
        this.setWidget("start_frame", start);
        const end = widget(this.node, "end_frame").value;
        this.setWidget("end_frame", end <= 0 ? this.total : clamp(end, start, this.total));
        const key = widget(this.node, "keyframe").value;
        this.setWidget("keyframe", key <= 0 ? 0 : clamp(key, 1, this.total));
    }

    // --- loading --------------------------------------------------------------

    async load(fileValue) {
        this.pause();
        this.info = null;
        this.strip = null;
        this.stripKey = "";
        this.useFrameImages = false;
        this.setPlayable(true);
        this.loadedFile = fileValue;
        this.video.removeAttribute("src");
        this.frameImage.style.display = "none";
        this.video.style.display = "block";
        if (!fileValue) {
            this.video.load();
            this.changed();
            return;
        }
        this.title.textContent = "Loading…";
        let info;
        try {
            const res = await api.fetchApi(`/video_trimmer/info?${new URLSearchParams({ filename: fileValue })}`);
            info = await res.json();
            if (!res.ok) throw new Error(info.error || res.statusText);
        } catch (err) {
            if (this.loadedFile !== fileValue) return;
            this.title.textContent = `Can't read video: ${err.message}`;
            this.draw();
            return;
        }
        if (this.loadedFile !== fileValue) return;  // a newer file was picked meanwhile
        this.info = info;
        if (this.resetOnLoad) {
            this.resetOnLoad = false;
            this.setWidget("start_frame", 1);
            this.setWidget("end_frame", info.frames);
            this.setWidget("keyframe", 0);
            this.playhead = 1;
        } else {
            this.normalizeWidgets();
            this.playhead = clamp(this.playhead, 1, info.frames);
        }
        this.node.properties.trim_playhead = this.playhead;
        const { filename, subfolder } = splitName(fileValue);
        this.video.src = api.apiURL(`/view?${new URLSearchParams({ filename, subfolder, type: "input" })}`);
        this.changed();
    }

    async upload(file) {
        const body = new FormData();
        body.append("image", file);
        body.append("type", "input");
        this.title.textContent = `Uploading ${file.name}…`;
        try {
            const res = await api.fetchApi("/upload/image", { method: "POST", body });
            if (!res.ok) throw new Error(res.statusText);
            const data = await res.json();
            const value = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
            const fileWidget = widget(this.node, "file");
            const values = fileWidget.options.values;
            if (Array.isArray(values) && !values.includes(value)) values.push(value);
            fileWidget.value = value;
            this.resetOnLoad = true;
            this.load(value);
        } catch (err) {
            this.title.textContent = `Upload failed: ${err.message}`;
        }
    }

    // --- preview --------------------------------------------------------------

    frameTime(f) {
        const t = this.info.times;
        const i = f - 1;
        const next = i + 1 < t.length ? t[i + 1] : this.info.duration;
        return (t[i] + next) / 2;  // middle of the frame's display interval
    }

    timeToFrame(time) {
        const t = this.info.times;
        let lo = 0, hi = t.length - 1;
        while (lo < hi) {
            const mid = (lo + hi + 1) >> 1;
            if (t[mid] <= time + 1e-6) lo = mid; else hi = mid - 1;
        }
        return lo + 1;
    }

    showPreview() {
        if (!this.info) {
            this.frameLabel.style.display = "none";
            return;
        }
        const name = this.focus === "keyframe" && !this.keyframe ? "playhead" : this.focus;
        const f = this.markerFrame(name);
        const label = { playhead: "Frame", start: "Start", end: "End", keyframe: "Keyframe" }[name];
        this.frameLabel.style.display = "block";
        this.frameLabel.textContent = `${label} ${f}`;
        this.frameLabel.style.color = name === "keyframe" ? KEYFRAME_COLOR : name === "playhead" ? "#fff" : TRIM_COLOR;
        if (this.playing) return;
        if (this.useFrameImages) {
            this.video.style.display = "none";
            this.frameImage.style.display = "block";
            const src = api.apiURL(`/video_trimmer/frame?${new URLSearchParams({ filename: this.loadedFile, frame: f })}`);
            if (this.frameImage.dataset.src !== src) {
                this.frameImage.dataset.src = src;
                this.frameImage.src = src;
            }
        } else if (this.video.readyState >= 1) {
            this.video.currentTime = this.frameTime(f);
        }
    }

    setPlayable(playable) {
        this.playButton.disabled = !playable;
        this.playButton.style.opacity = playable ? "1" : "0.35";
        this.playButton.title = playable
            ? "Play the trimmed range (loops)"
            : "This browser can't play the file's codec: frames come from the server, playback is off";
    }

    togglePlay() {
        if (this.playing) {
            this.pause();
            return;
        }
        if (!this.info || this.useFrameImages) return;
        if (this.playhead < this.start || this.playhead >= this.end) this.playhead = this.start;
        this.focus = "playhead";
        this.video.currentTime = this.frameTime(this.playhead);
        this.playing = true;
        this.playButton.textContent = "❚❚";
        this.video.play().catch(() => this.pause());
        const tick = () => {
            if (!this.playing) return;
            const f = this.timeToFrame(this.video.currentTime);
            if (f > this.end || this.video.ended) {
                this.video.currentTime = this.frameTime(this.start);
                if (this.video.paused) this.video.play().catch(() => this.pause());
                this.playhead = this.start;
            } else {
                this.playhead = Math.max(f, this.start);
            }
            this.updateReadout();
            this.draw();
            this.frameLabel.textContent = `Frame ${this.playhead}`;
            requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    }

    pause() {
        if (!this.playing) return;
        this.playing = false;
        this.video.pause();
        this.playButton.textContent = "▶";
        this.node.properties.trim_playhead = this.playhead;
        this.showPreview();
    }

    toggleKeyframe() {
        if (!this.info) return;
        if (this.keyframe) {
            this.setWidget("keyframe", 0);
            if (this.focus === "keyframe") this.focus = "playhead";
        } else {
            this.setWidget("keyframe", this.playhead);  // added at the playhead, then drag it
            this.focus = "keyframe";
        }
        this.changed();
    }

    updateReadout() {
        this.keyButton.textContent = this.keyframe ? "Remove keyframe" : "Add keyframe";
        this.keyButton.style.borderColor = this.keyframe ? "#444" : KEYFRAME_COLOR;
        if (!this.info) {
            this.readout.textContent = "";
            if (!this.loadedFile) this.title.textContent = "No video loaded";
            return;
        }
        const fps = this.info.fps;
        const fpsText = Number.isInteger(fps) ? fps : fps.toFixed(3);
        const n = this.end - this.start + 1;
        this.title.innerHTML = "";
        this.title.append(
            el("span", { color: "#eee" }, { textContent: `Your video has ${this.total} frames` }),
            el("span", { color: "#777" }, { textContent: `  ·  ${this.info.width}×${this.info.height} @ ${fpsText}fps` }),
        );
        this.readout.innerHTML = "";
        const item = (label, value, color) => el("span", { display: "inline-block", marginRight: "14px", whiteSpace: "nowrap" }, {
            innerHTML: `<span style="color:#888">${label}</span> <span style="color:${color}">${value}</span>`,
        });
        this.readout.append(
            item("Start Frame:", this.start, "#fff"),
            item("End Frame:", this.end, "#fff"),
            item("Duration:", `${n} Frames <span style="color:#777">(${(n / fps).toFixed(1)}s @ ${fpsText}fps)</span>`, "#fff"),
            item("Keyframe:", this.keyframe || "none", this.keyframe ? KEYFRAME_COLOR : "#666"),
        );
    }

    // --- timeline geometry ----------------------------------------------------

    get trackWidth() { return Math.max(1, this.canvas.clientWidth - 2 * PAD_X); }

    frameToX(f) {
        return this.total > 1 ? PAD_X + ((f - 1) / (this.total - 1)) * this.trackWidth : PAD_X + this.trackWidth / 2;
    }

    xToFrame(x) {
        return this.total > 1 ? clamp(Math.round(1 + ((x - PAD_X) / this.trackWidth) * (this.total - 1)), 1, this.total) : 1;
    }

    pointer(e) {
        const r = this.canvas.getBoundingClientRect();
        // DOM widgets are CSS-scaled with the graph zoom; convert to unscaled canvas px.
        // Grab and snap distances stay constant on screen at any zoom.
        this.zoom = r.width / this.canvas.clientWidth || 1;
        return [(e.clientX - r.left) / this.zoom, (e.clientY - r.top) / this.zoom];
    }

    hitTest(x, y) {
        const z = Math.min(1, this.zoom || 1);
        const near = (name, px) => Math.abs(x - this.frameToX(this.markerFrame(name))) <= px / z;
        const stripBottom = STRIP_TOP + STRIP_H;
        if (y < STRIP_TOP && near("playhead", 9)) return "playhead";
        if (this.keyframe && y > stripBottom - 4 && y < stripBottom + MARKER_ZONE && near("keyframe", 9)) return "keyframe";
        if (y >= STRIP_TOP - 2 && y <= stripBottom + 2) {
            const xs = this.frameToX(this.start), xe = this.frameToX(this.end);
            const ds = x <= xs + 3 ? Math.abs(x - (xs - HANDLE_W / 2)) : Infinity;
            const de = x >= xe - 3 ? Math.abs(x - (xe + HANDLE_W / 2)) : Infinity;
            if (Math.min(ds, de) <= HANDLE_W / z) return ds <= de ? "start" : "end";
        }
        if (near("playhead", 4)) return "playhead";
        if (this.keyframe && near("keyframe", 4)) return "keyframe";
        return null;
    }

    snap(name, x) {
        let frame = this.xToFrame(x);
        this.snapTarget = null;
        let best = SNAP_PX / (this.zoom || 1);
        for (const other of ["playhead", "keyframe", "start", "end"]) {
            if (other === name || (other === "keyframe" && !this.keyframe)) continue;
            if ((name === "start" && other === "end") || (name === "end" && other === "start")) continue;
            const f = this.markerFrame(other);
            const d = Math.abs(this.frameToX(f) - x);
            if (d <= best) {
                best = d;
                frame = f;
                this.snapTarget = other;
            }
        }
        return frame;
    }

    // --- interaction ----------------------------------------------------------

    onPointerDown(e) {
        if (!this.info || e.button !== 0) return;
        e.preventDefault();
        e.stopPropagation();
        this.canvas.focus();
        this.pause();
        const [x, y] = this.pointer(e);
        let name = this.hitTest(x, y);
        if (!name) {
            name = "playhead";  // click on the strip: jump the playhead there
            this.setMarker("playhead", this.xToFrame(x));
        }
        this.drag = { name, grab: x - this.frameToX(this.markerFrame(name)) };
        this.focus = name;
        this.canvas.setPointerCapture(e.pointerId);
        this.changed();
    }

    onPointerMove(e) {
        if (!this.info) return;
        const [x, y] = this.pointer(e);
        if (!this.drag) {
            const hit = this.hitTest(x, y);
            this.canvas.style.cursor = hit === "start" || hit === "end" ? "ew-resize" : hit ? "grab" : "pointer";
            return;
        }
        e.preventDefault();
        e.stopPropagation();
        this.canvas.style.cursor = "grabbing";
        this.setMarker(this.drag.name, this.snap(this.drag.name, x - this.drag.grab));
    }

    onPointerUp(e) {
        if (!this.drag) return;
        this.canvas.releasePointerCapture?.(e.pointerId);
        this.drag = null;
        this.snapTarget = null;
        this.draw();
    }

    onKeyDown(e) {
        if (!this.info) return;
        const step = { ArrowLeft: -1, ArrowRight: 1 }[e.key];
        if (e.key === " ") {
            this.togglePlay();
        } else if (step) {
            this.pause();
            const name = this.focus === "keyframe" && !this.keyframe ? "playhead" : this.focus;
            this.setMarker(name, this.markerFrame(name) + step * (e.shiftKey ? 10 : 1));
        } else {
            return;
        }
        e.preventDefault();
        e.stopPropagation();
    }

    // --- drawing --------------------------------------------------------------

    loadStrip() {
        const tileH = Math.round(STRIP_H * (window.devicePixelRatio || 1));
        const aspect = this.info.width && this.info.height ? this.info.width / this.info.height : 16 / 9;
        const count = clamp(Math.round(this.trackWidth / (STRIP_H * aspect)), 3, 40);
        const key = `${this.loadedFile}|${count}|${tileH}`;
        if (key === this.stripKey) return;
        this.stripKey = key;
        const img = new Image();
        img.onload = () => {
            if (this.stripKey !== key) return;
            this.strip = { img, count };
            this.draw();
        };
        img.src = api.apiURL(`/video_trimmer/strip?${new URLSearchParams({ filename: this.loadedFile, count, height: tileH })}`);
    }

    draw() {
        const c = this.canvas;
        const dpr = window.devicePixelRatio || 1;
        const W = c.clientWidth, H = CANVAS_H;
        if (!W) return;
        if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) {
            c.width = Math.round(W * dpr);
            c.height = Math.round(H * dpr);
        }
        const ctx = c.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, W, H);

        const x0 = PAD_X, x1 = PAD_X + this.trackWidth, top = STRIP_TOP, bottom = STRIP_TOP + STRIP_H;
        ctx.save();
        ctx.beginPath();
        ctx.roundRect(x0, top, x1 - x0, STRIP_H, 6);
        ctx.clip();
        ctx.fillStyle = "#1c1c1c";
        ctx.fillRect(x0, top, x1 - x0, STRIP_H);
        if (!this.info) {
            ctx.restore();
            return;
        }
        this.loadStrip();
        if (this.strip) {
            const { img, count } = this.strip;
            const tw = img.width / count, cell = (x1 - x0) / count;
            const scale = Math.max(cell / tw, STRIP_H / img.height);
            const sw = cell / scale, sh = STRIP_H / scale;
            for (let j = 0; j < count; j++) {
                ctx.drawImage(img, j * tw + (tw - sw) / 2, (img.height - sh) / 2, sw, sh, x0 + j * cell, top, cell + 0.5, STRIP_H);
            }
        }
        const xs = this.frameToX(this.start), xe = this.frameToX(this.end);
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(x0, top, xs - x0, STRIP_H);
        ctx.fillRect(xe, top, x1 - xe, STRIP_H);
        ctx.restore();

        // Trim box with a handle on each side.
        ctx.strokeStyle = TRIM_COLOR;
        ctx.lineWidth = 3;
        ctx.strokeRect(xs, top + 1.5, xe - xs, STRIP_H - 3);
        for (const [side, x] of [["start", xs - HANDLE_W], ["end", xe]]) {
            ctx.fillStyle = TRIM_COLOR;
            ctx.beginPath();
            ctx.roundRect(x, top, HANDLE_W, STRIP_H, side === "start" ? [5, 0, 0, 5] : [0, 5, 5, 0]);
            ctx.fill();
            ctx.fillStyle = "rgba(0,0,0,0.45)";
            ctx.fillRect(x + HANDLE_W / 2 - 1, top + STRIP_H / 2 - 8, 2, 16);
            if (this.snapTarget && this.drag?.name === side) {
                ctx.strokeStyle = "#fff";
                ctx.lineWidth = 1.5;
                ctx.strokeRect(x - 1, top - 1, HANDLE_W + 2, STRIP_H + 2);
            }
        }

        // Keyframe: cyan line with a diamond under the strip.
        if (this.keyframe) {
            const xk = this.frameToX(this.keyframe);
            ctx.fillStyle = KEYFRAME_COLOR;
            ctx.fillRect(xk - 1, top, 2, STRIP_H);
            ctx.beginPath();
            ctx.moveTo(xk, bottom + 1);
            ctx.lineTo(xk + 6, bottom + 7);
            ctx.lineTo(xk, bottom + 13);
            ctx.lineTo(xk - 6, bottom + 7);
            ctx.closePath();
            ctx.fill();
            if (this.focus === "keyframe" || this.snapTarget === "keyframe") {
                ctx.strokeStyle = "#fff";
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        }

        // Playhead: white line with a triangle on top.
        const xp = this.frameToX(this.playhead);
        ctx.fillStyle = PLAYHEAD_COLOR;
        ctx.fillRect(xp - 1, top - 2, 2, STRIP_H + 4);
        ctx.beginPath();
        ctx.moveTo(xp - 7, 0);
        ctx.lineTo(xp + 7, 0);
        ctx.lineTo(xp, top);
        ctx.closePath();
        ctx.fill();
        if (this.snapTarget === "playhead") {
            ctx.strokeStyle = TRIM_COLOR;
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        // Ruler: frame numbers at round steps, starting from frame 1 like the trimmer readout.
        const minGap = 44;
        const steps = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000];
        const pxPerFrame = this.total > 1 ? this.trackWidth / (this.total - 1) : this.trackWidth;
        const step = steps.find((s) => s * pxPerFrame >= minGap) || steps[steps.length - 1];
        ctx.fillStyle = "#888";
        ctx.font = "10px sans-serif";
        ctx.textBaseline = "bottom";
        const labelY = H - 2;
        const lastX = this.frameToX(this.total);
        for (let f = 1; f <= this.total; f += step) {
            const x = this.frameToX(f);
            if (f !== 1 && lastX - x < minGap * 0.8) break;  // leave room for the last frame's label
            ctx.textAlign = f === 1 ? "left" : "center";
            ctx.fillText(String(f), f === 1 ? x - 2 : x, labelY);
        }
        if (this.total > 1) {
            ctx.textAlign = "right";
            ctx.fillText(String(this.total), lastX + 2, labelY);
        }
    }
}

app.registerExtension({
    name: "VideoTrimmer.LoadVideoTrim",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            this.properties = this.properties || {};
            const trimmer = new VideoTrimmer(this);
            this.videoTrimmer = trimmer;

            const fileWidget = widget(this, "file");
            if (fileWidget) {
                const callback = fileWidget.callback;
                fileWidget.callback = function () {
                    const r = callback?.apply(this, arguments);
                    if (trimmer.loadedFile !== fileWidget.value) {
                        trimmer.resetOnLoad = true;
                        trimmer.load(fileWidget.value);
                    }
                    return r;
                };
            }
            for (const name of ["start_frame", "end_frame", "keyframe"]) {
                const w = widget(this, name);
                if (!w) continue;
                const callback = w.callback;
                w.callback = function () {
                    const r = callback?.apply(this, arguments);
                    trimmer.normalizeWidgets();
                    if (name !== "keyframe" || w.value) trimmer.focus = name === "start_frame" ? "start" : name === "end_frame" ? "end" : "keyframe";
                    trimmer.changed();
                    return r;
                };
            }

            this.addDOMWidget("video_trimmer", "video_trimmer", trimmer.element, {
                serialize: false,
                hideOnZoom: false,
                getMinHeight: () => 380,
            });
            this.setSize([Math.max(this.size[0], 520), Math.max(this.size[1], 760)]);
            // A new node has no saved state: load the default file once widgets settle.
            requestAnimationFrame(() => {
                if (!trimmer.loadedFile && fileWidget?.value) {
                    trimmer.resetOnLoad = true;
                    trimmer.load(fileWidget.value);
                }
            });
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            const trimmer = this.videoTrimmer;
            if (trimmer) {
                trimmer.playhead = this.properties?.trim_playhead || 1;
                trimmer.resetOnLoad = false;
                trimmer.load(widget(this, "file")?.value);
            }
            return r;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.videoTrimmer?.pause();
            return onRemoved?.apply(this, arguments);
        };

        // Drop a video file on the node to upload it.
        nodeType.prototype.onDragOver = function (e) {
            return !!e?.dataTransfer?.types?.includes?.("Files");
        };
        nodeType.prototype.onDragDrop = function (e) {
            const file = [...(e?.dataTransfer?.files || [])].find((f) => f.type.startsWith("video/") || /\.(mkv|mov|webm|mp4|avi|m4v)$/i.test(f.name));
            if (!file) return false;
            this.videoTrimmer?.upload(file);
            return true;
        };
    },
});
