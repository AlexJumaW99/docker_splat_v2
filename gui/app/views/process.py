"""
Process view — runs ns-process-data inside splat-ns.
"""
from __future__ import annotations

import shlex
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt, QObject, QRunnable, QThreadPool, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QStackedWidget, QVBoxLayout,
    QWidget,
)

from app import docker_service as dock
from app import warehouse as wh
from app.config import (
    CONTAINER_NAME, DATA_DIR, VIZER_CONTAINER, VIZER_VIEWER_PORT, WAREHOUSE_DIR,
)
from app.ffprobe import VideoInfo, probe
from app.jobs import JobManager, JobSpec
from app.manifest import ProcessParams, SourceManifest
from app.store import (
    ALLOWED_IMAGE_EXTS, INPUT_IMAGES_DIR, Project, ProjectStore,
    convert_heic_images, count_heic_images, count_images, source_kind,
)
from app.widgets import HelpBadge, ProjectBar, card, labeled, pill, section_title, set_pill


# ---------- help copy ----------
# Kept together so reviewers can audit the wording in one place. Each entry is
# (title, html-body) and describes what the setting does + how its range
# affects the downstream ns-process-data / COLMAP / ns-train pipeline.

_HELP = {
    "video": (
        "Source video",
        "The raw footage ns-process-data will extract frames from.<br><br>"
        "<b>Must live inside</b> the <code>data/</code> folder — that's the only "
        "path bind-mounted into the container. Steady handheld or drone orbits "
        "with plenty of overlap give the best reconstructions; fast shakes or "
        "rolling-shutter artifacts weaken COLMAP's feature matching."
    ),
    "images_dir": (
        "Source image folder",
        "A directory of still images COLMAP will calibrate cameras for.<br><br>"
        "<b>Convention</b>: drop folders under <code>data/input_images/</code>. "
        "Any path inside <code>data/</code> works (the container needs to see "
        "it), but keeping image sets in <code>input_images/</code> matches "
        "the rest of the docs.<br><br>"
        "Accepted extensions: <code>.jpg .jpeg .png .tif .tiff .cr2</code> "
        "(case-insensitive). For best results use 30+ overlapping views of a "
        "static scene with consistent lighting; sparse coverage or motion "
        "between shots will crater COLMAP's pose recovery."
    ),
    "eval_dir": (
        "Eval image folder (optional)",
        "A second folder of images held out from training and used to "
        "compute PSNR / SSIM on unseen views. Must live inside "
        "<code>data/</code>. Leave empty to let ns-process-data carve a "
        "split out of the main folder instead."
    ),
    "next_target": (
        "Next-run output folder",
        "Every “Run processing” writes to a fresh, never-overwritten "
        "folder under <code>data/processed/</code>. The first run for a "
        "project lands at <code>&lt;project&gt;_splat/</code>; subsequent "
        "runs auto-suffix <code>_2</code>, <code>_3</code>, … so old "
        "datasets stay intact and any training runs already pointing at "
        "them keep working.<br><br>"
        "After a successful run, the project is repointed at the new folder "
        "so the Train tab picks it up automatically."
    ),
    "preset": (
        "Parameter presets",
        "Save the current form values under a name, then recall them later for "
        "this same project. Useful when you're iterating on a single scene and "
        "want to try two or three parameter combinations without re-typing them. "
        "Switching presets overwrites all fields on the form."
    ),
    "frames": (
        "Target frames",
        "How many frames ns-process-data will evenly sample from the video.<br><br>"
        "<b>50–150</b> · fastest COLMAP, OK for short orbits with wide overlap.<br>"
        "<b>200–300</b> · balanced default for most handheld / drone clips.<br>"
        "<b>300–500</b> · denser coverage, slower COLMAP, better for long paths "
        "or fine detail.<br><br>"
        "More frames → longer feature-matching + bundle-adjust, but more "
        "supervision during training."
    ),
    "matching": (
        "Matching method",
        "How COLMAP decides which frame pairs to feature-match.<br><br>"
        "<b>sequential</b> · matches each frame to its neighbors in time. "
        "Fastest, assumes a roughly ordered video — correct default for "
        "continuous captures.<br>"
        "<b>vocab_tree</b> · uses a pretrained visual vocabulary to find loop "
        "closures. Slower, but recovers when the video revisits the same "
        "viewpoint.<br>"
        "<b>exhaustive</b> · matches every pair. Only use on tiny captures "
        "(&lt;100 frames); cost is O(N²)."
    ),
    "downscales": (
        "Downscale levels",
        "Number of half-resolution mipmap levels Nerfstudio writes alongside "
        "the full-size images.<br><br>"
        "<b>0</b> · full resolution only (fastest processing).<br>"
        "<b>3</b> · default — 1/2, 1/4, 1/8 levels available for multi-scale "
        "training.<br>"
        "<b>5</b> · adds 1/16, 1/32 — useful if you plan to train at "
        "deliberately low resolution to save memory."
    ),
    "verbose": (
        "Verbose output",
        "Enables <code>--verbose</code> on ns-process-data. Prints COLMAP's "
        "per-stage progress and the full feature-matching / mapper output to "
        "the log drawer. Leave on while you're still dialing in a scene; "
        "turn off once the pipeline is reliable."
    ),
    "camera_type": (
        "Camera type",
        "Lens model COLMAP calibrates against.<br><br>"
        "<b>perspective</b> · default pinhole + radial distortion. Use for "
        "most phone / DSLR / drone footage.<br>"
        "<b>fisheye</b> · wide-FOV lenses (GoPro, action cams).<br>"
        "<b>equirectangular</b> · 360° video; samples virtual perspective "
        "views from each frame.<br>"
        "<b>pinhole / simple_pinhole</b> · simpler calibrations when "
        "distortion is already removed."
    ),
    "crop": (
        "Crop factor (Top Bottom Left Right)",
        "Fraction of each edge to trim from every frame before COLMAP sees "
        "it. Values are 0.00–1.00.<br><br>"
        "<b>0.00</b> · no cropping (default).<br>"
        "<b>0.05–0.15</b> · removes dashboards, rig, timestamp overlays, or "
        "rolling-shutter artifacts at the edges.<br>"
        "<b>&gt; 0.3</b> · rarely useful; you'll throw away most of the image."
    ),
    "radius_crop": (
        "Circle-mask radius",
        "Fraction of the frame diagonal kept inside a circular mask. "
        "Pixels outside the circle are ignored.<br><br>"
        "<b>1.00</b> · no mask (default).<br>"
        "<b>0.85–0.95</b> · hides the vignetted / blurred corners typical of "
        "fisheye or ultra-wide lenses.<br>"
        "<b>&lt; 0.7</b> · very aggressive; keep only the sharp center."
    ),
    "equirect_samples": (
        "Frames per equirectangular view",
        "For 360° video, how many virtual perspective crops to sample from "
        "each source frame. ns-process-data only accepts 8 or 14.<br><br>"
        "<b>8</b> · 8 views per frame — default, lower feature-match cost.<br>"
        "<b>14</b> · denser sampling — stronger reconstruction, slower COLMAP."
    ),
    "refine_intrinsics": (
        "Refine intrinsics",
        "Lets COLMAP's bundle adjuster tweak focal length and distortion "
        "during optimization. <b>Recommended on</b> for any uncalibrated "
        "camera. Turn off only when you've measured the intrinsics yourself "
        "and want COLMAP to trust them verbatim."
    ),
    "same_dimensions": (
        "All frames same dimensions",
        "Tells COLMAP every extracted image shares the same width/height, "
        "so it can skip per-image metadata reads during feature matching. "
        "Leave on for video-sourced datasets — ns-process-data resizes every "
        "frame identically. Turn off only if you're feeding a mixed folder."
    ),
    "no_gpu": (
        "Disable GPU SIFT",
        "Forces COLMAP to run feature extraction on the CPU. A diagnostic "
        "switch — <b>much slower</b> (often 10–30×). Only enable to rule out "
        "a CUDA / driver issue when GPU SIFT crashes or produces empty "
        "feature sets."
    ),
    "seed": (
        "Train frame-selection seed",
        "Integer seed for the random subsample that picks train frames out "
        "of the video. <b>Leave as 'random'</b> for normal use. Set a fixed "
        "value when you need bit-for-bit reproducibility between runs on the "
        "same video."
    ),
    "eval_seed": (
        "Eval frame-selection seed",
        "Seed for the held-out eval split. Same guidance as the train seed: "
        "fix only when you need identical train/eval splits across runs."
    ),
    "use_sfm_depth": (
        "Export SfM depth maps",
        "Writes COLMAP's sparse 3D points as per-frame depth maps next to "
        "the images. Adds 10–30 s per frame during processing and extra disk "
        "space. Useful when you plan to train with a depth-supervised "
        "method; ignored by vanilla splatfacto."
    ),
    "eval_video": (
        "Eval video (optional)",
        "A second video processed alongside the main one and held out from "
        "training — used to compute PSNR / SSIM on unseen viewpoints. "
        "Must live inside <code>data/</code>. Leave empty to let "
        "ns-process-data pick eval frames from the main video instead."
    ),
    "advanced_mode": (
        "Advanced mode",
        "Expands the form to show COLMAP-level controls: camera model, "
        "edge / circle cropping, equirect samples, intrinsic refinement, "
        "GPU-SIFT toggles, seeds, SfM depth, and the raw-CLI-flags escape "
        "hatch. Leave off for typical video captures — the defaults are "
        "already tuned for handheld / drone footage."
    ),
    "extra_flags_proc": (
        "Raw CLI flags",
        "Appended verbatim to <code>ns-process-data</code> after every "
        "structured flag above. Use it when the GUI doesn't expose a flag "
        "you need. Anything you type here overrides the fields above if it "
        "names the same flag. Inspect the full option list with:<br>"
        "<code>docker exec splat-ns-v2 ns-process-data video --help</code>"
    ),
    "output": (
        "Output dataset",
        "Where the processed dataset lands on the host. This is automatically "
        "placed under <code>data/processed/&lt;project&gt;_splat/</code>. "
        "The container sees it as <code>/workspace/processed/…</code>; "
        "ns-train will read from the container-side path."
    ),
}


def _hb(key: str) -> HelpBadge:
    title, body = _HELP[key]
    return HelpBadge(title, body)


# ---------- helpers ----------

def _host_to_container(host_path: str | Path) -> str:
    p = Path(host_path).resolve()
    rel = p.relative_to(DATA_DIR.resolve())
    return f"/workspace/{rel.as_posix()}"


# Matching methods that ns-process-data video actually accepts. "spatial" was
# in this list historically but tyro rejects it ("invalid choice"); leaving it
# selectable would silently break a saved preset.
MATCH_METHODS = ["sequential", "vocab_tree", "exhaustive"]

# Camera models supported by ns-process-data --camera-type. Order matches the
# CLI help so users see the same dropdown order they'd find in the docs.
CAMERA_TYPES = ["perspective", "fisheye", "equirectangular", "pinhole", "simple_pinhole"]

# Only meaningful when --camera-type=equirectangular. ns-process-data only
# accepts these two values.
EQUIRECT_SAMPLES = [8, 14]

# Sentinel for the seed SpinBoxes — Qt renders this value as "random" (via
# setSpecialValueText) and we treat it as "don't emit a --random-seed flag".
SEED_RANDOM = -1


# ---------- async ffprobe ----------

class _ProbeSignals(QObject):
    done = Signal(object)   # VideoInfo | None


class _ProbeTask(QRunnable):
    def __init__(self, path: Path, sig: _ProbeSignals):
        super().__init__()
        self._path = path
        self._sig = sig
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        self._sig.done.emit(probe(self._path))


# ---------- async vizer status ----------

class _VizerStatusSignals(QObject):
    done = Signal(str)


class _VizerStatusTask(QRunnable):
    def __init__(self, signals: _VizerStatusSignals):
        super().__init__()
        self.setAutoDelete(True)
        self._signals = signals

    @Slot()
    def run(self) -> None:
        state = dock.container_state(VIZER_CONTAINER)
        try:
            self._signals.done.emit(state.status)
        except RuntimeError:
            pass


# ---------- empty state ----------

class _EmptyState(QWidget):
    def __init__(self, store: ProjectStore):
        super().__init__()
        self.setObjectName("Canvas")
        v = QVBoxLayout(self)
        v.setContentsMargins(40, 80, 40, 40)
        v.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        v.setSpacing(10)

        title = QLabel("No project yet")
        title.setObjectName("PageTitle")
        title.setAlignment(Qt.AlignCenter)

        sub = QLabel(
            "A project is one video + its processed dataset + its runs. "
            "Create one to start."
        )
        sub.setProperty("role", "caption")
        sub.setWordWrap(True)
        sub.setAlignment(Qt.AlignCenter)
        sub.setMaximumWidth(420)

        from app.widgets import NewProjectDialog
        btn = QPushButton("+  New project")
        btn.setProperty("role", "primary")
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda: NewProjectDialog(store, self).exec())

        v.addWidget(title)
        v.addWidget(sub, 0, Qt.AlignHCenter)
        v.addSpacing(16)
        v.addWidget(btn, 0, Qt.AlignHCenter)
        v.addStretch(1)


# ---------- imported state (no video to process) ----------

class _ImportedState(QWidget):
    """Active project was created from an already-processed folder, so there
    is nothing for this tab to do. Point the user at Train."""

    request_train = Signal()

    def __init__(self, project: Project):
        super().__init__()
        self.setObjectName("Canvas")
        v = QVBoxLayout(self)
        v.setContentsMargins(40, 80, 40, 40)
        v.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        v.setSpacing(10)

        title = QLabel("Imported dataset")
        title.setObjectName("PageTitle")
        title.setAlignment(Qt.AlignCenter)

        fmt = "COLMAP" if project.dataparser == "colmap" else "Nerfstudio"
        sub = QLabel(
            f"'{project.name}' was imported as a {fmt} dataset from\n"
            f"{project.processed_dir}\n\n"
            "There's no video to process — go straight to Train."
        )
        sub.setProperty("role", "caption")
        sub.setWordWrap(True)
        sub.setAlignment(Qt.AlignCenter)
        sub.setMaximumWidth(560)

        btn = QPushButton("Go to Train →")
        btn.setProperty("role", "primary")
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(self.request_train.emit)

        v.addWidget(title)
        v.addWidget(sub, 0, Qt.AlignHCenter)
        v.addSpacing(16)
        v.addWidget(btn, 0, Qt.AlignHCenter)
        v.addStretch(1)


# ---------- form ----------

class _Form(QWidget):
    """The per-project form. Rebuilt when the active project changes."""

    request_train = Signal()  # user clicked 'Go to Train'

    def __init__(self, store: ProjectStore, jobs: JobManager, project: Project):
        super().__init__()
        self.setObjectName("Canvas")
        self._store = store
        self._jobs = jobs
        self._project = project
        # _mode locks the form to either the video or images flow at
        # construction time. ProcessView._rebuild() is the only thing that
        # creates _Form instances and it gates on source_kind, so this
        # narrowing matches the caller's contract — there's no "imported"
        # case to handle here.
        kind = source_kind(project)
        assert kind in ("video", "images"), f"_Form built for kind={kind}"
        self._mode = kind
        # Output folder the next "Run processing" click will write into.
        # Now a ULID-keyed dir under WAREHOUSE_DIR. Stashed between _on_run
        # and _on_job_finished so the success handler knows which folder
        # to repoint the project at and where to write manifest.json.
        self._pending_target: Path | None = None
        # Run-scoped state captured at _on_run time and consumed by the
        # success branch of _on_job_finished to build the ProcessManifest.
        # All None outside of an active run.
        self._pending_source_id: str | None = None
        self._pending_run_id: str | None = None
        self._pending_params: ProcessParams | None = None
        self._pending_command: str | None = None
        self._pending_start_time: datetime | None = None
        self._pending_display_name: str | None = None
        self._running_mine = False
        # Async ffprobe is video-only; the images flow does a synchronous
        # iterdir() in _refresh_source_meta() and doesn't need a worker.
        self._probe_signals = _ProbeSignals()
        self._probe_signals.done.connect(self._on_probe_done)
        self._vizer_signals = _VizerStatusSignals()
        self._vizer_signals.done.connect(self._on_vizer_status)
        self._vizer_poll_pool = QThreadPool(self)
        self._vizer_poll_pool.setMaxThreadCount(1)

        self._build()
        self._load_project()
        self._wire_jobs()

        self._vizer_timer = QTimer(self)
        self._vizer_timer.setInterval(3000)
        self._vizer_timer.timeout.connect(self._poll_vizer)
        self._vizer_timer.start()
        self._poll_vizer()

        # Catch up if a job is already running (e.g. user switched projects
        # mid-run) so our UI reflects reality immediately.
        if self._jobs.is_running() and self._jobs.current is not None:
            self._on_job_started(self._jobs.current)

    # ---- build ----
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        body = QWidget(objectName="Canvas")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(32, 24, 32, 24)
        bl.setSpacing(14)

        # Header
        header_row = QHBoxLayout()
        title = QLabel("Process")
        title.setObjectName("PageTitle")
        sub_text = (
            "Prepare a video for splatfacto training."
            if self._mode == "video"
            else "Prepare a folder of images for splatfacto training."
        )
        sub = QLabel(sub_text)
        sub.setProperty("role", "caption")

        self.advanced_cb = QCheckBox("Advanced mode")
        self.advanced_cb.toggled.connect(self._on_advanced)

        adv_row = QHBoxLayout(); adv_row.setSpacing(6); adv_row.setContentsMargins(0, 0, 0, 0)
        adv_row.addWidget(self.advanced_cb)
        adv_row.addWidget(_hb("advanced_mode"))
        adv_wrap = QWidget(); adv_wrap.setLayout(adv_row)

        header_col = QVBoxLayout(); header_col.setSpacing(2)
        header_col.addWidget(title); header_col.addWidget(sub)
        header_row.addLayout(header_col, 1)
        header_row.addWidget(adv_wrap, 0, Qt.AlignBottom | Qt.AlignRight)
        bl.addLayout(header_row)

        bl.addWidget(self._source_card())
        bl.addWidget(self._params_card())
        bl.addWidget(self._output_card())
        bl.addWidget(self._actions_card())

        bl.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll)

    # ---- source card ----
    def _source_card(self) -> QFrame:
        """Pick-the-source card. Video mode shows a file picker + ffprobe
        readout; images mode shows a folder picker + image-count readout.
        Both modes share the same widget names (source_path_edit /
        source_meta_lbl / source_browse_btn) so downstream methods don't
        need to branch."""
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(10)
        v.addWidget(section_title("VIDEO" if self._mode == "video" else "IMAGE FOLDER"))

        row = QHBoxLayout(); row.setSpacing(8)
        self.source_path_edit = QLineEdit()
        self.source_path_edit.setReadOnly(True)
        self.source_path_edit.setProperty("role", "readout")
        row.addWidget(_hb("video" if self._mode == "video" else "images_dir"))
        self.source_browse_btn = QPushButton("Browse…")
        self.source_browse_btn.setProperty("role", "secondary")
        self.source_browse_btn.setCursor(Qt.PointingHandCursor)
        self.source_browse_btn.clicked.connect(self._on_browse_source)
        row.addWidget(self.source_path_edit, 1)
        row.addWidget(self.source_browse_btn)
        v.addLayout(row)

        self.source_meta_lbl = QLabel("")
        self.source_meta_lbl.setProperty("role", "caption")
        v.addWidget(self.source_meta_lbl)
        return c

    # ---- params card ----
    def _params_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(12)

        top = QHBoxLayout()
        top.addWidget(section_title("PARAMETERS"))
        top.addStretch(1)

        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(180)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)

        self.save_preset_btn = QPushButton("Save as preset…")
        self.save_preset_btn.setProperty("role", "ghost")
        self.save_preset_btn.setCursor(Qt.PointingHandCursor)
        self.save_preset_btn.clicked.connect(self._on_save_preset)

        top.addWidget(QLabel("Preset"))
        top.addWidget(_hb("preset"))
        top.addWidget(self.preset_combo)
        top.addWidget(self.save_preset_btn)
        v.addLayout(top)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14); grid.setVerticalSpacing(12)
        grid.setColumnStretch(1, 1)

        # Target frames — video only. ns-process-data images has no
        # equivalent flag (every input image is already a frame), so we
        # build the widgets unconditionally for preset round-tripping but
        # hide the row for images projects.
        self.frames_label = labeled("Target frames", *_HELP["frames"])
        grid.addWidget(self.frames_label, 0, 0)
        frames_row = QHBoxLayout(); frames_row.setSpacing(10)
        self.frames_slider = QSlider(Qt.Horizontal)
        self.frames_slider.setRange(50, 500); self.frames_slider.setSingleStep(10)
        self.frames_slider.setPageStep(25)
        self.frames_value = QSpinBox()
        self.frames_value.setRange(50, 500); self.frames_value.setSingleStep(10)
        self.frames_value.setMinimumWidth(80)
        self.frames_slider.valueChanged.connect(self.frames_value.setValue)
        self.frames_value.valueChanged.connect(self.frames_slider.setValue)
        frames_row.addWidget(self.frames_slider, 1)
        frames_row.addWidget(self.frames_value)
        self.frames_row_wrap = QWidget(); self.frames_row_wrap.setLayout(frames_row)
        grid.addWidget(self.frames_row_wrap, 0, 1)
        if self._mode == "images":
            self.frames_label.hide()
            self.frames_row_wrap.hide()

        # Matching method
        grid.addWidget(labeled("Matching method", *_HELP["matching"]), 1, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(MATCH_METHODS)
        grid.addWidget(self.method_combo, 1, 1)

        # Downscale levels
        grid.addWidget(labeled("Downscale levels", *_HELP["downscales"]), 2, 0)
        self.downscales_spin = QSpinBox()
        self.downscales_spin.setRange(0, 5)
        self.downscales_spin.setMaximumWidth(80)
        ds_row = QHBoxLayout(); ds_row.addWidget(self.downscales_spin); ds_row.addStretch(1)
        ds_wrap = QWidget(); ds_wrap.setLayout(ds_row)
        grid.addWidget(ds_wrap, 2, 1)

        # Verbose
        verbose_row = QHBoxLayout(); verbose_row.setSpacing(6); verbose_row.setContentsMargins(0, 0, 0, 0)
        self.verbose_cb = QCheckBox("Verbose output")
        verbose_row.addWidget(self.verbose_cb)
        verbose_row.addWidget(_hb("verbose"))
        verbose_row.addStretch(1)
        vw = QWidget(); vw.setLayout(verbose_row)
        grid.addWidget(vw, 3, 1)

        v.addLayout(grid)

        # Advanced panel — three structured sub-cards plus a raw-flags fallback.
        # Hidden by default; the "Advanced mode" header checkbox toggles it.
        self.advanced_wrap = QFrame()
        self.advanced_wrap.setProperty("role", "subcard")
        av = QVBoxLayout(self.advanced_wrap)
        av.setContentsMargins(0, 10, 0, 0)
        av.setSpacing(14)

        av.addWidget(self._adv_camera_subcard())
        av.addWidget(self._adv_colmap_subcard())
        av.addWidget(self._adv_extras_subcard())
        av.addWidget(self._adv_raw_subcard())

        self.advanced_wrap.hide()
        v.addWidget(self.advanced_wrap)
        return c

    # ---- advanced sub-cards ----

    def _adv_camera_subcard(self) -> QFrame:
        sc = QFrame()
        sc.setProperty("role", "subcard")
        sv = QVBoxLayout(sc)
        sv.setContentsMargins(14, 12, 14, 14); sv.setSpacing(10)
        sv.addWidget(section_title("CAMERA & CROPPING"))

        g = QGridLayout()
        g.setHorizontalSpacing(14); g.setVerticalSpacing(10)
        g.setColumnStretch(1, 1)

        # Camera type
        g.addWidget(labeled("Camera type", *_HELP["camera_type"]), 0, 0)
        self.camera_type_combo = QComboBox()
        self.camera_type_combo.addItems(CAMERA_TYPES)
        self.camera_type_combo.currentTextChanged.connect(self._on_camera_type_changed)
        g.addWidget(self.camera_type_combo, 0, 1)

        # Crop factor (top / bottom / left / right)
        g.addWidget(labeled("Crop (T B L R)", *_HELP["crop"]), 1, 0)
        crop_row = QHBoxLayout(); crop_row.setSpacing(6)
        self.crop_spins = []
        for _ in range(4):
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 1.0); sp.setSingleStep(0.05); sp.setDecimals(2)
            sp.setMaximumWidth(80)
            self.crop_spins.append(sp)
            crop_row.addWidget(sp)
        crop_row.addStretch(1)
        crop_wrap = QWidget(); crop_wrap.setLayout(crop_row)
        g.addWidget(crop_wrap, 1, 1)

        # Circle-mask radius
        g.addWidget(labeled("Circle-mask radius", *_HELP["radius_crop"]), 2, 0)
        self.radius_crop_spin = QDoubleSpinBox()
        self.radius_crop_spin.setRange(0.0, 1.0)
        self.radius_crop_spin.setSingleStep(0.05); self.radius_crop_spin.setDecimals(2)
        self.radius_crop_spin.setMaximumWidth(100)
        rc_row = QHBoxLayout(); rc_row.addWidget(self.radius_crop_spin); rc_row.addStretch(1)
        rc_wrap = QWidget(); rc_wrap.setLayout(rc_row)
        g.addWidget(rc_wrap, 2, 1)

        # Equirect samples — visible only for camera-type=equirectangular.
        self.equirect_lbl = labeled("Frames per equirect view", *_HELP["equirect_samples"])
        self.equirect_combo = QComboBox()
        self.equirect_combo.addItems([str(n) for n in EQUIRECT_SAMPLES])
        self.equirect_lbl.hide(); self.equirect_combo.hide()
        g.addWidget(self.equirect_lbl, 3, 0)
        g.addWidget(self.equirect_combo, 3, 1)

        sv.addLayout(g)

        hint = QLabel(
            "Crop trims pixels (0–1 fractions of each edge). Circle mask is "
            "useful for fisheye lenses that vignette the frame corners."
        )
        hint.setProperty("role", "caption")
        hint.setWordWrap(True)
        sv.addWidget(hint)
        return sc

    def _adv_colmap_subcard(self) -> QFrame:
        sc = QFrame()
        sc.setProperty("role", "subcard")
        sv = QVBoxLayout(sc)
        sv.setContentsMargins(14, 12, 14, 14); sv.setSpacing(10)
        sv.addWidget(section_title("COLMAP TUNING"))

        g = QGridLayout()
        g.setHorizontalSpacing(14); g.setVerticalSpacing(10)
        g.setColumnStretch(1, 1)

        def _cb_row(cb: QCheckBox, help_key: str) -> QWidget:
            r = QHBoxLayout(); r.setSpacing(6); r.setContentsMargins(0, 0, 0, 0)
            r.addWidget(cb); r.addWidget(_hb(help_key)); r.addStretch(1)
            w = QWidget(); w.setLayout(r); return w

        self.refine_intrinsics_cb = QCheckBox("Refine intrinsics (recommended)")
        g.addWidget(_cb_row(self.refine_intrinsics_cb, "refine_intrinsics"), 0, 0, 1, 2)

        self.same_dimensions_cb = QCheckBox("All frames are same dimensions (faster)")
        g.addWidget(_cb_row(self.same_dimensions_cb, "same_dimensions"), 1, 0, 1, 2)

        self.no_gpu_cb = QCheckBox("Disable GPU SIFT (diagnostic — much slower)")
        g.addWidget(_cb_row(self.no_gpu_cb, "no_gpu"), 2, 0, 1, 2)

        # Reproducibility seeds. -1 displays as "random" via setSpecialValueText
        # and is the signal that we shouldn't emit the corresponding flag.
        # ns-process-data images doesn't subsample frames (every input file
        # is already a frame), so the seed flags don't apply — we still
        # build the widgets for preset round-tripping but hide the rows.
        self.seed_label = labeled("Train frame-selection seed", *_HELP["seed"])
        g.addWidget(self.seed_label, 3, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(SEED_RANDOM, 2_147_483_647)
        self.seed_spin.setSpecialValueText("random")
        self.seed_spin.setMaximumWidth(140)
        seed_row = QHBoxLayout(); seed_row.addWidget(self.seed_spin); seed_row.addStretch(1)
        self.seed_wrap = QWidget(); self.seed_wrap.setLayout(seed_row)
        g.addWidget(self.seed_wrap, 3, 1)

        self.eval_seed_label = labeled("Eval frame-selection seed", *_HELP["eval_seed"])
        g.addWidget(self.eval_seed_label, 4, 0)
        self.eval_seed_spin = QSpinBox()
        self.eval_seed_spin.setRange(SEED_RANDOM, 2_147_483_647)
        self.eval_seed_spin.setSpecialValueText("random")
        self.eval_seed_spin.setMaximumWidth(140)
        es_row = QHBoxLayout(); es_row.addWidget(self.eval_seed_spin); es_row.addStretch(1)
        self.eval_seed_wrap = QWidget(); self.eval_seed_wrap.setLayout(es_row)
        g.addWidget(self.eval_seed_wrap, 4, 1)

        if self._mode == "images":
            self.seed_label.hide()
            self.seed_wrap.hide()
            self.eval_seed_label.hide()
            self.eval_seed_wrap.hide()

        sv.addLayout(g)
        return sc

    def _adv_extras_subcard(self) -> QFrame:
        sc = QFrame()
        sc.setProperty("role", "subcard")
        sv = QVBoxLayout(sc)
        sv.setContentsMargins(14, 12, 14, 14); sv.setSpacing(10)
        sv.addWidget(section_title("EXTRAS"))

        sfm_row = QHBoxLayout(); sfm_row.setSpacing(6); sfm_row.setContentsMargins(0, 0, 0, 0)
        self.use_sfm_depth_cb = QCheckBox("Export SfM depth maps")
        sfm_row.addWidget(self.use_sfm_depth_cb); sfm_row.addWidget(_hb("use_sfm_depth")); sfm_row.addStretch(1)
        sfm_wrap = QWidget(); sfm_wrap.setLayout(sfm_row)
        sv.addWidget(sfm_wrap)

        # Eval-data picker (optional). Stored as host path; converted at run
        # time. ns-process-data's --eval-data accepts a file in video mode and
        # a directory in images mode — same flag, different validator. The
        # widget name is generic (eval_data_*) and the browse handler
        # dispatches based on self._mode.
        eval_help_key = "eval_video" if self._mode == "video" else "eval_dir"
        eval_label = (
            "Eval video (optional)" if self._mode == "video"
            else "Eval folder (optional)"
        )
        sv.addWidget(labeled(eval_label, *_HELP[eval_help_key]))
        eval_row = QHBoxLayout(); eval_row.setSpacing(8)
        self.eval_data_edit = QLineEdit()
        self.eval_data_edit.setReadOnly(True)
        self.eval_data_edit.setProperty("role", "readout")
        self.eval_data_edit.setPlaceholderText("(none)")
        self.eval_data_browse = QPushButton("Browse…")
        self.eval_data_browse.setProperty("role", "secondary")
        self.eval_data_browse.setCursor(Qt.PointingHandCursor)
        self.eval_data_browse.clicked.connect(self._on_browse_eval_data)
        self.eval_data_clear = QPushButton("Clear")
        self.eval_data_clear.setProperty("role", "ghost")
        self.eval_data_clear.setCursor(Qt.PointingHandCursor)
        self.eval_data_clear.clicked.connect(lambda: self.eval_data_edit.setText(""))
        eval_row.addWidget(self.eval_data_edit, 1)
        eval_row.addWidget(self.eval_data_browse)
        eval_row.addWidget(self.eval_data_clear)
        sv.addLayout(eval_row)

        hint_text = (
            "Eval video, if set, is processed alongside the train video and "
            "held out from training. Must live inside the data/ bind mount."
            if self._mode == "video"
            else "Eval folder, if set, is processed alongside the train "
                 "images and held out from training. Must live inside the "
                 "data/ bind mount."
        )
        hint = QLabel(hint_text)
        hint.setProperty("role", "caption")
        hint.setWordWrap(True)
        sv.addWidget(hint)
        return sc

    def _adv_raw_subcard(self) -> QFrame:
        sc = QFrame()
        sc.setProperty("role", "subcard")
        sv = QVBoxLayout(sc)
        sv.setContentsMargins(14, 12, 14, 14); sv.setSpacing(8)
        raw_hdr = QHBoxLayout(); raw_hdr.setSpacing(6); raw_hdr.setContentsMargins(0, 0, 0, 0)
        raw_hdr.addWidget(section_title("RAW CLI FLAGS"))
        raw_hdr.addWidget(_hb("extra_flags_proc"))
        raw_hdr.addStretch(1)
        raw_wrap = QWidget(); raw_wrap.setLayout(raw_hdr)
        sv.addWidget(raw_wrap)

        self.extra_flags_edit = QLineEdit()
        self.extra_flags_edit.setPlaceholderText(
            "--skip-image-processing  …"
        )
        sv.addWidget(self.extra_flags_edit)

        hint = QLabel(
            "Appended verbatim to ns-process-data after the structured flags above. "
            "See: docker exec splat-ns-v2 ns-process-data video --help"
        )
        hint.setProperty("role", "caption")
        hint.setWordWrap(True)
        sv.addWidget(hint)
        return sc

    # ---- conditional visibility ----

    def _on_camera_type_changed(self, value: str) -> None:
        is_equirect = (value == "equirectangular")
        self.equirect_lbl.setVisible(is_equirect)
        self.equirect_combo.setVisible(is_equirect)


    def _on_browse_eval_data(self) -> None:
        """Eval-data browse dispatcher.

        ns-process-data's --eval-data flag accepts a file when the parent
        subcommand is `video` and a directory when it's `images`. We pick
        the right Qt dialog and validation rule for the current mode.
        """
        if self._mode == "video":
            path, _ = QFileDialog.getOpenFileName(
                self, "Pick an eval video",
                str(DATA_DIR),
                "Video files (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;All files (*.*)",
            )
            if not path:
                return
            try:
                Path(path).resolve().relative_to(DATA_DIR.resolve())
            except ValueError:
                QMessageBox.warning(
                    self, "Eval video outside data/",
                    f"Eval videos must live inside {DATA_DIR} so the container can see them."
                )
                return
            self.eval_data_edit.setText(path)
        else:
            INPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            path = QFileDialog.getExistingDirectory(
                self, "Pick an eval image folder",
                str(INPUT_IMAGES_DIR),
                QFileDialog.ShowDirsOnly,
            )
            if not path:
                return
            try:
                Path(path).resolve().relative_to(DATA_DIR.resolve())
            except ValueError:
                QMessageBox.warning(
                    self, "Eval folder outside data/",
                    f"Eval folders must live inside {DATA_DIR} so the container can see them."
                )
                return
            eval_path = Path(path)
            if count_heic_images(eval_path):
                convert_heic_images(eval_path)
            n, _ = count_images(eval_path)
            if n == 0:
                allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTS))
                QMessageBox.warning(
                    self, "No images in folder",
                    f"Folder contains no supported images.\nAccepted: {allowed}"
                )
                return
            self.eval_data_edit.setText(path)

    # ---- output card ----
    def _output_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(8)
        out_hdr = QHBoxLayout(); out_hdr.setSpacing(6); out_hdr.setContentsMargins(0, 0, 0, 0)
        out_hdr.addWidget(section_title("OUTPUT DATASET"))
        out_hdr.addWidget(_hb("next_target"))
        out_hdr.addStretch(1)
        out_wrap = QWidget(); out_wrap.setLayout(out_hdr)
        v.addWidget(out_wrap)
        self.output_lbl = QLineEdit()
        self.output_lbl.setReadOnly(True)
        self.output_lbl.setProperty("role", "readout")
        v.addWidget(self.output_lbl)
        # Caption updates via _refresh_next_target(): explains whether the
        # next Run will write into the existing processed_dir or auto-suffix
        # to a new sibling. Visible at all times so users see the policy.
        self.next_target_lbl = QLabel("")
        self.next_target_lbl.setProperty("role", "caption")
        self.next_target_lbl.setWordWrap(True)
        v.addWidget(self.next_target_lbl)
        hint = QLabel(
            "Host path. Container sees this as /workspace/processed/… "
            "automatically. Every run lands in a fresh folder; existing "
            "datasets are never overwritten."
        )
        hint.setProperty("role", "caption")
        hint.setWordWrap(True)
        v.addWidget(hint)
        return c

    # ---- next-run target preview ----
    def _refresh_next_target(self) -> None:
        """Update the caption that previews where the next Run will land.

        Output dirs are now ULID-keyed under WAREHOUSE_DIR/processed/, so
        we can't show a literal next target (ULIDs are minted at click
        time). The caption tells the user the policy and, when applicable,
        which prior dataset the project is currently pointing at."""
        current = self._project.processed_dir
        if current:
            existing = Path(current).name
            self.next_target_lbl.setText(
                f"Next run will write to {WAREHOUSE_DIR.name}/processed/<new ULID>/  "
                f"(current dataset: '{existing}')."
            )
        else:
            self.next_target_lbl.setText(
                f"Next run will write to {WAREHOUSE_DIR.name}/processed/<new ULID>/."
            )

    # ---- actions ----
    def _actions_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(10)

        row = QHBoxLayout(); row.setSpacing(10)
        self.run_btn = QPushButton("Run processing")
        self.run_btn.setProperty("role", "primary")
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self._on_run)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setProperty("role", "secondary")
        self.cancel_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setEnabled(False)

        self.status_pill = pill("Ready", "neutral")

        row.addWidget(self.run_btn)
        row.addWidget(self.cancel_btn)
        row.addStretch(1)
        row.addWidget(self.status_pill)

        self.ready_to_train = QFrame()
        self.ready_to_train.setProperty("role", "subcard")
        rtt = QHBoxLayout(self.ready_to_train)
        rtt.setContentsMargins(14, 10, 14, 10)
        ok = QLabel("✓ Processed dataset ready.")
        ok.setStyleSheet("color: #22C55E; font-weight: 600;")
        train_btn = QPushButton("Go to Train →")
        train_btn.setProperty("role", "primary")
        train_btn.setCursor(Qt.PointingHandCursor)
        train_btn.clicked.connect(self.request_train.emit)
        rtt.addWidget(ok)
        rtt.addStretch(1)
        rtt.addWidget(train_btn)
        self.ready_to_train.hide()

        # Splinter warning — populated by _check_sparse_splinter() on success.
        # COLMAP's incremental mapper occasionally produces multiple disjoint
        # submaps (sparse/0, sparse/1, …); ns-process-data always exports from
        # sparse/0, so if that one is the tiny dangling submap the dataset
        # silently ends up with only a handful of frames. We surface that here.
        self.splinter_warn = QFrame()
        self.splinter_warn.setProperty("role", "subcard")
        sw = QVBoxLayout(self.splinter_warn)
        sw.setContentsMargins(14, 10, 14, 10); sw.setSpacing(4)
        self.splinter_title = QLabel("Warning: COLMAP split into multiple submaps")
        self.splinter_title.setStyleSheet("color: #F59E0B; font-weight: 600;")
        self.splinter_body = QLabel("")
        self.splinter_body.setWordWrap(True)
        self.splinter_body.setProperty("role", "caption")
        self.splinter_body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sw.addWidget(self.splinter_title)
        sw.addWidget(self.splinter_body)
        self.splinter_warn.hide()

        v.addLayout(row)

        vizer_row = QHBoxLayout(); vizer_row.setSpacing(10)
        self.vizer_btn = QPushButton("Watch live in vizer")
        self.vizer_btn.setProperty("role", "secondary")
        self.vizer_btn.setCursor(Qt.PointingHandCursor)
        self.vizer_btn.clicked.connect(self._on_open_vizer)
        self.vizer_pill = pill("—", "neutral")
        vizer_row.addWidget(self.vizer_btn)
        vizer_row.addStretch(1)
        vizer_row.addWidget(self.vizer_pill)
        v.addLayout(vizer_row)

        v.addWidget(self.ready_to_train)
        v.addWidget(self.splinter_warn)
        return c

    # ---- advanced toggle ----
    def _on_advanced(self, checked: bool) -> None:
        self.advanced_wrap.setVisible(checked)

    # ---- project lifecycle ----
    def _load_project(self) -> None:
        p = self._project
        if self._mode == "video":
            assert p.video_path is not None  # narrowed by source_kind
            self.source_path_edit.setText(p.video_path)
        else:
            assert p.images_dir is not None
            self.source_path_edit.setText(p.images_dir)
        self.output_lbl.setText(p.processed_dir)
        # Source-meta readout (ffprobe for video, image count for images).
        # Video uses an async worker because ffprobe hits the disk + spawns
        # a subprocess; images is just iterdir() and doesn't need a thread.
        if self._mode == "video":
            self._probe_video(Path(p.video_path))
        else:
            self._refresh_images_meta(Path(p.images_dir))
        # Output card shows the host path of the next-run target so the
        # auto-suffix is visible BEFORE the user clicks Run, not just after.
        self._refresh_next_target()

        # Populate form from last params or sensible defaults.
        self._apply_params(p.last_process_params or {})

        self._refresh_presets()

    def _apply_params(self, params: dict) -> None:
        """Populate every widget from a params dict. Missing keys fall back to
        the same defaults ns-process-data would use, so legacy projects (saved
        before the advanced fields existed) load unchanged."""
        # --- basic ---
        self.frames_slider.setValue(int(params.get("num_frames_target", 250)))
        self.frames_value.setValue(int(params.get("num_frames_target", 250)))
        # ns-process-data's own --matching-method default differs per
        # subcommand: vocab_tree for images, and historically sequential for
        # video (continuous orbits match neighbors well). We follow upstream.
        default_method = "vocab_tree" if self._mode == "images" else "sequential"
        method = params.get("matching_method", default_method)
        idx = MATCH_METHODS.index(method) if method in MATCH_METHODS else 0
        self.method_combo.setCurrentIndex(idx)
        self.downscales_spin.setValue(int(params.get("num_downscales", 3)))
        self.verbose_cb.setChecked(bool(params.get("verbose", True)))
        self.extra_flags_edit.setText(params.get("extra_flags", ""))

        # --- camera & cropping ---
        camera_type = params.get("camera_type", "perspective")
        if camera_type in CAMERA_TYPES:
            self.camera_type_combo.setCurrentIndex(CAMERA_TYPES.index(camera_type))
        self._on_camera_type_changed(self.camera_type_combo.currentText())

        crop_keys = ("crop_top", "crop_bottom", "crop_left", "crop_right")
        for spin, key in zip(self.crop_spins, crop_keys):
            spin.setValue(float(params.get(key, 0.0)))
        self.radius_crop_spin.setValue(float(params.get("percent_radius_crop", 1.0)))

        eq = int(params.get("images_per_equirect", 8))
        if eq in EQUIRECT_SAMPLES:
            self.equirect_combo.setCurrentIndex(EQUIRECT_SAMPLES.index(eq))

        # --- COLMAP tuning ---
        self.refine_intrinsics_cb.setChecked(bool(params.get("refine_intrinsics", True)))
        self.same_dimensions_cb.setChecked(bool(params.get("same_dimensions", True)))
        # gpu_enabled is the user-facing concept; the CLI flag is --no-gpu.
        gpu_enabled = bool(params.get("gpu_enabled", True))
        self.no_gpu_cb.setChecked(not gpu_enabled)

        seed = params.get("random_seed", None)
        self.seed_spin.setValue(int(seed) if seed is not None else SEED_RANDOM)
        eval_seed = params.get("eval_random_seed", None)
        self.eval_seed_spin.setValue(int(eval_seed) if eval_seed is not None else SEED_RANDOM)

        # --- extras ---
        self.use_sfm_depth_cb.setChecked(bool(params.get("use_sfm_depth", False)))
        self.eval_data_edit.setText(params.get("eval_data") or "")

    def _refresh_presets(self) -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("— custom —", userData=None)
        for name in sorted(self._project.presets_process.keys()):
            self.preset_combo.addItem(name, userData=name)
        self.preset_combo.blockSignals(False)

    def _on_preset_selected(self, idx: int) -> None:
        name = self.preset_combo.itemData(idx)
        if not name:
            return
        preset = self._project.presets_process.get(name, {})
        if preset:
            self._apply_params(preset)

    def _on_save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        self._store.save_preset(self._project.id, "process", name, self._collect_params())
        self._refresh_presets()
        idx = self.preset_combo.findData(name)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)

    # ---- source path: browse + meta readouts ----
    def _on_browse_source(self) -> None:
        """Dispatcher — the source_browse_btn handler. Mode picks file vs
        folder picker so we don't end up with two buttons in the same row."""
        if self._mode == "video":
            self._on_browse_video()
        else:
            self._on_browse_images()

    def _on_browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick a video",
            str(DATA_DIR),
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;All files (*.*)",
        )
        if not path:
            return
        try:
            self._store.update_video_path(self._project.id, Path(path))
        except ValueError as e:
            QMessageBox.warning(self, "Video outside data/", str(e))
            return
        self.source_path_edit.setText(path)
        self._probe_video(Path(path))

    def _on_browse_images(self) -> None:
        # Default the picker to data/input_images/ — auto-create on first use
        # so the dialog has somewhere sensible to land, matching the New
        # Project dialog's behavior at widgets.py:_browse_images_folder.
        INPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        path = QFileDialog.getExistingDirectory(
            self, "Pick a folder of images",
            str(INPUT_IMAGES_DIR),
            QFileDialog.ShowDirsOnly,
        )
        if not path:
            return
        try:
            self._store.update_images_dir(self._project.id, Path(path))
        except ValueError as e:
            QMessageBox.warning(self, "Can't use folder", str(e))
            return
        self.source_path_edit.setText(path)
        self._refresh_images_meta(Path(path))

    def _probe_video(self, path: Path) -> None:
        self.source_meta_lbl.setText("reading video metadata…")
        QThreadPool.globalInstance().start(_ProbeTask(path, self._probe_signals))

    def _on_probe_done(self, info: VideoInfo | None) -> None:
        if info is None:
            self.source_meta_lbl.setText("video not found")
            return
        self.source_meta_lbl.setText(info.pretty())

    def _refresh_images_meta(self, path: Path) -> None:
        """Synchronous image-folder readout. Cheap enough to run on the GUI
        thread for any folder size we'd realistically ns-process-data on
        (a few thousand files at most)."""
        if not path.is_dir():
            self.source_meta_lbl.setText("folder not found")
            return
        n, exts = count_images(path)
        n_heic = count_heic_images(path)
        if n == 0 and n_heic == 0:
            allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTS))
            self.source_meta_lbl.setText(f"no supported images (need: {allowed})")
            return
        parts: list[str] = []
        if n:
            ext_summary = ", ".join(sorted(exts))
            parts.append(f"{n} image(s) · {ext_summary}")
        if n_heic:
            parts.append(f"{n_heic} HEIC → will auto-convert to JPEG")
        self.source_meta_lbl.setText("  ·  ".join(parts))

    # ---- collect + run ----
    def _collect_params(self) -> dict:
        seed_v = self.seed_spin.value()
        eval_seed_v = self.eval_seed_spin.value()
        return {
            # basic
            "num_frames_target": self.frames_value.value(),
            "matching_method": self.method_combo.currentText(),
            "num_downscales": self.downscales_spin.value(),
            "verbose": self.verbose_cb.isChecked(),
            "extra_flags": self.extra_flags_edit.text().strip(),
            # camera & cropping
            "camera_type": self.camera_type_combo.currentText(),
            "crop_top": self.crop_spins[0].value(),
            "crop_bottom": self.crop_spins[1].value(),
            "crop_left": self.crop_spins[2].value(),
            "crop_right": self.crop_spins[3].value(),
            "percent_radius_crop": self.radius_crop_spin.value(),
            "images_per_equirect": int(self.equirect_combo.currentText()),
            # COLMAP tuning
            "refine_intrinsics": self.refine_intrinsics_cb.isChecked(),
            "same_dimensions": self.same_dimensions_cb.isChecked(),
            "gpu_enabled": not self.no_gpu_cb.isChecked(),
            "random_seed": None if seed_v == SEED_RANDOM else seed_v,
            "eval_random_seed": None if eval_seed_v == SEED_RANDOM else eval_seed_v,
            # extras
            "use_sfm_depth": self.use_sfm_depth_cb.isChecked(),
            "eval_data": self.eval_data_edit.text().strip() or None,
        }

    def _build_argv(self, params: dict, c_source: str, c_out: str) -> list[str]:
        """Translate the form's params dict into an ns-process-data invocation.

        Only flags whose values differ from ns-process-data's own defaults are
        emitted, so the rendered command line stays identical to today's for a
        user who never touches the advanced panel. The subcommand and a few
        per-subcommand flags vary by self._mode; everything else is shared.

        May raise ValueError when a host-path field can't be mapped into the
        container's /workspace mount (e.g. eval video stored outside DATA_DIR).
        """
        argv = [
            "ns-process-data", self._mode,
            "--data", c_source,
            "--output-dir", c_out,
        ]
        # video-only: random subsample of frames from the input clip.
        # images mode treats every input file as a frame, so the flag is
        # rejected by tyro upstream.
        if self._mode == "video":
            argv.extend(["--num-frames-target", str(params["num_frames_target"])])
        argv.extend([
            "--matching-method", params["matching_method"],
            "--num-downscales", str(params["num_downscales"]),
        ])
        if params.get("verbose"):
            argv.append("--verbose")

        # Camera & cropping
        if params.get("camera_type", "perspective") != "perspective":
            argv.extend(["--camera-type", params["camera_type"]])

        crop = (
            float(params.get("crop_top", 0.0)),
            float(params.get("crop_bottom", 0.0)),
            float(params.get("crop_left", 0.0)),
            float(params.get("crop_right", 0.0)),
        )
        if any(c > 0.0 for c in crop):
            argv.extend(["--crop-factor", *(f"{c:.4f}" for c in crop)])

        radius = float(params.get("percent_radius_crop", 1.0))
        if radius < 1.0:
            argv.extend(["--percent-radius-crop", f"{radius:.4f}"])

        if params.get("camera_type") == "equirectangular":
            eq = int(params.get("images_per_equirect", 8))
            if eq != 8:
                argv.extend(["--images-per-equirect", str(eq)])

        # COLMAP tuning — emit only the negation flags when the user opted out
        # of a behaviour that's enabled by default.
        if not params.get("refine_intrinsics", True):
            argv.append("--no-refine-intrinsics")
        if not params.get("same_dimensions", True):
            argv.append("--no-same-dimensions")
        if not params.get("gpu_enabled", True):
            argv.append("--no-gpu")

        # video-only: seeds for the random train/eval frame split. Skipped
        # for images mode (every input image is a frame, no subsample).
        if self._mode == "video":
            seed = params.get("random_seed")
            if seed is not None:
                argv.extend(["--random-seed", str(int(seed))])
            eval_seed = params.get("eval_random_seed")
            if eval_seed is not None:
                argv.extend(["--eval-random-seed", str(int(eval_seed))])

        # Extras
        if params.get("use_sfm_depth"):
            argv.append("--use-sfm-depth")

        eval_data = params.get("eval_data")
        if eval_data:
            argv.extend(["--eval-data", _host_to_container(eval_data)])

        # Raw passthrough (last so user flags can override structured ones).
        extra_flags = params.get("extra_flags") or ""
        if extra_flags.strip():
            argv.extend(shlex.split(extra_flags))
        return argv

    def _on_run(self) -> None:
        # Disable the Run button SYNCHRONOUSLY so a quick double-click can't
        # mint two process ULIDs and start two redundant jobs in parallel.
        # The job-start path keeps the button disabled via _on_job_started;
        # any early-return path below uses the `started` flag + finally
        # clause to restore the idle state.
        self.run_btn.setEnabled(False)
        started = False
        try:
            if self._jobs.is_running():
                QMessageBox.information(self, "Busy",
                    "Another job is already running. Cancel it first.")
                return

            state = dock.container_state(CONTAINER_NAME)
            if state.status != "running":
                QMessageBox.warning(
                    self, "Container not running",
                    f"The {CONTAINER_NAME} container isn't running. "
                    "Start it from the Dashboard, then try again.",
                )
                return

            source = Path(self.source_path_edit.text())
            if self._mode == "video":
                if not source.is_file():
                    QMessageBox.warning(self, "Video not found", f"No such file:\n{source}")
                    return
            else:
                if not source.is_dir():
                    QMessageBox.warning(
                        self, "Image folder not found",
                        f"No such directory:\n{source}",
                    )
                    return
                n_heic = count_heic_images(source)
                if n_heic:
                    self.source_meta_lbl.setText(
                        f"Converting {n_heic} HEIC image(s) to JPEG…"
                    )
                    from PySide6.QtWidgets import QApplication
                    QApplication.processEvents()
                    converted = convert_heic_images(source)
                    self._refresh_images_meta(source)
                n, _ = count_images(source)
                if n == 0:
                    allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTS))
                    QMessageBox.warning(
                        self, "No images in folder",
                        f"Folder contains no supported images.\n"
                        f"Accepted extensions: {allowed}",
                    )
                    return

            params = self._collect_params()
            self._store.update_last_process_params(self._project.id, params)

            # Register the source under data_v2/sources/<ULID>/. Idempotent
            # by SHA256 of file (video) or by sorted (relpath, size) listing
            # (image folder) — re-running on the same source returns the
            # existing SourceManifest without copying. Slow on first call
            # for multi-GB videos (full copy + checksum); the status pill
            # makes the wait visible.
            set_pill(self.status_pill, "Registering source…", "warning")
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            try:
                if self._mode == "video":
                    src_manifest = wh.register_source_video(
                        source,
                        display_name=self._project.name or source.name,
                        video_info=probe(source),
                    )
                else:
                    n_imgs, exts = count_images(source)
                    n_heic = count_heic_images(source)
                    src_manifest = wh.register_source_image_folder(
                        source,
                        display_name=self._project.name or source.name,
                        image_count=n_imgs,
                        extensions=sorted(exts),
                        heic_originals_count=n_heic or None,
                    )
            except Exception as e:
                QMessageBox.warning(
                    self, "Source registration failed",
                    f"Couldn't register source under {WAREHOUSE_DIR}:\n{e}",
                )
                # We pushed "Registering source…" before the call; revert
                # so the pill doesn't get stuck pretending we're still working.
                set_pill(self.status_pill, "Ready", "neutral")
                return

            # Resolve the source's host path (the copy under
            # data_v2/sources/<ULID>/) and its container-side counterpart
            # under /warehouse. ns-process-data --data points at the copy,
            # not the user-picked original.
            src_rel = src_manifest.outputs.get("file") or src_manifest.outputs.get("directory")
            if src_rel is None:
                QMessageBox.warning(
                    self, "Source manifest invalid",
                    "Registered source manifest has no outputs.file or outputs.directory.",
                )
                return
            source_canonical = WAREHOUSE_DIR / src_rel

            # Mint a process ULID and reserve its output directory under
            # data_v2/processed/<ULID>/. ns-process-data creates the dir
            # itself when --output-dir doesn't exist; we just need the
            # parent (data_v2/processed/) to be present.
            proc_run_id = wh.mint_ulid()
            target = wh.run_dir("process", proc_run_id)
            # Pre-create the run dir as the host user. The container runs
            # as root and writes files INTO this dir; those files will be
            # root-owned, but the directory itself stays user-owned, so
            # the host-side manifest write in _on_job_finished succeeds.
            # Without this we'd hit EACCES on manifest.json.tmp because
            # the container's mkdir would land first and create the dir
            # root-owned (755).
            target.mkdir(parents=True, exist_ok=True)

            try:
                c_source = wh.host_to_container_warehouse(source_canonical)
                c_out = wh.host_to_container_warehouse(target)
            except ValueError as e:
                QMessageBox.warning(self, "Path translation failed", str(e))
                return

            try:
                argv = self._build_argv(params, c_source, c_out)
            except ValueError as e:
                # Raised by _host_to_container when the eval data lives
                # outside the bind mount, or by shlex.split on malformed
                # --raw flags.
                QMessageBox.warning(self, "Bad parameter", str(e))
                return

            display = f"docker exec {CONTAINER_NAME} " + " ".join(shlex.quote(a) for a in argv)

            spec = JobSpec(
                kind="process",
                display_cmd=display,
                container_argv=argv,
                kill_pattern="ns-process-data",
            )
            self._pending_target = target
            self._pending_source_id = src_manifest.run_id
            self._pending_run_id = proc_run_id
            try:
                self._pending_params = wh.process_params_from_form(params, self._mode)
            except Exception as e:
                # process_params_from_form validates literal-typed fields;
                # if the form somehow holds a bad camera_type / matcher,
                # we'd rather not run at all than write a malformed manifest.
                QMessageBox.warning(self, "Bad parameter", str(e))
                return
            self._pending_command = display
            self._pending_start_time = datetime.now(timezone.utc)
            self._pending_display_name = self._project.name or source.name
            self.ready_to_train.hide()
            self.splinter_warn.hide()
            self.output_lbl.setText(str(target))
            self._jobs.start(spec)
            started = True
        finally:
            if not started:
                # Any pre-start bailout: discard the pending target and let
                # the user click Run again.
                self._pending_target = None
                self.run_btn.setEnabled(True)

    def _on_cancel(self) -> None:
        self._jobs.cancel()

    # ---- job signals ----
    def _wire_jobs(self) -> None:
        self._jobs.started.connect(self._on_job_started)
        self._jobs.finished.connect(self._on_job_finished)
        self._jobs.cancelled.connect(self._on_job_cancelled)
        self._jobs.failed.connect(self._on_job_failed)

    def _on_job_started(self, spec: JobSpec) -> None:
        if spec.kind != "process":
            self._running_mine = False
            self.run_btn.setEnabled(False)
            self.cancel_btn.setEnabled(False)
            set_pill(self.status_pill, "Other job running", "warning")
            return
        self._running_mine = True
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        set_pill(self.status_pill, "● Running", "warning")

    def _on_job_finished(self, code: int) -> None:
        if not self._running_mine:
            self._reset_idle()
            return
        self._running_mine = False
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if code == 0:
            # The job wrote into _pending_target (data_v2/processed/<ULID>/).
            # Two things happen on success:
            #   1. Write manifest.json into the target — the canonical
            #      "this run succeeded" signal for any external consumer
            #      walking the warehouse. Atomic via .tmp + rename, so a
            #      crash during step 2 leaves the manifest valid.
            #   2. Repoint the project at the new target so the Train tab
            #      keeps working off projects.json today (we'll migrate it
            #      to the warehouse contract in a later turn).
            target = self._pending_target
            if (
                target is not None
                and self._pending_source_id is not None
                and self._pending_run_id is not None
                and self._pending_params is not None
                and self._pending_start_time is not None
            ):
                duration = (
                    datetime.now(timezone.utc) - self._pending_start_time
                ).total_seconds()
                try:
                    wh.write_process_manifest(
                        output_dir=target,
                        run_id=self._pending_run_id,
                        parent_source_id=self._pending_source_id,
                        display_name=self._pending_display_name or target.name,
                        params=self._pending_params,
                        command=self._pending_command or "",
                        duration_seconds=duration,
                        start_time=self._pending_start_time,
                    )
                except Exception as e:
                    # Manifest write failed — surface to the user but don't
                    # treat the run itself as failed. The on-disk dataset
                    # is intact; only the lineage record is missing, and
                    # a future scan_catalog pass can backfill it.
                    QMessageBox.warning(
                        self, "Manifest write failed",
                        f"ns-process-data succeeded but the manifest "
                        f"couldn't be written:\n{e}\n\nDataset is still "
                        f"usable; lineage will need to be backfilled.",
                    )
                try:
                    self._store.update_processed_dir(self._project.id, target)
                except ValueError:
                    # update_processed_dir validates the path is under
                    # one of the known data roots. If it raises here the
                    # warehouse root isn't registered with the store yet;
                    # the manifest still landed correctly, so it's safe
                    # to no-op and let projects.json catch up later.
                    pass
                else:
                    # Refresh derived UI: output_lbl already showed `target`
                    # since _on_run, but read it back from the project record
                    # in case the store normalized the path.
                    self.output_lbl.setText(self._project.processed_dir)
            self._clear_pending_state()
            set_pill(self.status_pill, "Done", "success")
            self.ready_to_train.show()
            self._check_sparse_splinter()
            # The just-completed target is now on disk; the next Run will
            # mint a fresh ULID anyway, but the caption shows the policy.
            self._refresh_next_target()
        else:
            # A non-zero exit means the run failed — do NOT repoint the
            # project and do NOT write a manifest (no manifest = canonical
            # "failed/incomplete" marker). The orphaned partial folder at
            # _pending_target stays on disk for inspection / manual cleanup.
            self._clear_pending_state()
            set_pill(self.status_pill, f"Failed · exit {code}", "error")
            # Restore output_lbl to the project's still-active processed_dir
            # so the user isn't misled into thinking the failed run succeeded.
            self.output_lbl.setText(self._project.processed_dir)
            self._refresh_next_target()

    def _on_job_cancelled(self) -> None:
        if not self._running_mine:
            self._reset_idle()
            return
        self._running_mine = False
        self._clear_pending_state()
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        set_pill(self.status_pill, "Cancelled", "error")
        self.output_lbl.setText(self._project.processed_dir)
        # A cancelled job may have left a partial folder at _pending_target;
        # the absence of manifest.json marks it incomplete for any consumer.
        self._refresh_next_target()

    def _on_job_failed(self, msg: str) -> None:
        if not self._running_mine:
            self._reset_idle()
            return
        self._running_mine = False
        self._clear_pending_state()
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        set_pill(self.status_pill, "Failed to start", "error")
        self.output_lbl.setText(self._project.processed_dir)
        self._refresh_next_target()

    def _clear_pending_state(self) -> None:
        """Drop all per-run state stashed by _on_run. Called from every
        terminal job-signal handler (finished/cancelled/failed) so we
        never carry stale source/run IDs into the next click of Run."""
        self._pending_target = None
        self._pending_source_id = None
        self._pending_run_id = None
        self._pending_params = None
        self._pending_command = None
        self._pending_start_time = None
        self._pending_display_name = None

    def _reset_idle(self) -> None:
        """Another kind's job just ended — restore our idle UI."""
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        set_pill(self.status_pill, "Ready", "neutral")

    # ---- splinter detection ----
    def _check_sparse_splinter(self) -> None:
        """After a successful run, warn when ns-process-data exported the
        wrong COLMAP submap.

        ns-process-data always writes transforms.json from `colmap/sparse/0`,
        but COLMAP's incremental mapper can split a reconstruction into
        multiple disjoint submaps when the scene has repeated near-duplicate
        views (e.g. several similar vehicles in one hangar) or when vocab_tree
        matching pairs up visually similar but geometrically unrelated frames.
        If sparse/0 happens to be the tiny splinter, transforms.json silently
        ends up with only a handful of frames while the real reconstruction
        sits unused in sparse/1+.

        We stay silent on any probe failure — a false warning is worse than
        none, and a genuinely splintered dataset will also fail visibly in
        viser the moment the user tries to train.
        """
        sparse_dir = Path(self._project.processed_dir) / "colmap" / "sparse"
        if not sparse_dir.is_dir():
            return
        try:
            subs = sorted(
                p for p in sparse_dir.iterdir()
                if p.is_dir() and p.name.isdigit()
            )
        except OSError:
            return
        if len(subs) < 2:
            return

        counts: dict[str, int] = {}
        for p in subs:
            try:
                c_path = _host_to_container(p)
            except ValueError:
                return
            n = dock.colmap_registered_image_count(c_path)
            if n is None:
                return  # probe failed — stay silent
            counts[p.name] = n

        largest = max(counts, key=counts.get)
        if largest == "0":
            return

        parts = ", ".join(f"sparse/{k}={v}" for k, v in sorted(counts.items()))
        fix_cmd = (
            f"cd {sparse_dir} && mv 0 _tiny && mv {largest} 0  "
            f"# then re-run processing with --skip-colmap --skip-image-processing"
        )
        self.splinter_body.setText(
            f"COLMAP produced {len(subs)} disjoint submaps ({parts}).\n"
            f"ns-process-data exported transforms.json from sparse/0, but "
            f"sparse/{largest} is larger — only sparse/0's {counts['0']} "
            f"frames are in the dataset. To promote the larger submap:\n"
            f"  {fix_cmd}"
        )
        self.splinter_warn.show()

    # ---- vizer live viewer ----
    def _poll_vizer(self) -> None:
        if self._vizer_poll_pool.activeThreadCount() > 0:
            return
        self._vizer_poll_pool.start(_VizerStatusTask(self._vizer_signals))

    def _on_vizer_status(self, status: str) -> None:
        if status == "running":
            set_pill(self.vizer_pill, "● Running", "success")
        elif status in ("exited", "created"):
            set_pill(self.vizer_pill, "Stopped", "warning")
        elif status == "missing":
            set_pill(self.vizer_pill, "Not found", "error")
        else:
            set_pill(self.vizer_pill, status, "neutral")

    def _on_open_vizer(self) -> None:
        state = dock.container_state(VIZER_CONTAINER)
        if state.status == "missing":
            QMessageBox.warning(
                self, "Vizer container not found",
                f"No container named '{VIZER_CONTAINER}' exists.\n\n"
                "Create it first — see the vizer README or run:\n"
                "  cd ~/docker-stacks/vizer && docker compose up -d",
            )
            return
        if state.status != "running":
            try:
                dock.start_container(VIZER_CONTAINER)
            except dock.DockerError as e:
                QMessageBox.warning(
                    self, "Couldn't start vizer",
                    f"Failed to start '{VIZER_CONTAINER}':\n{e}",
                )
                return
            set_pill(self.vizer_pill, "● Starting…", "warning")
        webbrowser.open(f"http://localhost:{VIZER_VIEWER_PORT}")


# ---------- top-level view ----------

class ProcessView(QWidget):
    """Top-level Process tab. Host for ProjectBar + per-project form (or empty state)."""

    request_navigate = Signal(int)  # tab index to switch to

    def __init__(self, store: ProjectStore, jobs: JobManager):
        super().__init__()
        self.setObjectName("Canvas")
        self._store = store
        self._jobs = jobs

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.project_bar = ProjectBar(store, jobs)
        outer.addWidget(self.project_bar)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)

        self._rebuild()
        self._store.active_changed.connect(lambda _p: self._rebuild())
        self._store.projects_changed.connect(self._rebuild)

    def _rebuild(self) -> None:
        # clear existing
        while self.stack.count():
            w = self.stack.widget(0)
            self.stack.removeWidget(w)
            w.deleteLater()

        active = self._store.active
        if active is None:
            self.stack.addWidget(_EmptyState(self._store))
            return
        # Dispatch on source_kind so a freshly-created images project
        # (video_path=None, images_dir set) gets _Form, not _ImportedState.
        kind = source_kind(active)
        if kind == "imported":
            gate = _ImportedState(active)
            gate.request_train.connect(lambda: self.request_navigate.emit(2))
            self.stack.addWidget(gate)
            return
        form = _Form(self._store, self._jobs, active)
        form.request_train.connect(lambda: self.request_navigate.emit(2))  # Train tab idx
        self.stack.addWidget(form)
