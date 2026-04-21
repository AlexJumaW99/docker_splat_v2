"""
Process view — runs ns-process-data inside splat-ns.
"""
from __future__ import annotations

import shlex
from pathlib import Path

from PySide6.QtCore import Qt, QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QStackedWidget, QVBoxLayout,
    QWidget,
)

from app import docker_service as dock
from app.config import CONTAINER_NAME, DATA_DIR
from app.ffprobe import VideoInfo, probe
from app.jobs import JobManager, JobSpec
from app.store import Project, ProjectStore
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
        self._running_mine = False
        self._probe_signals = _ProbeSignals()
        self._probe_signals.done.connect(self._on_probe_done)

        self._build()
        self._load_project()
        self._wire_jobs()

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
        sub = QLabel("Prepare a video for splatfacto training.")
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

        bl.addWidget(self._video_card())
        bl.addWidget(self._params_card())
        bl.addWidget(self._output_card())
        bl.addWidget(self._actions_card())

        bl.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll)

    # ---- video card ----
    def _video_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(10)
        v.addWidget(section_title("VIDEO"))

        row = QHBoxLayout(); row.setSpacing(8)
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setReadOnly(True)
        self.video_path_edit.setProperty("role", "readout")
        row.addWidget(_hb("video"))
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.setProperty("role", "secondary")
        self.browse_btn.setCursor(Qt.PointingHandCursor)
        self.browse_btn.clicked.connect(self._on_browse_video)
        row.addWidget(self.video_path_edit, 1)
        row.addWidget(self.browse_btn)
        v.addLayout(row)

        self.meta_lbl = QLabel("")
        self.meta_lbl.setProperty("role", "caption")
        v.addWidget(self.meta_lbl)
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

        # Target frames
        grid.addWidget(labeled("Target frames", *_HELP["frames"]), 0, 0)
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
        w = QWidget(); w.setLayout(frames_row)
        grid.addWidget(w, 0, 1)

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
        g.addWidget(labeled("Train frame-selection seed", *_HELP["seed"]), 3, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(SEED_RANDOM, 2_147_483_647)
        self.seed_spin.setSpecialValueText("random")
        self.seed_spin.setMaximumWidth(140)
        seed_row = QHBoxLayout(); seed_row.addWidget(self.seed_spin); seed_row.addStretch(1)
        seed_wrap = QWidget(); seed_wrap.setLayout(seed_row)
        g.addWidget(seed_wrap, 3, 1)

        g.addWidget(labeled("Eval frame-selection seed", *_HELP["eval_seed"]), 4, 0)
        self.eval_seed_spin = QSpinBox()
        self.eval_seed_spin.setRange(SEED_RANDOM, 2_147_483_647)
        self.eval_seed_spin.setSpecialValueText("random")
        self.eval_seed_spin.setMaximumWidth(140)
        es_row = QHBoxLayout(); es_row.addWidget(self.eval_seed_spin); es_row.addStretch(1)
        es_wrap = QWidget(); es_wrap.setLayout(es_row)
        g.addWidget(es_wrap, 4, 1)

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

        # Eval video picker (optional). Stored as host path; converted at run time.
        sv.addWidget(labeled("Eval video (optional)", *_HELP["eval_video"]))
        eval_row = QHBoxLayout(); eval_row.setSpacing(8)
        self.eval_video_edit = QLineEdit()
        self.eval_video_edit.setReadOnly(True)
        self.eval_video_edit.setProperty("role", "readout")
        self.eval_video_edit.setPlaceholderText("(none)")
        self.eval_video_browse = QPushButton("Browse…")
        self.eval_video_browse.setProperty("role", "secondary")
        self.eval_video_browse.setCursor(Qt.PointingHandCursor)
        self.eval_video_browse.clicked.connect(self._on_browse_eval_video)
        self.eval_video_clear = QPushButton("Clear")
        self.eval_video_clear.setProperty("role", "ghost")
        self.eval_video_clear.setCursor(Qt.PointingHandCursor)
        self.eval_video_clear.clicked.connect(lambda: self.eval_video_edit.setText(""))
        eval_row.addWidget(self.eval_video_edit, 1)
        eval_row.addWidget(self.eval_video_browse)
        eval_row.addWidget(self.eval_video_clear)
        sv.addLayout(eval_row)

        hint = QLabel(
            "Eval video, if set, is processed alongside the train video and "
            "held out from training. Must live inside the data/ bind mount."
        )
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


    def _on_browse_eval_video(self) -> None:
        from app.config import DATA_DIR as _DD
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick an eval video",
            str(_DD),
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;All files (*.*)",
        )
        if not path:
            return
        try:
            Path(path).resolve().relative_to(_DD.resolve())
        except ValueError:
            QMessageBox.warning(
                self, "Eval video outside data/",
                f"Eval videos must live inside {_DD} so the container can see them."
            )
            return
        self.eval_video_edit.setText(path)

    # ---- output card ----
    def _output_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(8)
        out_hdr = QHBoxLayout(); out_hdr.setSpacing(6); out_hdr.setContentsMargins(0, 0, 0, 0)
        out_hdr.addWidget(section_title("OUTPUT DATASET"))
        out_hdr.addWidget(_hb("output"))
        out_hdr.addStretch(1)
        out_wrap = QWidget(); out_wrap.setLayout(out_hdr)
        v.addWidget(out_wrap)
        self.output_lbl = QLineEdit()
        self.output_lbl.setReadOnly(True)
        self.output_lbl.setProperty("role", "readout")
        v.addWidget(self.output_lbl)
        hint = QLabel(
            "Host path. Container sees this as /workspace/processed/… automatically."
        )
        hint.setProperty("role", "caption")
        v.addWidget(hint)
        return c

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
        v.addWidget(self.ready_to_train)
        v.addWidget(self.splinter_warn)
        return c

    # ---- advanced toggle ----
    def _on_advanced(self, checked: bool) -> None:
        self.advanced_wrap.setVisible(checked)

    # ---- project lifecycle ----
    def _load_project(self) -> None:
        p = self._project
        self.video_path_edit.setText(p.video_path)
        self.output_lbl.setText(p.processed_dir)

        # Populate form from last params or sensible defaults.
        self._apply_params(p.last_process_params or {})

        self._refresh_presets()
        self._probe_video(Path(p.video_path))

    def _apply_params(self, params: dict) -> None:
        """Populate every widget from a params dict. Missing keys fall back to
        the same defaults ns-process-data would use, so legacy projects (saved
        before the advanced fields existed) load unchanged."""
        # --- basic ---
        self.frames_slider.setValue(int(params.get("num_frames_target", 250)))
        self.frames_value.setValue(int(params.get("num_frames_target", 250)))
        method = params.get("matching_method", "sequential")
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
        self.eval_video_edit.setText(params.get("eval_data") or "")

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

    # ---- video probe ----
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
        self.video_path_edit.setText(path)
        self._probe_video(Path(path))

    def _probe_video(self, path: Path) -> None:
        self.meta_lbl.setText("reading video metadata…")
        QThreadPool.globalInstance().start(_ProbeTask(path, self._probe_signals))

    def _on_probe_done(self, info: VideoInfo | None) -> None:
        if info is None:
            self.meta_lbl.setText("video not found")
            return
        self.meta_lbl.setText(info.pretty())

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
            "eval_data": self.eval_video_edit.text().strip() or None,
        }

    def _build_argv(self, params: dict, c_video: str, c_out: str) -> list[str]:
        """Translate the form's params dict into an ns-process-data invocation.

        Only flags whose values differ from ns-process-data's own defaults are
        emitted, so the rendered command line stays identical to today's for a
        user who never touches the advanced panel.

        May raise ValueError when a host-path field can't be mapped into the
        container's /workspace mount (e.g. eval video stored outside DATA_DIR).
        """
        argv = [
            "ns-process-data", "video",
            "--data", c_video,
            "--output-dir", c_out,
            "--num-frames-target", str(params["num_frames_target"]),
            "--matching-method", params["matching_method"],
            "--num-downscales", str(params["num_downscales"]),
        ]
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

        video = Path(self.video_path_edit.text())
        if not video.exists():
            QMessageBox.warning(self, "Video not found", f"No such file:\n{video}")
            return

        params = self._collect_params()
        self._store.update_last_process_params(self._project.id, params)

        try:
            c_video = _host_to_container(video)
            c_out = _host_to_container(self._project.processed_dir)
        except ValueError as e:
            QMessageBox.warning(self, "Path outside data/", str(e))
            return

        try:
            argv = self._build_argv(params, c_video, c_out)
        except ValueError as e:
            # Raised by _host_to_container when the eval video lives outside
            # the bind mount, or by shlex.split on malformed --raw flags.
            QMessageBox.warning(self, "Bad parameter", str(e))
            return

        display = f"docker exec {CONTAINER_NAME} " + " ".join(shlex.quote(a) for a in argv)

        spec = JobSpec(
            kind="process",
            display_cmd=display,
            container_argv=argv,
            kill_pattern="ns-process-data",
        )
        self.ready_to_train.hide()
        self.splinter_warn.hide()
        self._jobs.start(spec)

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
            set_pill(self.status_pill, "Done", "success")
            self.ready_to_train.show()
            self._check_sparse_splinter()
        else:
            set_pill(self.status_pill, f"Failed · exit {code}", "error")

    def _on_job_cancelled(self) -> None:
        if not self._running_mine:
            self._reset_idle()
            return
        self._running_mine = False
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        set_pill(self.status_pill, "Cancelled", "error")

    def _on_job_failed(self, msg: str) -> None:
        if not self._running_mine:
            self._reset_idle()
            return
        self._running_mine = False
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        set_pill(self.status_pill, "Failed to start", "error")

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

        self.project_bar = ProjectBar(store)
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
        if active.video_path is None:
            gate = _ImportedState(active)
            gate.request_train.connect(lambda: self.request_navigate.emit(2))
            self.stack.addWidget(gate)
            return
        form = _Form(self._store, self._jobs, active)
        form.request_train.connect(lambda: self.request_navigate.emit(2))  # Train tab idx
        self.stack.addWidget(form)
