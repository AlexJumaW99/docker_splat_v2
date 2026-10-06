"""
Train view — runs ns-train inside splat-ns.

Mirrors the Process tab: ProjectBar + per-project form, with a not-processed
gate for projects that haven't been through ns-process-data yet. Presets live
per-project in presets_train. Job signals are filtered by kind so that a
concurrent process/train run can't cross-wire UI state.
"""
from __future__ import annotations

import re
import shlex
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QSlider,
    QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

from app import docker_service as dock
from app import nvidia_service as nv
from app import train_options as topt
from app import warehouse as wh
from app.config import CONTAINER_NAME, VISER_PORT, WAREHOUSE_DIR
from app.jobs import JobManager, JobSpec
from app.store import Project, ProjectStore, detect_colmap_path, source_kind
from app.widgets import HelpBadge, ProjectBar, card, labeled, pill, section_title, set_pill


# ---------- help copy ----------
_HELP = {
    "dataset": (
        "Processed dataset",
        "The host-side folder containing the ns-process-data output for this "
        "project, passed to ns-train via <code>--data</code>. The marker file "
        "(<code>transforms.json</code> for nerfstudio-data, "
        "<code>cameras.bin</code> for colmap) is what gates training.<br><br>"
        "<b>Poses</b> \u00b7 how many extracted frames received a camera pose. "
        "ns-process-data exits 0 even when COLMAP registered 2 of 75 frames, "
        "so this is the number to check before training: green \u2265 90\u202f%, "
        "amber 50\u201390\u202f%, red below 50\u202f% (Run asks for confirmation)."
    ),
    "preset": (
        "Training presets",
        "Named snapshots of every field on this form, stored per-project in "
        "<code>presets_train</code>. Switching presets overwrites the form; "
        "use 'Save as preset\u2026' to capture the current values."
    ),
    "method": (
        "Method",
        "Which Nerfstudio model ns-train optimizes.<br><br>"
        "<b>splatfacto</b> \u00b7 default 3D Gaussian Splatting recipe \u2014 fast, "
        "quality scales with iterations.<br>"
        "<b>splatfacto-big</b> \u00b7 higher gaussian budget; better detail, more "
        "VRAM.<br>"
        "<b>splatfacto-mcmc</b> \u00b7 MCMC-based densification with a fixed "
        "gaussian budget (<i>Max gaussians</i>); often higher-quality but "
        "slower.<br>"
        "<b>splatfacto-w-light</b> \u00b7 in-the-wild variant with a per-image "
        "appearance model; runs on any processed dataset.<br>"
        "<b>splatfacto-w</b> \u00b7 full in-the-wild model (appearance + "
        "background + transient masking). Needs the COLMAP sparse model "
        "with perspective cameras; fisheye datasets are refused. Trains at "
        "\u22641600 px using the dataset's downscaled copies."
    ),
    "max_gs_num": (
        "Max gaussians",
        "How many gaussians <b>splatfacto-mcmc</b> ends up with "
        "(<code>--pipeline.model.max-gs-num</code>). The other methods let "
        "the scene decide the count; MCMC grows the model 5\u202f% every 100 "
        "steps (steps 500\u201325\u202f000) until it holds exactly this many, so it "
        "is a budget you choose. nerfstudio's default is 1\u202fM; the method's "
        "authors set it per scene, from 1.1\u202fM to 5.9\u202fM.<br><br>"
        "Measured on a 266-frame drone scene at 960\u202fpx, 30\u202f000 steps "
        "(RTX 5070 Ti):<br>"
        "<b>1\u202fM</b> \u00b7 PSNR 35.0 \u00b7 2.6\u202fGiB peak VRAM \u00b7 10\u202fmin \u00b7 "
        "238\u202fMB PLY \u00b7 129\u202ffps<br>"
        "<b>2\u202fM</b> \u00b7 PSNR 35.4 \u00b7 4.2\u202fGiB \u00b7 16\u202fmin \u00b7 457\u202fMB \u00b7 87\u202ffps<br>"
        "<b>4\u202fM</b> \u00b7 PSNR 35.7 \u00b7 7.6\u202fGiB \u00b7 28\u202fmin \u00b7 847\u202fMB \u00b7 54\u202ffps<br>"
        "Two identical 1\u202fM runs differed by 0.04\u202fdB, so the gain is real, "
        "but each doubling buys less. 3\u202fM was not run (its figures are "
        "interpolated) and nothing above 960\u202fpx was measured.<br><br>"
        "A short run can stop before the budget is reached: 7\u202f000 steps "
        "grow the model to at most ~23\u00d7 its SfM point count. The exported "
        "PLY holds slightly fewer gaussians than the budget because ns-export "
        "drops those below 1/255 opacity (4\u202f% at 1\u202fM, 15\u202f% at 4\u202fM)."
    ),
    "iters": (
        "Max iterations",
        "Total ns-train optimization steps. Every preset's internal schedule "
        "(densification until step 15\u202f000, or 25\u202f000 for mcmc; learning-"
        "rate decay over 30\u202f000) is an absolute step count, so short runs end "
        "mid-schedule.<br><br>"
        "<b>Preview \u00b7 7\u202f000</b> \u00b7 a first look (measured: PSNR 31.0 in 87\u202fs "
        "at 960\u202fpx).<br>"
        "<b>Standard \u00b7 30\u202f000</b> \u00b7 nerfstudio's default and this tab's "
        "default; the same scene reached PSNR 33.9 in 370\u202fs.<br>"
        "<b>Long \u00b7 55\u202f000</b> \u00b7 splatfacto-w's upstream default (its robust "
        "masking starts at 6\u202f000 and densification runs to 20\u202f000).<br>"
        "Beyond 30\u202f000 the built-in splatfacto variants show diminishing returns."
    ),
    "save_step": (
        "Steps per save",
        "Interval between checkpoint dumps. Each checkpoint is a full "
        ".ckpt under the run's <code>nerfstudio_models/</code> folder \u2014 "
        "smaller values = more disk but finer-grained recovery if training "
        "crashes. Default 2000 is a reasonable trade-off.<br><br>"
        "A 4K splatfacto-big run with <b>10000</b> here ran out of memory at "
        "step ~8500 and left no checkpoint at all; values above 5000 are "
        "flagged for that reason."
    ),
    "resolution": (
        "Training resolution",
        "Which copy of the frames ns-train optimises against. "
        "<b>A splat is only sharp at the resolution it was trained at</b>: on "
        "the same 4K scene, a model trained at 960\u202fpx scored PSNR 33.9 viewed "
        "at 960\u202fpx but 28.2 viewed at 1920\u202fpx; the 1920\u202fpx model scored "
        "32.5 at 1920 (2.4\u00d7 the training time).<br><br>"
        "<b>Auto</b> \u00b7 nerfstudio's own rule: the smallest "
        "<code>images_N</code> copy whose longest side is \u2264 1600\u202fpx \u2014 "
        "for 4K footage that is 960\u202fpx.<br>"
        "<b>images_N entries</b> \u00b7 emitted as "
        "<code>--downscale-factor N</code> on the dataparser.<br>"
        "<b>rescaled at load</b> \u00b7 the dataset has no such folder; frames "
        "are resized while loading (<code>camera-res-scale-factor</code>). "
        "Use <i>Add downscale levels</i> to create the folders once instead."
    ),
    "cache_images": (
        "Image cache",
        "Where nerfstudio keeps the training frames: on the GPU (fastest) or "
        "in host RAM (saves VRAM, about a third slower per step at 4K).<br><br>"
        "<b>Auto</b> \u00b7 GPU unless the frames exceed "
        f"{topt.CACHE_CPU_THRESHOLD_GIB:g}\u202fGiB at the chosen resolution, "
        "then CPU. Measured: 266 frames at 4K are 6.2\u202fGiB on the GPU and "
        "pushed splatfacto-big to 11.6\u202fGiB by step 6500; with the CPU cache "
        "the same run peaked at 5.6\u202fGiB (581\u202fs \u2192 781\u202fs for 6500 steps)."
    ),
    "eval_steps": (
        "Eval cadence",
        "How often nerfstudio renders a held-out image (and, less often, all "
        "of them) for TensorBoard. nerfstudio's default is <b>100</b>: a full-"
        "resolution render plus PSNR/SSIM/LPIPS every 100 steps, and ~180\u202fMB "
        "of TensorBoard images per 1000 steps at 4K. <b>1000</b> keeps the "
        "metrics and the live viewer at a tenth of that cost.<br><br>"
        "The first eval image used to be the VRAM high-water mark of a run "
        "(+12.4\u202fGiB at 1920\u202fpx): cuDNN autotuning the LPIPS convolutions. "
        "Image v4.8 switches that autotuning off (same speed, 11.2 \u2192 2.4\u202fGiB "
        "peak), so cadence no longer decides the peak."
    ),
    "rasterize_mode": (
        "Rasterize mode",
        "<b>Method default</b> \u00b7 classic for splatfacto / -big / -mcmc, "
        "antialiased for splatfacto-w / -w-light.<br>"
        "<b>classic</b> \u00b7 the original 3DGS rasterizer; PLYs render the same "
        "in every web viewer.<br>"
        "<b>antialiased</b> \u00b7 Mip-Splatting-style opacity compensation. "
        "Measured on the same scene: PSNR 34.9 vs 33.9 at native resolution "
        "and 30.2 vs 28.2 when viewed at twice the training resolution. "
        "nerfstudio warns that a PLY exported from an antialiased model does "
        "not match classic-mode viewers \u2014 keep classic if the PLY is the "
        "deliverable."
    ),
    "vis": (
        "Visualization",
        "Where training telemetry goes while ns-train runs.<br><br>"
        "<b>viewer+tensorboard</b> \u00b7 default \u2014 live viser viewer on "
        f":{VISER_PORT} <i>and</i> TB event files on disk.<br>"
        "<b>viewer</b> \u00b7 viser only, no TB logs.<br>"
        "<b>tensorboard</b> \u00b7 no live viewer (headless runs, lowest "
        "overhead)."
    ),
    "quit_on_completion": (
        "Quit viewer when training finishes",
        "Passes <code>--viewer.quit-on-train-completion True</code>. By "
        "default the viser server keeps running after the last iteration so "
        "you can explore the scene; enable this for unattended / batch runs "
        "so the process exits cleanly when done."
    ),
    "advanced_mode": (
        "Advanced mode",
        "Reveals the <b>Raw CLI flags</b> box at the bottom of the form. "
        "Everything the GUI doesn't expose as a field goes there — any token "
        "you type is appended verbatim to <code>ns-train</code> before the "
        "dataparser subcommand. Leave off unless you have a specific flag "
        "in mind; the defaults above are already sensible."
    ),
    "extra_flags": (
        "Raw CLI flags",
        "Appended verbatim to <code>ns-train</code> before the dataparser "
        "subcommand. Use for model / pipeline flags the GUI doesn't "
        "surface, e.g. <code>--pipeline.model.cull-alpha-thresh 0.005</code>. "
        "Inspect full options with:<br>"
        "<code>docker exec splat-ns-v2 ns-train splatfacto --help</code>"
    ),
    "output": (
        "Output run",
        "Host folder where ns-train writes this project's runs. Each "
        "launch creates a new <code>&lt;method&gt;/&lt;timestamp&gt;/</code> "
        "subfolder so concurrent or repeated runs never clobber each other."
    ),
}


def _hb(key: str) -> HelpBadge:
    title, body = _HELP[key]
    return HelpBadge(title, body)


# ---------- constants ----------

METHODS = [
    "splatfacto", "splatfacto-big", "splatfacto-mcmc",
    "splatfacto-w", "splatfacto-w-light",
]
VIS_MODES = ["viewer+tensorboard", "viewer", "tensorboard"]
CACHE_MODES = ["auto", "gpu", "cpu"]
RASTER_MODES = ["default", "classic", "antialiased"]
_CACHE_LABELS = {"auto": "Auto (recommended)", "gpu": "GPU", "cpu": "CPU (host RAM)"}
_RASTER_LABELS = {"default": "Method default", "classic": "classic", "antialiased": "antialiased"}
# Steps-per-save values above this get an inline warning (see help text).
_SAVE_STEP_WARN = 5000
# Iteration quick picks: (label, steps). Measured basis in _HELP["iters"].
_ITER_PRESETS = [("Preview · 7k", 7000), ("Standard · 30k", 30000), ("Long · 55k", 55000)]
# Upstream TrainerConfig.max_num_iterations per method (nerfstudio
# method_configs.py / splatfactow_config.py).
_UPSTREAM_ITERS = {"splatfacto": 30000, "splatfacto-big": 30000, "splatfacto-mcmc": 30000,
                   "splatfacto-w": 55000, "splatfacto-w-light": 30000}


# nerfstudio prints a rich Panel with "🎉 Training Finished 🎉" in the title
# when ns-train completes its final iteration. When the user runs with
# --viewer.quit-on-train-completion False (our default), the process stays
# alive past this banner to keep the viser viewer up. We watch stdout for
# this marker so the manifest can be finalized at the *real* end-of-training
# event, not at process exit. Match on the bare text (no emoji) so we're
# robust against ANSI/encoding surprises.
_TRAINING_DONE_RE = re.compile(r"\bTraining Finished\b")


# ---------- helpers ----------


def _colmap_rel(project: Project) -> str:
    """Relative path to the COLMAP model dir for a project. Stored on the
    project at import time; fall back to Nerfstudio's own default so a
    corrupted record never silently trains on the wrong folder."""
    return project.colmap_path or "colmap/sparse/0"


def _marker_file(project: Project) -> Path:
    """The filesystem marker we check to know a dataset is ready for training.
    Kept in one place so _is_trainable, the dataset-card readout, and any
    future dataparser all share the same definition."""
    root = Path(project.processed_dir)
    if project.dataparser == "colmap":
        return root / _colmap_rel(project) / "cameras.bin"
    # nerfstudio-data (the default)
    return root / "transforms.json"


def _is_trainable(project: Project) -> bool:
    """True if the project's processed folder currently holds the artifacts
    the project's chosen dataparser requires."""
    return _marker_file(project).is_file()


def _dataset_meta_text(project: Project) -> str:
    """One-line summary for the DATASET card — format + frame count."""
    if project.dataparser == "colmap":
        head = f"✓ COLMAP format ({_colmap_rel(project)}/cameras.bin)"
    else:
        head = "✓ Nerfstudio format (transforms.json)"
    n = _count_frames(project.processed_dir)
    return f"{head}  ·  {n} frames" if n is not None else head


def _count_frames(processed_dir: str) -> int | None:
    from app.manifest import ProcessManifest, load_manifest
    manifest_path = Path(processed_dir) / "manifest.json"
    if manifest_path.is_file():
        try:
            m = load_manifest(manifest_path)
            if isinstance(m, ProcessManifest):
                count = m.metrics.get("extracted_frame_count")
                if count is not None:
                    return int(count)
        except Exception:
            pass
    images = Path(processed_dir) / "images"
    if not images.is_dir():
        return None
    return sum(1 for p in images.iterdir() if p.is_file())


# ---------- empty state (no project) ----------

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
            "Training needs a processed dataset. "
            "Create a project on the Process tab to get started."
        )
        sub.setProperty("role", "caption")
        sub.setWordWrap(True)
        sub.setAlignment(Qt.AlignCenter)
        sub.setMaximumWidth(440)

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


# ---------- not-processed gate ----------

class _NotProcessedState(QWidget):
    """Active project's processed folder is missing or incomplete. Behaviour
    differs for imported vs video-sourced projects: an imported project has
    no source video to re-process, so we don't point the user at Process."""

    request_process = Signal()

    def __init__(self, project: Project):
        super().__init__()
        self.setObjectName("Canvas")
        v = QVBoxLayout(self)
        v.setContentsMargins(40, 80, 40, 40)
        v.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        v.setSpacing(10)

        # source_kind == "imported" means the project was created via the
        # "Import processed dataset" flow — there's no source to re-process,
        # so we point the user at the broken folder rather than at Process.
        # video- and images-sourced projects can both be re-processed.
        imported = source_kind(project) == "imported"

        title = QLabel("Imported dataset missing" if imported else "Not processed yet")
        title.setObjectName("PageTitle")
        title.setAlignment(Qt.AlignCenter)

        if imported:
            msg = (
                f"'{project.name}' was imported from\n{project.processed_dir}\n"
                "but that folder is gone or no longer holds the expected files. "
                "Restore it, or create a new project pointing at a valid folder."
            )
        else:
            msg = (
                f"'{project.name}' has no processed dataset yet. "
                "Run it through Process first, then come back."
            )
        sub = QLabel(msg)
        sub.setProperty("role", "caption")
        sub.setWordWrap(True)
        sub.setAlignment(Qt.AlignCenter)
        sub.setMaximumWidth(520)

        v.addWidget(title)
        v.addWidget(sub, 0, Qt.AlignHCenter)

        if not imported:
            v.addSpacing(16)
            btn = QPushButton("Go to Process →")
            btn.setProperty("role", "primary")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(self.request_process.emit)
            v.addWidget(btn, 0, Qt.AlignHCenter)

        v.addStretch(1)


# ---------- form ----------

class _Form(QWidget):
    """Per-project training form. Rebuilt when the active project changes."""

    request_viewers = Signal()
    request_process = Signal()

    def __init__(self, store: ProjectStore, jobs: JobManager, project: Project):
        super().__init__()
        self.setObjectName("Canvas")
        self._store = store
        self._jobs = jobs
        self._project = project
        self._running_mine = False

        # Output dir reserved at _on_run time and consumed by the success
        # branch of _on_job_finished. Holds the absolute host path of
        # data_v2/trained/<train_ULID>/. All pending fields cleared via
        # _clear_pending_state() at every terminal handler.
        self._pending_target: Path | None = None
        self._pending_run_id: str | None = None
        self._pending_parent_process_id: str | None = None
        self._pending_form_params: dict | None = None
        self._pending_dataparser: str | None = None  # effective dataparser subcommand
        self._pending_command: str | None = None
        self._pending_start_time: datetime | None = None
        self._pending_display_name: str | None = None
        # ns-train picks its own `<YYYY-MM-DD_HHMMSS>` slug AFTER it starts,
        # so timestamp_dir is captured at manifest-write time by scanning the
        # output dir — not stashed here.

        # Set True the moment we see nerfstudio's "Training Finished" banner
        # in stdout. _on_job_line uses it to write the manifest BEFORE the
        # process exits — necessary because viewer-stays-open runs never
        # terminate until the user clicks Cancel. _on_job_finished checks
        # this flag and skips its own manifest write so we don't double-write.
        # Reset to False at the top of _on_run for each new attempt.
        self._training_finalized = False

        # Training-resolution dropdown entries for the current dataset
        # (rebuilt by _refresh_resolution) and whether the running job is
        # our own "Add downscale levels" helper rather than a training run.
        self._resolution_options: list[topt.ResolutionOption] = []
        self._downscale_mine = False

        # Poll for viser readiness (port 7007) once training has started.
        self._viser_poll = QTimer(self)
        self._viser_poll.setInterval(1000)
        self._viser_poll.timeout.connect(self._check_viser)
        # After "Training Finished": poll nvidia-smi so the card can show what
        # the idle viewer process is still holding.
        self._gpu_held_poll = QTimer(self)
        self._gpu_held_poll.setInterval(5000)
        self._gpu_held_poll.timeout.connect(self._refresh_gpu_held)

        self._build()
        self._load_project()
        self._wire_jobs()

        # If a job is already running when this form is constructed (e.g. user
        # switched projects mid-run), catch up to that state.
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
        title = QLabel("Train")
        title.setObjectName("PageTitle")
        sub = QLabel("Train a gaussian splat from the processed dataset.")
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

        bl.addWidget(self._dataset_card())
        bl.addWidget(self._params_card())
        bl.addWidget(self._output_card())
        bl.addWidget(self._actions_card())

        bl.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll)

    # ---- dataset card ----
    def _dataset_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(8)
        ds_hdr = QHBoxLayout(); ds_hdr.setSpacing(6); ds_hdr.setContentsMargins(0, 0, 0, 0)
        ds_hdr.addWidget(section_title("DATASET"))
        ds_hdr.addWidget(_hb("dataset"))
        ds_hdr.addStretch(1)
        ds_wrap = QWidget(); ds_wrap.setLayout(ds_hdr)
        v.addWidget(ds_wrap)

        self.dataset_path = QLineEdit()
        self.dataset_path.setReadOnly(True)
        self.dataset_path.setProperty("role", "readout")
        v.addWidget(self.dataset_path)

        meta_row = QHBoxLayout(); meta_row.setSpacing(10); meta_row.setContentsMargins(0, 0, 0, 0)
        self.dataset_meta = QLabel("")
        self.dataset_meta.setProperty("role", "caption")
        # Pose-recovery pill: how many extracted frames got a camera. Colour
        # follows train_options.PoseRecovery.level (success/warning/error).
        self.pose_pill = pill("Poses: —", "neutral")
        meta_row.addWidget(self.dataset_meta, 1)
        meta_row.addWidget(self.pose_pill, 0, Qt.AlignRight)
        meta_wrap = QWidget(); meta_wrap.setLayout(meta_row)
        v.addWidget(meta_wrap)

        self.levels_caption = QLabel("")
        self.levels_caption.setProperty("role", "caption")
        self.levels_caption.setWordWrap(True)
        v.addWidget(self.levels_caption)
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

        # Method
        grid.addWidget(labeled("Method", *_HELP["method"]), 0, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(METHODS)
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        grid.addWidget(self.method_combo, 0, 1)

        # Max gaussians — splatfacto-mcmc only; the row is shown / hidden and
        # its cost readout refreshed in _refresh_max_gs().
        self.max_gs_label = labeled("Max gaussians", *_HELP["max_gs_num"])
        grid.addWidget(self.max_gs_label, 1, 0)
        gs_col = QVBoxLayout(); gs_col.setSpacing(4); gs_col.setContentsMargins(0, 0, 0, 0)
        self.max_gs_combo = QComboBox()
        for n in topt.MAX_GS_NUM_CHOICES:
            self.max_gs_combo.addItem(topt.max_gs_num_label(n), userData=n)
        self.max_gs_combo.currentIndexChanged.connect(self._refresh_max_gs)
        self.max_gs_caption = QLabel("")
        self.max_gs_caption.setProperty("role", "caption")
        self.max_gs_caption.setWordWrap(True)
        gs_col.addWidget(self.max_gs_combo)
        gs_col.addWidget(self.max_gs_caption)
        self.max_gs_wrap = QWidget(); self.max_gs_wrap.setLayout(gs_col)
        grid.addWidget(self.max_gs_wrap, 1, 1)

        # Training resolution — populated per dataset in _refresh_resolution().
        grid.addWidget(labeled("Training resolution", *_HELP["resolution"]), 2, 0)
        res_col = QVBoxLayout(); res_col.setSpacing(4); res_col.setContentsMargins(0, 0, 0, 0)
        res_row = QHBoxLayout(); res_row.setSpacing(10); res_row.setContentsMargins(0, 0, 0, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.setMinimumWidth(320)
        self.resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)
        self.add_levels_btn = QPushButton("Add downscale levels…")
        self.add_levels_btn.setProperty("role", "secondary")
        self.add_levels_btn.setCursor(Qt.PointingHandCursor)
        self.add_levels_btn.setToolTip(
            "Create images_2 / images_4 / images_8 next to images/ with "
            "nerfstudio's own downscaler (ffmpeg, inside the container), so "
            "the resolution entries above become folder-backed. One-off; "
            "about 3 minutes for 266 4K frames."
        )
        self.add_levels_btn.clicked.connect(self._on_add_levels)
        res_row.addWidget(self.resolution_combo, 1)
        res_row.addWidget(self.add_levels_btn, 0)
        res_wrap = QWidget(); res_wrap.setLayout(res_row)
        # Live readout: frame cache size at this resolution + what Auto cache picks.
        self.resolution_caption = QLabel("")
        self.resolution_caption.setProperty("role", "caption")
        self.resolution_caption.setWordWrap(True)
        res_col.addWidget(res_wrap)
        res_col.addWidget(self.resolution_caption)
        res_col_wrap = QWidget(); res_col_wrap.setLayout(res_col)
        grid.addWidget(res_col_wrap, 2, 1)

        # Max iterations
        grid.addWidget(labeled("Max iterations", *_HELP["iters"]), 3, 0)
        it_col = QVBoxLayout(); it_col.setSpacing(6); it_col.setContentsMargins(0, 0, 0, 0)
        it_row = QHBoxLayout(); it_row.setSpacing(10); it_row.setContentsMargins(0, 0, 0, 0)
        self.iters_slider = QSlider(Qt.Horizontal)
        self.iters_slider.setRange(1000, 60000)
        self.iters_slider.setSingleStep(500)
        self.iters_slider.setPageStep(1000)
        self.iters_value = QSpinBox()
        self.iters_value.setRange(1000, 60000)
        self.iters_value.setSingleStep(500)
        self.iters_value.setMinimumWidth(90)
        self.iters_slider.valueChanged.connect(self.iters_value.setValue)
        self.iters_value.valueChanged.connect(self.iters_slider.setValue)
        self.iters_value.valueChanged.connect(self._refresh_iters_caption)
        it_row.addWidget(self.iters_slider, 1)
        it_row.addWidget(self.iters_value)
        it_wrap = QWidget(); it_wrap.setLayout(it_row)
        # Quick picks + a caption naming the method's upstream default.
        chips = QHBoxLayout(); chips.setSpacing(8); chips.setContentsMargins(0, 0, 0, 0)
        self.iter_chip_btns: list[QPushButton] = []
        for label, steps in _ITER_PRESETS:
            b = QPushButton(label)
            b.setProperty("role", "ghost")
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(f"Set Max iterations to {steps:,}")
            b.clicked.connect(lambda _c=False, s=steps: self.iters_value.setValue(s))
            chips.addWidget(b)
            self.iter_chip_btns.append(b)
        self.iters_caption = QLabel("")
        self.iters_caption.setProperty("role", "caption")
        chips.addWidget(self.iters_caption, 1)
        chips_wrap = QWidget(); chips_wrap.setLayout(chips)
        it_col.addWidget(it_wrap)
        it_col.addWidget(chips_wrap)
        it_col_wrap = QWidget(); it_col_wrap.setLayout(it_col)
        grid.addWidget(it_col_wrap, 3, 1)

        # Steps per save (+ inline warning above _SAVE_STEP_WARN)
        grid.addWidget(labeled("Steps per save", *_HELP["save_step"]), 4, 0)
        self.save_step_spin = QSpinBox()
        self.save_step_spin.setRange(100, 10000)
        self.save_step_spin.setSingleStep(100)
        self.save_step_spin.setMaximumWidth(100)
        self.save_step_spin.valueChanged.connect(self._on_save_step_changed)
        self.save_step_warn = QLabel("")
        self.save_step_warn.setProperty("role", "caption")
        ss_row = QHBoxLayout(); ss_row.setSpacing(10)
        ss_row.addWidget(self.save_step_spin); ss_row.addWidget(self.save_step_warn, 1)
        ss_wrap = QWidget(); ss_wrap.setLayout(ss_row)
        grid.addWidget(ss_wrap, 4, 1)

        # Visualization
        grid.addWidget(labeled("Visualization", *_HELP["vis"]), 5, 0)
        self.vis_combo = QComboBox()
        self.vis_combo.addItems(VIS_MODES)
        grid.addWidget(self.vis_combo, 5, 1)

        # Quit on completion
        qc_row = QHBoxLayout(); qc_row.setSpacing(6); qc_row.setContentsMargins(0, 0, 0, 0)
        self.quit_on_completion_cb = QCheckBox("Quit viewer when training finishes")
        qc_row.addWidget(self.quit_on_completion_cb); qc_row.addWidget(_hb("quit_on_completion")); qc_row.addStretch(1)
        qc_wrap = QWidget(); qc_wrap.setLayout(qc_row)
        grid.addWidget(qc_wrap, 6, 1)

        v.addLayout(grid)

        # Advanced panel
        self.advanced_wrap = QFrame()
        self.advanced_wrap.setProperty("role", "subcard")
        av = QVBoxLayout(self.advanced_wrap)
        av.setContentsMargins(0, 8, 0, 0); av.setSpacing(8)

        mem_hdr = QHBoxLayout(); mem_hdr.setSpacing(6); mem_hdr.setContentsMargins(0, 0, 0, 0)
        mem_hdr.addWidget(section_title("ADVANCED · MEMORY & EVAL"))
        mem_hdr.addStretch(1)
        mem_wrap = QWidget(); mem_wrap.setLayout(mem_hdr)
        av.addWidget(mem_wrap)

        agrid = QGridLayout()
        agrid.setHorizontalSpacing(14); agrid.setVerticalSpacing(12)
        agrid.setColumnStretch(1, 1)

        agrid.addWidget(labeled("Image cache", *_HELP["cache_images"]), 0, 0)
        self.cache_combo = QComboBox()
        for m in CACHE_MODES:
            self.cache_combo.addItem(_CACHE_LABELS[m], userData=m)
        self.cache_combo.currentIndexChanged.connect(lambda _i: self._refresh_resolution_caption())
        agrid.addWidget(self.cache_combo, 0, 1)

        agrid.addWidget(labeled("Eval cadence", *_HELP["eval_steps"]), 1, 0)
        ev_row = QHBoxLayout(); ev_row.setSpacing(8); ev_row.setContentsMargins(0, 0, 0, 0)
        self.eval_image_spin = QSpinBox()
        self.eval_image_spin.setRange(50, 100000); self.eval_image_spin.setSingleStep(100)
        self.eval_image_spin.setValue(topt.DEFAULT_STEPS_PER_EVAL_IMAGE)
        self.eval_image_spin.setMaximumWidth(110)
        self.eval_all_spin = QSpinBox()
        self.eval_all_spin.setRange(100, 100000); self.eval_all_spin.setSingleStep(500)
        self.eval_all_spin.setValue(topt.DEFAULT_STEPS_PER_EVAL_ALL_IMAGES)
        self.eval_all_spin.setMaximumWidth(110)
        lbl_a = QLabel("one image every"); lbl_a.setProperty("role", "caption")
        lbl_b = QLabel("steps · all images every"); lbl_b.setProperty("role", "caption")
        lbl_c = QLabel("steps"); lbl_c.setProperty("role", "caption")
        for w in (lbl_a, self.eval_image_spin, lbl_b, self.eval_all_spin, lbl_c):
            ev_row.addWidget(w)
        ev_row.addStretch(1)
        ev_wrap = QWidget(); ev_wrap.setLayout(ev_row)
        agrid.addWidget(ev_wrap, 1, 1)

        agrid.addWidget(labeled("Rasterize mode", *_HELP["rasterize_mode"]), 2, 0)
        self.raster_combo = QComboBox()
        for m in RASTER_MODES:
            self.raster_combo.addItem(_RASTER_LABELS[m], userData=m)
        agrid.addWidget(self.raster_combo, 2, 1)
        av.addLayout(agrid)

        adv_hdr = QHBoxLayout(); adv_hdr.setSpacing(6); adv_hdr.setContentsMargins(0, 8, 0, 0)
        adv_hdr.addWidget(section_title("ADVANCED · RAW CLI FLAGS"))
        adv_hdr.addWidget(_hb("extra_flags"))
        adv_hdr.addStretch(1)
        adv_wrap = QWidget(); adv_wrap.setLayout(adv_hdr)
        av.addWidget(adv_wrap)

        self.extra_flags_edit = QLineEdit()
        self.extra_flags_edit.setPlaceholderText(
            "--pipeline.model.cull-alpha-thresh 0.005 …"
        )
        av.addWidget(self.extra_flags_edit)

        hint = QLabel(
            "Passed through verbatim to ns-train. "
            "See: docker exec splat-ns-v2 ns-train splatfacto --help"
        )
        hint.setProperty("role", "caption")
        hint.setWordWrap(True)
        av.addWidget(hint)

        self.advanced_wrap.hide()
        v.addWidget(self.advanced_wrap)
        return c

    # ---- output card ----
    def _output_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(8)
        out_hdr = QHBoxLayout(); out_hdr.setSpacing(6); out_hdr.setContentsMargins(0, 0, 0, 0)
        out_hdr.addWidget(section_title("OUTPUT RUN"))
        out_hdr.addWidget(_hb("output"))
        out_hdr.addStretch(1)
        out_wrap = QWidget(); out_wrap.setLayout(out_hdr)
        v.addWidget(out_wrap)
        self.output_lbl = QLineEdit()
        self.output_lbl.setReadOnly(True)
        self.output_lbl.setProperty("role", "readout")
        v.addWidget(self.output_lbl)
        hint = QLabel(
            f"Each Run training writes to {WAREHOUSE_DIR.name}/trained/<new ULID>/. "
            "ns-train nests its own <experiment>/<method>/<timestamp>/ subfolder "
            "underneath; the ULID dir holds the manifest plus that nested run."
        )
        hint.setProperty("role", "caption")
        hint.setWordWrap(True)
        v.addWidget(hint)
        return c

    # ---- actions ----
    def _actions_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(10)

        row = QHBoxLayout(); row.setSpacing(10)
        self.run_btn = QPushButton("Run training")
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

        # Viser-ready banner — shows once port 7007 starts listening.
        self.viser_card = QFrame()
        self.viser_card.setProperty("role", "subcard")
        vcol = QVBoxLayout(self.viser_card)
        vcol.setContentsMargins(14, 10, 14, 10); vcol.setSpacing(6)
        vs = QHBoxLayout(); vs.setContentsMargins(0, 0, 0, 0)
        self.viser_ok_lbl = QLabel("✓ Viser server is live.")
        self.viser_ok_lbl.setStyleSheet("color: #22C55E; font-weight: 600;")
        self.open_viser_btn = QPushButton(f"Open http://localhost:{VISER_PORT}")
        self.open_viser_btn.setProperty("role", "primary")
        self.open_viser_btn.setCursor(Qt.PointingHandCursor)
        self.open_viser_btn.clicked.connect(self._open_viser_in_browser)
        self.viewers_btn = QPushButton("Go to Viewers →")
        self.viewers_btn.setProperty("role", "secondary")
        self.viewers_btn.setCursor(Qt.PointingHandCursor)
        self.viewers_btn.clicked.connect(self.request_viewers.emit)
        vs.addWidget(self.viser_ok_lbl)
        vs.addStretch(1)
        vs.addWidget(self.open_viser_btn)
        vs.addWidget(self.viewers_btn)
        vs_wrap = QWidget(); vs_wrap.setLayout(vs)
        # Shown after the "Training Finished" banner while the process stays
        # alive for the viewer: how much GPU memory it is still holding, and a
        # one-click release. A finished run once held 10.7 GB for 15 hours.
        held = QHBoxLayout(); held.setContentsMargins(0, 0, 0, 0); held.setSpacing(10)
        self.gpu_held_lbl = QLabel("")
        self.gpu_held_lbl.setProperty("role", "caption")
        self.gpu_held_lbl.setWordWrap(True)
        self.release_gpu_btn = QPushButton("Release GPU (stop viewer)")
        self.release_gpu_btn.setProperty("role", "secondary")
        self.release_gpu_btn.setCursor(Qt.PointingHandCursor)
        self.release_gpu_btn.setToolTip(
            "Stops the finished ns-train process that is only serving the viewer. "
            "The checkpoint and manifest are already on disk; reopen the run from the Viewers tab."
        )
        self.release_gpu_btn.clicked.connect(self._on_cancel)
        held.addWidget(self.gpu_held_lbl, 1)
        held.addWidget(self.release_gpu_btn, 0)
        self.gpu_held_wrap = QWidget(); self.gpu_held_wrap.setLayout(held)
        self.gpu_held_wrap.hide()
        vcol.addWidget(vs_wrap)
        vcol.addWidget(self.gpu_held_wrap)
        self.viser_card.hide()

        v.addLayout(row)
        v.addWidget(self.viser_card)
        return c

    # ---- advanced toggle ----
    def _on_advanced(self, checked: bool) -> None:
        self.advanced_wrap.setVisible(checked)

    # ---- project lifecycle ----
    def _load_project(self) -> None:
        p = self._project
        self.dataset_path.setText(p.processed_dir)
        self._refresh_dataset_card()
        self._refresh_resolution()

        # Output path isn't knowable until _on_run mints a ULID. Show the
        # policy here so the user sees where the next run will land.
        self._refresh_next_target()

        self._apply_params(p.last_train_params or {})
        self._refresh_presets()

    def _apply_params(self, params: dict) -> None:
        """Push a saved form dict (last-used params or a preset) into the
        widgets. Unknown / missing keys fall back to the defaults, so presets
        saved before the Tier-2 controls existed load unchanged."""
        method = params.get("method", "splatfacto")
        if method in METHODS:
            self.method_combo.setCurrentIndex(METHODS.index(method))
        iters = int(params.get("max_num_iterations", 30000))
        self.iters_slider.setValue(iters); self.iters_value.setValue(iters)
        self.save_step_spin.setValue(int(params.get("steps_per_save", 2000)))
        vis = params.get("vis", "viewer+tensorboard")
        if vis in VIS_MODES:
            self.vis_combo.setCurrentIndex(VIS_MODES.index(vis))
        self.quit_on_completion_cb.setChecked(
            bool(params.get("quit_on_train_completion", False))
        )
        self.extra_flags_edit.setText(params.get("extra_flags", ""))
        self._refresh_iters_caption()

        res_idx = self.resolution_combo.findData(params.get("resolution", "auto"))
        self.resolution_combo.setCurrentIndex(res_idx if res_idx >= 0 else 0)
        cache_idx = self.cache_combo.findData(params.get("cache_images", "auto"))
        self.cache_combo.setCurrentIndex(cache_idx if cache_idx >= 0 else 0)
        self.eval_image_spin.setValue(
            int(params.get("steps_per_eval_image", topt.DEFAULT_STEPS_PER_EVAL_IMAGE))
        )
        self.eval_all_spin.setValue(
            int(params.get("steps_per_eval_all_images", topt.DEFAULT_STEPS_PER_EVAL_ALL_IMAGES))
        )
        raster_idx = self.raster_combo.findData(params.get("rasterize_mode", "default"))
        self.raster_combo.setCurrentIndex(raster_idx if raster_idx >= 0 else 0)
        gs_idx = self.max_gs_combo.findData(params.get("max_gs_num", topt.MAX_GS_NUM_DEFAULT))
        self.max_gs_combo.setCurrentIndex(gs_idx if gs_idx >= 0 else 0)
        self._refresh_resolution_caption()
        self._refresh_max_gs()

    def _on_method_changed(self, _text: str) -> None:
        self._refresh_iters_caption()
        self._refresh_max_gs()

    def _refresh_max_gs(self, *_args) -> None:
        """Max gaussians only applies to splatfacto-mcmc: show its row for
        that method and keep the cost readout in step with the chosen budget
        and training resolution."""
        is_mcmc = self.method_combo.currentText() == topt.MCMC_METHOD
        self.max_gs_label.setVisible(is_mcmc)
        self.max_gs_wrap.setVisible(is_mcmc)
        if not is_mcmc:
            return
        opt = self._current_resolution()
        budget = self.max_gs_combo.currentData() or topt.MAX_GS_NUM_DEFAULT
        self.max_gs_caption.setText(
            topt.max_gs_num_caption(int(budget), max(opt.width, opt.height) if opt else None)
        )

    def _refresh_iters_caption(self, *_args) -> None:
        method = self.method_combo.currentText()
        upstream = _UPSTREAM_ITERS.get(method, 30000)
        cur = self.iters_value.value()
        if cur == upstream:
            self.iters_caption.setText(f"= {method}'s upstream default")
        elif cur < upstream:
            self.iters_caption.setText(
                f"{method}'s upstream default is {upstream:,} — shorter runs stop mid-schedule"
            )
        else:
            self.iters_caption.setText(f"{method}'s upstream default is {upstream:,}")

    # ---- dataset card / resolution helpers ----
    def _refresh_dataset_card(self) -> None:
        p = self._project
        self.dataset_meta.setText(_dataset_meta_text(p))
        pr = topt.pose_recovery(Path(p.processed_dir))
        if pr.ratio is None:
            set_pill(self.pose_pill, "Poses: unknown", "neutral")
        else:
            set_pill(
                self.pose_pill,
                f"Poses: {pr.registered} of {pr.extracted} frames ({pr.ratio:.0%})",
                pr.level,
            )
        levels = topt.dataset_image_levels(Path(p.processed_dir))
        if levels:
            full = levels[1] if 1 in levels else None
            names = ", ".join(f"images_{f} ({levels[f].width}×{levels[f].height})" for f in sorted(levels) if f > 1)
            head = f"Frames {full.width}×{full.height}" if full else "Frames"
            self.levels_caption.setText(
                f"{head} · downscaled copies: {names if names else 'none — use Add downscale levels'}"
            )
        else:
            self.levels_caption.setText("")

    def _refresh_resolution(self) -> None:
        """Rebuild the Training-resolution dropdown from what is on disk,
        keeping the current selection when its key still exists."""
        current = self.resolution_combo.currentData()
        self._resolution_options = topt.resolution_options(Path(self._project.processed_dir))
        self.resolution_combo.blockSignals(True)
        self.resolution_combo.clear()
        for o in self._resolution_options:
            self.resolution_combo.addItem(o.label, userData=o.key)
        idx = self.resolution_combo.findData(current) if current else -1
        self.resolution_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.resolution_combo.blockSignals(False)
        self.add_levels_btn.setVisible(
            bool(topt.missing_levels(Path(self._project.processed_dir)))
        )
        self._refresh_resolution_caption()
        self._refresh_max_gs()

    def _current_resolution(self) -> topt.ResolutionOption | None:
        return topt.find_option(self._resolution_options, self.resolution_combo.currentData())

    def _frame_count(self) -> int:
        n = _count_frames(self._project.processed_dir)
        return int(n) if n else 0

    def _refresh_resolution_caption(self) -> None:
        opt = self._current_resolution()
        if opt is None:
            self.resolution_caption.setText("")
            return
        frames = self._frame_count()
        gib = topt.estimate_cache_gib(frames, opt.width, opt.height)
        mode = self.cache_combo.currentData() or "auto"
        effective = topt.choose_cache_images(mode, frames, opt.width, opt.height) or "gpu"
        where = "GPU" if effective == "gpu" else "host RAM (CPU cache)"
        note = ""
        if not opt.via_folder and not opt.is_auto and opt.factor > 1:
            note = " · frames are resized while loading; Add downscale levels makes this a folder-backed copy"
        self.resolution_caption.setText(
            f"{frames} frames × {opt.width}×{opt.height} ≈ {gib:.2f} GiB of image cache → kept on {where}{note}"
        )

    def _on_resolution_changed(self, _idx: int) -> None:
        self._refresh_resolution_caption()
        self._refresh_max_gs()

    def _on_save_step_changed(self, value: int) -> None:
        if value > _SAVE_STEP_WARN:
            self.save_step_warn.setText(
                f"⚠ a crash before step {value} leaves no checkpoint — consider ≤ {_SAVE_STEP_WARN}"
            )
            self.save_step_warn.setStyleSheet("color: #8A5A00;")
        else:
            self.save_step_warn.setText("")
            self.save_step_warn.setStyleSheet("")

    def _resolved_params(self, params: dict) -> dict:
        """Turn the dropdown choices into the effective values that go into
        the command line and the TrainManifest. Kept separate from the raw
        form dict so presets stay portable between datasets."""
        opt = self._current_resolution()
        frames = self._frame_count()
        cache_eff = (
            topt.choose_cache_images(params["cache_images"], frames, opt.width, opt.height)
            if opt is not None else None
        )
        raster = params["rasterize_mode"]
        try:
            extras = shlex.split(params.get("extra_flags") or "")
        except ValueError:  # _on_run rejects unbalanced quotes before this matters
            extras = []
        return {
            "resolution_label": opt.label if opt else "",
            "train_width": opt.width if opt else None,
            "train_height": opt.height if opt else None,
            "downscale_factor": (
                opt.factor if (opt and opt.via_folder and not opt.is_auto and opt.factor > 1) else None
            ),
            "camera_res_scale_factor": (
                1.0 / opt.factor if (opt and not opt.via_folder and not opt.is_auto and opt.factor > 1) else None
            ),
            "cache_images_effective": cache_eff,
            "rasterize_mode_effective": None if raster == "default" else raster,
            "max_gs_num_effective": topt.choose_max_gs_num(
                params["method"], params.get("max_gs_num"), extras
            ),
        }

    def _on_add_levels(self) -> None:
        """Run nerfstudio's own downscaler inside the container to create the
        missing images_2/4/8 folders next to images/. A JobManager job (kind
        'downscale') so it streams into the log drawer and can be cancelled."""
        if self._jobs.is_running():
            QMessageBox.information(self, "Busy", "Another job is already running. Cancel it first.")
            return
        state = dock.container_state(CONTAINER_NAME)
        if state.status != "running":
            QMessageBox.warning(self, "Container not running",
                                f"The {CONTAINER_NAME} container isn't running. Start it from the Dashboard.")
            return
        processed = Path(self._project.processed_dir)
        missing = topt.missing_levels(processed)
        if not missing:
            return
        try:
            c_data = wh.host_to_container_warehouse(processed)
        except ValueError:
            try:
                from app.views.process import _host_to_container
                c_data = _host_to_container(processed)
            except Exception as e:
                QMessageBox.warning(self, "Path translation failed", str(e))
                return
        # downscale_images(image_dir, num_downscales) writes images_2 .. images_2^n
        # beside image_dir; 3 levels reaches images_8 (the deepest the Train
        # tab offers). Existing folders are rewritten identically.
        n_levels = max(1, int(round(__import__("math").log2(max(missing)))))
        code = (
            "from pathlib import Path; "
            "from nerfstudio.process_data.process_data_utils import downscale_images; "
            f"print(downscale_images(Path({c_data + '/images'!r}), {n_levels}, verbose=True))"
        )
        argv = ["python", "-c", code]
        display = f"docker exec {CONTAINER_NAME} python -c '<downscale_images {c_data}/images, {n_levels} levels>'"
        resp = QMessageBox.question(
            self, "Add downscale levels",
            f"Create {', '.join(f'images_{f}' for f in missing)} next to images/ for "
            f"'{self._project.name}' using nerfstudio's downscaler (ffmpeg in the container)?\n\n"
            "This is a one-off and typically takes a few minutes for ~250 4K frames.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if resp != QMessageBox.Yes:
            return
        self._downscale_mine = True
        self._jobs.start(JobSpec(
            kind="downscale", display_cmd=display, container_argv=argv,
            kill_pattern="downscale_images",
            log_path=processed / "downscale.log",
        ))

    def _refresh_presets(self) -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("— custom —", userData=None)
        for name in sorted(self._project.presets_train.keys()):
            self.preset_combo.addItem(name, userData=name)
        self.preset_combo.blockSignals(False)

    def _on_preset_selected(self, idx: int) -> None:
        name = self.preset_combo.itemData(idx)
        if not name:
            return
        preset = self._project.presets_train.get(name, {})
        if not preset:
            return
        self._apply_params(preset)

    def _on_save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        self._store.save_preset(
            self._project.id, "train", name, self._collect_params()
        )
        self._refresh_presets()
        idx = self.preset_combo.findData(name)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)

    # ---- collect + run ----
    def _collect_params(self) -> dict:
        return {
            "method": self.method_combo.currentText(),
            "max_num_iterations": self.iters_value.value(),
            "steps_per_save": self.save_step_spin.value(),
            "vis": self.vis_combo.currentText(),
            "quit_on_train_completion": self.quit_on_completion_cb.isChecked(),
            "extra_flags": self.extra_flags_edit.text().strip(),
            # Tier-2 controls (keys, not resolved values — see _resolved_params)
            "resolution": self.resolution_combo.currentData() or "auto",
            "cache_images": self.cache_combo.currentData() or "auto",
            "steps_per_eval_image": self.eval_image_spin.value(),
            "steps_per_eval_all_images": self.eval_all_spin.value(),
            "rasterize_mode": self.raster_combo.currentData() or "default",
            # splatfacto-mcmc budget; kept on the form for every method so a
            # preset survives a method switch (see _resolved_params).
            "max_gs_num": int(self.max_gs_combo.currentData() or topt.MAX_GS_NUM_DEFAULT),
        }

    def _on_run(self) -> None:
        # Disable the Run button SYNCHRONOUSLY so a quick double-click can't
        # mint two train ULIDs and start two redundant jobs in parallel.
        # The job-start path keeps it disabled via _on_job_started; any
        # early-return below uses the `started` flag + finally clause to
        # restore the idle state.
        self.run_btn.setEnabled(False)
        # Fresh attempt: clear any "training finished" flag leftover from a
        # previous run that already finalized. (The flag intentionally
        # survives _clear_pending_state so the banner-detection path and the
        # process-exit path can both consult it.)
        self._training_finalized = False
        started = False
        try:
            if self._jobs.is_running():
                QMessageBox.information(
                    self, "Busy",
                    "Another job is already running. Cancel it first.",
                )
                return

            state = dock.container_state(CONTAINER_NAME)
            if state.status != "running":
                QMessageBox.warning(
                    self, "Container not running",
                    f"The {CONTAINER_NAME} container isn't running. "
                    "Start it from the Dashboard, then try again.",
                )
                return

            if not _is_trainable(self._project):
                marker = _marker_file(self._project)
                QMessageBox.warning(
                    self, "Missing dataset",
                    f"Expected file not found:\n{marker}\n\n"
                    "Restore the dataset folder, or (for video- or images-sourced "
                    "projects) re-run Process.",
                )
                # Imported projects have no source to re-process — only offer
                # the jump-to-Process shortcut for video / images projects.
                if source_kind(self._project) != "imported":
                    self.request_process.emit()
                return

            # Block legacy data/processed/*_splat/ datasets (no parent
            # ProcessManifest = nothing to point TrainManifest.parent at,
            # and the schema requires parent). Backfill is a separate task.
            processed_path = Path(self._project.processed_dir).resolve()
            try:
                is_warehouse = processed_path.is_relative_to(WAREHOUSE_DIR.resolve())
            except ValueError:
                is_warehouse = False
            if not is_warehouse:
                QMessageBox.warning(
                    self, "Legacy dataset — reprocess required",
                    f"'{self._project.name}' is pointed at\n"
                    f"{processed_path}\n\n"
                    "which predates the warehouse and has no manifest.json. "
                    "TrainManifest needs a declared parent ProcessManifest, "
                    "so this dataset can't be trained until you reprocess it "
                    "via the Process tab.",
                )
                if source_kind(self._project) != "imported":
                    self.request_process.emit()
                return
            parent_manifest = wh.find_parent_process_manifest(processed_path)
            if parent_manifest is None:
                QMessageBox.warning(
                    self, "Missing parent manifest",
                    f"No manifest.json found at\n{processed_path}/manifest.json\n\n"
                    "Reprocess via the Process tab so a lineage record exists.",
                )
                if source_kind(self._project) != "imported":
                    self.request_process.emit()
                return

            params = self._collect_params()

            # ---- pre-flight (nothing written to disk yet) ----
            try:
                extras = shlex.split(params["extra_flags"]) if params["extra_flags"] else []
            except ValueError as e:
                QMessageBox.warning(self, "Bad extra flags", str(e))
                return

            colmap_rel: str | None = None
            if params["method"] == "splatfacto-w":
                # splatfacto-w's registered default is NerfWDataParser whose
                # colmap_path defaults to "dense/sparse"; the Train tab
                # overrides it via the splatfactow_dataparser subcommand.
                # splatfacto-w-light is deliberately NOT routed that way: its
                # upstream default is nerfstudio-data, which the author
                # recommends for non-phototourism captures.
                colmap_rel = detect_colmap_path(Path(self._project.processed_dir))
                if not colmap_rel:
                    QMessageBox.warning(
                        self, "splatfacto-w needs a COLMAP sparse model",
                        f"'{self._project.name}' has no "
                        "sparse/0/cameras.bin or colmap/sparse/0/cameras.bin, "
                        "so splatfacto-w's dataparser cannot load it.\n\n"
                        "Use splatfacto-w-light (works on any processed "
                        "dataset), or reprocess the source through the "
                        "Process tab to get a COLMAP model.",
                    )
                    return
            elif self._project.dataparser == "colmap":
                # Always forward the stored path rather than relying on
                # ColmapDataParserConfig's default ("colmap/sparse/0"), which
                # doesn't match the plain-COLMAP ("sparse/0") layout.
                colmap_rel = _colmap_rel(self._project)

            # Pose-recovery gate: ns-process-data exits 0 even when COLMAP
            # registered a handful of frames. Below POSE_RECOVERY_BLOCK_RATIO
            # the result is not worth the GPU time without a conscious yes.
            pr = topt.pose_recovery(Path(self._project.processed_dir))
            if pr.ratio is not None and pr.ratio < topt.POSE_RECOVERY_BLOCK_RATIO:
                resp = QMessageBox.question(
                    self, "Few frames have poses",
                    f"Only {pr.registered} of {pr.extracted} extracted frames "
                    f"({pr.ratio:.0%}) received a camera pose during processing. "
                    "A splat trained on this will cover a fraction of the scene.\n\n"
                    "Usually fixable by reprocessing with more frames, exhaustive "
                    "matching, or --sfm-tool hloc.\n\nTrain anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if resp != QMessageBox.Yes:
                    return

            # Free-VRAM pre-flight. The usual culprit is a finished run whose
            # viewer is still open (10.7 GB held for 15 h once); the port check
            # below only catches it when the viewer is requested again.
            stats = nv.query_gpu()
            if stats is not None:
                free_gib = (stats.mem_total_mb - stats.mem_used_mb) / 1024
                # A splatfacto-mcmc budget above the default needs more than
                # the generic 3 GiB floor: use the peak measured for it.
                budget = topt.choose_max_gs_num(params["method"], params.get("max_gs_num"), extras)
                cost = topt.mcmc_budget_cost(budget) if budget else None
                need_gib = max(3.0, cost.peak_vram_gib if cost else 0.0)
                budget_note = (
                    f"splatfacto-mcmc with {topt.max_gs_num_label(budget)} gaussians needs about "
                    f"{cost.peak_vram_gib:.1f} GiB at {topt.MCMC_MEASURED_WIDTH} px "
                    f"({'measured' if cost.measured else 'interpolated'}). "
                    if cost and cost.peak_vram_gib > 3.0 else ""
                )
                if free_gib < need_gib:
                    resp = QMessageBox.question(
                        self, "Little GPU memory free",
                        f"Only {free_gib:.1f} GiB of {stats.mem_total_mb / 1024:.1f} GiB VRAM is free "
                        f"({stats.mem_used_mb / 1024:.1f} GiB in use). {budget_note}A viewer from a finished run, "
                        "ns-viewer, or another process is probably holding it — see the Viewers tab.\n\n"
                        "Start training anyway?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                    )
                    if resp != QMessageBox.Yes:
                        return

            resolution = self._current_resolution()
            resolved = self._resolved_params(params)
            # Everything the manifest needs: the raw form keys plus the
            # effective values. Stored on the pending sidecar too, so a
            # backfill reconstructs the same TrainParams.
            form_params = {**params, **resolved}
            # The dataparser subcommand ns-train actually receives — recorded
            # as such in the manifest (splatfacto-w uses its own parser).
            effective_dataparser = (
                "splatfactow_dataparser" if params["method"] == "splatfacto-w"
                else self._project.dataparser
            )

            # If the viewer was requested, make sure nothing INSIDE the container
            # is already bound to viser's port. The host-side port is always held
            # by docker-proxy while splat-ns is running — that's the port mapping
            # itself and is expected. A real conflict only exists when a leftover
            # ns-train/ns-viewer is still holding :VISER_PORT inside the container.
            if "viewer" in params["vis"]:
                busy = dock.container_port_in_use(VISER_PORT)
                if busy is True:
                    resp = QMessageBox.question(
                        self, "Viser port is busy inside the container",
                        f"Something inside {CONTAINER_NAME} is already bound to "
                        f":{VISER_PORT} — most likely a leftover ns-train or "
                        "ns-viewer from a previous run.\n\n"
                        "Terminate it (SIGTERM → SIGKILL) and continue?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes,
                    )
                    if resp != QMessageBox.Yes:
                        return
                    if not dock.free_container_port(VISER_PORT):
                        QMessageBox.warning(
                            self, "Couldn't free port",
                            f"Port {VISER_PORT} inside {CONTAINER_NAME} is still "
                            "busy after SIGKILL. Inspect with:\n\n"
                            f"  docker exec {CONTAINER_NAME} ps -ef\n\n"
                            "and stop the offender manually.",
                        )
                        return

            self._store.update_last_train_params(self._project.id, params)

            # Mint a train ULID and reserve its output directory under
            # data_v2/trained/<ULID>/. Pre-create as the host user so the
            # post-run manifest.json write (host-side) doesn't EACCES against
            # a root-owned dir created by the container.
            train_run_id = wh.mint_ulid()
            target = wh.run_dir("train", train_run_id)
            target.mkdir(parents=True, exist_ok=True)

            try:
                c_data = wh.host_to_container_warehouse(processed_path)
                c_out = wh.host_to_container_warehouse(target)
            except ValueError as e:
                QMessageBox.warning(self, "Path translation failed", str(e))
                return

            argv = topt.build_train_argv(
                method=params["method"],
                data=c_data,
                output_dir=c_out,
                experiment_name=self._project.id,
                max_num_iterations=int(params["max_num_iterations"]),
                steps_per_save=int(params["steps_per_save"]),
                vis=params["vis"],
                quit_on_train_completion=bool(params["quit_on_train_completion"]),
                steps_per_eval_image=int(params["steps_per_eval_image"]),
                steps_per_eval_all_images=int(params["steps_per_eval_all_images"]),
                cache_images=resolved["cache_images_effective"],
                rasterize_mode=resolved["rasterize_mode_effective"],
                max_gs_num=resolved["max_gs_num_effective"],
                resolution=resolution,
                extra_flags=extras,
                project_dataparser=self._project.dataparser,
                colmap_path=colmap_rel,
            )

            display = (
                f"docker exec {CONTAINER_NAME} "
                + " ".join(shlex.quote(a) for a in argv)
            )

            spec = JobSpec(
                kind="train",
                display_cmd=display,
                container_argv=argv,
                kill_pattern="ns-train",
                # Every stdout line lands next to pending.json / manifest.json,
                # so a failed or cancelled run leaves evidence on disk.
                log_path=target / "job.log",
            )
            self._pending_target = target
            self._pending_run_id = train_run_id
            self._pending_parent_process_id = parent_manifest.run_id
            self._pending_form_params = form_params
            self._pending_dataparser = effective_dataparser
            self._pending_command = display
            self._pending_start_time = datetime.now(timezone.utc)
            self._pending_display_name = self._project.name or self._project.id
            # Write-ahead log: persist the lineage breadcrumb to disk BEFORE
            # starting the job. If the GUI dies between now and job-finish
            # (a real failure mode that produced the existing orphan), the
            # backfill helper can finalize the manifest from this sidecar
            # plus the on-disk config.yml. Wide try/except: a sidecar write
            # failure is non-fatal — we'd rather miss the breadcrumb than
            # refuse to run training.
            try:
                wh.write_pending_train(
                    target,
                    run_id=train_run_id,
                    parent_process_id=parent_manifest.run_id,
                    started_at=self._pending_start_time,
                    command=display,
                    display_name=self._pending_display_name,
                    form_params=form_params,
                    dataparser=effective_dataparser,
                    experiment_name=self._project.id,
                )
            except Exception as e:
                # Surface but don't block — print to stderr-equivalent via
                # the job log line stream so it's at least visible in the
                # output drawer.
                self._jobs.line.emit(
                    f"[warn] pending sidecar write failed: {e!r}"
                )
            self.viser_card.hide()
            self.output_lbl.setText(str(target))
            self._jobs.start(spec)
            started = True
        finally:
            if not started:
                # Any pre-start bailout: discard the pending target and let
                # the user click Run again.
                self._clear_pending_state()
                self.run_btn.setEnabled(True)

    def _on_cancel(self) -> None:
        self._jobs.cancel()

    # ---- viser readiness ----
    def _check_viser(self) -> None:
        # Probe inside the container, not the host — docker-proxy's host-side
        # listen is a false positive (it accepts before the backend exists).
        if dock.container_port_in_use(VISER_PORT) is True:
            self._viser_poll.stop()
            self.viser_card.show()

    def _open_viser_in_browser(self) -> None:
        webbrowser.open(f"http://localhost:{VISER_PORT}")

    def _refresh_gpu_held(self) -> None:
        stats = nv.query_gpu()
        if stats is None:
            self.gpu_held_lbl.setText("The finished run keeps the GPU allocated while its viewer is open.")
            return
        used, total = stats.mem_used_mb / 1024, stats.mem_total_mb / 1024
        self.gpu_held_lbl.setText(
            f"GPU memory in use: {used:.1f} of {total:.1f} GiB while this viewer stays open — "
            f"the next training run only gets what is left."
        )

    def _stop_gpu_held(self) -> None:
        self._gpu_held_poll.stop()
        self.gpu_held_wrap.hide()
        self.viser_ok_lbl.setText("✓ Viser server is live.")

    # ---- job signals (filtered by kind) ----
    def _wire_jobs(self) -> None:
        self._jobs.started.connect(self._on_job_started)
        self._jobs.line.connect(self._on_job_line)
        self._jobs.finished.connect(self._on_job_finished)
        self._jobs.cancelled.connect(self._on_job_cancelled)
        self._jobs.failed.connect(self._on_job_failed)

    def _on_job_line(self, line: str) -> None:
        """Watch ns-train's stdout for the 'Training Finished' banner that
        nerfstudio prints when the last iteration completes. With
        --viewer.quit-on-train-completion False (our default), the process
        stays alive past this point to keep viser running, so the
        process-exit handler will never fire unless the user clicks Cancel.
        Detecting the banner lets us finalize the manifest at the real
        end-of-training event instead."""
        if self._training_finalized:
            return
        if not self._running_mine:
            return
        if _TRAINING_DONE_RE.search(line) is None:
            return
        wrote = self._finalize_training_run()
        if wrote:
            # Process is still alive (viewer kept open). Surface the new
            # state so the user knows the lineage was recorded — clicking
            # Cancel from here is now just dismissing the viewer.
            set_pill(self.status_pill, "✓ Trained · viewer alive", "success")
            self.viser_ok_lbl.setText("✓ Training finished — viewer still live.")
            self.gpu_held_wrap.show()
            self.viser_card.show()
            self._refresh_gpu_held()
            self._gpu_held_poll.start()

    def _on_job_started(self, spec: JobSpec) -> None:
        if spec.kind == "downscale" and self._downscale_mine:
            self._running_mine = False
            self.run_btn.setEnabled(False)
            self.add_levels_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            set_pill(self.status_pill, "● Adding downscale levels", "warning")
            return
        if spec.kind != "train":
            self._running_mine = False
            self.run_btn.setEnabled(False)
            self.cancel_btn.setEnabled(False)
            set_pill(self.status_pill, "Other job running", "warning")
            return
        self._running_mine = True
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        set_pill(self.status_pill, "● Training", "warning")
        if "viewer" in self._collect_params()["vis"]:
            self._viser_poll.start()

    def _finalize_training_run(self) -> bool:
        """Persist the TrainManifest, clear the pending sidecar, and update
        last_train_run_id. Idempotent: safe to call from both the
        Training-Finished-banner detector AND the process-exit handler;
        the first caller wins, the second is a no-op (see
        `self._training_finalized`).

        Returns True if the manifest was newly written, False otherwise."""

        if self._training_finalized:
            return False
        target = self._pending_target
        if (
            target is None
            or self._pending_run_id is None
            or self._pending_parent_process_id is None
            or self._pending_form_params is None
            or self._pending_start_time is None
        ):
            return False

        duration = (
            datetime.now(timezone.utc) - self._pending_start_time
        ).total_seconds()
        nested = wh.find_train_run_subdir(target)
        # `nested` is .../<train_ULID>/<exp_name>/<method>/<ts>/ — we only
        # need the leaf name for params.timestamp_dir.
        timestamp_dir = nested.name if nested is not None else ""

        try:
            train_params = wh.train_params_from_form(
                self._pending_form_params,
                experiment_name=self._project.id,
                dataparser=self._pending_dataparser or self._project.dataparser,
                timestamp_dir=timestamp_dir,
            )
            wh.write_train_manifest(
                output_dir=target,
                run_id=self._pending_run_id,
                parent_process_id=self._pending_parent_process_id,
                display_name=self._pending_display_name or target.name,
                params=train_params,
                command=self._pending_command or "",
                duration_seconds=duration,
                start_time=self._pending_start_time,
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Manifest write failed",
                f"ns-train succeeded but the TrainManifest couldn't "
                f"be written:\n{e}\n\nThe run on disk is intact; "
                f"the pending sidecar will let a future backfill pass "
                f"reconstruct the manifest.",
            )
            return False

        # Manifest persisted — pending sidecar's job is done. Removing it
        # preserves the "manifest.json present = success" invariant.
        try:
            wh.clear_pending_train(target)
        except Exception:
            pass
        self._store.update_last_train_run_id(
            self._project.id, self._pending_run_id
        )
        self._training_finalized = True
        return True

    def _on_job_finished(self, code: int) -> None:
        if self._downscale_mine:
            self._downscale_mine = False
            self.add_levels_btn.setEnabled(True)
            # New images_N folders exist now: rebuild the dropdown so the
            # rescale-at-load entries turn into folder-backed ones.
            self._refresh_dataset_card()
            self._refresh_resolution()
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            set_pill(self.status_pill, "Downscale levels added" if code == 0 else f"Downscale failed · exit {code}",
                     "success" if code == 0 else "error")
            return
        if not self._running_mine:
            self._reset_idle()
            return
        self._running_mine = False
        self._viser_poll.stop()
        self._stop_gpu_held()
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if self._training_finalized:
            # Banner detector already wrote the manifest while the process
            # was alive; this exit is just the viewer being closed. Don't
            # try to re-write — the lineage record is intact.
            self._clear_pending_state()
            set_pill(self.status_pill, "Done", "success")
            return
        if code == 0:
            # ns-train exited with code 0 but we never saw the "Training
            # Finished" banner (e.g., user cancelled cleanly, or ns-train
            # version doesn't print that exact text). Treat as success and
            # write the manifest if we can.
            self._finalize_training_run()
            self._clear_pending_state()
            set_pill(self.status_pill, "Done", "success")
        else:
            # Non-zero exit: do NOT write a manifest. The pending sidecar
            # remains as the canonical "attempted but unfinalized" marker.
            self._clear_pending_state()
            set_pill(self.status_pill, f"Failed · exit {code}", "error")

    def _on_job_cancelled(self) -> None:
        if self._downscale_mine:
            self._downscale_mine = False
            self.add_levels_btn.setEnabled(True)
            self._refresh_resolution()
            self._reset_idle()
            set_pill(self.status_pill, "Downscale cancelled", "error")
            return
        if not self._running_mine:
            self._reset_idle()
            return
        self._running_mine = False
        self._viser_poll.stop()
        self._stop_gpu_held()
        self.viser_card.hide()
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if self._training_finalized:
            # The banner already fired and the manifest is on disk. This
            # cancel is just the user closing the viewer after training
            # completed — show success, not Cancelled.
            set_pill(self.status_pill, "Done", "success")
        else:
            set_pill(self.status_pill, "Cancelled", "error")
        self._clear_pending_state()

    def _on_job_failed(self, msg: str) -> None:
        if self._downscale_mine:
            self._downscale_mine = False
            self.add_levels_btn.setEnabled(True)
            self._reset_idle()
            set_pill(self.status_pill, "Downscale failed to start", "error")
            return
        if not self._running_mine:
            self._reset_idle()
            return
        self._running_mine = False
        self._viser_poll.stop()
        self._stop_gpu_held()
        self.viser_card.hide()
        self._clear_pending_state()
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        set_pill(self.status_pill, "Failed to start", "error")

    def _clear_pending_state(self) -> None:
        """Drop all per-run state stashed by _on_run. Called from every
        terminal job-signal handler (finished/cancelled/failed) and on
        pre-start bailout so we never carry stale IDs into the next click."""
        self._pending_target = None
        self._pending_run_id = None
        self._pending_parent_process_id = None
        self._pending_form_params = None
        self._pending_dataparser = None
        self._pending_command = None
        self._pending_start_time = None
        self._pending_display_name = None

    def _refresh_next_target(self) -> None:
        """Update the output readout with the warehouse policy: every Run
        mints a fresh ULID under data_v2/trained/. Updated again to the
        actual host path once _on_run reserves a target."""
        self.output_lbl.setText(
            f"{WAREHOUSE_DIR}/trained/<new ULID>/"
        )

    def _reset_idle(self) -> None:
        """Called when another kind's job ends — restore our idle UI."""
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        set_pill(self.status_pill, "Ready", "neutral")


# ---------- top-level view ----------

class TrainView(QWidget):
    """Top-level Train tab. Hosts ProjectBar + per-project form / gates."""

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

        # When a process job succeeds, the dataset may have just become valid;
        # re-evaluate so the not-processed gate drops out automatically.
        self._last_started_kind: str | None = None
        self._jobs.started.connect(self._on_any_started)
        self._jobs.finished.connect(self._on_any_finished)

    def _on_any_started(self, spec: JobSpec) -> None:
        self._last_started_kind = spec.kind

    def _on_any_finished(self, code: int) -> None:
        was_process = self._last_started_kind == "process"
        self._last_started_kind = None
        if code != 0 or not was_process:
            return
        # Only rebuild if the user is still on the gate. A live _Form owns
        # in-flight edits we don't want to discard.
        cur = self.stack.currentWidget() if self.stack.count() else None
        if not isinstance(cur, _Form):
            self._rebuild()

    def _rebuild(self) -> None:
        while self.stack.count():
            w = self.stack.widget(0)
            self.stack.removeWidget(w)
            w.deleteLater()

        active = self._store.active
        if active is None:
            self.stack.addWidget(_EmptyState(self._store))
            return

        if not _is_trainable(active):
            gate = _NotProcessedState(active)
            gate.request_process.connect(lambda: self.request_navigate.emit(1))
            self.stack.addWidget(gate)
            return

        form = _Form(self._store, self._jobs, active)
        form.request_viewers.connect(lambda: self.request_navigate.emit(4))
        form.request_process.connect(lambda: self.request_navigate.emit(1))
        self.stack.addWidget(form)
