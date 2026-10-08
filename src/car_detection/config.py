"""Central runtime defaults shared by both YOLO workflows."""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
VIDEO_DIR = PROJECT_ROOT / "assets" / "videos"


@dataclass(frozen=True)
class TrackerDefaults:
    ema_alpha: float = 0.6
    confirm_hits: int = 3
    confirm_window: int = 8
    max_missed: int = 12
    match_iou: float = 0.15
    max_center_distance: float = 1.25
    box_alpha: float = 0.65


@dataclass(frozen=True)
class PipelineDefaults:
    model_path: Path = MODEL_DIR / "yolo" / "best.pt"
    vehicle_classes: tuple[int, ...] = (1,)
    device: str = "auto"
    image_size: int = 512
    confidence: float = 0.5
    roi_padding: int = 256
    max_roi_crops: int = 4
    min_roi_size: int = 512
    min_motion_speed: float = 30.0
    min_motion_displacement: float = 8.0
    detect_every: int = 2
    tile_width_mm: float = 600.0
    tile_height_mm: float = 600.0
    columns: int = 2
    rows: int = 3
    output_dir: Path = PROJECT_ROOT / "onsite_sessions"


@dataclass(frozen=True)
class RealtimeDefaults:
    device: str = "auto"
    image_size: int = 768
    confidence: float = 0.2
    red_cone_confidence: float = 0.25
    red_cone_high_confidence: float = 0.55
    red_color_minimum: float = 0.03
    toy_car_confidence: float = 0.5
    filter_padding: int = 256
    filter_min_area: int = 20
    filter_dilation: int = 12
    temporal_mode: str = "tracker"
    red_cone_detect_every: int = 5
    red_cone_motion_threshold: float = 0.10
    motion_threshold: float = 0.0015
    motion_pixel_threshold: int = 20
    motion_dilation: int = 7
    motion_roi: bool = True
    motion_roi_min_area: int = 40
    motion_roi_padding: int = 128
    detect_every: int = 2


class ConfigError(ValueError):
    """Raised when command-line settings are internally inconsistent."""


@dataclass
class PipelineConfig:
    source: str
    tile_width_mm: float
    tile_height_mm: float
    cols: int
    rows: int
    measured: bool
    model: Path
    vehicle_class: list[int]
    device: str
    imgsz: int
    conf: float
    no_warmup: bool
    roi_padding: int
    max_roi_crops: int
    min_roi_size: int
    min_motion_speed: float
    min_motion_displacement: float
    detect_every: int
    max_frames: int | None
    save_results: bool
    output_dir: Path
    last_detect_perf: dict[str, float | int] = field(default_factory=dict)

    @classmethod
    def from_namespace(cls, namespace: Any) -> "PipelineConfig":
        return cls(**vars(namespace))


@dataclass
class RealtimeConfig:
    model: Path
    source: str
    output_video: Path | None
    device: str
    imgsz: int
    conf: float
    red_cone_conf: float
    red_cone_high_conf: float
    red_color_min: float
    toy_car_conf: float
    filter: str
    filter_padding: int
    filter_min_area: int
    filter_dilation: int
    temporal_mode: str
    ema_alpha: float
    confirm_hits: int
    confirm_window: int
    max_missed: int
    match_iou: float
    max_center_distance: float
    box_alpha: float
    motion_gate: bool
    motion_threshold: float
    motion_pixel_threshold: int
    motion_dilation: int
    motion_roi: bool
    motion_roi_min_area: int
    motion_roi_padding: int
    detect_every: int
    red_cone_detect_every: int
    red_cone_motion_threshold: float
    max_frames: int | None
    display: bool
    draw_roi: bool
    profile: bool
    strict_red_color: bool = True

    @classmethod
    def from_namespace(cls, namespace: Any) -> "RealtimeConfig":
        return cls(**vars(namespace))


TRACKER_DEFAULTS = TrackerDefaults()
PIPELINE_DEFAULTS = PipelineDefaults()
REALTIME_DEFAULTS = RealtimeDefaults()


def _probability(value: float, name: str, *, allow_zero: bool = False) -> None:
    lower_valid = 0.0 <= value if allow_zero else 0.0 < value
    if not lower_valid or value > 1.0:
        minimum = "[0, 1]" if allow_zero else "(0, 1]"
        raise ConfigError(f"{name} must be in {minimum}")


def validate_pipeline_config(config: PipelineConfig) -> None:
    if config.imgsz < 32:
        raise ConfigError("Image size must be at least 32")
    _probability(config.conf, "Confidence")
    if any(class_id < 0 for class_id in config.vehicle_class):
        raise ConfigError("Vehicle class IDs must be non-negative")
    if not math.isfinite(config.min_motion_speed) or config.min_motion_speed < 0:
        raise ConfigError("Minimum movement speed must be finite and non-negative")
    if config.detect_every < 1 or not math.isfinite(config.min_motion_displacement):
        raise ConfigError(
            "Detection interval must be positive and minimum displacement finite"
        )
    if config.min_motion_displacement < 0:
        raise ConfigError("Minimum displacement must be non-negative")
    if config.roi_padding < 0:
        raise ConfigError("ROI padding must be non-negative")
    if config.max_roi_crops < 1 or config.min_roi_size < 32:
        raise ConfigError(
            "Maximum crop count must be positive and minimum ROI size must be at least 32"
        )
    if min(config.tile_width_mm, config.tile_height_mm, config.cols, config.rows) <= 0:
        raise ConfigError("Dimensions must be positive")


def validate_realtime_config(config: RealtimeConfig) -> None:
    if config.imgsz < 32:
        raise ConfigError("--imgsz must be at least 32")
    _probability(config.conf, "--conf")
    _probability(config.red_cone_conf, "--red-cone-conf")
    _probability(config.red_cone_high_conf, "--red-cone-high-conf")
    if config.red_cone_high_conf < config.red_cone_conf:
        raise ConfigError("red-cone-high-conf must be >= red-cone-conf")
    _probability(config.red_color_min, "--red-color-min", allow_zero=True)
    _probability(config.toy_car_conf, "--toy-car-conf")
    _probability(config.ema_alpha, "--ema-alpha")
    if config.confirm_hits <= 0 or config.confirm_window < config.confirm_hits:
        raise ConfigError("confirm-window must be >= confirm-hits > 0")
    if config.max_missed < 0:
        raise ConfigError("--max-missed must be non-negative")
    if config.detect_every <= 0 or config.red_cone_detect_every <= 0:
        raise ConfigError("detection intervals must be greater than 0")
    _probability(config.red_cone_motion_threshold, "--red-cone-motion-threshold")
    if config.max_center_distance <= 0:
        raise ConfigError("max-center-distance must be greater than 0")
    _probability(config.match_iou, "--match-iou", allow_zero=True)
    _probability(config.box_alpha, "--box-alpha")
    _probability(config.motion_threshold, "--motion-threshold")
    if config.motion_pixel_threshold <= 0 or config.motion_dilation <= 0:
        raise ConfigError("motion pixel threshold and dilation must be greater than 0")
    if config.filter_min_area <= 0 or config.filter_dilation <= 0:
        raise ConfigError("filter area and dilation must be greater than 0")
    if config.filter_padding < 0:
        raise ConfigError("filter padding must be non-negative")
    if config.motion_roi_min_area <= 0 or config.motion_roi_padding < 0:
        raise ConfigError("motion ROI area must be greater than 0 and padding non-negative")
    if config.max_frames is not None and config.max_frames < 1:
        raise ConfigError("--max-frames must be greater than 0")
