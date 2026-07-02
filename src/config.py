import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class QualityThresholds:
    loudness_target_lufs: float = -14.0
    loudness_tolerance_lu: float = 10.0
    clipping_max_percent: float = 0.1
    snr_min_db: float = 9.0
    min_speaker_count: int = 1
    max_speaker_count: int = 2
    min_speech_duration_s: float = 10.0

@dataclass
class ModelParams:
    vad_threshold: float = 0.5
    vad_min_speech_duration_ms: int = 250
    vad_min_silence_duration_ms: int = 100
    vad_speech_pad_ms: int = 30
    speaker_clustering_distance_threshold: float = 0.15
    speaker_chunk_size: int = 24000
    speaker_hop_size: int = 8000
    langid_model: str = "speechbrain/lang-id-voxlingua107-ecapa"
    langid_confidence_threshold: float = 0.7
    langid_chunk_count: int = 5

@dataclass
class PathsConfig:
    checkpoint_dir: str = "checkpoints"
    reports_output_dir: str = "reports_output"
    pretrained_models_dir: str = "pretrained_models"

@dataclass
class OutputConfig:
    generate_html_dashboard: bool = True
    generate_png_plots: bool = True
    generate_pdf_plots: bool = True
    generate_json_summary: bool = True
    generate_csv_detailed: bool = True
    plot_dpi: int = 300
    plot_style: str = "seaborn-v0_8"
    plot_font_scale: float = 1.5

@dataclass
class Config:
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    models: ModelParams = field(default_factory=ModelParams)
    paths: PathsConfig = field(default_factory=PathsConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    protocol_language_map: Dict[str, str] = field(default_factory=dict)
    site_codes: List[str] = field(default_factory=lambda: ["HUA", "IQT", "LIM", "TUM"])

def load_config(config_path: Optional[str] = None) -> Config:
    """Load config from YAML file, falling back to defaults."""
    if config_path is None or not os.path.exists(config_path):
        return Config()

    import yaml
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f) or {}

    config = Config()

    if 'quality_thresholds' in raw:
        qt = raw['quality_thresholds']
        config.quality = QualityThresholds(
            loudness_target_lufs=qt.get('loudness_target_lufs', -14.0),
            loudness_tolerance_lu=qt.get('loudness_tolerance_lu', 10.0),
            clipping_max_percent=qt.get('clipping_max_percent', 0.1),
            snr_min_db=qt.get('snr_min_db', 9.0),
            min_speaker_count=qt.get('min_speaker_count', 1),
            max_speaker_count=qt.get('max_speaker_count', 2),
            min_speech_duration_s=qt.get('min_speech_duration_s', 10.0),
        )

    if 'models' in raw:
        m = raw['models']
        config.models = ModelParams(
            vad_threshold=m.get('vad_threshold', 0.5),
            vad_min_speech_duration_ms=m.get('vad_min_speech_duration_ms', 250),
            vad_min_silence_duration_ms=m.get('vad_min_silence_duration_ms', 100),
            vad_speech_pad_ms=m.get('vad_speech_pad_ms', 30),
            speaker_clustering_distance_threshold=m.get('speaker_clustering_distance_threshold', 0.15),
            speaker_chunk_size=m.get('speaker_chunk_size', 24000),
            speaker_hop_size=m.get('speaker_hop_size', 8000),
            langid_model=m.get('langid_model', "speechbrain/lang-id-voxlingua107-ecapa"),
            langid_confidence_threshold=m.get('langid_confidence_threshold', 0.7),
            langid_chunk_count=m.get('langid_chunk_count', 5),
        )

    if 'paths' in raw:
        p = raw['paths']
        config.paths = PathsConfig(
            checkpoint_dir=p.get('checkpoint_dir', 'checkpoints'),
            reports_output_dir=p.get('reports_output_dir', 'reports_output'),
            pretrained_models_dir=p.get('pretrained_models_dir', 'pretrained_models'),
        )

    if 'outputs' in raw:
        o = raw['outputs']
        config.outputs = OutputConfig(
            generate_html_dashboard=o.get('generate_html_dashboard', True),
            generate_png_plots=o.get('generate_png_plots', True),
            generate_pdf_plots=o.get('generate_pdf_plots', True),
            generate_json_summary=o.get('generate_json_summary', True),
            generate_csv_detailed=o.get('generate_csv_detailed', True),
            plot_dpi=o.get('plot_dpi', 300),
            plot_style=o.get('plot_style', 'seaborn-v0_8'),
            plot_font_scale=o.get('plot_font_scale', 1.5),
        )

    config.protocol_language_map = raw.get('protocol_language_map', {})
    config.site_codes = raw.get('site_codes', ["HUA", "IQT", "LIM", "TUM"])

    return config
