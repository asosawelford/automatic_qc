import json
import logging
import numpy as np
from typing import Dict, Any, Optional

from src.standardize import load_and_standardize_audio
from src.level_analysis import analyze_levels
from src.noise_analysis import analyze_snr
from src.speaker_count import estimate_speaker_count
from src.vad_utils import analyze_speech_activity
from src.config import Config

logger = logging.getLogger(__name__)


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def generate_quality_report(
    file_path: str,
    config: Config,
    skip_speaker_count: bool = False,
    skip_langid: bool = False,
) -> Dict[str, Any]:
    """Run the full analysis pipeline on a single audio file."""
    import os

    report = {
        "source_file": os.path.basename(file_path),
        "processing_status": "ERROR",
        "error_message": None,
        "analysis_results": {},
        "quality_assessment": {},
    }

    try:
        sample_rate = 16000
        audio_array = load_and_standardize_audio(
            file_path, sample_rate=sample_rate, max_duration_secs=3000000
        )

        if audio_array is None:
            report["error_message"] = "Failed during audio standardization (FFmpeg error)."
            return report

        level_report = analyze_levels(audio_array, sample_rate)
        snr_report = analyze_snr(audio_array, sample_rate)

        speaker_report = None
        if not skip_speaker_count:
            speaker_report = estimate_speaker_count(
                audio_array, sample_rate,
                distance_threshold=config.models.speaker_clustering_distance_threshold,
            )

        speech_duration = analyze_speech_activity(audio_array, sample_rate)

        if level_report is None or snr_report is None:
            report["error_message"] = "Level or noise analysis failed."
            return report

        report["analysis_results"] = {
            "level": level_report,
            "noise": snr_report,
            "speaker": speaker_report,
            "speech_duration": speech_duration,
        }

        # Language detection (if enabled)
        if not skip_langid and config.protocol_language_map:
            try:
                from src.language_detection import detect_language, detect_code_switching
                lang_result = detect_language(audio_array, sample_rate, config)
                code_switch = detect_code_switching(audio_array, sample_rate, config)
                report["analysis_results"]["language"] = lang_result
                report["analysis_results"]["code_switching"] = code_switch
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
                report["analysis_results"]["language"] = None
                report["analysis_results"]["code_switching"] = None

        # Quality assessment
        q = config.quality
        loudness_ok = (
            (q.loudness_target_lufs - q.loudness_tolerance_lu)
            <= level_report["integrated_lufs"]
            <= (q.loudness_target_lufs + q.loudness_tolerance_lu)
        )
        clipping_ok = level_report["clipping_percent"] < q.clipping_max_percent
        snr_ok = snr_report["snr_db"] >= q.snr_min_db

        if speaker_report is not None:
            speaker_count_ok = (
                q.min_speaker_count <= speaker_report["estimated_speakers"] <= q.max_speaker_count
            )
        else:
            speaker_count_ok = True

        duration_ok = speech_duration > q.min_speech_duration_s

        overall_pass = all([loudness_ok, clipping_ok, snr_ok, speaker_count_ok, duration_ok])

        report["quality_assessment"] = {
            "loudness_ok": bool(loudness_ok),
            "clipping_ok": bool(clipping_ok),
            "snr_ok": bool(snr_ok),
            "speaker_count_ok": bool(speaker_count_ok),
            "duration_ok": bool(duration_ok),
            "overall_pass": bool(overall_pass),
        }

        report["processing_status"] = "SUCCESS"

    except Exception as e:
        report["error_message"] = f"An unexpected error occurred: {str(e)}"
        logger.error(f"Error processing {file_path}: {e}", exc_info=True)

    return report
