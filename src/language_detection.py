import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, List

from src.config import Config

logger = logging.getLogger(__name__)

_LANGID_MODEL = None


def _load_langid_model(config: Config):
    """Load and cache the SpeechBrain language identification model."""
    global _LANGID_MODEL
    if _LANGID_MODEL is None:
        from speechbrain.inference import EncoderClassifier
        logger.info("Initializing language identification model...")
        _LANGID_MODEL = EncoderClassifier.from_hparams(
            source=config.models.langid_model,
            savedir=f"{config.paths.pretrained_models_dir}/lang-id-voxlingua107-ecapa",
        )
    return _LANGID_MODEL


def detect_language(
    audio_array: np.ndarray,
    sample_rate: int,
    config: Config,
) -> Optional[Dict[str, Any]]:
    """
    Detect the primary language spoken in an audio recording.

    Returns dict with 'detected_language', 'confidence', or None on failure.
    """
    if len(audio_array) < sample_rate * 2:
        return {"detected_language": None, "confidence": 0.0, "note": "audio_too_short"}

    model = _load_langid_model(config)

    audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float()

    with torch.no_grad():
        prediction = model.classify_batch(audio_tensor)

    # prediction returns (posterior, score, index, label)
    score = prediction[1].item()
    label = prediction[3][0]

    return {
        "detected_language": label,
        "confidence": score,
    }


def detect_code_switching(
    audio_array: np.ndarray,
    sample_rate: int,
    config: Config,
) -> Optional[Dict[str, Any]]:
    """
    Detect code-switching by running LangID on non-overlapping chunks.

    Returns dict with 'code_switching_detected' boolean flag.
    """
    chunk_count = config.models.langid_chunk_count
    min_chunk_samples = sample_rate * 3  # at least 3 seconds per chunk

    total_samples = len(audio_array)
    if total_samples < min_chunk_samples * 2:
        return {"code_switching_detected": False, "note": "audio_too_short_for_detection"}

    model = _load_langid_model(config)

    chunk_size = total_samples // chunk_count
    if chunk_size < min_chunk_samples:
        chunk_count = total_samples // min_chunk_samples
        chunk_size = total_samples // max(chunk_count, 1)

    if chunk_count < 2:
        return {"code_switching_detected": False, "note": "insufficient_chunks"}

    detected_languages = []
    threshold = config.models.langid_confidence_threshold

    for i in range(chunk_count):
        start = i * chunk_size
        end = start + chunk_size
        chunk = audio_array[start:end]

        audio_tensor = torch.from_numpy(chunk).unsqueeze(0).float()
        with torch.no_grad():
            prediction = model.classify_batch(audio_tensor)

        score = prediction[1].item()
        label = prediction[3][0]

        if score >= threshold:
            detected_languages.append(label)

    unique_languages = set(detected_languages)
    code_switching = len(unique_languages) > 1

    return {
        "code_switching_detected": code_switching,
        "languages_detected": list(unique_languages),
        "chunks_analyzed": chunk_count,
    }
