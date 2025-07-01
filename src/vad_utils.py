import torch
import numpy as np
from typing import List, Dict

_VAD_MODEL = None
_VAD_UTILS = None

def _load_vad_model():
    """Loads the Silero VAD model and utils, caching them globally."""
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is None:
        print("Initializing Silero VAD model (first run)...")
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        _VAD_MODEL = model
        _VAD_UTILS = utils
    return _VAD_MODEL, _VAD_UTILS

def get_speech_timestamps(audio_array: np.ndarray, sample_rate: int) -> List[Dict[str, int]]:
    """
    Uses Silero VAD to get speech segments from an audio array.

    Args:
        audio_array (np.ndarray): Mono float32 audio.
        sample_rate (int): Must be 16000.

    Returns:
        List[Dict[str, int]]: A list of dictionaries, each with 'start' and 'end' sample indices.
    """
    if sample_rate != 16000:
        raise ValueError("Silero VAD requires a sample rate of 16000 Hz.")
    
    model, utils = _load_vad_model()
    get_ts_function = utils[0]
    
    audio_tensor = torch.from_numpy(audio_array)
    speech_timestamps = get_ts_function(audio_tensor, model, sampling_rate=sample_rate)
    
    return speech_timestamps
