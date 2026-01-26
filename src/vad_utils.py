import torch
import numpy as np
from typing import List, Dict, Any
import librosa

# --- VAD Hyperparams ---
VAD_SAMPLING_RATE = 16000
VAD_PARAMS = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 100,
    "window_size_samples": 512,
    "speech_pad_ms": 30
}


# Global cache for the model to avoid reloading it on every call.
_VAD_MODEL = None
_VAD_UTILS = None

def _load_vad_model():
    """Loads the Silero VAD model and utils, caching them globally."""
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is None:
        print("Initializing Silero VAD model (will run only once)...")
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        _VAD_MODEL = model
        _VAD_UTILS = utils
    return _VAD_MODEL, _VAD_UTILS

def get_speech_timestamps(
    audio_array: np.ndarray,
    sample_rate: int,
    **vad_kwargs: Any
) -> List[Dict[str, int]]:
    """
    Uses Silero VAD to get speech segments from an audio array.

    Args:
        audio_array (np.ndarray): Mono float32 audio.
        sample_rate (int): The sample rate of the audio array. Must be 16000.
        **vad_kwargs: Arbitrary keyword arguments passed directly to the
                      Silero VAD `get_speech_timestamps` function.
                      Common parameters include:
                      - threshold (float): Speech confidence threshold.
                      - min_speech_duration_ms (int): Minimum duration of a speech chunk.
                      - min_silence_duration_ms (int): Minimum duration of a silence chunk.
                      - speech_pad_ms (int): Padding to add to speech chunks.

    Returns:
        List[Dict[str, int]]: A list of dictionaries, each with 'start' and 'end' sample indices.
    """
    if sample_rate != 16000:
        raise ValueError(f"Silero VAD requires a sample rate of 16000 Hz, but got {sample_rate} Hz.")
    
    model, utils = _load_vad_model()
    get_ts_function = utils[0]
    
    # Ensure audio is a tensor
    audio_tensor = torch.from_numpy(audio_array).float()
    
    # Pass the keyword arguments directly to the VAD function
    speech_timestamps = get_ts_function(
        audio_tensor,
        model,
        sampling_rate=sample_rate,
        **vad_kwargs  # This unpacks the dictionary into keyword arguments
    )
    
    return speech_timestamps

def analyze_speech_activity(y, sr):
    """Analyzes total speech duration and noise level using your VAD utility."""
    # Resample audio for VAD
    if sr != VAD_SAMPLING_RATE:
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=VAD_SAMPLING_RATE)
    else:
        y_resampled = y

    try:
        # Call the function from your module, passing the parameters
        speech_timestamps = get_speech_timestamps(
            y_resampled, VAD_SAMPLING_RATE, **VAD_PARAMS
        )
    except Exception as e:
        print(f"  -> VAD processing failed: {e}")
        return {"speech_duration_s": -1}

    # The rest of the logic remains the same
    total_speech_samples = sum(d['end'] - d['start'] for d in speech_timestamps)
    speech_duration_s = total_speech_samples / VAD_SAMPLING_RATE
        
    return speech_duration_s
