import numpy as np
from typing import Dict, Optional
from src.vad_utils import get_speech_timestamps # <-- Import from our new utility

def analyze_snr(audio_array: np.ndarray, sample_rate: int) -> Optional[Dict[str, float]]:
    """
    Estimates the Signal-to-Noise Ratio (SNR) of an audio array using Silero VAD.

    Args:
        audio_array (np.ndarray): A mono, float32 audio array normalized to [-1.0, 1.0].
        sample_rate (int): The sample rate of the audio array (must be 16000 for Silero VAD).

    Returns:
        Optional[Dict[str, float]]: A dictionary with the estimated 'snr_db', or None on error.
    """
    if sample_rate != 16000:
        raise ValueError("Silero VAD requires a sample rate of 16000 Hz.")
    
    speech_timestamps = get_speech_timestamps(audio_array, sample_rate)

    # 2. Create a boolean mask for speech segments
    speech_mask = np.zeros_like(audio_array, dtype=bool)
    for segment in speech_timestamps:
        start = segment['start']
        end = segment['end']
        speech_mask[start:end] = True
        
    # 3. Handle edge cases
    num_speech_samples = np.sum(speech_mask)
    if num_speech_samples == 0:
        print("Warning: No speech detected in the audio. Cannot calculate SNR.")
        return {'snr_db': -np.inf} # Or some other indicator for "no signal"
    
    num_noise_samples = len(audio_array) - num_speech_samples
    if num_noise_samples == 0:
        print("Warning: No noise/silence detected. The entire file is speech.")
        # This could mean a very noisy file or continuous speech. SNR is effectively infinite.
        return {'snr_db': np.inf}

    # 4. Calculate power
    # Power is the mean of the squared sample values
    power_speech_plus_noise = np.mean(audio_array[speech_mask] ** 2)
    power_noise = np.mean(audio_array[~speech_mask] ** 2)

    # Add a small epsilon to prevent division by zero or log of zero
    epsilon = 1e-10
    
    if power_noise < epsilon:
        # If noise power is virtually zero, SNR is extremely high
        return {'snr_db': np.inf}

    # 5. Calculate SNR
    # Estimated signal power is the total power during speech minus the noise power
    power_signal = power_speech_plus_noise - power_noise
    
    # If noise is louder than speech+noise (a VAD error), clamp signal power to 0
    if power_signal < 0:
        power_signal = epsilon

    snr = 10 * np.log10(power_signal / power_noise)
    
    return {'snr_db': snr}