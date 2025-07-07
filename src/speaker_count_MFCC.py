import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from typing import Dict, Optional
from src.vad_utils import get_speech_timestamps

def estimate_speaker_count(
    audio_array: np.ndarray, 
    sample_rate: int,
    max_speakers: int = 3
) -> Optional[Dict[str, int]]:
    """
    Estimates the number of speakers using MFCCs and Gaussian Mixture Models.
    ...
    """
    speech_timestamps = get_speech_timestamps(audio_array, sample_rate)
    
    if not speech_timestamps:
        print("Warning: No speech detected, cannot estimate speaker count.")
        return {'estimated_speakers': 0}

    speech_segments = [audio_array[ts['start']:ts['end']] for ts in speech_timestamps]
    speech_audio = np.concatenate(speech_segments)

    # Check if speech audio is effectively silent after VAD
    if np.max(np.abs(speech_audio)) < 1e-4:
        print("Warning: Speech segments are silent, cannot estimate speaker count.")
        return {'estimated_speakers': 0}

    mfccs = librosa.feature.mfcc(y=speech_audio, sr=sample_rate, n_mfcc=13).T

    # Not enough distinct frames to analyze
    if mfccs.shape[0] < 2:
        return {'estimated_speakers': 1}

    n_components_range = range(1, max_speakers + 1)
    bics = []
    
    for n_components in n_components_range:
        # --- THE FIXES ---
        # 1. More robust sanity check: We need more data points than components.
        if mfccs.shape[0] < n_components:
            break # Can't test this or higher component counts

        # 2. Add regularization (reg_covar) to prevent matrix errors on uniform data.
        gmm = GaussianMixture(
            n_components=n_components, 
            covariance_type='full', 
            random_state=0,
            reg_covar=1e-6 # This is the key to stability
        )
        try:
            gmm.fit(mfccs)
            bics.append(gmm.bic(mfccs))
        except ValueError:
            # If it still fails, we can't consider this n_component.
            # Append a very large number so it won't be chosen as the minimum.
            bics.append(np.inf)
            continue
            
    if not bics or np.all(np.isinf(bics)):
        # This can happen if all GMM fits fail, e.g., on very short/weird audio.
        print("Warning: Could not fit any GMM. Defaulting to 1 speaker.")
        return {'estimated_speakers': 1}

    # The number of speakers is the one with the lowest BIC score
    estimated_speakers = n_components_range[np.argmin(bics)]

    return {'estimated_speakers': estimated_speakers}
