import ffmpeg
import numpy as np
import re
from typing import Dict, Optional

def analyze_levels(audio_array: np.ndarray, sample_rate: int) -> Optional[Dict[str, float]]:
    """
    Analyzes the loudness and clipping of a standardized audio array.
    ...
    """
    if np.max(np.abs(audio_array)) == 0:
        print("Warning: Audio array is silent. Skipping level analysis.")
        return {
            'integrated_lufs': -70.0,
            'loudness_range_lu': 0.0,
            'true_peak_dbfs': -99.0,
            'clipping_percent': 0.0
        }

    # --- Part 1: Loudness Analysis with ebur128 ---
    try:
        audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()

        _, stderr = (
            ffmpeg
            .input('pipe:', format='s16le', ac=1, ar=sample_rate)
            .filter('ebur128', peak='true')
            .output('-', format='null')
            .run(input=audio_bytes, capture_stdout=True, capture_stderr=True)
        )
        
        stderr_str = stderr.decode()
        
        i_matches = re.findall(r"I:\s+([-\d\.]+) LUFS", stderr_str)
        lra_matches = re.findall(r"LRA:\s+([-\d\.]+) LU", stderr_str)
        
        # --- THE FIX ---
        # Your FFmpeg version prints the true peak value on a line starting with "Peak:".
        # This regex now correctly targets that line.
        tp_matches = re.findall(r"Peak:\s+([-\d\.]+) dBFS", stderr_str)

        # Added more specific error messages for easier debugging in the future
        if not i_matches:
            raise ValueError("Could not parse Integrated Loudness (I) from FFmpeg.")
        if not lra_matches:
            raise ValueError("Could not parse Loudness Range (LRA) from FFmpeg.")
        if not tp_matches:
            raise ValueError("Could not parse True Peak (Peak) from FFmpeg.")

        loudness_results = {
            'integrated_lufs': float(i_matches[-1]),
            'loudness_range_lu': float(lra_matches[-1]),
            'true_peak_dbfs': float(tp_matches[-1]),
        }

    except (ffmpeg.Error, ValueError) as e:
        print(f"Error during loudness analysis: {e}")
        if 'stderr_str' in locals():
            print("--- FFmpeg stderr for debugging ---")
            print(stderr_str)
            print("---------------------------------")
        return None
        
    # --- Part 2: Clipping Analysis ---
    clipping_threshold = 0.99
    num_clipped_samples = np.sum(np.abs(audio_array) >= clipping_threshold)
    clipping_percent = (num_clipped_samples / len(audio_array)) * 100
    
    loudness_results['clipping_percent'] = clipping_percent

    return loudness_results