import ffmpeg
import numpy as np

def load_and_standardize_audio(
    audio_path: str, 
    sample_rate: int = 16000, 
    max_duration_secs: int = 300000,
    crop_method: str = 'start'
) -> np.ndarray:
    """
    Loads an audio file from any format, standardizes it, and returns it as a NumPy array.

    This function uses a robust two-step in-memory process:
    1. First, it converts the ENTIRE audio file to a standardized PCM format in memory.
       This avoids `probe` errors on unusual or corrupted input formats.
    2. Then, it checks the duration of the in-memory audio and crops it using
       fast NumPy slicing if necessary.

    Args:
        audio_path (str): Path to the input audio file.
        sample_rate (int): The target sample rate. Defaults to 16000.
        max_duration_secs (int): Maximum duration. If longer, it will be cropped.
        crop_method (str): How to crop. 'start' or 'middle'. Defaults to 'start'.

    Returns:
        np.ndarray: The standardized audio waveform as a NumPy array.
                    Returns None if there is an error.
    """
    try:
        # Step 1: Decode the ENTIRE audio file into a raw PCM byte stream in memory.
        # We do not specify duration ('t') here, we load the whole thing.
        full_audio_bytes, _ = (
            ffmpeg
            .input(audio_path)
            .output(
                'pipe:',            # Write to stdout pipe
                format='s16le',     # Raw 16-bit signed little-endian PCM
                ac=1,               # Mono
                ar=sample_rate      # Target sample rate
            )
            .run(capture_stdout=True, capture_stderr=True)
        )

        # Interpret the raw byte buffer as a NumPy array
        full_audio_array = np.frombuffer(full_audio_bytes, dtype=np.int16)

    except ffmpeg.Error as e:
        print(f"Error processing {audio_path} with FFmpeg:")
        print(e.stderr.decode())
        return None

    # Step 2: Now that we have the full audio in memory, perform duration checks and cropping.
    num_samples = len(full_audio_array)
    original_duration = num_samples / sample_rate
    
    final_audio_array = full_audio_array

    if original_duration > max_duration_secs:
        max_samples = int(max_duration_secs * sample_rate)
        
        if crop_method == 'middle':
            start_index = (num_samples - max_samples) // 2
            end_index = start_index + max_samples
            final_audio_array = full_audio_array[start_index:end_index]
        else: # Default to 'start'
            final_audio_array = full_audio_array[:max_samples]

    # Convert final audio array to float32 and normalize to [-1.0, 1.0]
    audio_array_float = final_audio_array.astype(np.float32) / 32768.0

    return audio_array_float

