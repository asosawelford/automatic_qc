"""
Preprocessing involves format/codec/sr normalization and lenght shortening
"""
import os
import sys
import ffmpeg
from tqdm import tqdm # Import tqdm

# ==============================================================================
# FUNCTION 1: Now with a 'verbose' flag to control printing
# ==============================================================================
def convert_and_crop_safely(audio_file, save_directory, duration_secs=30, crop_method='start', verbose=False):
    """
    Safely converts and crops an audio file. Now quieter by default.
    Set verbose=True to see step-by-step prints for debugging.
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    filename = os.path.basename(audio_file)
    base, _ = os.path.splitext(filename)
    output_file = os.path.join(save_directory, base + '.wav')

    # Step 1: Initial conversion
    if verbose: print(f"-> Step 1/2: Converting '{filename}' to full-length WAV...")
    try:
        (
        ffmpeg.input(audio_file)
        .output(output_file, format='wav', acodec='pcm_s16le', ar=16000, ac=1)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        # Use tqdm.write for error logging to avoid breaking the progress bar
        tqdm.write(f"ERROR: Could not convert '{filename}'. FFmpeg error:", file=sys.stderr)
        tqdm.write(e.stderr.decode(), file=sys.stderr)
        return None
    
    # Step 2: Probe and crop
    if verbose: print(f"-> Step 2/2: Checking duration and cropping '{os.path.basename(output_file)}'...")
    try:
        probe = ffmpeg.probe(output_file)
        original_duration = float(probe['streams'][0]['duration'])

        if original_duration > duration_secs:
            if verbose: print(f"   Duration is {original_duration:.2f}s. Cropping to {duration_secs}s.")
            temp_output_file = os.path.join(save_directory, "temp_" + base + ".wav")
            input_args = {}
            if crop_method == 'middle':
                start_time = max(0, (original_duration / 2) - (duration_secs / 2))
                input_args['ss'] = start_time

            (
            ffmpeg.input(output_file, **input_args)
            .output(temp_output_file, t=duration_secs, c='copy')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
            )

            os.remove(output_file)
            os.rename(temp_output_file, output_file)
            if verbose: print("   Successfully cropped.")
        else:
            if verbose: print(f"   Duration is {original_duration:.2f}s. No cropping needed.")

    except (ffmpeg.Error, KeyError, IndexError) as e:
        tqdm.write(f"WARNING: Error during cropping of '{os.path.basename(output_file)}': {e}", file=sys.stderr)
        pass
        
    return output_file

# ==============================================================================
# FUNCTION 2: Now featuring a TQDM progress bar
# ==============================================================================
def process_audio_folder(source_folder, output_folder, duration_secs=30, crop_method='start'):
    if not os.path.isdir(source_folder):
        print(f"Error: Source folder not found at '{source_folder}'")
        return

    allowed_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.webm', '.opus')
    audio_files_to_process = [
        f for f in os.listdir(source_folder) if f.lower().endswith(allowed_extensions)
    ]
    
    if not audio_files_to_process:
        print(f"No audio files with allowed extensions found in '{source_folder}'.")
        return

    print(f"Found {len(audio_files_to_process)} audio files. Starting processing...")
    
    success_count, failure_count = 0, 0
    
    # --- TQDM LOOP ---
    # Wrap the list of files with tqdm() to create the progress bar
    for filename in tqdm(audio_files_to_process, desc="Processing files", unit="file"):
        input_filepath = os.path.join(source_folder, filename)
        
        result_path = convert_and_crop_safely(
            audio_file=input_filepath,
            save_directory=output_folder,
            duration_secs=duration_secs,
            crop_method=crop_method,
            verbose=False # Set to False for clean output
        )
        
        if result_path:
            success_count += 1
        else:
            failure_count += 1
            
    # Final summary
    print("\n=======================================")
    print("           Processing Complete")
    print("=======================================")
    print(f"Successfully processed: {success_count} files")
    print(f"Failed to process:      {failure_count} files")
    print(f"Processed files are in: '{output_folder}'")

# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================


if __name__ == '__main__':
    # --- CONFIGURE YOUR PATHS HERE ---
    # Replace these with the actual paths on your system
    SOURCE_DIRECTORY = '/home/aleph/Downloads/adresso/train/cn'
    PROCESSED_DIRECTORY = '/home/aleph/Downloads/adresso_prepro'

    # --- RUN THE BATCH PROCESSING ---
    # To use, simply create the folders and run the script.
    # For example:
    # 1. Create a folder named 'clinical_audio_raw' and put some audio files in it.
    # 2. Change SOURCE_DIRECTORY to 'clinical_audio_raw'.
    # 3. Change PROCESSED_DIRECTORY to 'clinical_audio_processed'.
    # 4. Run the script: python your_script_name.py
    
    process_audio_folder(
        source_folder=SOURCE_DIRECTORY,
        output_folder=PROCESSED_DIRECTORY,
        duration_secs=30,      # All files will be max 30 seconds
        crop_method='middle'   # Take the middle 30s portion
    )
