import os
import sys
import json
import numpy as np
from typing import Dict, Any

# --- Import our analysis modules ---
from src.standardize import load_and_standardize_audio
from src.level_analysis import analyze_levels
from src.noise_analysis import analyze_snr
from src.speaker_count import estimate_speaker_count

# --- Define Quality Thresholds in one place for easy tuning ---
LOUDNESS_TARGET_LUFS = -19.0
LOUDNESS_TOLERANCE_LU = 4.0  # Allows a range from -23 to -15 LUFS
CLIPPING_MAX_PERCENT = 0.1
SNR_MIN_DB = 15.0
EXPECTED_SPEAKER_COUNT = 1

class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy types.
    This is necessary because the JSON library doesn't know how to handle
    NumPy's specific float and integer types by default.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

def generate_quality_report(file_path: str) -> Dict[str, Any]:
    """
    Runs the full analysis pipeline on a single audio file and returns a structured report.
    """
    # --- Initialize the report dictionary ---
    report = {
        "source_file": os.path.basename(file_path),
        "processing_status": "ERROR",
        "error_message": None,
        "analysis_results": {},
        "quality_assessment": {}
    }

    try:
        # --- Step 1: Standardize Audio ---
        sample_rate = 16000
        audio_array = load_and_standardize_audio(file_path, sample_rate=sample_rate, max_duration_secs=30)
        
        if audio_array is None:
            report["error_message"] = "Failed during audio standardization (FFmpeg error)."
            return report

        # --- Steps 2, 3, 4: Run all analyses ---
        level_report = analyze_levels(audio_array, sample_rate)
        snr_report = analyze_snr(audio_array, sample_rate)
        speaker_report = estimate_speaker_count(audio_array, sample_rate)

        # Check for analysis failures
        if not all([level_report, snr_report, speaker_report]):
             report["error_message"] = "One or more analysis modules failed."
             return report
        
        report["analysis_results"] = {
            "level": level_report,
            "noise": snr_report,
            "speaker": speaker_report
        }

        # --- Step 5: Apply Quality Logic ---
        loudness_ok = (LOUDNESS_TARGET_LUFS - LOUDNESS_TOLERANCE_LU) <= level_report['integrated_lufs'] <= (LOUDNESS_TARGET_LUFS + LOUDNESS_TOLERANCE_LU)
        clipping_ok = level_report['clipping_percent'] < CLIPPING_MAX_PERCENT
        snr_ok = snr_report['snr_db'] >= SNR_MIN_DB
        speaker_count_ok = speaker_report['estimated_speakers'] == EXPECTED_SPEAKER_COUNT
        
        overall_pass = all([loudness_ok, clipping_ok, snr_ok, speaker_count_ok])

        report["quality_assessment"] = {
            "loudness_ok": bool(loudness_ok),
            "clipping_ok": bool(clipping_ok),
            "snr_ok": bool(snr_ok),
            "speaker_count_ok": bool(speaker_count_ok),
            "overall_pass": bool(overall_pass)
        }
        
        report["processing_status"] = "SUCCESS"

    except Exception as e:
        report["error_message"] = f"An unexpected error occurred: {str(e)}"
    
    return report

def process_path(input_path: str, output_dir: str):
    """
    Processes a single file or all audio files in a directory.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input path does not exist: {input_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Find all audio files to process
    files_to_process = []
    if os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm')):
                files_to_process.append(os.path.join(input_path, filename))
    else:
        files_to_process.append(input_path)
    
    print(f"Found {len(files_to_process)} audio file(s) to process.")

    # Process each file
    for file_path in files_to_process:
        print(f"\nProcessing: {os.path.basename(file_path)}...")
        
        report_data = generate_quality_report(file_path)
        
        # Define output JSON path
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_json_path = os.path.join(output_dir, f"{base_name}_report.json")
        
        # Save the report as a JSON file
        with open(output_json_path, 'w') as f:
            json.dump(report_data, f, indent=4, cls=NumpyJSONEncoder)
        
        print(f"-> Report saved to: {output_json_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n--- Audio Quality Assessment Pipeline ---")
        print("Usage:")
        print("  python main.py <path_to_audio_file_or_directory>")
        print("\nOptional:")
        print("  python main.py <input_path> <output_directory>")
        sys.exit(1)

    input_path = sys.argv[1]
    
    # If output directory is not specified, create one named 'reports'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "reports"
    
    process_path(input_path, output_dir)
    print("\nProcessing complete.")