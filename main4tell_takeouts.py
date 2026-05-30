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
from src.vad_utils import analyze_speech_activity


# --- Quality Thresholds ---
LOUDNESS_TARGET_LUFS = -14.0
LOUDNESS_TOLERANCE_LU = 10.0  # Allows a range from -24 to -4 LUFS
CLIPPING_MAX_PERCENT = 0.1
SNR_MIN_DB = 9.0
EXPECTED_SPEAKER_COUNT = 1
MIN_SPEECH_DURATION_S = 5

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
        audio_array = load_and_standardize_audio(file_path, sample_rate=sample_rate, max_duration_secs=3000000)
        
        if audio_array is None:
            report["error_message"] = "Failed during audio standardization (FFmpeg error)."
            return report

        # --- Steps 2, 3, 4: Run all analyses ---
        level_report = analyze_levels(audio_array, sample_rate)
        snr_report = analyze_snr(audio_array, sample_rate)
        speaker_report = estimate_speaker_count(audio_array, sample_rate)
        speech_duration = analyze_speech_activity(audio_array, sample_rate)

        # Check for analysis failures
        if not all([level_report, snr_report, speaker_report, speech_duration]):
             report["error_message"] = "One or more analysis modules failed."
             return report
        
        report["analysis_results"] = {
            "level": level_report,
            "noise": snr_report,
            "speaker": speaker_report,
            "speech_duration": speech_duration,
        }

        # --- Step 5: Apply Quality Logic ---
        loudness_ok = (LOUDNESS_TARGET_LUFS - LOUDNESS_TOLERANCE_LU) <= level_report['integrated_lufs'] <= (LOUDNESS_TARGET_LUFS + LOUDNESS_TOLERANCE_LU)
        clipping_ok = level_report['clipping_percent'] < CLIPPING_MAX_PERCENT
        snr_ok = snr_report['snr_db'] >= SNR_MIN_DB
        speaker_count_ok = speaker_report['estimated_speakers'] == EXPECTED_SPEAKER_COUNT
        duration_ok =  speech_duration > MIN_SPEECH_DURATION_S

        overall_pass = all([loudness_ok, clipping_ok, snr_ok, speaker_count_ok, duration_ok])

        report["quality_assessment"] = {
            "loudness_ok": bool(loudness_ok),
            "clipping_ok": bool(clipping_ok),
            "snr_ok": bool(snr_ok),
            "speaker_count_ok": bool(speaker_count_ok),
            "duration_ok": bool(duration_ok),
            "overall_pass": bool(overall_pass)
        }
        
        report["processing_status"] = "SUCCESS"

    except Exception as e:
        report["error_message"] = f"An unexpected error occurred: {str(e)}"
    
    return report

def process_path(input_path: str, output_dir: str):
    """
    Processes all audio files in a nested directory structure.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input path does not exist: {input_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Scanning {input_path} for audio files...")

    # Process each folder recursively using os.walk
    for root, dirs, files in os.walk(input_path):
        if "audio.webm" in files:
            audio_path = os.path.join(root, "audio.webm")
            json_path = os.path.join(root, "feature_extraction.json")
            
            # 1. Get the participant code so we don't overwrite files!
            participant_code = "Unknown"
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        participant_code = metadata.get("participant_code", "Unknown")
                except Exception as e:
                    print(f"Warning: Could not read JSON for {root}: {e}")

            print(f"\nProcessing: Participant {participant_code} (File: {audio_path})")
            
            # 2. Run the quality report
            report_data = generate_quality_report(audio_path)
            
            # Optional: inject the participant code into the report itself
            report_data["participant_code"] = participant_code 
            
            # 3. Define output JSON path using the participant code
            # Example: UHDACNMSCM00054_quality_report.json
            output_json_name = f"{participant_code}_quality_report.json"
            
            # Handle edge case: if participant has multiple audios, avoid overwriting
            counter = 1
            output_json_path = os.path.join(output_dir, output_json_name)
            while os.path.exists(output_json_path):
                output_json_name = f"{participant_code}_{counter}_quality_report.json"
                output_json_path = os.path.join(output_dir, output_json_name)
                counter += 1

            # 4. Save the report
            with open(output_json_path, 'w', encoding='utf-8') as f:
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