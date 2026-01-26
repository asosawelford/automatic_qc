import os
import json
import random
import shutil
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration ---
FOLDER_PATH = "db/impact_analysis_v2"
SITES = ["LIM", "HUA", "TUM", "IQT"]

# --- New Combined Analysis Criteria ---
ALLOWED_SPEAKERS = [1, 2]
MIN_DURATION_S = 10.0

def analyze_with_combined_criteria(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return print("ERROR")

    pass_counts = {site: 0 for site in SITES}
    fail_counts = {site: 0 for site in SITES}
    failure_reasons = {'Poor SNR': 0, 'Invalid Speaker Count': 0, 'Insufficient Duration': 0}
    total_files = 0

    for filename in os.listdir(folder_path):
        if not filename.endswith('.json'): continue
        total_files += 1
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        
        analysis = data.get('analysis_results', {})
        assessment = data.get('quality_assessment', {})
        source_file = data.get('source_file', 'UNKNOWN_')
        site_code = source_file[:3]
        if site_code not in SITES: continue

        num_speakers = analysis.get('speaker', {}).get('estimated_speakers', 0)
        speaker_ok = num_speakers in ALLOWED_SPEAKERS
        speech_duration = analysis.get('speech_duration', 0)
        duration_ok = speech_duration > MIN_DURATION_S
        snr_ok = assessment.get('snr_ok', False)
        
        overall_pass_final = speaker_ok and duration_ok and snr_ok

        if not snr_ok:
            print(filename)

analyze_with_combined_criteria(FOLDER_PATH)