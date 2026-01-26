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
    passed_audio_paths = []  # ✅ new list to store usable audios

    for filename in os.listdir(folder_path):
        if not filename.endswith('.json'):
            continue
        total_files += 1
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        analysis = data.get('analysis_results', {})
        assessment = data.get('quality_assessment', {})
        source_file = data.get('source_file', 'UNKNOWN_')
        site_code = source_file[:3]
        if site_code not in SITES:
            continue

        num_speakers = analysis.get('speaker', {}).get('estimated_speakers', 0)
        speaker_ok = num_speakers in ALLOWED_SPEAKERS
        speech_duration = analysis.get('speech_duration', 0)
        duration_ok = speech_duration > MIN_DURATION_S
        snr_ok = assessment.get('snr_ok', False)
        
        overall_pass_final = speaker_ok and duration_ok and snr_ok

        if overall_pass_final:
            pass_counts[site_code] += 1
            passed_audio_paths.append(source_file)  # ✅ store path of usable audio
        else:
            fail_counts[site_code] += 1
            if not speaker_ok:
                failure_reasons['Invalid Speaker Count'] += 1
            if not duration_ok:
                failure_reasons['Insufficient Duration'] += 1
            if not snr_ok:
                failure_reasons['Poor SNR'] += 1

    # ✅ Save usable audio paths to txt file
    output_path = os.path.join(folder_path, "usable_audios_combined_check.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for path in passed_audio_paths:
            f.write(path + "\n")

    print(f"\nSaved list of usable audios to: {output_path}")

    print_report(total_files, pass_counts, fail_counts, failure_reasons)
    generate_plots(pass_counts, fail_counts, failure_reasons)

def print_report(total_files, pass_counts, fail_counts, failure_reasons):
    total_pass = sum(pass_counts.values())
    total_fail = sum(fail_counts.values())
    print("--- Audio Analysis Report (Combined Criteria) ---")
    print(f"\nAnalysis based on: Speaker Count in {ALLOWED_SPEAKERS}, Duration > {MIN_DURATION_S}s, AND SNR OK")
    print(f"\nTotal audio files analyzed: {total_files}")
    print(f"\n1. Usable Audios (Passed Combined Check):")
    print(f"   - Total Passed: {total_pass} ({total_pass/total_files:.1%})")
    for site, count in pass_counts.items(): print(f"     - {site}: {count}")
    print(f"\n2. Unusable Audios (Failed Combined Check):")
    print(f"   - Total Failed: {total_fail} ({total_fail/total_files:.1%})")
    for site, count in fail_counts.items(): print(f"     - {site}: {count}")
    print("\n3. Reasons for Failure (based on combined rules):")
    if total_fail > 0:
        sorted_reasons = sorted(failure_reasons.items(), key=lambda item: item[1], reverse=True)
        for reason, count in sorted_reasons:
            print(f"   - {reason}: {count} failures ({count/total_fail:.1%})")
    else:
        print("   - No failures recorded!")
    print("\n--- End of Report ---")

# --- UPDATED PLOT FUNCTION ---
def generate_plots(pass_counts, fail_counts, failure_reasons):
    """
    Generates and displays plots with increased font sizes for better readability.
    """
    # --- FONT SIZE CONFIGURATION ---
    scaler = 2
    TITLE_FONTSIZE = 20*scaler
    SUBTITLE_FONTSIZE = 16*scaler
    AXIS_LABEL_FONTSIZE = 14*scaler
    LEGEND_FONTSIZE = 12*scaler
    TICK_LABEL_FONTSIZE = 12*scaler
    ANNOTATION_FONTSIZE = 11*scaler

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8)) # Increased figure size slightly for more space
    
    # --- Main Title ---
    fig.suptitle(
        'Audio Usability Summary (Combined Criteria)', 
        fontsize=TITLE_FONTSIZE, 
        fontweight='bold'
    )
    
    # --- Plot 1: Pass/Fail counts by Site ---
    sites = list(SITES)
    ax1.bar(sites, [pass_counts[s] for s in sites], label='Pass', color='g')
    ax1.bar(sites, [fail_counts[s] for s in sites], bottom=[pass_counts[s] for s in sites], label='Fail', color='r')
    
    ax1.set_title('Audio Usability by Site', fontsize=SUBTITLE_FONTSIZE)
    ax1.set_ylabel('Number of Audios', fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_xlabel('Site', fontsize=AXIS_LABEL_FONTSIZE)
    ax1.legend(fontsize=LEGEND_FONTSIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot 2: Failure Reasons ---
    sorted_reasons = sorted([(k, v) for k, v in failure_reasons.items() if v > 0], key=lambda item: item[1])
    
    if sorted_reasons:
        reason_labels = [r[0] for r in sorted_reasons]
        reason_counts = [r[1] for r in sorted_reasons]
        
        ax2.barh(reason_labels, reason_counts, color='coral')
        ax2.set_title('Reasons for Failure', fontsize=SUBTITLE_FONTSIZE)
        ax2.set_xlabel('Number of Failures', fontsize=AXIS_LABEL_FONTSIZE)
        ax2.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
        
        for index, value in enumerate(reason_counts):
            ax2.text(value, index, f' {value}', va='center', fontsize=ANNOTATION_FONTSIZE)
            
    else:
        ax2.text(0.5, 0.5, 'No Failures to Report!', ha='center', va='center', fontsize=14)
        ax2.set_title('Reasons for Failure', fontsize=SUBTITLE_FONTSIZE)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig("audio_combined_criteria_summary_large_font.png")
    print("\nPlots saved to 'audio_combined_criteria_summary_large_font.png'")
    plt.show()

if __name__ == "__main__":
    analyze_with_combined_criteria(FOLDER_PATH)
