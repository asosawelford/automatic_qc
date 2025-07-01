# Automatic Speech Quality Assessment Pipeline

This project provides a command-line tool for automatically assessing the technical quality of speech recordings. It analyzes audio files for common issues like background noise, improper recording levels, clipping, and multiple speakers, generating a structured JSON report for each file.

The primary goal is to create a fast, reliable, and automated first-pass quality check, ensuring that audio data meets a certain standard before being used for more intensive tasks like Automatic Speech Recognition (ASR), research, or archival.

## Features

-   **Flexible Input:** Processes any audio format supported by FFmpeg (e.g., `.wav`, `.mp3`, `.m4a`, `.flac`).
-   **Bulk Processing:** Analyze a single audio file or an entire directory of files in one command.
-   **Comprehensive Analysis:**
    -   **Loudness & Level:** Measures perceived loudness using the **EBU R 128** standard (`LUFS`) and detects audio **clipping**.
    -   **Background Noise:** Estimates the **Signal-to-Noise Ratio (SNR)** in decibels (dB) using Silero VAD.
    -   **Speaker Count:** Provides an estimate of the number of speakers using a fast and CPU-friendly **MFCC + GMM** clustering method.
-   **Structured Output:** Generates a detailed `.json` report for each audio file, containing raw metrics and a final quality assessment (`PASS`/`FAIL`).
-   **Robust & Modular:** Built with a clean, modular structure for easy maintenance and extension.

## Requirements

### 1. System Dependencies

-   **Python** (tested with 3.10.12)
-   **FFmpeg:** This tool is essential for audio decoding and analysis. It must be installed on your system and accessible from the command line (i.e., in your system's PATH).
    -   **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install ffmpeg`
    -   **macOS (using Homebrew):** `brew install ffmpeg`
    -   **Windows:** Download the executable from the [official FFmpeg website](https://ffmpeg.org/download.html) and add its `bin` directory to your system's PATH environment variable.

### 2. Python Packages

All required Python packages are listed in the `requirements.txt` file.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/asosawelford/automatic_cq.git
    cd automatic_cq
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv .env
    source .env/bin/activate
    # On Windows, use: .env\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The script is run from the command line via `main.py`.

#### Basic Usage

-   **To see the help message:**
    ```bash
    python main.py
    ```

-   **To analyze a single audio file:**
    ```bash
    python main.py /path/to/your/audio.mp3
    ```
    This will create a `reports/` directory in your current location and save `audio_report.json` inside it.

-   **To analyze all audio files in a directory:**
    ```bash
    python main.py /path/to/your/audio_folder/
    ```
    This will process every supported audio file in the folder and generate a corresponding `_report.json` for each.

#### Specifying an Output Directory

You can specify a custom directory for the output reports as a second argument.

```bash
python main.py /path/to/your/audio_folder/ /path/to/my/json_outputs/
```

## Output Format

For each audio file processed, a `.json` file is generated containing a detailed report.

**Example `_report.json` file:**
```json
{
    "source_file": "good_quality_test.wav",
    "processing_status": "SUCCESS",
    "error_message": null,
    "analysis_results": {
        "level": {
            "integrated_lufs": -18.75,
            "loudness_range_lu": 8.5,
            "true_peak_dbfs": -2.1,
            "clipping_percent": 0.0
        },
        "noise": {
            "snr_db": 25.41
        },
        "speaker": {
            "estimated_speakers": 1
        }
    },
    "quality_assessment": {
        "loudness_ok": true,
        "clipping_ok": true,
        "snr_ok": true,
        "speaker_count_ok": true,
        "overall_pass": true
    }
}
```

## Project Structure
```
├── main.py                 # Main executable script
├── reports/                # Default directory for output JSON reports
├── requirements.txt        # List of Python dependencies
└── src/
    ├── __init__.py
    ├── standardize.py        # Handles audio loading and standardization
    ├── level_analysis.py     # Measures loudness and clipping
    ├── noise_analysis.py     # Estimates Signal-to-Noise Ratio
    ├── speaker_count.py      # Estimates the number of speakers
    └── vad_utils.py          # Voice Activity Detection utility
```