# Automatic Speech Quality Assessment Pipeline

Automated audio quality assessment for clinical speech recordings in the TELL platform. Analyzes recordings for technical issues (noise, clipping, loudness, speaker count, language) and generates structured reports with growth tracking across bi-weekly monitoring runs.

## Features

- **Bi-weekly monitoring**: Checkpoint-based incremental processing with comparative metrics across runs
- **Language detection**: Identifies spoken language and flags mismatches against expected protocol language
- **Code-switching detection**: Detects language switches within a recording
- **Quality analysis**: Loudness (EBU R128), clipping, SNR, speaker count, speech duration
- **Growth tracking**: New participants, new recordings, growth rate, breakdown by site/protocol
- **Multiple outputs**: Interactive HTML dashboard, PNG/PDF plots, JSON summary, CSV detailed report
- **Configurable**: YAML config for all thresholds, protocol-language mappings, and model parameters
- **Flexible input**: Processes any audio format supported by FFmpeg

## Requirements

### System Dependencies

- **Python** 3.10+
- **FFmpeg** (must be in PATH)
  - Linux: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
- **GPU** (optional): CUDA-capable GPU accelerates speaker embedding and language detection

### Python Packages

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `speechbrain`, `ffmpeg-python`, `scikit-learn`, `matplotlib`, `pyyaml`

## Installation

```bash
git clone https://github.com/asosawelford/automatic_cq.git
cd automatic_cq

python3 -m venv .env
source .env/bin/activate

pip install -r requirements.txt
cp config.example.yaml config.yaml  # Edit to customize thresholds and protocol mappings
```

## Usage

### Bi-weekly Monitoring (primary workflow)

```bash
# First run — processes all audio, creates initial checkpoint
python monitor.py /path/to/takeout_export/ --verbose

# Subsequent runs — only processes new files, compares to previous run
python monitor.py /path/to/takeout_export_new/ --verbose

# Force reprocess everything
python monitor.py /path/to/takeout/ --force-reprocess

# Skip expensive analyses for faster testing
python monitor.py /path/to/takeout/ --skip-langid --skip-speaker-count
```

**Full options:**

```
python monitor.py <takeout-path> [options]

Options:
  --config <path>           Path to config.yaml (default: config.yaml)
  --output-dir <path>       Output directory for reports
  --checkpoint-dir <path>   Checkpoint directory
  --force-reprocess         Reprocess all files, ignoring checkpoint
  --skip-langid             Skip language detection
  --skip-speaker-count      Skip speaker count analysis
  --export-to-backend       Output Django-ready JSON
  --verbose                 Enable detailed logging
```

### Single File Analysis

```bash
python analyze_single.py /path/to/audio.webm
python analyze_single.py recording.mp3 --output-dir my_reports/
```

### Batch Analysis (no state tracking)

```bash
python analyze_batch.py /path/to/audio_folder/
python analyze_batch.py /path/to/folder/ --output-dir results/ --skip-langid
```

### Legacy Commands (preserved for backward compatibility)

```bash
python main.py /path/to/audio.mp3                    # Single file or directory
python main4tell_takeouts.py /path/to/takeout/ out/  # TELL takeout processing
```

## Configuration

All settings are in `config.yaml` (copy from `config.example.yaml`):

```yaml
quality_thresholds:
  loudness_target_lufs: -14.0
  loudness_tolerance_lu: 10.0      # Allows -24 to -4 LUFS
  clipping_max_percent: 0.1
  snr_min_db: 9.0
  min_speaker_count: 1
  max_speaker_count: 2             # Allows examiner + participant
  min_speech_duration_s: 10

protocol_language_map:
  "Recuerdo_agradable": "es-PE"
  "Rutina": "es-PE"
  "Re-narración_de_video": "es-PE"
  # Add protocols as needed

models:
  langid_model: "speechbrain/lang-id-voxlingua107-ecapa"
  langid_confidence_threshold: 0.7
  speaker_clustering_distance_threshold: 0.15

paths:
  checkpoint_dir: "checkpoints"
  reports_output_dir: "reports_output"
```

Use `extract_protocol_names.py` to discover all protocol names in your data before filling the language map.

## Output

### Directory Structure

```
automatic_qc/
├── reports_output/
│   └── 2026-07-02/
│       ├── summary_report.json      # Aggregate metrics
│       ├── detailed_report.csv      # One row per audio file
│       ├── dashboard.html           # Interactive HTML dashboard
│       ├── quality_report.png       # Multi-panel overview plot
│       ├── quality_report.pdf
│       ├── comparison.png           # Current vs previous run (if applicable)
│       └── comparison.pdf
└── checkpoints/
    ├── checkpoint_2026-06-18.json
    └── checkpoint_2026-07-02.json
```

### Report JSON Format

```json
{
    "source_file": "audio.webm",
    "participant_code": "HUA0004_0083P",
    "protocol_name": "Recuerdo_agradable",
    "processing_status": "SUCCESS",
    "error_message": null,
    "analysis_results": {
        "level": {
            "integrated_lufs": -18.75,
            "loudness_range_lu": 8.5,
            "true_peak_dbfs": -2.1,
            "clipping_percent": 0.0
        },
        "noise": { "snr_db": 25.41 },
        "speaker": { "estimated_speakers": 1 },
        "speech_duration": 45.2,
        "language": {
            "detected_language": "es",
            "confidence": 0.94
        },
        "code_switching": {
            "code_switching_detected": false,
            "languages_detected": ["es"],
            "chunks_analyzed": 5
        }
    },
    "quality_assessment": {
        "loudness_ok": true,
        "clipping_ok": true,
        "snr_ok": true,
        "speaker_count_ok": true,
        "duration_ok": true,
        "overall_pass": true
    },
    "language_mismatch": false
}
```

## Project Structure

```
automatic_qc/
├── monitor.py                  # Bi-weekly monitoring CLI (main entry point)
├── analyze_single.py           # Single-file analysis CLI
├── analyze_batch.py            # Batch analysis CLI (no state tracking)
├── config.yaml                 # Active configuration
├── config.example.yaml         # Configuration template with comments
├── requirements.txt            # Python dependencies
├── main.py                     # Legacy: single file/directory analysis
├── main4tell_takeouts.py       # Legacy: TELL takeout processing
├── analyze_reportsv2.py        # Legacy: aggregate report analysis
├── extract_protocol_names.py   # Helper: discover protocol names in data
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration loading (YAML + dataclass defaults)
│   ├── pipeline.py             # Core analysis pipeline (generate_quality_report)
│   ├── checkpoint.py           # Checkpoint persistence for incremental processing
│   ├── metrics.py              # Aggregate, growth, and comparative metrics
│   ├── language_detection.py   # Language ID and code-switching detection
│   ├── plotting.py             # Static matplotlib plots (PNG/PDF)
│   ├── dashboard.py            # Self-contained HTML dashboard generation
│   ├── standardize.py          # Audio loading/resampling via FFmpeg
│   ├── level_analysis.py       # Loudness (EBU R128) and clipping detection
│   ├── noise_analysis.py       # SNR estimation using Silero VAD
│   ├── speaker_count.py        # X-vector embeddings + agglomerative clustering
│   ├── speaker_count_MFCC.py   # Alternative: MFCC + GMM (faster, CPU-only)
│   └── vad_utils.py            # Voice Activity Detection (Silero VAD)
├── checkpoints/                # Run checkpoints (auto-created)
├── reports_output/             # Generated reports (auto-created)
└── pretrained_models/          # Cached model weights (auto-downloaded)
```

## Example Workflow

```bash
# Week 1 (June 18)
python monitor.py /data/takeout_2026-06-18/ --verbose
# Output:
#   Processed 487 recordings, 312 participants
#   Pass rate: 76.2%
#   No previous run for comparison

# Week 3 (July 2)
python monitor.py /data/takeout_2026-07-02/ --verbose
# Output:
#   Processed 523 recordings (+36 new), 338 participants (+26 new)
#   Growth rate: 7.4%
#   Average SNR: 12.8 dB (+0.5 dB from previous run)
#   Pass rate: 78.3% (+2.1% from previous run)
#   Language mismatches: 3
#   Outputs saved to: reports_output/2026-07-02/
```

## Quality Thresholds

| Metric | Criterion | Default |
|--------|-----------|---------|
| Loudness | Within target +/- tolerance | -14 +/- 10 LUFS |
| Clipping | Below maximum percent | < 0.1% |
| SNR | Above minimum | >= 9.0 dB |
| Speaker count | Within range | 1-2 speakers |
| Speech duration | Above minimum | > 10 seconds |

## Notes

- Site codes (HUA, IQT, LIM, TUM) are extracted from the first 3 characters of `participant_code`
- Models are downloaded on first run and cached in `pretrained_models/`
- The checkpoint system uses MD5 hashes of file paths for deduplication
- GPU is used automatically if CUDA is available; falls back to CPU
