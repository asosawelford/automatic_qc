# Automatic QC Refactoring - Quick Summary

## What's Changing

**From**: One-off batch audio quality checker  
**To**: Bi-weekly monitoring dashboard with growth tracking and comparative metrics

## New Features

### 1. Language Analysis
- **Language Mismatch Detection**: Compare detected language vs. expected (from protocol mapping)
- **Code-Switching Detection**: Binary flag if speaker switches languages mid-recording
- **Model**: SpeechBrain Language ID (balanced accuracy/speed)

### 2. Growth Metrics
- Track new participants and recordings between runs
- Breakdown by site code (HUA, IQT, LIM, TUM) and protocol name
- Calculate growth rates

### 3. Comparative Metrics
- Compare current run vs. previous run:
  - Average SNR (±X dB)
  - Pass rate percentage (±X%)
  - Language mismatch rate
  - Code-switching rate
  - Speech duration trends

### 4. State Persistence
- JSON checkpoints (one per run)
- Skip already-processed files
- Enable historical comparisons

### 5. Multiple Output Formats
- **HTML Dashboard**: Interactive single-page with charts
- **PNG/PDF Plots**: Static multi-panel figures
- **JSON/CSV**: Machine-readable for downstream processing

### 6. Configuration System
- `config.yaml` for all thresholds, protocol→language mappings, model settings
- No more hardcoded constants

## Bug Fixes

1. **CUDA tensor conversion** - Add `.cpu()` before `.numpy()` throughout
2. **Code duplication** - Refactor main.py and main4tell_takeouts.py
3. **Speaker count threshold** - Change from "exactly 1" to "1 or 2"

## New CLI

```bash
# Bi-weekly monitoring (new)
python monitor.py /path/to/takeout/ --verbose

# Single file (refactored)
python analyze_single.py audio.mp3

# Batch without state tracking (refactored)
python analyze_batch.py /path/to/audio_folder/
```

## Technical Specs

- **Target runtime**: <30 min for ~500 recordings
- **Hardware**: GPU (12GB RAM) now, AWS later
- **Architecture**: Hybrid CLI + backend-ready design
- **State**: JSON checkpoints in `checkpoints/` directory
- **Outputs**: Organized by date in `reports_output/YYYY-MM-DD/`

## Example Workflow

```bash
# Week 1
python monitor.py /data/takeout_2026-06-10/
# → Processes 487 recordings, 312 participants
# → No previous run for comparison

# Week 3 (2 weeks later)
python monitor.py /data/takeout_2026-06-24/
# → Processes 523 recordings (+36 new)
# → 338 participants (+26 new)
# → Growth rate: 7.4%
# → SNR: 12.8 dB (+0.5 from last run)
# → Pass rate: 78.3% (+2.1% from last run)
```

## Files to Review

- `REFACTORING_PROMPT.md` - Full detailed requirements (give this to Opus)
- Current codebase in `automatic_qc/`
- `/home/aleph/tell/CLAUDE.md` - "Automatic QC Architecture" section
