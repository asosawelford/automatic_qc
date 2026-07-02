# Automatic QC Pipeline Refactoring - Requirements Document

## Context

You are refactoring an existing automated audio quality assessment pipeline for clinical speech recordings in the TELL platform. The current pipeline (`automatic_qc/`) analyzes individual audio files for technical quality issues (loudness, clipping, SNR, speaker count, speech duration).

**Current codebase location**: `/home/aleph/tell/automatic_qc/`

Review the existing code, particularly:
- `main.py` - Standard CLI entry point
- `main4tell_takeouts.py` - Processes TELL participant export directory structures
- `src/` modules - Individual analysis components
- `analyze_reportsv2.py` - Current aggregation tool
- `README.md` - Current documentation
- `/home/aleph/tell/CLAUDE.md` - Project-wide documentation (has detailed "Automatic QC Architecture" section)

## New Requirements

### 1. Periodic Execution Model

**Goal**: Transform from one-off batch processing to bi-weekly monitoring with historical tracking.

**Specifics**:
- Tool will run every 2 weeks on TELL takeout exports
- Must track which recordings were processed in previous runs (avoid re-processing)
- Generate comparative metrics between current run and previous run(s)
- Show growth trends over time

### 2. Input Format: TELL Takeout Structure

**Directory structure** (based on `main4tell_takeouts.py` expectations):
```
<takeout-root>/
├── <participant-folder-1>/
│   ├── audio.webm                      # Audio file
│   ├── feature_extraction.json         # Metadata (contains participant_code, protocol_name)
│   └── [other files...]
├── <participant-folder-2>/
│   ├── audio.webm
│   ├── feature_extraction.json
│   └── ...
└── ...
```

**Key fields in `feature_extraction.json`**:
- `participant_code` (e.g., "UHDACNMSCM00067", "HUA0004_0083P")
- `protocol_name` (e.g., "Recuerdo_agradable", "Rutina", "Re-narración_de_video", "Lámina_1")

### 3. Enhanced Metrics

#### 3.1 Language Mismatch Detection

**Requirement**: Detect when audio language doesn't match expected language for the protocol.

**Implementation approach**:
- Create a **protocol-to-language mapping dictionary** (configurable, likely in a separate config file or top of main script)
  - Example: `{"Recuerdo_agradable": "es-PE", "Rutina": "es-PE", "Re-narración_de_video": "es-PE", "Lámina_1": "es-PE", ...}`
- Use **language identification model** to detect actual spoken language in audio
  - **Recommended**: SpeechBrain Language Identification (`speechbrain/lang-id-voxlingua107-ecapa`) - good balance of accuracy and speed
  - **Alternative**: Whisper's language detection (slower but very accurate)
  - **Not recommended**: fastText (too simplistic for clinical speech)
- Output: Binary flag `language_mismatch: true/false` + detected language code

**Edge cases**:
- If protocol_name not in mapping, skip language check (log warning)
- If audio too short for reliable LangID, mark as `language_mismatch: null` (indeterminate)

#### 3.2 Code-Switching Detection

**Requirement**: Detect if speaker switches languages mid-recording (binary flag only, no timestamps needed).

**Implementation approach**:
- After language identification, analyze **confidence scores** across audio segments
- If multiple languages detected with high confidence in different parts → flag as code-switching
- **Simple heuristic**: Run LangID on 3-5 non-overlapping chunks; if >1 language detected with >70% confidence → code-switching
- Output: Binary flag `code_switching_detected: true/false`

**Constraint**: Must not be computationally expensive. Use existing LangID model results where possible.

#### 3.3 Enhanced Audio Quality Metrics

**Existing metrics to keep**:
- ✅ Loudness (EBU R128 LUFS)
- ✅ Clipping percentage
- ✅ Signal-to-Noise Ratio (SNR in dB)
- ✅ Speech duration (minimum threshold)

**Update speaker count logic**:
- Current: Expects exactly 1 speaker
- **New**: Allow up to 2 speakers (clinical protocols may have examiner + participant)
- Change threshold: `speaker_count_ok = (1 <= estimated_speakers <= 2)`

#### 3.4 Database Growth Metrics

**Requirement**: Track overall growth of the "database" (TELL takeout collection) between runs.

**Metrics to compute**:
- **New participants** since last run (unique participant_codes not seen before)
- **New recordings** since last run (total new audio files processed)
- **Total participants** to date (cumulative unique participant_codes)
- **Total recordings** to date (cumulative audio files)
- **Growth rate**: `(new_recordings / total_recordings_previous_run) * 100` %

**Breakdown by**:
- Site code (first 3 characters of participant_code, e.g., "HUA", "IQT", "LIM", "TUM")
- Protocol name (from feature_extraction.json)

#### 3.5 Comparative Metrics Across Runs

**Requirement**: Compare quality metrics between current run and previous run.

**Metrics to compare**:
- Average SNR (current vs. previous)
- Pass rate percentage (overall_pass) (current vs. previous)
- Language mismatch rate (current vs. previous)
- Code-switching rate (current vs. previous)
- Average speech duration (current vs. previous)
- Multi-speaker recording rate (current vs. previous)

**Output format**: Show delta/trend, e.g., "SNR: 12.3 dB (+1.5 dB from last run)"

### 4. State Persistence (Checkpoint System)

**Requirement**: Track processing history to avoid re-analyzing same files and enable comparative metrics.

**Implementation**:
- Use **JSON checkpoint files** (one per run)
- Checkpoint filename format: `checkpoint_<YYYY-MM-DD>.json`
- Checkpoint should contain:
  - `run_date`: ISO timestamp of run
  - `processed_files`: List of processed audio file paths or unique identifiers (participant_code + audio filename hash?)
  - `aggregate_metrics`: Summary statistics for this run (counts, averages, pass rates, etc.)
  - `detailed_reports`: Optional (could link to individual quality reports directory)

**Checkpoint storage location**: `automatic_qc/checkpoints/` directory

**Logic**:
- On each run, load most recent checkpoint
- Skip files already in `processed_files` (unless `--force-reprocess` flag)
- After processing, generate new checkpoint with updated metrics

### 5. Output Formats

**Generate all three formats** (we'll choose best one later and remove others):

#### 5.1 Interactive HTML Dashboard
- Single-page HTML with embedded charts (use Plotly or similar)
- Sections:
  - Overview: Total counts, growth metrics, quality summary
  - Quality trends: Line charts comparing metrics across runs
  - Site breakdown: Bar charts by site code
  - Protocol breakdown: Bar charts by protocol name
  - Failure analysis: Pie chart or bar chart of failure reasons
  - Language analysis: Mismatch/code-switching rates

#### 5.2 Static Plots (PNG/PDF)
- Similar to current `analyze_reportsv2.py` output
- Multi-panel figure with:
  - Growth charts (participants, recordings over time)
  - Quality metrics comparison (current vs. previous)
  - Site/protocol breakdowns
- Save as high-res PNG and PDF

#### 5.3 JSON/CSV for Downstream Processing
- **JSON summary**: `summary_report_<YYYY-MM-DD>.json` with all aggregate metrics
- **CSV detailed**: `detailed_report_<YYYY-MM-DD>.csv` with one row per audio file (all metrics as columns)

**Output directory structure**:
```
automatic_qc/
├── reports_output/
│   └── <YYYY-MM-DD>/
│       ├── summary_report.json
│       ├── detailed_report.csv
│       ├── dashboard.html
│       ├── quality_trends.png
│       └── quality_trends.pdf
└── checkpoints/
    ├── checkpoint_2026-06-10.json
    ├── checkpoint_2026-06-24.json
    └── ...
```

### 6. Integration Design (Hybrid CLI + Backend-Ready)

**Primary interface**: CLI tool (standalone Python script)

**Future integration hook**: Design with Django backend integration in mind

**Implementation**:
- Keep core analysis logic in standalone modules (current `src/` structure is good)
- Add optional `--export-to-backend` flag that outputs Django-ready JSON format
- Include Django model suggestions in comments (e.g., "# Could be stored in AudioQuality model with ForeignKey to Audio")
- Keep database-agnostic: use dataclasses or simple dicts, not Django models yet

### 7. Computational Budget

**Target hardware**:
- **Current**: Desktop with GPU (12GB RAM, CUDA available)
- **Future**: AWS deployment (scalable compute)

**Constraints**:
- Language ID should be GPU-accelerated but not require >8GB VRAM
- Total runtime target: <30 minutes for ~500 recordings
- Optimize for batch processing (load models once, process all files)

### 8. Critical Bug Fixes (Must Address)

#### 8.1 CUDA Tensor Conversion Error
**Current issue**: Many quality reports fail with error:
```
"can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first."
```

**Location**: Likely in `src/speaker_count.py` or `src/vad_utils.py`

**Fix**: Add `.cpu()` before `.numpy()` for all PyTorch tensors when converting to NumPy arrays.

**Example**:
```python
# Before (broken):
embedding.squeeze().numpy()

# After (fixed):
embedding.squeeze().cpu().numpy()
```

Test thoroughly on GPU to ensure all tensor conversions are fixed.

#### 8.2 Code Organization

**Issues**:
- `main.py` and `main4tell_takeouts.py` have significant code duplication
- No clear separation between "single file analysis" and "batch monitoring" workflows

**Proposed refactoring**:
- Extract common logic into `src/pipeline.py` or similar
- Create separate entry points:
  - `analyze_single.py` - Analyze one file (current main.py behavior)
  - `analyze_batch.py` - Analyze directory without state tracking
  - `monitor.py` - New bi-weekly monitoring mode with checkpoints and comparative metrics
- Keep `main.py` as a dispatcher or deprecate it

### 9. Configuration Management

**Create a configuration system** instead of hardcoded constants.

**Suggested approach**: `config.yaml` file with sections:
```yaml
quality_thresholds:
  loudness_target_lufs: -14.0
  loudness_tolerance_lu: 10.0
  clipping_max_percent: 0.1
  snr_min_db: 9.0
  min_speaker_count: 1
  max_speaker_count: 2
  min_speech_duration_s: 10

protocol_language_map:
  "Recuerdo_agradable": "es-PE"
  "Rutina": "es-PE"
  "Re-narración_de_video": "es-PE"
  "Lámina_1": "es-PE"
  # Add more protocols as needed

models:
  vad_threshold: 0.5
  speaker_clustering_distance_threshold: 0.15
  langid_confidence_threshold: 0.7

paths:
  checkpoint_dir: "checkpoints"
  reports_output_dir: "reports_output"
  pretrained_models_dir: "pretrained_models"
```

Load with PyYAML or similar.

### 10. CLI Interface Design

**New CLI for bi-weekly monitoring mode**:

```bash
python monitor.py <takeout-path> [options]

Required:
  <takeout-path>              Path to TELL takeout export directory

Options:
  --config <path>             Path to config.yaml (default: config.yaml)
  --output-dir <path>         Output directory for reports (default: reports_output/)
  --checkpoint-dir <path>     Checkpoint directory (default: checkpoints/)
  --force-reprocess           Reprocess all files, ignoring checkpoint
  --skip-langid               Skip language detection (faster, for testing)
  --skip-speaker-count        Skip speaker count analysis (faster, for testing)
  --export-to-backend         Output Django-ready JSON for future backend integration
  --verbose                   Enable detailed logging
```

**Existing CLIs** (keep for backward compatibility, refactor internals):
```bash
# Single file analysis
python analyze_single.py <audio-file> [--output-dir <dir>]

# Batch analysis (no state tracking)
python analyze_batch.py <audio-directory> [--output-dir <dir>]
```

### 11. Deliverables Checklist

Please provide:

- [ ] **Refactored codebase** with clean module structure
- [ ] **Bug fixes**: CUDA tensor conversion errors resolved
- [ ] **New features**: Language mismatch, code-switching detection, enhanced metrics
- [ ] **Checkpoint system**: JSON-based state persistence
- [ ] **Comparative metrics**: Current vs. previous run analysis
- [ ] **Three output formats**: HTML dashboard, PNG/PDF plots, JSON/CSV exports
- [ ] **Configuration system**: YAML config file with all thresholds/settings
- [ ] **Updated CLI**: New `monitor.py` entry point with comprehensive options
- [ ] **Documentation**: 
  - Updated README.md with new usage examples
  - Inline code comments explaining key logic
  - Example config.yaml with comments
  - Sample output files (1-2 examples of each format)
- [ ] **Testing**: 
  - Verify GPU compatibility (CUDA tensor fixes)
  - Test on small sample dataset (~10 files)
  - Validate checkpoint persistence works across runs
  - Ensure all three output formats generate correctly

### 12. Implementation Priorities

**Phase 1 (Critical - Do First)**:
1. Fix CUDA tensor conversion bugs
2. Add language detection module
3. Implement checkpoint system
4. Refactor code duplication between main.py variants

**Phase 2 (Core Features)**:
5. Implement code-switching detection
6. Add growth metrics tracking
7. Add comparative metrics (current vs. previous)
8. Create configuration system (YAML)

**Phase 3 (Output & Polish)**:
9. Generate HTML dashboard
10. Generate PNG/PDF plots
11. Generate JSON/CSV exports
12. Update documentation
13. Create example outputs

### 13. Code Quality Guidelines

- **Type hints**: Use Python type hints for all function signatures
- **Docstrings**: Google-style docstrings for all modules, classes, and functions
- **Error handling**: Graceful failure for individual files (don't crash entire batch if one file fails)
- **Logging**: Use Python `logging` module, not `print()` statements (except for main CLI output)
- **Performance**: Minimize model loading (load once, cache globally)
- **Modularity**: Each analysis module should be independently testable
- **Backward compatibility**: Existing quality report JSON format should be extended, not broken

### 14. Example Workflow

**Scenario**: User runs bi-weekly monitoring for the first time.

```bash
# First run (June 10, 2026)
python monitor.py /path/to/takeout_2026-06-10/ --verbose

# Output:
# - Processes all audio files (no checkpoint exists)
# - Generates checkpoint_2026-06-10.json with metadata and metrics
# - Creates reports_output/2026-06-10/ with all three output formats
# - Console shows: "Processed 487 recordings, 312 participants. No previous run for comparison."

# Second run (June 24, 2026)
python monitor.py /path/to/takeout_2026-06-24/ --verbose

# Output:
# - Loads checkpoint_2026-06-10.json
# - Skips already-processed files, processes only new ones
# - Generates checkpoint_2026-06-24.json
# - Creates reports_output/2026-06-24/ with comparison to June 10 run
# - Console shows:
#   "Processed 523 recordings (+36 new), 338 participants (+26 new).
#    Growth rate: 7.4%
#    Average SNR: 12.8 dB (+0.5 dB from previous run)
#    Pass rate: 78.3% (+2.1% from previous run)"
```

### 15. Notes & Context

- Original repo is at: https://github.com/asosawelford/automatic_cq.git
- This is part of the TELL (Technology for Early Language Learning) platform
- Clinical protocols involve examiner + participant, so 2 speakers is acceptable
- Site codes (HUA, IQT, LIM, TUM) correspond to different research sites in Peru
- Language variants are primarily Spanish dialects (es-PE, es-AR, etc.)
- This tool will eventually be integrated into Django backend but should remain CLI-capable

### 16. Questions to Address During Implementation

If you encounter ambiguities, use reasonable defaults and document your assumptions. Specifically:

- **Language mapping**: If a protocol_name isn't in the config map, should it error or skip LangID? (Recommend: skip with warning)
- **Checkpoint uniqueness**: Should checkpoints be per-run-date or allow multiple runs per day? (Recommend: timestamp down to second if needed)
- **Memory management**: For very large takeouts (>1000 files), should reports be streamed to disk or accumulated in memory? (Recommend: stream to disk)
- **Site code extraction**: What if participant_code doesn't start with a known site code? (Recommend: categorize as "UNKNOWN" site)

---

## Summary

Transform `automatic_qc/` from a one-off audio quality checker into a **bi-weekly monitoring dashboard** with:
- ✅ Language mismatch & code-switching detection
- ✅ Growth tracking (new participants, recordings)
- ✅ Comparative metrics across runs
- ✅ State persistence via JSON checkpoints
- ✅ Three output formats (HTML, plots, JSON/CSV)
- ✅ CUDA bug fixes
- ✅ Clean, configurable, extensible architecture

**Estimated scope**: Medium refactoring (~1500-2000 lines of new/modified code across 10-15 files). Focus on correctness, maintainability, and performance.

Good luck! 🚀
