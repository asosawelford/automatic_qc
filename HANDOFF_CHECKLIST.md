# Handoff Checklist for Claude Opus Refactoring

## Before You Start

- [ ] Read `REFACTORING_SUMMARY.md` for quick overview
- [ ] Read `REFACTORING_PROMPT.md` in full (this is what you'll give to Opus)
- [ ] Review current codebase structure in `automatic_qc/`
- [ ] Check `/home/aleph/tell/CLAUDE.md` - "Automatic QC Architecture" section

## Protocol → Language Mapping

**Action needed**: Create/populate the protocol language mapping dictionary in config.yaml

Here are some protocols extracted from the codebase. You'll need to map each to language codes:

```yaml
protocol_language_map:
  "Recuerdo_agradable": "es-PE"        # Pleasant memory recall (Spanish)
  "Rutina": "es-PE"                    # Daily routine (Spanish)
  "Re-narración_de_video": "es-PE"    # Video re-narration (Spanish)
  "Lámina_1": "es-PE"                  # Picture description (Spanish)
  # Add more protocols as you discover them in the takeout data
```

**How to find all protocols**:
```bash
cd /home/aleph/tell/automatic_qc
find . -name "feature_extraction.json" -exec grep -h "protocol_name" {} \; | sort -u
```

## Sample Data for Testing

**Recommended test workflow**:
1. Start with a small subset (~10-20 recordings) from the takeout
2. Run first iteration to verify basic functionality
3. Expand to full dataset once bugs are fixed

**Sample data location** (if available):
- Check `automatic_qc/db/` for existing test data
- Or create a `test_data/` folder with a few sample recordings

## Giving the Prompt to Claude Opus

### Option 1: Copy the entire REFACTORING_PROMPT.md
```
# In Claude.ai or Claude Code with Opus model
Upload or paste the contents of REFACTORING_PROMPT.md

Then say:
"Please implement this refactoring. Start with Phase 1 (critical fixes),
then Phase 2 (core features), then Phase 3 (output & polish).

After each phase, show me what you've done and wait for my approval
before continuing to the next phase."
```

### Option 2: Iterative approach
```
Phase 1 only:
"Review the automatic_qc/ codebase and implement Phase 1 from the refactoring requirements:
1. Fix CUDA tensor conversion bugs
2. Add language detection module
3. Implement checkpoint system
4. Refactor code duplication between main.py variants

Use the requirements in REFACTORING_PROMPT.md as your guide."
```

## After Opus Completes Phase 1

**Test checklist**:
- [ ] Run on GPU - verify no CUDA errors
- [ ] Check language detection works on sample audio
- [ ] Verify checkpoint saves/loads correctly
- [ ] Test that main.py duplication is resolved

**Command to test**:
```bash
cd /home/aleph/tell/automatic_qc
python monitor.py /path/to/small_test_takeout/ --verbose
```

## After Opus Completes Phase 2

**Test checklist**:
- [ ] Code-switching detection returns boolean
- [ ] Growth metrics calculate correctly (new participants, recordings)
- [ ] Comparative metrics work (need 2 runs to test)
- [ ] config.yaml loads and overrides work

**Command to test comparative metrics**:
```bash
# First run
python monitor.py /path/to/takeout_week1/ --output-dir reports_output/

# Second run (should compare to first)
python monitor.py /path/to/takeout_week3/ --output-dir reports_output/
```

## After Opus Completes Phase 3

**Test checklist**:
- [ ] HTML dashboard opens in browser and is interactive
- [ ] PNG plots generate and look good
- [ ] PDF plots generate
- [ ] JSON summary is well-structured
- [ ] CSV detailed report has all columns
- [ ] README.md is updated with new examples

**Output to review**:
```bash
ls -lh reports_output/YYYY-MM-DD/
# Should see: dashboard.html, *.png, *.pdf, summary_report.json, detailed_report.csv
```

## Integration Testing

Once all phases complete:

1. **Full run on real data**:
   ```bash
   python monitor.py /path/to/real_takeout/ --verbose
   ```

2. **Check all outputs**:
   - Open dashboard.html in browser
   - Review plots visually
   - Spot-check JSON/CSV data

3. **Performance test**:
   - Time the run (should be <30 min for ~500 recordings)
   - Monitor GPU memory usage

4. **Edge cases**:
   - Unknown protocol_name (should skip LangID gracefully)
   - Empty audio files (should fail gracefully)
   - Malformed feature_extraction.json (should log warning, continue)

## Questions to Ask Opus If Needed

- "How did you handle protocol_names not in the language mapping?"
- "What happens if checkpoint file is corrupted?"
- "Can I customize the HTML dashboard theme/colors?"
- "How do I add a new quality metric without breaking existing checkpoints?"

## Final Deliverables to Verify

- [ ] Refactored code runs without errors
- [ ] All 3 output formats generate
- [ ] Checkpoints persist correctly
- [ ] Documentation updated (README, docstrings, config examples)
- [ ] CUDA bug fixed and tested on GPU
- [ ] Example outputs included in repo

## Post-Refactoring: Backend Integration

Once CLI tool is stable, next steps:
1. Design Django models (AudioQuality, QualityCheckpoint, etc.)
2. Create management command that wraps monitor.py
3. Add API endpoints to fetch quality reports
4. Build frontend dashboard (React) to visualize trends

---

## Quick Reference: Project Structure After Refactoring

```
automatic_qc/
├── src/
│   ├── pipeline.py              # Core refactored analysis pipeline
│   ├── langid.py                # NEW: Language detection
│   ├── checkpoint.py            # NEW: State persistence
│   ├── metrics.py               # NEW: Comparative metrics
│   └── [existing modules...]
├── monitor.py                   # NEW: Main bi-weekly monitoring CLI
├── analyze_single.py            # Refactored single-file analysis
├── analyze_batch.py             # Refactored batch analysis
├── config.yaml                  # NEW: Configuration file
├── config.example.yaml          # NEW: Example config with comments
├── reports_output/              # NEW: Organized by date
│   └── YYYY-MM-DD/
│       ├── dashboard.html
│       ├── quality_trends.png
│       ├── quality_trends.pdf
│       ├── summary_report.json
│       └── detailed_report.csv
├── checkpoints/                 # NEW: JSON checkpoints
│   └── checkpoint_YYYY-MM-DD.json
├── REFACTORING_PROMPT.md        # Requirements doc
├── REFACTORING_SUMMARY.md       # Quick overview
├── HANDOFF_CHECKLIST.md         # This file
└── [existing files...]
```

Good luck with the refactoring! 🎉
