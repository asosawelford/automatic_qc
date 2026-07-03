#!/usr/bin/env python3
"""
Bi-weekly monitoring entry point for TELL audio quality assessment.

Usage:
    python monitor.py <takeout-path> [options]
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

from src.config import Config, load_config
from src.pipeline import generate_quality_report, NumpyJSONEncoder
from src.checkpoint import (
    get_latest_checkpoint,
    get_processed_ids,
    save_checkpoint,
    compute_file_id,
)
from src.metrics import (
    compute_aggregate_metrics,
    compute_growth_metrics,
    compute_comparative_metrics,
)

logger = logging.getLogger("monitor")


def discover_audio_files(
    takeout_path: str,
    from_date: str = None,
    to_date: str = None,
) -> List[Dict[str, Any]]:
    """Scan takeout directory for audio files and their metadata.

    Args:
        takeout_path: Root directory to scan.
        from_date: Include only recordings on or after this date (YYYY-MM-DD).
        to_date: Include only recordings on or before this date (YYYY-MM-DD).
    """
    entries = []
    skipped = 0

    for root, dirs, files in os.walk(takeout_path):
        if "audio.webm" in files:
            audio_path = os.path.join(root, "audio.webm")
            metadata_path = os.path.join(root, "feature_extraction.json")

            participant_code = "Unknown"
            protocol_name = "Unknown"
            recording_date = None

            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    participant_code = metadata.get("participant_code", "Unknown")
                    protocol_name = metadata.get("protocol_name", "Unknown")
                    # Extract date from query_timestamp_start or audio_file name
                    timestamp = metadata.get("query_timestamp_start", "")
                    if timestamp:
                        recording_date = timestamp.split(" ")[0]
                    else:
                        # Fallback: parse from audio_file field (format: CODE__YYYYMMDD__Task__hash.webm)
                        audio_filename = metadata.get("audio_file", "")
                        parts = audio_filename.split("__")
                        if len(parts) >= 2 and len(parts[1]) == 8 and parts[1].isdigit():
                            d = parts[1]
                            recording_date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
                except Exception as e:
                    logger.warning(f"Could not read metadata: {metadata_path}: {e}")

            # Date filtering
            if from_date and recording_date and recording_date < from_date:
                skipped += 1
                continue
            if to_date and recording_date and recording_date > to_date:
                skipped += 1
                continue

            entries.append({
                "audio_path": audio_path,
                "participant_code": participant_code,
                "protocol_name": protocol_name,
                "recording_date": recording_date,
            })

    if skipped > 0:
        logger.info(f"Filtered out {skipped} recordings outside date range [{from_date or '...'} to {to_date or '...'}]")

    return entries


def check_language_mismatch(
    report: Dict[str, Any],
    protocol_name: str,
    protocol_language_map: Dict[str, str],
) -> bool:
    """Check if detected language mismatches expected language for protocol."""
    expected = protocol_language_map.get(protocol_name)
    if expected is None:
        return False  # No mapping, skip check

    lang_result = report.get("analysis_results", {}).get("language")
    if lang_result is None or lang_result.get("detected_language") is None:
        return False  # Detection failed, can't determine mismatch

    detected = lang_result["detected_language"]
    # Compare language family (first 2 chars: "es", "en", "pt")
    return detected[:2].lower() != expected[:2].lower()


def run_monitor(
    takeout_path: str,
    config: Config,
    force_reprocess: bool = False,
    skip_langid: bool = False,
    skip_speaker_count: bool = False,
    verbose: bool = False,
    from_date: str = None,
    to_date: str = None,
) -> Dict[str, Any]:
    """Main monitoring pipeline."""
    run_date = datetime.now().strftime("%Y-%m-%d")

    # Load previous checkpoint
    checkpoint_dir = config.paths.checkpoint_dir
    prev_checkpoint = get_latest_checkpoint(checkpoint_dir)
    prev_processed = get_processed_ids(prev_checkpoint)

    if prev_checkpoint:
        logger.info(f"Loaded previous checkpoint: {prev_checkpoint.get('run_date', 'unknown')}")
        logger.info(f"Previously processed: {len(prev_processed)} files")
    else:
        logger.info("No previous checkpoint found. Processing all files.")

    # Discover files
    entries = discover_audio_files(takeout_path, from_date=from_date, to_date=to_date)
    logger.info(f"Found {len(entries)} audio files in takeout.")

    # Filter already processed
    to_process = []
    all_file_ids = []

    for entry in entries:
        file_id = compute_file_id(entry["participant_code"], entry["audio_path"])
        all_file_ids.append(file_id)

        if not force_reprocess and file_id in prev_processed:
            continue
        to_process.append((entry, file_id))

    logger.info(f"New files to process: {len(to_process)} (skipping {len(entries) - len(to_process)} already processed)")

    # Process files with incremental checkpointing
    CHECKPOINT_INTERVAL = 50
    reports = []
    newly_processed_ids = set()

    for i, (entry, file_id) in enumerate(to_process, 1):
        if verbose:
            logger.info(f"[{i}/{len(to_process)}] Processing: {entry['participant_code']} - {entry['protocol_name']}")

        report = generate_quality_report(
            entry["audio_path"],
            config=config,
            skip_speaker_count=skip_speaker_count,
            skip_langid=skip_langid,
        )

        report["participant_code"] = entry["participant_code"]
        report["protocol_name"] = entry["protocol_name"]
        report["file_id"] = file_id

        # Check language mismatch
        if not skip_langid:
            mismatch = check_language_mismatch(report, entry["protocol_name"], config.protocol_language_map)
            report["language_mismatch"] = mismatch

        reports.append(report)
        newly_processed_ids.add(file_id)

        # Save incremental checkpoint every N files
        if i % CHECKPOINT_INTERVAL == 0:
            partial_processed = list(prev_processed | newly_processed_ids)
            partial_aggregate = compute_aggregate_metrics(reports, config.site_codes)
            save_checkpoint(checkpoint_dir, partial_processed, partial_aggregate, run_date)
            logger.info(f"Incremental checkpoint saved ({i}/{len(to_process)} processed)")

    # Final metrics and checkpoint
    participants = {r.get("participant_code") for r in reports if r.get("participant_code") != "Unknown"}
    aggregate = compute_aggregate_metrics(reports, config.site_codes)
    growth = compute_growth_metrics(aggregate, prev_checkpoint, all_file_ids, participants)
    comparison = compute_comparative_metrics(aggregate, prev_checkpoint)

    all_processed = list(prev_processed | newly_processed_ids)
    save_checkpoint(checkpoint_dir, all_processed, aggregate, run_date)

    # Build final result
    result = {
        "run_date": run_date,
        "aggregate_metrics": aggregate,
        "growth_metrics": growth,
        "comparative_metrics": comparison,
        "reports": reports,
    }

    # Print summary
    _print_summary(aggregate, growth, comparison)

    return result


def _print_summary(aggregate, growth, comparison):
    """Print a human-readable summary to console."""
    print("\n" + "=" * 60)
    print("  TELL Audio Quality Monitoring Report")
    print("=" * 60)

    print(f"\n  Processed: {aggregate['total_files']} recordings, {aggregate['unique_participants']} participants")
    print(f"  Errors: {aggregate['errors']}")
    print(f"  Pass rate: {aggregate['pass_rate']:.1%}")

    if aggregate.get("avg_snr_db") is not None:
        print(f"  Average SNR: {aggregate['avg_snr_db']:.1f} dB")
    if aggregate.get("avg_speech_duration_s") is not None:
        print(f"  Average speech duration: {aggregate['avg_speech_duration_s']:.1f} s")

    if aggregate.get("language_mismatch_count", 0) > 0:
        print(f"  Language mismatches: {aggregate['language_mismatch_count']}")
    if aggregate.get("code_switching_count", 0) > 0:
        print(f"  Code-switching detected: {aggregate['code_switching_count']}")

    # Growth
    if growth:
        print(f"\n  --- Growth ---")
        if growth.get("is_first_run"):
            print("  First run - no comparison available.")
        else:
            print(f"  New recordings: +{growth['new_recordings']}")
            if growth.get("growth_rate_percent") is not None:
                print(f"  Growth rate: {growth['growth_rate_percent']:.1f}%")

    # Comparison
    if comparison:
        print(f"\n  --- Comparison with previous run ---")
        for key, val in comparison.items():
            if val and val.get("delta") is not None:
                label = key.replace("_", " ").title()
                delta = val["delta"]
                current = val["current"]
                if "rate" in key:
                    print(f"  {label}: {current:.1%} ({delta:+.1%})")
                elif "snr" in key:
                    print(f"  {label}: {current:.1f} dB ({delta:+.1f} dB)")
                elif "duration" in key:
                    print(f"  {label}: {current:.1f} s ({delta:+.1f} s)")

    # Failure reasons
    reasons = aggregate.get("failure_reasons", {})
    if reasons:
        print(f"\n  --- Failure reasons ---")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="TELL Audio Quality Monitoring - Bi-weekly Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("takeout_path", help="Path to TELL takeout export directory")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output-dir", help="Output directory for reports")
    parser.add_argument("--checkpoint-dir", help="Checkpoint directory")
    parser.add_argument("--from-date", help="Only include recordings on or after this date (YYYY-MM-DD)")
    parser.add_argument("--to-date", help="Only include recordings on or before this date (YYYY-MM-DD)")
    parser.add_argument("--force-reprocess", action="store_true", help="Reprocess all files, ignoring checkpoint")
    parser.add_argument("--skip-langid", action="store_true", help="Skip language detection")
    parser.add_argument("--skip-speaker-count", action="store_true", help="Skip speaker count analysis")
    parser.add_argument("--export-to-backend", action="store_true", help="Output Django-ready JSON")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config = load_config(args.config)

    # Override paths from CLI
    if args.output_dir:
        config.paths.reports_output_dir = args.output_dir
    if args.checkpoint_dir:
        config.paths.checkpoint_dir = args.checkpoint_dir

    # Validate input
    if not os.path.isdir(args.takeout_path):
        print(f"Error: Takeout path does not exist: {args.takeout_path}")
        sys.exit(1)

    # Run monitoring
    result = run_monitor(
        takeout_path=args.takeout_path,
        config=config,
        force_reprocess=args.force_reprocess,
        skip_langid=args.skip_langid,
        skip_speaker_count=args.skip_speaker_count,
        verbose=args.verbose,
        from_date=args.from_date,
        to_date=args.to_date,
    )

    # Generate outputs
    run_date = result["run_date"]
    output_dir = os.path.join(config.paths.reports_output_dir, run_date)
    os.makedirs(output_dir, exist_ok=True)

    # JSON summary
    if config.outputs.generate_json_summary:
        summary_path = os.path.join(output_dir, "summary_report.json")
        summary = {
            "run_date": run_date,
            "aggregate_metrics": result["aggregate_metrics"],
            "growth_metrics": result["growth_metrics"],
            "comparative_metrics": result["comparative_metrics"],
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, cls=NumpyJSONEncoder)
        logger.info(f"Summary saved: {summary_path}")

    # CSV detailed
    if config.outputs.generate_csv_detailed:
        _export_csv(result["reports"], output_dir)

    # Plots
    if config.outputs.generate_png_plots or config.outputs.generate_pdf_plots:
        try:
            from src.plotting import generate_plots
            generate_plots(result, output_dir, config)
        except ImportError:
            logger.warning("Plotting module not available. Skipping plots.")

    # HTML dashboard
    if config.outputs.generate_html_dashboard:
        try:
            from src.dashboard import generate_dashboard
            generate_dashboard(result, output_dir, config)
        except ImportError:
            logger.warning("Dashboard module not available. Skipping HTML dashboard.")

    print(f"\nOutputs saved to: {output_dir}")


def _export_csv(reports: List[Dict[str, Any]], output_dir: str):
    """Export detailed per-file report as CSV."""
    import csv

    path = os.path.join(output_dir, "detailed_report.csv")

    fieldnames = [
        "participant_code", "protocol_name", "source_file", "processing_status",
        "error_message", "integrated_lufs", "clipping_percent", "snr_db",
        "estimated_speakers", "speech_duration_s",
        "loudness_ok", "clipping_ok", "snr_ok", "speaker_count_ok", "duration_ok", "overall_pass",
        "detected_language", "language_confidence", "language_mismatch", "code_switching_detected",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for r in reports:
            ar = r.get("analysis_results", {})
            qa = r.get("quality_assessment", {})
            lang = ar.get("language") or {}
            cs = ar.get("code_switching") or {}
            level = ar.get("level") or {}
            noise = ar.get("noise") or {}
            speaker = ar.get("speaker") or {}

            row = {
                "participant_code": r.get("participant_code", ""),
                "protocol_name": r.get("protocol_name", ""),
                "source_file": r.get("source_file", ""),
                "processing_status": r.get("processing_status", ""),
                "error_message": r.get("error_message", ""),
                "integrated_lufs": level.get("integrated_lufs", ""),
                "clipping_percent": level.get("clipping_percent", ""),
                "snr_db": noise.get("snr_db", ""),
                "estimated_speakers": speaker.get("estimated_speakers", ""),
                "speech_duration_s": ar.get("speech_duration", ""),
                "loudness_ok": qa.get("loudness_ok", ""),
                "clipping_ok": qa.get("clipping_ok", ""),
                "snr_ok": qa.get("snr_ok", ""),
                "speaker_count_ok": qa.get("speaker_count_ok", ""),
                "duration_ok": qa.get("duration_ok", ""),
                "overall_pass": qa.get("overall_pass", ""),
                "detected_language": lang.get("detected_language", ""),
                "language_confidence": lang.get("confidence", ""),
                "language_mismatch": r.get("language_mismatch", ""),
                "code_switching_detected": cs.get("code_switching_detected", ""),
            }
            writer.writerow(row)

    logger.info(f"Detailed CSV saved: {path}")


if __name__ == "__main__":
    main()
