#!/usr/bin/env python3
"""Batch analyze audio files without checkpoint tracking."""
import argparse
import json
import os
import sys

from src.config import load_config
from src.pipeline import generate_quality_report, NumpyJSONEncoder


def main():
    parser = argparse.ArgumentParser(description="Batch analyze audio files (no state tracking)")
    parser.add_argument("audio_directory", help="Path to directory with audio files")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--skip-langid", action="store_true", help="Skip language detection")
    parser.add_argument("--skip-speaker-count", action="store_true", help="Skip speaker count")

    args = parser.parse_args()

    if not os.path.isdir(args.audio_directory):
        print(f"Error: Directory not found: {args.audio_directory}")
        sys.exit(1)

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    audio_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm')
    files = [
        os.path.join(args.audio_directory, f)
        for f in os.listdir(args.audio_directory)
        if f.lower().endswith(audio_extensions)
    ]

    print(f"Found {len(files)} audio file(s) to process.")

    passed = 0
    failed = 0
    errors = 0

    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {os.path.basename(file_path)}...", end=" ")

        report = generate_quality_report(
            file_path,
            config=config,
            skip_langid=args.skip_langid,
            skip_speaker_count=args.skip_speaker_count,
        )

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(args.output_dir, f"{base_name}_report.json")

        with open(output_path, "w") as f:
            json.dump(report, f, indent=4, cls=NumpyJSONEncoder)

        if report["processing_status"] == "SUCCESS":
            if report["quality_assessment"].get("overall_pass"):
                passed += 1
                print("PASS")
            else:
                failed += 1
                print("FAIL")
        else:
            errors += 1
            print(f"ERROR: {report.get('error_message', 'unknown')}")

    print(f"\nDone. Passed: {passed}, Failed: {failed}, Errors: {errors}")
    print(f"Reports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
