#!/usr/bin/env python3
"""Analyze a single audio file for quality."""
import argparse
import json
import os
import sys

from src.config import load_config
from src.pipeline import generate_quality_report, NumpyJSONEncoder


def main():
    parser = argparse.ArgumentParser(description="Analyze a single audio file for quality")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--skip-langid", action="store_true", help="Skip language detection")
    parser.add_argument("--skip-speaker-count", action="store_true", help="Skip speaker count")

    args = parser.parse_args()

    if not os.path.isfile(args.audio_file):
        print(f"Error: File not found: {args.audio_file}")
        sys.exit(1)

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Analyzing: {args.audio_file}")
    report = generate_quality_report(
        args.audio_file,
        config=config,
        skip_langid=args.skip_langid,
        skip_speaker_count=args.skip_speaker_count,
    )

    base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
    output_path = os.path.join(args.output_dir, f"{base_name}_report.json")

    with open(output_path, "w") as f:
        json.dump(report, f, indent=4, cls=NumpyJSONEncoder)

    print(f"Report saved: {output_path}")

    qa = report.get("quality_assessment", {})
    if report["processing_status"] == "SUCCESS":
        status = "PASS" if qa.get("overall_pass") else "FAIL"
        print(f"Result: {status}")
    else:
        print(f"Status: ERROR - {report.get('error_message')}")


if __name__ == "__main__":
    main()
