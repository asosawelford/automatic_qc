#!/usr/bin/env python3
"""
Extract all unique protocol_name values from TELL takeout feature_extraction.json files.
Use this to build the protocol_language_map in config.yaml.

Usage:
    python extract_protocol_names.py /path/to/takeout/
    python extract_protocol_names.py /path/to/takeout/ --output protocols.txt
"""

import os
import sys
import json
from collections import Counter
from pathlib import Path


def extract_protocol_names(takeout_path: str) -> Counter:
    """
    Walk through takeout directory and extract all protocol_name values.

    Returns:
        Counter object with protocol_name frequencies
    """
    protocol_names = Counter()

    for root, dirs, files in os.walk(takeout_path):
        if "feature_extraction.json" in files:
            json_path = os.path.join(root, "feature_extraction.json")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    protocol_name = data.get("protocol_name")
                    if protocol_name:
                        protocol_names[protocol_name] += 1
            except (json.JSONDecodeError, KeyError, IOError) as e:
                print(f"Warning: Could not read {json_path}: {e}", file=sys.stderr)

    return protocol_names


def generate_yaml_template(protocol_names: Counter) -> str:
    """
    Generate a YAML template for protocol_language_map.
    """
    lines = ["protocol_language_map:"]
    for protocol, count in sorted(protocol_names.items(), key=lambda x: -x[1]):
        # Default to es-PE (Spanish Peru), user should review and correct
        lines.append(f'  "{protocol}": "es-PE"  # Found {count} times - TODO: Verify language')

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    takeout_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(takeout_path):
        print(f"Error: Path does not exist: {takeout_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {takeout_path} for protocol names...", file=sys.stderr)

    protocol_names = extract_protocol_names(takeout_path)

    if not protocol_names:
        print("No protocol names found!", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(protocol_names)} unique protocol names:", file=sys.stderr)
    print(f"Total recordings: {sum(protocol_names.values())}", file=sys.stderr)
    print("\n" + "="*60, file=sys.stderr)

    yaml_template = generate_yaml_template(protocol_names)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(yaml_template)
        print(f"\nYAML template saved to: {output_file}", file=sys.stderr)
    else:
        print("\n" + yaml_template)

    print("\n" + "="*60, file=sys.stderr)
    print("\nNext steps:", file=sys.stderr)
    print("1. Review the protocol names above", file=sys.stderr)
    print("2. For each protocol, determine the expected language code", file=sys.stderr)
    print("3. Update config.yaml with the correct language mappings", file=sys.stderr)
    print("\nLanguage codes examples:", file=sys.stderr)
    print("  - Spanish (Peru): es-PE", file=sys.stderr)
    print("  - Spanish (Argentina): es-AR", file=sys.stderr)
    print("  - English (US): en-US", file=sys.stderr)
    print("  - Portuguese (Brazil): pt-BR", file=sys.stderr)


if __name__ == "__main__":
    main()
