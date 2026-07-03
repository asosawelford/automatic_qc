#!/usr/bin/env python3
"""Plot histogram of recordings per month from a TELL takeout directory."""
import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_dates(takeout_path: str) -> list:
    """Extract (month, protocol) tuples from feature_extraction.json files."""
    records = []
    for root, dirs, files in os.walk(takeout_path):
        if "feature_extraction.json" in files:
            path = os.path.join(root, "feature_extraction.json")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                month = None
                # Try query_timestamp_start first
                timestamp = data.get("query_timestamp_start", "")
                if timestamp:
                    date_str = timestamp.split(" ")[0]  # YYYY-MM-DD
                    month = date_str[:7]  # YYYY-MM
                else:
                    # Fallback: parse from audio_file (CODE__YYYYMMDD__Task__hash.webm)
                    audio_file = data.get("audio_file", "")
                    parts = audio_file.split("__")
                    if len(parts) >= 2 and len(parts[1]) == 8 and parts[1].isdigit():
                        d = parts[1]
                        month = f"{d[:4]}-{d[4:6]}"

                if month:
                    protocol = data.get("protocol_name", "unknown")
                    records.append((month, protocol))
            except Exception:
                continue
    return records


def main():
    parser = argparse.ArgumentParser(description="Plot recordings per month")
    parser.add_argument("takeout_path", help="Path to takeout directory")
    parser.add_argument("--output", default="recordings_by_month.png", help="Output image path")
    args = parser.parse_args()

    if not os.path.isdir(args.takeout_path):
        print(f"Error: {args.takeout_path} not found")
        return

    print(f"Scanning {args.takeout_path}...")
    records = extract_dates(args.takeout_path)
    print(f"Found {len(records)} recordings with date info.")

    if not records:
        print("No dates found.")
        return

    # Overall counts
    month_counts = Counter(month for month, _ in records)
    months = sorted(month_counts.keys())
    values = [month_counts[m] for m in months]

    # Per-protocol counts
    protocols = sorted(set(p for _, p in records))
    protocol_month_counts = {p: Counter() for p in protocols}
    for month, protocol in records:
        protocol_month_counts[protocol][month] += 1

    # Print table — overall
    print(f"\n{'Month':<10} {'Total':>6}  " + "  ".join(f"{p[:20]:>20}" for p in protocols))
    print("-" * (18 + 22 * len(protocols)))
    max_val = max(values)
    for month in months:
        total = month_counts[month]
        per_proto = "  ".join(f"{protocol_month_counts[p][month]:>20}" for p in protocols)
        bar = "█" * int(total / max_val * 20)
        print(f"{month:<10} {total:>6}  {per_proto}  {bar}")
    print(f"\n{'Total':<10} {sum(values):>6}  " + "  ".join(f"{sum(protocol_month_counts[p].values()):>20}" for p in protocols))

    # Plot — stacked bar chart by protocol
    colors = ["#2b6cb0", "#e53e3e", "#38a169", "#d69e2e", "#9f7aea", "#ed8936", "#4299e1"]

    fig, ax = plt.subplots(figsize=(max(8, len(months) * 0.9), 6))
    x = range(len(months))
    bottoms = [0] * len(months)

    for i, protocol in enumerate(protocols):
        vals = [protocol_month_counts[protocol][m] for m in months]
        color = colors[i % len(colors)]
        ax.bar(x, vals, bottom=bottoms, label=protocol, color=color, edgecolor="white", linewidth=0.5)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Recordings")
    ax.set_title(f"Recordings per Month by Protocol (total: {sum(values)})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Total labels on top
    for i, total in enumerate(values):
        ax.text(i, bottoms[i] + max_val * 0.01, str(total), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nPlot saved: {args.output}")


if __name__ == "__main__":
    main()
