"""Static plot generation for the QC monitoring pipeline.

Generates multi-panel matplotlib figures summarizing quality metrics,
saved as PNG and/or PDF.
"""

import os
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_plots(result: Dict[str, Any], output_dir: str, config) -> None:
    """Generate static quality report plots.

    Creates:
      - quality_report.png/pdf: 4-panel overview figure
      - comparison.png/pdf: comparative metrics (if available)

    Args:
        result: Pipeline result dict with aggregate_metrics, growth_metrics, etc.
        output_dir: Directory to write output files.
        config: Config instance with outputs.plot_dpi, outputs.plot_style, etc.
    """
    os.makedirs(output_dir, exist_ok=True)

    dpi = config.outputs.plot_dpi
    style = config.outputs.plot_style
    font_scale = config.outputs.plot_font_scale

    try:
        plt.style.use(style)
    except OSError:
        pass

    plt.rcParams.update({
        "font.size": 10 * font_scale,
        "axes.titlesize": 12 * font_scale,
        "axes.labelsize": 10 * font_scale,
        "xtick.labelsize": 8 * font_scale,
        "ytick.labelsize": 8 * font_scale,
    })

    agg = result.get("aggregate_metrics", {})

    _generate_overview_figure(agg, result, output_dir, dpi, config)

    comparative = result.get("comparative_metrics")
    if comparative:
        _generate_comparison_figure(comparative, output_dir, dpi, config)


def _generate_overview_figure(
    agg: Dict[str, Any],
    result: Dict[str, Any],
    output_dir: str,
    dpi: int,
    config,
) -> None:
    """Generate the 4-panel overview figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Audio Quality Report - {result.get('run_date', 'N/A')}",
        fontsize=14 * config.outputs.plot_font_scale,
        fontweight="bold",
    )

    _plot_pass_fail_by_site(axes[0, 0], agg, config)
    _plot_failure_reasons(axes[0, 1], agg)
    _plot_quality_summary(axes[1, 0], agg, result, config)
    _plot_protocol_breakdown(axes[1, 1], agg)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if config.outputs.generate_png_plots:
        fig.savefig(
            os.path.join(output_dir, "quality_report.png"),
            dpi=dpi,
            bbox_inches="tight",
        )
    if config.outputs.generate_pdf_plots:
        fig.savefig(
            os.path.join(output_dir, "quality_report.pdf"),
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def _plot_pass_fail_by_site(ax, agg: Dict[str, Any], config) -> None:
    """Panel 1: Pass/Fail stacked bar chart by site."""
    by_site = agg.get("by_site", {})
    pass_data = by_site.get("pass", {})
    fail_data = by_site.get("fail", {})

    sites = sorted(set(list(pass_data.keys()) + list(fail_data.keys())))
    if not sites:
        sites = config.site_codes

    pass_counts = [pass_data.get(s, 0) for s in sites]
    fail_counts = [fail_data.get(s, 0) for s in sites]

    if not sites:
        ax.text(0.5, 0.5, "No site data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("Pass/Fail by Site")
        return

    x = np.arange(len(sites))
    width = 0.6

    ax.bar(x, pass_counts, width, label="Pass", color="#4CAF50")
    ax.bar(x, fail_counts, width, bottom=pass_counts, label="Fail", color="#F44336")

    ax.set_xlabel("Site")
    ax.set_ylabel("Count")
    ax.set_title("Pass/Fail by Site")
    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.legend()

    for i, (p, f) in enumerate(zip(pass_counts, fail_counts)):
        total = p + f
        if total > 0:
            ax.text(i, total + 0.5, str(total), ha="center", va="bottom", fontsize=8)


def _plot_failure_reasons(ax, agg: Dict[str, Any]) -> None:
    """Panel 2: Failure reasons horizontal bar chart."""
    reasons = agg.get("failure_reasons", {})

    if not reasons:
        ax.text(0.5, 0.5, "No failures recorded", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("Failure Reasons")
        return

    sorted_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
    labels = [r[0].replace("_", " ").title() for r in sorted_reasons]
    values = [r[1] for r in sorted_reasons]

    colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(labels)))

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Count")
    ax.set_title("Failure Reasons")
    ax.invert_yaxis()

    for i, v in enumerate(values):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=8)


def _plot_quality_summary(ax, agg: Dict[str, Any], result: Dict[str, Any], config) -> None:
    """Panel 3: Quality metrics summary as a text panel."""
    ax.axis("off")
    ax.set_title("Quality Metrics Summary")

    total = agg.get("total_files", 0)
    successful = agg.get("successful", 0)
    errors = agg.get("errors", 0)
    passed = agg.get("passed", 0)
    failed = agg.get("failed", 0)
    pass_rate = agg.get("pass_rate", 0.0)
    avg_snr = agg.get("avg_snr_db")
    avg_duration = agg.get("avg_speech_duration_s")
    unique_parts = agg.get("unique_participants", 0)
    lang_mismatch = agg.get("language_mismatch_count", 0)
    lang_rate = agg.get("language_mismatch_rate", 0.0)
    code_switch = agg.get("code_switching_count", 0)
    multi_spk = agg.get("multi_speaker_count", 0)

    growth = result.get("growth_metrics", {})
    new_recs = growth.get("new_recordings", 0) if growth else 0
    growth_rate = growth.get("growth_rate_percent") if growth else None

    lines = [
        f"Total files processed: {total}",
        f"Successful: {successful}  |  Errors: {errors}",
        f"Passed: {passed}  |  Failed: {failed}",
        f"Pass rate: {pass_rate:.1%}",
        "",
        f"Unique participants: {unique_parts}",
        f"Avg SNR: {avg_snr:.1f} dB" if avg_snr is not None else "Avg SNR: N/A",
        f"Avg speech duration: {avg_duration:.1f} s" if avg_duration is not None else "Avg speech duration: N/A",
        "",
        f"Language mismatches: {lang_mismatch} ({lang_rate:.1%})",
        f"Code switching: {code_switch}",
        f"Multi-speaker: {multi_spk}",
    ]

    if new_recs > 0 or growth_rate is not None:
        lines.append("")
        lines.append(f"New recordings: {new_recs}")
        if growth_rate is not None:
            lines.append(f"Growth rate: {growth_rate:.1f}%")

    text = "\n".join(lines)
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes,
        fontsize=9 * config.outputs.plot_font_scale if hasattr(config, "outputs") else 9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8),
    )


def _plot_protocol_breakdown(ax, agg: Dict[str, Any]) -> None:
    """Panel 4: Top 10 protocols by total count."""
    by_protocol = agg.get("by_protocol", {})
    pass_data = by_protocol.get("pass", {})
    fail_data = by_protocol.get("fail", {})

    all_protocols = sorted(
        set(list(pass_data.keys()) + list(fail_data.keys())),
        key=lambda p: pass_data.get(p, 0) + fail_data.get(p, 0),
        reverse=True,
    )

    if not all_protocols:
        ax.text(0.5, 0.5, "No protocol data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("Top Protocols")
        return

    top_protocols = all_protocols[:10]
    pass_counts = [pass_data.get(p, 0) for p in top_protocols]
    fail_counts = [fail_data.get(p, 0) for p in top_protocols]

    # Truncate long protocol names
    display_names = [p[:20] + "..." if len(p) > 20 else p for p in top_protocols]

    y_pos = np.arange(len(top_protocols))
    bar_height = 0.6

    ax.barh(y_pos, pass_counts, bar_height, label="Pass", color="#4CAF50")
    ax.barh(y_pos, fail_counts, bar_height, left=pass_counts, label="Fail", color="#F44336")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names)
    ax.set_xlabel("Count")
    ax.set_title("Top 10 Protocols")
    ax.legend(loc="lower right")
    ax.invert_yaxis()


def _generate_comparison_figure(
    comparative: Dict[str, Any],
    output_dir: str,
    dpi: int,
    config,
) -> None:
    """Generate comparison figure showing current vs previous metrics."""
    metrics_to_plot = {}
    for key, val in comparative.items():
        if isinstance(val, dict) and "current" in val and "previous" in val:
            current = val.get("current")
            previous = val.get("previous")
            if current is not None and previous is not None:
                metrics_to_plot[key] = val

    if not metrics_to_plot:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "Comparison: Current vs Previous Run",
        fontsize=14 * config.outputs.plot_font_scale,
        fontweight="bold",
    )

    labels = []
    current_vals = []
    previous_vals = []

    for key, val in metrics_to_plot.items():
        label = key.replace("_", " ").title()
        labels.append(label)
        current_vals.append(val["current"])
        previous_vals.append(val["previous"])

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, previous_vals, width, label="Previous", color="#90CAF9")
    bars2 = ax.bar(x + width / 2, current_vals, width, label="Current", color="#1565C0")

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()

    # Add delta annotations
    for i, key in enumerate(metrics_to_plot):
        delta = metrics_to_plot[key].get("delta")
        if delta is not None:
            color = "#4CAF50" if delta >= 0 else "#F44336"
            sign = "+" if delta >= 0 else ""
            ax.annotate(
                f"{sign}{delta:.2f}",
                xy=(x[i] + width / 2, current_vals[i]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=color,
                fontweight="bold",
            )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if config.outputs.generate_png_plots:
        fig.savefig(
            os.path.join(output_dir, "comparison.png"),
            dpi=dpi,
            bbox_inches="tight",
        )
    if config.outputs.generate_pdf_plots:
        fig.savefig(
            os.path.join(output_dir, "comparison.pdf"),
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)
