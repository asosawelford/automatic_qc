"""HTML dashboard generation for the QC monitoring pipeline.

Generates a self-contained single-page HTML dashboard with embedded CSS
and SVG/CSS-based charts. No external dependencies required at runtime.
"""

import html
import json
import os
from typing import Any, Dict, List, Optional


def generate_dashboard(result: Dict[str, Any], output_dir: str, config) -> None:
    """Generate a self-contained HTML dashboard.

    Creates dashboard.html in output_dir with inline CSS/JS and CSS-based
    bar charts. Opens cleanly in any modern browser with no external
    dependencies.

    Args:
        result: Pipeline result dict with aggregate_metrics, growth_metrics, etc.
        output_dir: Directory to write dashboard.html.
        config: Config instance (used for site_codes reference).
    """
    os.makedirs(output_dir, exist_ok=True)

    agg = result.get("aggregate_metrics", {})
    growth = result.get("growth_metrics", {})
    comparative = result.get("comparative_metrics")
    run_date = result.get("run_date", "N/A")

    sections = [
        _build_overview_section(agg, growth, run_date),
        _build_site_section(agg, config),
        _build_failure_section(agg),
        _build_language_section(agg),
        _build_growth_section(growth),
    ]

    if comparative:
        sections.append(_build_comparison_section(comparative))

    body_content = "\n".join(sections)
    html_content = _wrap_html(body_content, run_date)

    output_path = os.path.join(output_dir, "dashboard.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def _wrap_html(body_content: str, run_date: str) -> str:
    """Wrap body content in full HTML document with embedded styles."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Quality Dashboard - {_esc(run_date)}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #f5f7fa;
            color: #2d3748;
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        header h1 {{
            font-size: 1.75rem;
            margin-bottom: 0.25rem;
        }}
        header .subtitle {{
            opacity: 0.85;
            font-size: 0.95rem;
        }}
        .section {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
        .section h2 {{
            font-size: 1.25rem;
            color: #1a365d;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        .metric-card {{
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }}
        .metric-card .value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: #2b6cb0;
        }}
        .metric-card .label {{
            font-size: 0.8rem;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.25rem;
        }}
        .metric-card.success .value {{ color: #38a169; }}
        .metric-card.danger .value {{ color: #e53e3e; }}
        .metric-card.warning .value {{ color: #d69e2e; }}

        .bar-chart {{
            margin: 1rem 0;
        }}
        .bar-row {{
            display: flex;
            align-items: center;
            margin-bottom: 0.6rem;
        }}
        .bar-label {{
            width: 140px;
            font-size: 0.85rem;
            color: #4a5568;
            flex-shrink: 0;
            text-align: right;
            padding-right: 0.75rem;
        }}
        .bar-track {{
            flex: 1;
            height: 24px;
            background: #edf2f7;
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }}
        .bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            padding-left: 8px;
        }}
        .bar-fill .bar-value {{
            font-size: 0.75rem;
            color: white;
            font-weight: 600;
            white-space: nowrap;
        }}
        .bar-value-outside {{
            font-size: 0.75rem;
            color: #4a5568;
            margin-left: 0.5rem;
            flex-shrink: 0;
        }}

        .stacked-bar-chart {{
            margin: 1rem 0;
        }}
        .stacked-row {{
            display: flex;
            align-items: center;
            margin-bottom: 0.6rem;
        }}
        .stacked-label {{
            width: 80px;
            font-size: 0.85rem;
            color: #4a5568;
            flex-shrink: 0;
            text-align: right;
            padding-right: 0.75rem;
            font-weight: 600;
        }}
        .stacked-track {{
            flex: 1;
            height: 28px;
            background: #edf2f7;
            border-radius: 4px;
            overflow: hidden;
            display: flex;
        }}
        .stacked-segment {{
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            color: white;
            font-weight: 600;
            min-width: 0;
            overflow: hidden;
        }}
        .stacked-segment.pass {{ background: #48bb78; }}
        .stacked-segment.fail {{ background: #fc8181; }}
        .stacked-total {{
            font-size: 0.8rem;
            color: #4a5568;
            margin-left: 0.5rem;
            flex-shrink: 0;
            width: 40px;
        }}

        .legend {{
            display: flex;
            gap: 1.5rem;
            margin-bottom: 0.75rem;
            font-size: 0.85rem;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }}
        .legend-dot.pass {{ background: #48bb78; }}
        .legend-dot.fail {{ background: #fc8181; }}

        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
        }}
        .comparison-card {{
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
        }}
        .comparison-card .comp-title {{
            font-size: 0.85rem;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}
        .comparison-card .comp-values {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
        }}
        .comparison-card .comp-current {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #2b6cb0;
        }}
        .comparison-card .comp-previous {{
            font-size: 0.9rem;
            color: #a0aec0;
        }}
        .comparison-card .comp-delta {{
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 0.25rem;
        }}
        .comp-delta.positive {{ color: #38a169; }}
        .comp-delta.negative {{ color: #e53e3e; }}
        .comp-delta.neutral {{ color: #718096; }}

        .empty-state {{
            text-align: center;
            color: #a0aec0;
            padding: 2rem;
            font-style: italic;
        }}

        @media (max-width: 768px) {{
            body {{ padding: 1rem; }}
            .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .bar-label {{ width: 100px; font-size: 0.75rem; }}
            .stacked-label {{ width: 60px; font-size: 0.75rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Audio Quality Dashboard</h1>
            <div class="subtitle">Monitoring report generated on {_esc(run_date)}</div>
        </header>
        {body_content}
    </div>
</body>
</html>"""


def _build_overview_section(agg: Dict[str, Any], growth: Dict[str, Any], run_date: str) -> str:
    """Build the Overview section with key metrics."""
    total = agg.get("total_files", 0)
    successful = agg.get("successful", 0)
    errors = agg.get("errors", 0)
    passed = agg.get("passed", 0)
    failed = agg.get("failed", 0)
    pass_rate = agg.get("pass_rate", 0.0)
    unique_parts = agg.get("unique_participants", 0)
    avg_snr = agg.get("avg_snr_db")
    avg_duration = agg.get("avg_speech_duration_s")

    pass_rate_class = "success" if pass_rate >= 0.8 else ("warning" if pass_rate >= 0.6 else "danger")
    snr_str = f"{avg_snr:.1f} dB" if avg_snr is not None else "N/A"
    duration_str = f"{avg_duration:.1f} s" if avg_duration is not None else "N/A"

    cards = [
        _metric_card(str(total), "Total Files", ""),
        _metric_card(str(passed), "Passed", "success"),
        _metric_card(str(failed), "Failed", "danger"),
        _metric_card(f"{pass_rate:.0%}", "Pass Rate", pass_rate_class),
        _metric_card(str(unique_parts), "Participants", ""),
        _metric_card(snr_str, "Avg SNR", ""),
        _metric_card(duration_str, "Avg Duration", ""),
        _metric_card(str(errors), "Errors", "danger" if errors > 0 else ""),
    ]

    return f"""<div class="section">
    <h2>Overview</h2>
    <div class="metrics-grid">
        {"".join(cards)}
    </div>
</div>"""


def _build_site_section(agg: Dict[str, Any], config) -> str:
    """Build the Quality by Site section with stacked bars."""
    by_site = agg.get("by_site", {})
    pass_data = by_site.get("pass", {})
    fail_data = by_site.get("fail", {})

    sites = sorted(set(list(pass_data.keys()) + list(fail_data.keys())))
    if not sites:
        sites = config.site_codes

    if not sites or (not pass_data and not fail_data):
        return """<div class="section">
    <h2>Quality by Site</h2>
    <div class="empty-state">No site data available</div>
</div>"""

    max_total = max(
        (pass_data.get(s, 0) + fail_data.get(s, 0) for s in sites),
        default=1,
    )
    if max_total == 0:
        max_total = 1

    rows = []
    for site in sites:
        p = pass_data.get(site, 0)
        f = fail_data.get(site, 0)
        total = p + f
        pass_pct = (p / max_total * 100) if max_total > 0 else 0
        fail_pct = (f / max_total * 100) if max_total > 0 else 0

        pass_segment = f'<div class="stacked-segment pass" style="width:{pass_pct}%">{p if p > 0 else ""}</div>' if p > 0 else ""
        fail_segment = f'<div class="stacked-segment fail" style="width:{fail_pct}%">{f if f > 0 else ""}</div>' if f > 0 else ""

        rows.append(f"""<div class="stacked-row">
            <div class="stacked-label">{_esc(site)}</div>
            <div class="stacked-track">{pass_segment}{fail_segment}</div>
            <div class="stacked-total">{total}</div>
        </div>""")

    return f"""<div class="section">
    <h2>Quality by Site</h2>
    <div class="legend">
        <div class="legend-item"><div class="legend-dot pass"></div>Pass</div>
        <div class="legend-item"><div class="legend-dot fail"></div>Fail</div>
    </div>
    <div class="stacked-bar-chart">
        {"".join(rows)}
    </div>
</div>"""


def _build_failure_section(agg: Dict[str, Any]) -> str:
    """Build the Failure Analysis section with horizontal bar chart."""
    reasons = agg.get("failure_reasons", {})

    if not reasons:
        return """<div class="section">
    <h2>Failure Analysis</h2>
    <div class="empty-state">No failures recorded</div>
</div>"""

    sorted_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
    max_val = max(v for _, v in sorted_reasons) if sorted_reasons else 1
    if max_val == 0:
        max_val = 1

    colors = ["#e53e3e", "#ed8936", "#ecc94b", "#9f7aea", "#4299e1", "#38b2ac", "#ed64a6"]

    rows = []
    for i, (reason, count) in enumerate(sorted_reasons):
        pct = count / max_val * 100
        color = colors[i % len(colors)]
        label = reason.replace("_", " ").title()

        rows.append(f"""<div class="bar-row">
            <div class="bar-label">{_esc(label)}</div>
            <div class="bar-track">
                <div class="bar-fill" style="width:{pct}%; background:{color};">
                    <span class="bar-value">{count}</span>
                </div>
            </div>
        </div>""")

    return f"""<div class="section">
    <h2>Failure Analysis</h2>
    <div class="bar-chart">
        {"".join(rows)}
    </div>
</div>"""


def _build_language_section(agg: Dict[str, Any]) -> str:
    """Build the Language Analysis section."""
    lang_mismatch = agg.get("language_mismatch_count", 0)
    lang_rate = agg.get("language_mismatch_rate", 0.0)
    code_switch = agg.get("code_switching_count", 0)
    code_rate = agg.get("code_switching_rate", 0.0)
    multi_spk = agg.get("multi_speaker_count", 0)
    multi_rate = agg.get("multi_speaker_rate", 0.0)

    mismatch_class = "danger" if lang_rate > 0.1 else ("warning" if lang_rate > 0.05 else "success")
    code_class = "warning" if code_rate > 0.05 else "success"
    multi_class = "danger" if multi_rate > 0.1 else ("warning" if multi_rate > 0.05 else "success")

    cards = [
        _metric_card(str(lang_mismatch), "Language Mismatches", mismatch_class),
        _metric_card(f"{lang_rate:.1%}", "Mismatch Rate", mismatch_class),
        _metric_card(str(code_switch), "Code Switching", code_class),
        _metric_card(f"{code_rate:.1%}", "Switch Rate", code_class),
        _metric_card(str(multi_spk), "Multi-Speaker", multi_class),
        _metric_card(f"{multi_rate:.1%}", "Multi-Spk Rate", multi_class),
    ]

    return f"""<div class="section">
    <h2>Language Analysis</h2>
    <div class="metrics-grid">
        {"".join(cards)}
    </div>
</div>"""


def _build_growth_section(growth: Dict[str, Any]) -> str:
    """Build the Growth Metrics section."""
    if not growth:
        return """<div class="section">
    <h2>Growth Metrics</h2>
    <div class="empty-state">No growth data available</div>
</div>"""

    is_first = growth.get("is_first_run", True)
    new_recs = growth.get("new_recordings", 0)
    total_recs = growth.get("total_recordings", 0)
    growth_rate = growth.get("growth_rate_percent")

    if is_first:
        growth_str = "First run"
        growth_class = ""
    elif growth_rate is not None:
        growth_str = f"{growth_rate:+.1f}%"
        growth_class = "success" if growth_rate > 0 else ("danger" if growth_rate < 0 else "")
    else:
        growth_str = "N/A"
        growth_class = ""

    cards = [
        _metric_card(str(total_recs), "Total Recordings", ""),
        _metric_card(str(new_recs), "New Recordings", "success" if new_recs > 0 else ""),
        _metric_card(growth_str, "Growth Rate", growth_class),
    ]

    note = ""
    if is_first:
        note = '<p style="color:#718096; font-size:0.85rem; margin-top:0.75rem;">This is the first monitoring run. Comparative metrics will be available after the next run.</p>'

    return f"""<div class="section">
    <h2>Growth Metrics</h2>
    <div class="metrics-grid">
        {"".join(cards)}
    </div>
    {note}
</div>"""


def _build_comparison_section(comparative: Dict[str, Any]) -> str:
    """Build the Comparison section showing current vs previous."""
    if not comparative:
        return ""

    cards = []
    for key, val in comparative.items():
        if not isinstance(val, dict):
            continue
        current = val.get("current")
        previous = val.get("previous")
        delta = val.get("delta")

        if current is None:
            continue

        label = key.replace("_", " ").title()
        current_str = _format_metric_value(key, current)
        previous_str = _format_metric_value(key, previous) if previous is not None else "N/A"

        delta_html = ""
        if delta is not None:
            if delta > 0:
                delta_class = "positive"
                delta_str = f"+{_format_metric_value(key, delta)}"
            elif delta < 0:
                delta_class = "negative"
                delta_str = _format_metric_value(key, delta)
            else:
                delta_class = "neutral"
                delta_str = "No change"
            delta_html = f'<div class="comp-delta {delta_class}">{_esc(delta_str)}</div>'

        cards.append(f"""<div class="comparison-card">
            <div class="comp-title">{_esc(label)}</div>
            <div class="comp-values">
                <span class="comp-current">{_esc(current_str)}</span>
                <span class="comp-previous">prev: {_esc(previous_str)}</span>
            </div>
            {delta_html}
        </div>""")

    if not cards:
        return ""

    return f"""<div class="section">
    <h2>Comparison with Previous Run</h2>
    <div class="comparison-grid">
        {"".join(cards)}
    </div>
</div>"""


def _metric_card(value: str, label: str, css_class: str = "") -> str:
    """Build a single metric card HTML snippet."""
    class_attr = f' {css_class}' if css_class else ""
    return f"""<div class="metric-card{class_attr}">
        <div class="value">{_esc(value)}</div>
        <div class="label">{_esc(label)}</div>
    </div>"""


def _format_metric_value(key: str, value) -> str:
    """Format a metric value for display based on its key name."""
    if value is None:
        return "N/A"
    if "rate" in key and isinstance(value, float):
        return f"{value:.1%}"
    if "percent" in key and isinstance(value, float):
        return f"{value:.1f}%"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html.escape(str(text))
