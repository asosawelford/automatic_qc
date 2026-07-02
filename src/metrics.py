import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


def compute_aggregate_metrics(reports: List[Dict[str, Any]], site_codes: List[str]) -> Dict[str, Any]:
    """Compute aggregate quality metrics from a list of per-file reports."""
    total = len(reports)
    if total == 0:
        return {"total_files": 0}

    passed = sum(1 for r in reports if r.get("quality_assessment", {}).get("overall_pass", False))
    errors = sum(1 for r in reports if r.get("processing_status") == "ERROR")

    snr_values = []
    duration_values = []
    speaker_counts = []
    language_mismatches = 0
    code_switches = 0

    failure_reasons = defaultdict(int)
    site_pass = defaultdict(int)
    site_fail = defaultdict(int)
    protocol_pass = defaultdict(int)
    protocol_fail = defaultdict(int)
    participants = set()

    for r in reports:
        participant = r.get("participant_code", "Unknown")
        participants.add(participant)

        site = participant[:3] if len(participant) >= 3 else "UNK"
        protocol = r.get("protocol_name", "Unknown")

        qa = r.get("quality_assessment", {})
        ar = r.get("analysis_results", {})

        if r.get("processing_status") != "SUCCESS":
            continue

        # SNR
        noise = ar.get("noise")
        if noise and "snr_db" in noise:
            snr_val = noise["snr_db"]
            if snr_val not in (float("inf"), float("-inf")):
                snr_values.append(snr_val)

        # Duration
        duration = ar.get("speech_duration")
        if isinstance(duration, (int, float)) and duration > 0:
            duration_values.append(duration)

        # Speaker count
        speaker = ar.get("speaker")
        if speaker and "estimated_speakers" in speaker:
            speaker_counts.append(speaker["estimated_speakers"])

        # Language
        lang = ar.get("language")
        if lang and lang.get("detected_language"):
            # Check for mismatch (requires protocol_name in report)
            pass  # Mismatch is flagged per-report in pipeline

        lang_mismatch_flag = r.get("language_mismatch")
        if lang_mismatch_flag is True:
            language_mismatches += 1

        cs = ar.get("code_switching")
        if cs and cs.get("code_switching_detected"):
            code_switches += 1

        # Pass/fail by site and protocol
        if qa.get("overall_pass"):
            site_pass[site] += 1
            protocol_pass[protocol] += 1
        else:
            site_fail[site] += 1
            protocol_fail[protocol] += 1
            if not qa.get("loudness_ok", True):
                failure_reasons["loudness"] += 1
            if not qa.get("clipping_ok", True):
                failure_reasons["clipping"] += 1
            if not qa.get("snr_ok", True):
                failure_reasons["snr"] += 1
            if not qa.get("speaker_count_ok", True):
                failure_reasons["speaker_count"] += 1
            if not qa.get("duration_ok", True):
                failure_reasons["duration"] += 1

    successful = total - errors

    return {
        "total_files": total,
        "successful": successful,
        "errors": errors,
        "passed": passed,
        "failed": successful - passed,
        "pass_rate": passed / successful if successful > 0 else 0.0,
        "unique_participants": len(participants),
        "avg_snr_db": sum(snr_values) / len(snr_values) if snr_values else None,
        "avg_speech_duration_s": sum(duration_values) / len(duration_values) if duration_values else None,
        "language_mismatch_count": language_mismatches,
        "language_mismatch_rate": language_mismatches / successful if successful > 0 else 0.0,
        "code_switching_count": code_switches,
        "code_switching_rate": code_switches / successful if successful > 0 else 0.0,
        "multi_speaker_count": sum(1 for s in speaker_counts if s > 1),
        "multi_speaker_rate": sum(1 for s in speaker_counts if s > 1) / len(speaker_counts) if speaker_counts else 0.0,
        "failure_reasons": dict(failure_reasons),
        "by_site": {
            "pass": dict(site_pass),
            "fail": dict(site_fail),
        },
        "by_protocol": {
            "pass": dict(protocol_pass),
            "fail": dict(protocol_fail),
        },
    }


def compute_growth_metrics(
    current_metrics: Dict[str, Any],
    previous_checkpoint: Optional[Dict[str, Any]],
    current_processed_ids: List[str],
    current_participants: set,
) -> Dict[str, Any]:
    """Compute growth metrics comparing current run to previous."""
    if previous_checkpoint is None:
        return {
            "is_first_run": True,
            "new_recordings": current_metrics["total_files"],
            "new_participants": current_metrics["unique_participants"],
            "total_recordings": current_metrics["total_files"],
            "total_participants": current_metrics["unique_participants"],
            "growth_rate_percent": None,
        }

    prev_metrics = previous_checkpoint.get("aggregate_metrics", {})
    prev_ids = set(previous_checkpoint.get("processed_files", []))

    new_ids = set(current_processed_ids) - prev_ids
    total_ids = prev_ids | set(current_processed_ids)

    prev_total = prev_metrics.get("total_files", len(prev_ids))
    growth_rate = (len(new_ids) / prev_total * 100) if prev_total > 0 else None

    return {
        "is_first_run": False,
        "new_recordings": len(new_ids),
        "new_participants": None,  # Would need participant set from prev checkpoint
        "total_recordings": len(total_ids),
        "total_participants": current_metrics["unique_participants"],
        "growth_rate_percent": growth_rate,
    }


def compute_comparative_metrics(
    current_metrics: Dict[str, Any],
    previous_checkpoint: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Compare current metrics to previous run. Returns None if no previous run."""
    if previous_checkpoint is None:
        return None

    prev = previous_checkpoint.get("aggregate_metrics", {})

    def _delta(current_val, prev_key):
        prev_val = prev.get(prev_key)
        if current_val is None or prev_val is None:
            return {"current": current_val, "previous": prev_val, "delta": None}
        return {
            "current": current_val,
            "previous": prev_val,
            "delta": current_val - prev_val,
        }

    return {
        "avg_snr_db": _delta(current_metrics.get("avg_snr_db"), "avg_snr_db"),
        "pass_rate": _delta(current_metrics.get("pass_rate"), "pass_rate"),
        "language_mismatch_rate": _delta(current_metrics.get("language_mismatch_rate"), "language_mismatch_rate"),
        "code_switching_rate": _delta(current_metrics.get("code_switching_rate"), "code_switching_rate"),
        "avg_speech_duration_s": _delta(current_metrics.get("avg_speech_duration_s"), "avg_speech_duration_s"),
        "multi_speaker_rate": _delta(current_metrics.get("multi_speaker_rate"), "multi_speaker_rate"),
    }
