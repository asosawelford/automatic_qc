import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Set

logger = logging.getLogger(__name__)


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
    """Load the most recent checkpoint file from the directory."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".json")],
        reverse=True,
    )

    if not checkpoints:
        return None

    path = os.path.join(checkpoint_dir, checkpoints[0])
    with open(path, "r") as f:
        return json.load(f)


def get_processed_ids(checkpoint: Optional[Dict[str, Any]]) -> Set[str]:
    """Extract the set of processed file identifiers from a checkpoint."""
    if checkpoint is None:
        return set()
    return set(checkpoint.get("processed_files", []))


def save_checkpoint(
    checkpoint_dir: str,
    processed_files: list,
    aggregate_metrics: Dict[str, Any],
    run_date: Optional[str] = None,
) -> str:
    """Save a new checkpoint file. Returns the path written."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    if run_date is None:
        run_date = datetime.now().strftime("%Y-%m-%d")

    # Handle multiple runs per day
    filename = f"checkpoint_{run_date}.json"
    path = os.path.join(checkpoint_dir, filename)
    counter = 1
    while os.path.exists(path):
        filename = f"checkpoint_{run_date}_{counter}.json"
        path = os.path.join(checkpoint_dir, filename)
        counter += 1

    checkpoint = {
        "run_date": run_date,
        "run_timestamp": datetime.now().isoformat(),
        "processed_files": processed_files,
        "total_processed": len(processed_files),
        "aggregate_metrics": aggregate_metrics,
    }

    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)

    logger.info(f"Checkpoint saved: {path}")
    return path


def compute_file_id(participant_code: str, audio_path: str) -> str:
    """Generate a unique identifier for a processed audio file."""
    import hashlib
    path_hash = hashlib.md5(audio_path.encode()).hexdigest()[:8]
    return f"{participant_code}_{path_hash}"
