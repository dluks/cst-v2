"""Utility functions for managing training run IDs."""

import datetime
from pathlib import Path


def now() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_run_id() -> str:
    """Generate a new run ID based on current timestamp."""
    return f"run_{now()}"


def get_latest_run_id(base_dir: Path) -> str | None:
    """
    Find the most recent run ID in a directory.

    Run IDs are formatted as 'run_YYYYMMDD_HHMMSS' which sorts lexicographically
    in chronological order.

    Args:
        base_dir: Directory containing run subdirectories

    Returns:
        The most recent run ID, or None if no runs exist
    """
    if not base_dir.exists():
        return None

    run_dirs = sorted(
        [
            d.name
            for d in base_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ],
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None
