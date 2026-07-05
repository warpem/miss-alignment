"""Read cluster configuration from environment variables.

load_cluster_config() is called only when --n-cluster-workers is set.
It raises RuntimeError immediately if either required env var is absent,
so the user gets a clear error rather than a silent fallback to local mode.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClusterConfig:
    submit: str
    submit_job_id_regex: str
    cancel: str
    script_path: Path


def load_cluster_config() -> ClusterConfig:
    """Return ClusterConfig. Raises RuntimeError if env vars are missing."""
    config_path_str = os.environ.get("MISS_CLUSTER_CONFIG")
    if not config_path_str:
        raise RuntimeError(
            "MISS_CLUSTER_CONFIG environment variable is required when "
            "--n-cluster-workers is set. Point it to a JSON file with "
            "'submit', 'submit_job_id_regex', and 'cancel' keys."
        )

    script_path_str = os.environ.get("MISS_CLUSTER_SCRIPT")
    if not script_path_str:
        raise RuntimeError(
            "MISS_CLUSTER_SCRIPT environment variable is required when "
            "--n-cluster-workers is set. Point it to a shell script template "
            "containing a {{command}} placeholder."
        )

    config_path = Path(config_path_str)
    script_path = Path(script_path_str)

    if not config_path.exists():
        raise FileNotFoundError(f"MISS_CLUSTER_CONFIG not found: {config_path}")
    if not script_path.exists():
        raise FileNotFoundError(f"MISS_CLUSTER_SCRIPT not found: {script_path}")

    data = json.loads(config_path.read_text())
    return ClusterConfig(
        submit=data["submit"],
        submit_job_id_regex=data["submit_job_id_regex"],
        cancel=data["cancel"],
        script_path=script_path,
    )
