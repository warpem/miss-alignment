"""Tests for the distributed worker provisioner (replaces _parallel.py tests)."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miss_alignment.distributed.provisioner import LocalProvisioner


def test_local_provisioner_spawns_worker_per_device(tmp_path):
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0, 1, 2])
        p.ensure_workers(n_workers=10)

        assert mock_popen.call_count == 3


def test_local_provisioner_does_not_double_spawn(tmp_path):
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0])
        p.ensure_workers(n_workers=5)
        p.ensure_workers(n_workers=5)

        assert mock_popen.call_count == 1


def test_local_provisioner_shutdown_terminates(tmp_path):
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0])
        p.ensure_workers(n_workers=5)
        p.shutdown()

        mock_proc.terminate.assert_called()
