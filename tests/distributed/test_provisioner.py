import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miss_alignment.distributed.config import ClusterConfig
from miss_alignment.distributed.provisioner import ClusterProvisioner, LocalProvisioner


def test_local_provisioner_spawns_one_process_per_device(tmp_path):
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0, 1, 2])
        p.ensure_workers(n_workers=10)

        assert mock_popen.call_count == 3
        # verify device args appear in calls
        all_args = [str(c) for c in mock_popen.call_args_list]
        assert any("0" in s for s in all_args)


def test_local_provisioner_does_not_respawn_running(tmp_path):
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0])
        p.ensure_workers(n_workers=5)
        p.ensure_workers(n_workers=5)

        assert mock_popen.call_count == 1


def test_local_provisioner_respawns_dead_worker(tmp_path):
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # exited
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0])
        p.ensure_workers(n_workers=5)
        p.ensure_workers(n_workers=5)  # should respawn because poll() != None

        assert mock_popen.call_count == 2


def test_local_provisioner_shutdown_terminates(tmp_path):
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0])
        p.ensure_workers(n_workers=5)
        p.shutdown()

        mock_proc.terminate.assert_called()


def test_cluster_provisioner_submits_n_workers_jobs(tmp_path):
    script = tmp_path / "worker.sh"
    script.write_text("#!/bin/bash\n{{command}}\n")
    cfg = ClusterConfig(
        submit="sbatch {{script_path}}",
        submit_job_id_regex=r"Submitted batch job (\d+)",
        cancel="scancel {{job_id}}",
        script_path=script,
    )

    submitted = []

    def fake_run(cmd, **kwargs):
        submitted.append(cmd)
        result = MagicMock()
        result.stdout = "Submitted batch job 12345\n"
        return result

    with patch(
        "miss_alignment.distributed.provisioner.subprocess.run", side_effect=fake_run
    ):
        p = ClusterProvisioner(queue_dir=tmp_path, config=cfg)
        p.ensure_workers(n_workers=4)

    assert len(submitted) == 4


def test_cluster_provisioner_cancels_on_shutdown(tmp_path):
    script = tmp_path / "worker.sh"
    script.write_text("#!/bin/bash\n{{command}}\n")
    cfg = ClusterConfig(
        submit="sbatch {{script_path}}",
        submit_job_id_regex=r"Submitted batch job (\d+)",
        cancel="scancel {{job_id}}",
        script_path=script,
    )

    cancel_calls = []

    def fake_run(cmd, **kwargs):
        if "sbatch" in cmd:
            result = MagicMock()
            result.stdout = "Submitted batch job 99999\n"
            return result
        cancel_calls.append(cmd)
        return MagicMock()

    with patch(
        "miss_alignment.distributed.provisioner.subprocess.run", side_effect=fake_run
    ):
        p = ClusterProvisioner(queue_dir=tmp_path, config=cfg)
        p.ensure_workers(n_workers=3)
        p.shutdown()

    assert len(cancel_calls) == 3
    assert all("scancel" in c for c in cancel_calls)
