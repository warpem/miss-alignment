import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miss_alignment.distributed.config import ClusterConfig
from miss_alignment.distributed.provisioner import (
    ClusterProvisioner,
    CompositeProvisioner,
    LocalProvisioner,
    _parse_status_output,
)


def _cluster_cfg(tmp_path) -> ClusterConfig:
    script = tmp_path / "worker.sh"
    script.write_text("#!/bin/bash\n{{command}}\n")
    return ClusterConfig(
        submit="sbatch {{script_path}}",
        submit_job_id_regex=r"Submitted batch job (\d+)",
        cancel="scancel {{job_id}}",
        status_list="squeue -u $USER -h -o '%i,%T'",
        script_path=script,
    )


# ---------------------------------------------------------------------------
# Status parser unit tests
# ---------------------------------------------------------------------------

def test_parse_status_output_slurm_alive():
    output = "12345,PENDING\n12346,RUNNING\n12347,COMPLETING\n"
    our_ids = {"12345", "12346", "12347", "99999"}
    alive = _parse_status_output(output, our_ids, "slurm", [], "")
    assert alive == {"12345", "12346", "12347"}


def test_parse_status_output_slurm_terminal():
    output = "12345,COMPLETED\n12346,FAILED\n12347,CANCELLED\n"
    our_ids = {"12345", "12346", "12347"}
    alive = _parse_status_output(output, our_ids, "slurm", [], "")
    assert alive == set()


def test_parse_status_output_lsf_alive():
    output = "12345,PEND\n12346,RUN\n"
    our_ids = {"12345", "12346"}
    alive = _parse_status_output(output, our_ids, "lsf", [], "")
    assert alive == {"12345", "12346"}


def test_parse_status_output_pbs_alive():
    output = "12345,Q\n12346,R\n12347,C\n"
    our_ids = {"12345", "12346", "12347"}
    alive = _parse_status_output(output, our_ids, "pbs", [], "")
    assert alive == {"12345", "12346"}


def test_parse_status_output_sge_alive():
    output = "12345,qw\n12346,r\n"
    our_ids = {"12345", "12346"}
    alive = _parse_status_output(output, our_ids, "sge", [], "")
    assert alive == {"12345", "12346"}


def test_parse_status_output_auto_detects_slurm():
    output = "12345,PENDING\n12346,RUNNING\n"
    our_ids = {"12345", "12346"}
    alive = _parse_status_output(output, our_ids, "auto", [], "")
    assert alive == {"12345", "12346"}


def test_parse_status_output_ignores_unknown_ids():
    output = "99999,RUNNING\n"
    our_ids = {"12345"}
    alive = _parse_status_output(output, our_ids, "slurm", [], "")
    assert alive == set()


def test_parse_status_output_no_status_token_treated_as_alive():
    """A job present in output with no comma is assumed alive."""
    output = "12345\n"
    our_ids = {"12345"}
    alive = _parse_status_output(output, our_ids, "slurm", [], "")
    assert alive == {"12345"}


def test_parse_status_output_custom():
    output = "12345,INPROGRESS\n12346,DONE\n"
    our_ids = {"12345", "12346"}
    alive = _parse_status_output(output, our_ids, "custom", ["INPROGRESS"], "")
    assert alive == {"12345"}


# ---------------------------------------------------------------------------
# LocalProvisioner tests
# ---------------------------------------------------------------------------

def test_local_provisioner_spawns_one_process_per_device(tmp_path):
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0, 1, 2])
        p.ensure_workers(n_workers=10)

        assert mock_popen.call_count == 3
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
        p.ensure_workers(n_workers=5)

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


# ---------------------------------------------------------------------------
# ClusterProvisioner tests
# ---------------------------------------------------------------------------

def test_cluster_provisioner_submits_n_workers_jobs(tmp_path):
    cfg = _cluster_cfg(tmp_path)
    submitted = []

    def fake_run(cmd, **kwargs):
        submitted.append(cmd)
        result = MagicMock()
        result.stdout = "Submitted batch job 12345\n"
        return result

    with patch(
        "miss_alignment.distributed.provisioner.subprocess.run", side_effect=fake_run
    ), patch.object(ClusterProvisioner, "_alive_job_ids", return_value=set()):
        p = ClusterProvisioner(queue_dir=tmp_path, config=cfg)
        p.ensure_workers(n_workers=4)

    assert len(submitted) == 4


def test_cluster_provisioner_does_not_resubmit_alive_jobs(tmp_path):
    cfg = _cluster_cfg(tmp_path)
    submitted = []

    def fake_run(cmd, **kwargs):
        submitted.append(cmd)
        result = MagicMock()
        result.stdout = "Submitted batch job 12345\n"
        return result

    with patch(
        "miss_alignment.distributed.provisioner.subprocess.run", side_effect=fake_run
    ), patch.object(
        ClusterProvisioner, "_alive_job_ids", return_value={"1", "2", "3", "4"}
    ):
        p = ClusterProvisioner(queue_dir=tmp_path, config=cfg)
        p.ensure_workers(n_workers=4)

    assert len(submitted) == 0


def test_cluster_provisioner_replenishes_preempted_jobs(tmp_path):
    cfg = _cluster_cfg(tmp_path)
    submitted = []

    def fake_run(cmd, **kwargs):
        submitted.append(cmd)
        result = MagicMock()
        result.stdout = "Submitted batch job 99999\n"
        return result

    with patch(
        "miss_alignment.distributed.provisioner.subprocess.run", side_effect=fake_run
    ), patch.object(
        ClusterProvisioner, "_alive_job_ids", return_value={"1", "2"}
    ):
        p = ClusterProvisioner(queue_dir=tmp_path, config=cfg)
        p.ensure_workers(n_workers=4)

    assert len(submitted) == 2


def test_cluster_provisioner_cancels_on_shutdown(tmp_path):
    cfg = _cluster_cfg(tmp_path)
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
    ), patch.object(ClusterProvisioner, "_alive_job_ids", return_value=set()):
        p = ClusterProvisioner(queue_dir=tmp_path, config=cfg)
        p.ensure_workers(n_workers=3)
        p.shutdown()

    assert len(cancel_calls) == 3
    assert all("scancel" in c for c in cancel_calls)


def test_cluster_provisioner_prunes_terminated_jobs(tmp_path):
    """_alive_job_ids prunes job IDs no longer in scheduler output."""
    cfg = _cluster_cfg(tmp_path)

    # Simulate: 3 jobs submitted, only 2 alive in scheduler
    status_output = "11111,PENDING\n22222,RUNNING\n"

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.stdout = status_output
        result.returncode = 0
        return result

    with patch(
        "miss_alignment.distributed.provisioner.subprocess.run", side_effect=fake_run
    ):
        p = ClusterProvisioner(queue_dir=tmp_path, config=cfg)
        p._job_ids = ["11111", "22222", "33333"]
        alive = p._alive_job_ids()

    assert alive == {"11111", "22222"}
    assert p._job_ids == ["11111", "22222"]


# ---------------------------------------------------------------------------
# CompositeProvisioner tests
# ---------------------------------------------------------------------------

def test_composite_provisioner_delegates_to_all(tmp_path):
    a = MagicMock(spec=LocalProvisioner)
    b = MagicMock(spec=LocalProvisioner)
    p = CompositeProvisioner([a, b])

    p.ensure_workers(5)
    a.ensure_workers.assert_called_once_with(5)
    b.ensure_workers.assert_called_once_with(5)

    p.shutdown()
    a.shutdown.assert_called_once()
    b.shutdown.assert_called_once()
