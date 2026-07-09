import sys
import time
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

def test_parse_status_output_slurm_running_and_pending():
    output = "12345,PENDING\n12346,RUNNING\n12347,COMPLETING\n"
    our_ids = {"12345", "12346", "12347", "99999"}
    states = _parse_status_output(output, our_ids, "slurm", [], "")
    assert states == {"12345": "pending", "12346": "running", "12347": "running"}


def test_parse_status_output_slurm_terminal_absent():
    """Terminal jobs (COMPLETED/FAILED) are absent from the result, not included."""
    output = "12345,COMPLETED\n12346,FAILED\n12347,CANCELLED\n"
    our_ids = {"12345", "12346", "12347"}
    states = _parse_status_output(output, our_ids, "slurm", [], "")
    assert states == {}


def test_parse_status_output_lsf():
    output = "12345,PEND\n12346,RUN\n"
    our_ids = {"12345", "12346"}
    states = _parse_status_output(output, our_ids, "lsf", [], "")
    assert states == {"12345": "pending", "12346": "running"}


def test_parse_status_output_pbs():
    output = "12345,Q\n12346,R\n12347,C\n"
    our_ids = {"12345", "12346", "12347"}
    states = _parse_status_output(output, our_ids, "pbs", [], "")
    assert states == {"12345": "pending", "12346": "running"}


def test_parse_status_output_sge():
    output = "12345,qw\n12346,r\n"
    our_ids = {"12345", "12346"}
    states = _parse_status_output(output, our_ids, "sge", [], "")
    assert states == {"12345": "pending", "12346": "running"}


def test_parse_status_output_auto_detects_slurm():
    output = "12345,PENDING\n12346,RUNNING\n"
    our_ids = {"12345", "12346"}
    states = _parse_status_output(output, our_ids, "auto", [], "")
    assert states == {"12345": "pending", "12346": "running"}


def test_parse_status_output_ignores_unknown_ids():
    output = "99999,RUNNING\n"
    our_ids = {"12345"}
    states = _parse_status_output(output, our_ids, "slurm", [], "")
    assert states == {}


def test_parse_status_output_no_status_token_treated_as_pending():
    """A job present in output with no comma is assumed pending."""
    output = "12345\n"
    our_ids = {"12345"}
    states = _parse_status_output(output, our_ids, "slurm", [], "")
    assert states == {"12345": "pending"}


def test_parse_status_output_custom():
    """Custom: alive statuses appear as 'pending'; others are absent."""
    output = "12345,INPROGRESS\n12346,DONE\n"
    our_ids = {"12345", "12346"}
    states = _parse_status_output(output, our_ids, "custom", ["INPROGRESS"], "")
    assert states == {"12345": "pending"}


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
    """ensure_workers submits until _job_submit_time reaches target."""
    cfg = _cluster_cfg(tmp_path)
    call_num = [0]

    def fake_run(cmd, **kwargs):
        call_num[0] += 1
        result = MagicMock()
        result.stdout = (
            "" if "squeue" in cmd
            else f"Submitted batch job {call_num[0]}\n"
        )
        return result

    with patch(
        "miss_alignment.distributed.provisioner.subprocess.run", side_effect=fake_run
    ):
        p = ClusterProvisioner(queue_dir=tmp_path, config=cfg)
        p.ensure_workers(n_workers=4)

    assert len(p._job_submit_time) == 4


def test_cluster_provisioner_does_not_resubmit_within_grace(tmp_path):
    """Jobs recently submitted but absent from squeue are not double-counted."""
    cfg = _cluster_cfg(tmp_path)

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.stdout = ""  # squeue returns nothing (jobs not yet registered)
        return result

    with patch(
        "miss_alignment.distributed.provisioner.subprocess.run", side_effect=fake_run
    ):
        p = ClusterProvisioner(queue_dir=tmp_path, config=cfg)
        now = time.time()
        p._job_submit_time = {"1": now, "2": now, "3": now, "4": now}
        p.ensure_workers(n_workers=4)

    assert len(p._job_submit_time) == 4  # no new submissions


def test_cluster_provisioner_prunes_grace_expired_absent_jobs(tmp_path):
    """Jobs absent from squeue past the grace period are pruned."""
    cfg = _cluster_cfg(tmp_path)

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.stdout = "3,RUNNING\n"  # only job 3 visible
        return result

    with patch(
        "miss_alignment.distributed.provisioner.subprocess.run", side_effect=fake_run
    ):
        p = ClusterProvisioner(queue_dir=tmp_path, config=cfg)
        old = time.time() - 200  # well past grace period
        now = time.time()
        p._job_submit_time = {"1": old, "2": old, "3": now}
        states = p._query_job_states()

    # Old absent jobs pruned; recent absent job kept; visible job in states
    assert "1" not in p._job_submit_time
    assert "2" not in p._job_submit_time
    assert "3" in p._job_submit_time
    assert states == {"3": "running"}


def test_cluster_provisioner_absent_within_grace_not_pruned(tmp_path):
    """Jobs absent from squeue within the grace period are kept."""
    cfg = _cluster_cfg(tmp_path)

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.stdout = "11111,RUNNING\n"
        return result

    with patch(
        "miss_alignment.distributed.provisioner.subprocess.run", side_effect=fake_run
    ):
        p = ClusterProvisioner(queue_dir=tmp_path, config=cfg)
        now = time.time()
        p._job_submit_time = {"11111": now, "22222": now, "33333": now}
        states = p._query_job_states()

    assert states == {"11111": "running"}
    assert set(p._job_submit_time.keys()) == {"11111", "22222", "33333"}


def test_cluster_provisioner_replenishes_grace_expired_jobs(tmp_path):
    """ensure_workers resubmits for jobs that have aged out."""
    cfg = _cluster_cfg(tmp_path)
    submitted = []

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        if "squeue" in cmd:
            result.stdout = "3,RUNNING\n4,RUNNING\n"
        else:
            submitted.append(cmd)
            result.stdout = f"Submitted batch job {len(submitted) + 10}\n"
        return result

    with patch(
        "miss_alignment.distributed.provisioner.subprocess.run", side_effect=fake_run
    ):
        p = ClusterProvisioner(queue_dir=tmp_path, config=cfg)
        old = time.time() - 200
        now = time.time()
        p._job_submit_time = {"1": old, "2": old, "3": now, "4": now}
        p.ensure_workers(n_workers=4)

    assert len(submitted) == 2  # replaced the 2 expired-absent jobs


def test_cluster_provisioner_worker_counts_split_running_pending(tmp_path):
    cfg = _cluster_cfg(tmp_path)

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.stdout = "1,RUNNING\n2,RUNNING\n3,PENDING\n"
        return result

    with patch(
        "miss_alignment.distributed.provisioner.subprocess.run", side_effect=fake_run
    ):
        p = ClusterProvisioner(queue_dir=tmp_path, config=cfg)
        now = time.time()
        # 5 tracked: 3 visible (2 running, 1 pending), 2 not yet visible
        p._job_submit_time = {"1": now, "2": now, "3": now, "4": now, "5": now}
        counts = p.worker_counts_by_type()

    assert counts["cluster-running"] == 2
    assert counts["cluster-pending"] == 3  # 1 explicit + 2 not-yet-visible


def test_cluster_provisioner_cancels_on_shutdown(tmp_path):
    cfg = _cluster_cfg(tmp_path)
    cancel_calls = []
    submit_count = [0]

    def fake_run(cmd, **kwargs):
        if "sbatch" in cmd:
            submit_count[0] += 1
            result = MagicMock()
            result.stdout = f"Submitted batch job {submit_count[0]}\n"
            return result
        if "squeue" in cmd:
            result = MagicMock()
            result.stdout = ""
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
