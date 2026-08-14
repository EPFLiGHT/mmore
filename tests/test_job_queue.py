import threading
import time

import pytest

from mmore.job_queue import DuplicateJobError, JobQueue, QueueFullError


def _wait_for(manager, job_id, status, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get(job_id)
        if job and job.status == status:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach {status}")


def test_submit_runs_to_done():
    manager = JobQueue(devices=["cpu"])
    job_id = manager.submit("f1", "a.pdf", lambda device: {"device": device, "n": 1})
    job = _wait_for(manager, job_id, "done")
    assert job.result == {"device": "cpu", "n": 1}
    manager.shutdown()


def test_failure_marks_failed():
    manager = JobQueue(devices=["cpu"])

    def boom(device):
        raise ValueError("error")

    job_id = manager.submit("f1", "a.pdf", boom)
    job = _wait_for(manager, job_id, "failed")
    assert job.error == "error"
    manager.shutdown()


def test_duplicate_file_id_while_in_flight():
    manager = JobQueue(devices=["cpu"])
    release = threading.Event()

    def block(device):
        release.wait()
        return {}

    job_id = manager.submit("same", "a.pdf", block)
    _wait_for(manager, job_id, "processing")
    with pytest.raises(DuplicateJobError):
        manager.submit("same", "b.pdf", lambda device: {})

    release.set()
    _wait_for(manager, job_id, "done")
    manager.shutdown()


def test_queue_full():
    manager = JobQueue(devices=["cpu"], max_queue_size=1)
    release = threading.Event()

    def block(device):
        release.wait()
        return {}

    job_id = manager.submit("f1", "a.pdf", block)
    _wait_for(manager, job_id, "processing")
    with pytest.raises(QueueFullError):
        manager.submit("f2", "b.pdf", lambda device: {})

    release.set()
    manager.shutdown()


def test_single_device_serializes():
    manager = JobQueue(devices=["cpu"])
    running = []
    max_seen = []
    lock = threading.Lock()

    def work(device):
        with lock:
            running.append(1)
            max_seen.append(len(running))
        time.sleep(0.05)
        with lock:
            running.pop()
        return {}

    ids = [manager.submit(f"f{i}", "a.pdf", work) for i in range(4)]
    for job_id in ids:
        _wait_for(manager, job_id, "done")

    assert max(max_seen) == 1
    manager.shutdown()
