"""In-memory async job runner for the indexer API.

One shared queue, one worker per GPU. Each job checks out a device for its whole
run so the per-GPU models are never double-booked. A running job may also borrow
the GPUs that sit fully idle while nothing else is waiting, to split its own work. State is in-memory only and
lost on restart, the logs are the durable record.
"""

import logging
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)

# Keep finished jobs queryable for a while before dropping (2h)
JOB_RETENTION_SECONDS = 7200


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.DONE, JobStatus.FAILED)


class DuplicateJobError(Exception):
    """A job for this file id is already queued or running."""


class QueueFullError(Exception):
    """Too many jobs pending, the upload should be retried later."""


@dataclass
class Job:
    id: str
    file_id: str
    filename: str
    status: JobStatus = JobStatus.QUEUED
    device: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


def _detect_devices() -> list[str]:
    import torch

    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return ["mps"]
    return ["cpu"]


class JobQueue:
    def __init__(
        self,
        jobs_per_gpu: int = 1,
        max_queue_size: Optional[int] = None,
        devices: Optional[list[str]] = None,
    ):
        self.devices = devices or _detect_devices()
        self.jobs_per_gpu = jobs_per_gpu
        self.n_workers = len(self.devices) * jobs_per_gpu
        self.max_queue_size = (
            max_queue_size if max_queue_size is not None else self.n_workers * 10
        )

        # Free job slots per device
        self._free_device_slots: Counter[str] = Counter(
            {device: jobs_per_gpu for device in self.devices}
        )
        self._slots_changed = threading.Condition()

        self._executor = ThreadPoolExecutor(max_workers=self.n_workers)
        self._jobs: dict[str, Job] = {}
        self._reserved: set[str] = set()
        self._lock = threading.Lock()

        logger.info(
            "[JobQueue] ready: %d worker(s) on %s, max_queue=%d",
            self.n_workers,
            self.devices,
            self.max_queue_size,
        )

    def submit(
        self, file_id: str, filename: str, work_fn: Callable[[str], dict]
    ) -> str:
        """Queue work_fn(device) and return a job id immediately.

        Raises DuplicateJobError if file_id is in flight, QueueFullError if the
        queue is saturated.
        """
        job_id = uuid.uuid4().hex
        with self._lock:
            self._evict_old()
            if file_id in self._reserved:
                raise DuplicateJobError(file_id)
            if self._pending_count() >= self.max_queue_size:
                raise QueueFullError()
            self._reserved.add(file_id)
            self._jobs[job_id] = Job(id=job_id, file_id=file_id, filename=filename)

        logger.info(
            "[JobQueue] job %s queued (file_id=%s, filename=%s)",
            job_id,
            file_id,
            filename,
        )
        self._executor.submit(self._run, job_id, work_fn)
        return job_id

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    @contextmanager
    def borrow_idle_devices(self, exclude: str, limit: int) -> Iterator[list[str]]:
        """Lend up to `limit` extra devices to a running job, without waiting.

        Only devices with every slot free are lent, and only while no other job
        is queued.
        """
        borrowed: list[str] = []
        with self._lock, self._slots_changed:
            if limit > 0 and not self._has_queued_jobs():
                for device in self.devices:
                    if (
                        device != exclude
                        and self._free_device_slots[device] == self.jobs_per_gpu
                    ):
                        borrowed.append(device)
                borrowed = borrowed[:limit]
                for device in borrowed:
                    self._free_device_slots[device] = 0
        if borrowed:
            logger.info("[JobQueue] lent idle devices %s", borrowed)
        try:
            yield borrowed
        finally:
            if borrowed:
                with self._slots_changed:
                    for device in borrowed:
                        self._free_device_slots[device] = self.jobs_per_gpu
                    self._slots_changed.notify_all()

    def _acquire_device(self) -> str:
        with self._slots_changed:
            while True:
                device = max(self.devices, key=lambda d: self._free_device_slots[d])
                if self._free_device_slots[device] > 0:
                    self._free_device_slots[device] -= 1
                    return device
                self._slots_changed.wait()

    def _release_device(self, device: str) -> None:
        with self._slots_changed:
            self._free_device_slots[device] += 1
            self._slots_changed.notify_all()

    def _free_slot_count(self) -> int:
        with self._slots_changed:
            return sum(self._free_device_slots.values())

    def _run(self, job_id: str, work_fn: Callable[[str], dict]) -> None:
        job = self._jobs[job_id]
        device = self._acquire_device()
        job.device = device
        job.status = JobStatus.PROCESSING
        job.started_at = time.time()
        logger.info(
            "[JobQueue] job %s processing (gpu=%s), free slots: %d/%d",
            job_id,
            device,
            self._free_slot_count(),
            self.n_workers,
        )

        result = None
        error = None
        try:
            result = work_fn(device)
        except Exception as e:
            error = e
        finally:
            self._release_device(device)
            with self._lock:
                self._reserved.discard(job.file_id)
            job.finished_at = time.time()

        if error is None:
            job.result = result
            job.status = JobStatus.DONE
            logger.info(
                "[JobQueue] job %s done (gpu=%s, result=%s)", job_id, device, result
            )
        else:
            job.error = str(error)
            job.status = JobStatus.FAILED
            logger.error(
                "[JobQueue] job %s failed (gpu=%s): %s",
                job_id,
                device,
                error,
                exc_info=error,
            )

    def _has_queued_jobs(self) -> bool:
        return any(j.status == JobStatus.QUEUED for j in self._jobs.values())

    def _pending_count(self) -> int:
        return sum(not j.status.is_terminal for j in self._jobs.values())

    def _evict_old(self) -> None:
        now = time.time()
        stale = [
            j.id
            for j in self._jobs.values()
            if j.finished_at and now - j.finished_at > JOB_RETENTION_SECONDS
        ]
        for job_id in stale:
            self._jobs.pop(job_id, None)
