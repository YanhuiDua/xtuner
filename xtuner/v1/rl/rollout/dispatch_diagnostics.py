import asyncio
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

import requests


CONTROLLER_ACTIVE = "controller_active"
WORKER_SUBMITTED = "worker_submitted"
WORKER_INFLIGHT = "worker_inflight"
HTTP_WAITING = "http_waiting"
HTTP_INFLIGHT = "http_inflight"


@dataclass(frozen=True)
class DispatchDiagnosticsConfig:
    enabled: bool = False
    log_interval_seconds: float = 30.0
    grace_seconds: float = 2.0
    engine_metrics_timeout_seconds: float = 5.0

    @classmethod
    def from_rollout_config(cls, config: Any) -> "DispatchDiagnosticsConfig":
        return cls(
            enabled=bool(getattr(config, "rollout_dispatch_diagnostics", False)),
            log_interval_seconds=float(getattr(config, "rollout_dispatch_log_interval_seconds", 30.0)),
            grace_seconds=float(getattr(config, "rollout_dispatch_grace_seconds", 2.0)),
        )


@dataclass
class _DispatchRecord:
    request_id: str
    stage: str
    created_at: float
    stage_started_at: float


class DispatchStageTracker:
    """Small in-process lifecycle tracker for rollout dispatch diagnostics."""

    def __init__(
        self,
        *,
        name: str,
        config: DispatchDiagnosticsConfig,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self.metadata = metadata or {}
        self._active: dict[str, _DispatchRecord] = {}
        self._entered_total_by_stage: Counter[str] = Counter()
        self._finished_total_by_status: Counter[str] = Counter()
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def enter_worker(self, request_id: str) -> None:
        self._enter(request_id, WORKER_INFLIGHT)

    def enter_http_waiting(self, request_id: str) -> None:
        self._enter(request_id, HTTP_WAITING)

    def enter_http_inflight(self, request_id: str) -> None:
        self._enter(request_id, HTTP_INFLIGHT)

    def finish(self, request_id: str, status: Any = None) -> None:
        if not self.enabled:
            return
        normalized_status = _normalize_status(status)
        with self._lock:
            if self._active.pop(request_id, None) is None:
                return
            self._finished_total_by_status[normalized_status] += 1

    def _enter(self, request_id: str, stage: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            existing = self._active.get(request_id)
            created_at = existing.created_at if existing is not None else now
            self._active[request_id] = _DispatchRecord(
                request_id=request_id,
                stage=stage,
                created_at=created_at,
                stage_started_at=now,
            )
            self._entered_total_by_stage[stage] += 1

    def snapshot(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "name": self.name,
                "metadata": dict(self.metadata),
                "active_by_stage": {},
                "entered_total_by_stage": {},
                "finished_total_by_status": {},
                "oldest_active_s_by_stage": {},
            }

        now = time.perf_counter()
        with self._lock:
            active_records = list(self._active.values())
            active_by_stage = Counter(record.stage for record in active_records)
            oldest_by_stage: dict[str, float] = {}
            for record in active_records:
                age = now - record.stage_started_at
                oldest_by_stage[record.stage] = max(oldest_by_stage.get(record.stage, 0.0), age)
            return {
                "enabled": True,
                "name": self.name,
                "metadata": dict(self.metadata),
                "active_by_stage": dict(active_by_stage),
                "entered_total_by_stage": dict(self._entered_total_by_stage),
                "finished_total_by_status": dict(self._finished_total_by_status),
                "oldest_active_s_by_stage": oldest_by_stage,
            }


class AggregateDispatchCounter:
    """Aggregate counters for controller-side dispatch stages.

    This deliberately does not keep request ids. It is meant for coarse controller diagnostics such as "submitted to
    worker but not yet visible in worker.generate".
    """

    def __init__(self, *, enabled: bool, name: str, metadata: dict[str, Any] | None = None) -> None:
        self.enabled = enabled
        self.name = name
        self.metadata = metadata or {}
        self._active_by_stage: Counter[str] = Counter()
        self._entered_total_by_stage: Counter[str] = Counter()
        self._finished_total_by_stage: Counter[str] = Counter()
        self._stage_first_active_at: dict[str, float] = {}
        self._lock = threading.RLock()

    def enter(self, stage: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            if self._active_by_stage[stage] == 0:
                self._stage_first_active_at[stage] = now
            self._active_by_stage[stage] += 1
            self._entered_total_by_stage[stage] += 1

    def finish(self, stage: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._active_by_stage[stage] <= 0:
                return
            self._active_by_stage[stage] -= 1
            self._finished_total_by_stage[stage] += 1
            if self._active_by_stage[stage] == 0:
                self._stage_first_active_at.pop(stage, None)

    def snapshot(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "name": self.name,
                "metadata": dict(self.metadata),
                "active_by_stage": {},
                "entered_total_by_stage": {},
                "finished_total_by_stage": {},
                "oldest_active_s_by_stage": {},
            }
        now = time.perf_counter()
        with self._lock:
            oldest_by_stage = {stage: now - started_at for stage, started_at in self._stage_first_active_at.items()}
            return {
                "enabled": True,
                "name": self.name,
                "metadata": dict(self.metadata),
                "active_by_stage": dict(+self._active_by_stage),
                "entered_total_by_stage": dict(self._entered_total_by_stage),
                "finished_total_by_stage": dict(self._finished_total_by_stage),
                "oldest_active_s_by_stage": oldest_by_stage,
            }


class DispatchHttpGate:
    """Visible HTTP concurrency gate used before entering httpx."""

    def __init__(self, *, limit: int, tracker: DispatchStageTracker) -> None:
        self.limit = max(1, int(limit))
        self._tracker = tracker
        self._semaphore = asyncio.Semaphore(self.limit)

    @asynccontextmanager
    async def slot(self, request_id: str) -> AsyncIterator[None]:
        acquired = False
        self._tracker.enter_http_waiting(request_id)
        try:
            await self._semaphore.acquire()
            acquired = True
            self._tracker.enter_http_inflight(request_id)
            yield
        finally:
            if acquired:
                self._semaphore.release()


class EngineMetricsReader:
    LMDEPLOY_METRIC_NAMES = {
        "api_routed": "lmdeploy:num_api_requests_routed",
        "api_waiting": "lmdeploy:num_api_requests_waiting",
        "engine_running": "lmdeploy:num_requests_running",
        "engine_waiting": "lmdeploy:num_requests_waiting",
        "failed": "lmdeploy:num_requests_failed",
    }

    def __init__(self, *, timeout_seconds: float = 5.0) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch_lmdeploy(self, server_urls: list[str]) -> dict[str, Any]:
        totals = dict.fromkeys(self.LMDEPLOY_METRIC_NAMES, 0.0)
        errors: dict[str, str] = {}
        for url in server_urls:
            try:
                response = requests.get(f"{url}/metrics", timeout=self.timeout_seconds)
                response.raise_for_status()
            except Exception as exc:
                errors[url] = str(exc)
                continue
            text = response.text
            for key, metric_name in self.LMDEPLOY_METRIC_NAMES.items():
                totals[key] += _parse_prometheus_gauge(text, metric_name)
        return {
            "backend": "lmdeploy",
            "metrics": totals,
            "errors": errors,
        }


def build_rollout_request_id(rollout_state: Any, *, attempt: int | None = None) -> str:
    message_uid = getattr(rollout_state, "message_uid", None)
    uid = getattr(rollout_state, "uid", None)
    session_uid = getattr(rollout_state, "session_uid", None)
    base = f"msg={message_uid}:uid={uid}:session={session_uid}"
    if attempt is None:
        return base
    return f"{base}:attempt={attempt}"


def summarize_dispatch_diagnostics(
    *,
    controller: dict[str, Any] | None = None,
    router: dict[str, Any] | None = None,
    workers: list[dict[str, Any]],
    engine: dict[str, Any] | None = None,
) -> dict[str, Any]:
    worker_active = Counter()
    worker_oldest: dict[str, float] = {}
    worker_entered_total = Counter()
    worker_finished_total = Counter()

    for worker in workers:
        worker_active.update(worker.get("active_by_stage", {}))
        worker_entered_total.update(worker.get("entered_total_by_stage", {}))
        worker_finished_total.update(worker.get("finished_total_by_status", {}))
        for stage, age in worker.get("oldest_active_s_by_stage", {}).items():
            worker_oldest[stage] = max(worker_oldest.get(stage, 0.0), float(age))

    controller = controller or {}
    controller_active = controller.get("active_by_stage", {})
    controller_oldest = controller.get("oldest_active_s_by_stage", {})
    router = router or {}
    engine_metrics = (engine or {}).get("metrics", {})
    engine_visible = (
        engine_metrics.get("api_routed", 0.0)
        + engine_metrics.get("api_waiting", 0.0)
        + engine_metrics.get("engine_running", 0.0)
        + engine_metrics.get("engine_waiting", 0.0)
    )

    worker_inflight = int(worker_active.get(WORKER_INFLIGHT, 0))
    worker_submitted = int(controller_active.get(WORKER_SUBMITTED, 0))
    return {
        "controller_active": int(controller_active.get(CONTROLLER_ACTIVE, 0)),
        "worker_submitted": worker_submitted,
        "worker_actor_pending": max(0, worker_submitted - worker_inflight),
        "oldest_worker_submitted_s": float(controller_oldest.get(WORKER_SUBMITTED, 0.0)),
        "active_workers": int(router.get("active_workers", 0)),
        "total_workers": int(router.get("total_workers", 0)),
        "session_count": int(router.get("session_count", 0)),
        "worker_inflight": worker_inflight,
        "http_waiting": int(worker_active.get(HTTP_WAITING, 0)),
        "http_inflight": int(worker_active.get(HTTP_INFLIGHT, 0)),
        "oldest_http_waiting_s": float(worker_oldest.get(HTTP_WAITING, 0.0)),
        "oldest_http_inflight_s": float(worker_oldest.get(HTTP_INFLIGHT, 0.0)),
        "worker_entered_total_by_stage": dict(worker_entered_total),
        "worker_finished_total_by_status": dict(worker_finished_total),
        "engine_visible": float(engine_visible),
        "engine_metrics": dict(engine_metrics),
        "engine_errors": dict((engine or {}).get("errors", {})),
    }


def analyze_dispatch_summary(summary: dict[str, Any], config: DispatchDiagnosticsConfig) -> list[str]:
    warnings: list[str] = []
    if summary["worker_actor_pending"] > 0 and summary["oldest_worker_submitted_s"] >= config.grace_seconds:
        warnings.append(
            "rollout requests were submitted by controller but are not visible in worker.generate yet: "
            f"worker_actor_pending={summary['worker_actor_pending']} "
            f"oldest={summary['oldest_worker_submitted_s']:.3f}s"
        )

    if summary["http_waiting"] > 0 and summary["oldest_http_waiting_s"] >= config.grace_seconds:
        warnings.append(
            "rollout HTTP requests are waiting for local worker slots: "
            f"http_waiting={summary['http_waiting']} oldest={summary['oldest_http_waiting_s']:.3f}s"
        )

    if (
        summary.get("engine_metrics")
        and summary["http_inflight"] > summary["engine_visible"]
        and summary["oldest_http_inflight_s"] >= config.grace_seconds
    ):
        warnings.append(
            "rollout HTTP inflight requests are not visible in inference engine metrics: "
            f"http_inflight={summary['http_inflight']} engine_visible={summary['engine_visible']:.0f} "
            f"oldest={summary['oldest_http_inflight_s']:.3f}s"
        )

    engine_metrics = summary.get("engine_metrics", {})
    if engine_metrics.get("api_waiting", 0.0) > 0:
        warnings.append(f"lmdeploy API has waiting requests: api_waiting={engine_metrics['api_waiting']:.0f}")
    if engine_metrics.get("engine_waiting", 0.0) > 0:
        warnings.append(
            f"lmdeploy engine has scheduler waiting requests: engine_waiting={engine_metrics['engine_waiting']:.0f}"
        )
    if summary.get("engine_errors"):
        warnings.append(f"failed to fetch inference engine metrics: {summary['engine_errors']}")
    return warnings


def format_dispatch_summary(summary: dict[str, Any], warnings: list[str] | None = None) -> str:
    engine_metrics = summary.get("engine_metrics", {})
    warning_suffix = "" if not warnings else f" warnings={warnings}"
    return (
        "rollout_dispatch "
        f"active_workers={summary['active_workers']}/{summary['total_workers']} "
        f"sessions={summary['session_count']} "
        f"controller_active={summary['controller_active']} "
        f"worker_submitted={summary['worker_submitted']} "
        f"worker_actor_pending={summary['worker_actor_pending']} "
        f"worker_inflight={summary['worker_inflight']} "
        f"http_waiting={summary['http_waiting']} "
        f"http_inflight={summary['http_inflight']} "
        f"engine_visible={summary['engine_visible']:.0f} "
        f"api_waiting={engine_metrics.get('api_waiting', 0.0):.0f} "
        f"api_routed={engine_metrics.get('api_routed', 0.0):.0f} "
        f"engine_waiting={engine_metrics.get('engine_waiting', 0.0):.0f} "
        f"engine_running={engine_metrics.get('engine_running', 0.0):.0f}"
        f"{warning_suffix}"
    )


def _parse_prometheus_gauge(text: str, metric_name: str) -> float:
    total = 0.0
    for line in text.splitlines():
        if not line.startswith(metric_name):
            continue
        try:
            total += float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
    return total


def _normalize_status(status: Any) -> str:
    if status is None:
        return "unknown"
    value = getattr(status, "value", status)
    return str(value)
