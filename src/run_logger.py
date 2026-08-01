#!/usr/bin/env python3
"""Simple run logger for model scripts."""

from __future__ import annotations

import json
import os
import sys
import time
import atexit
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def append_run_log(
    output_dir: Path,
    script_name: str,
    started_at_utc: str,
    ended_at_utc: str,
    duration_seconds: float,
    status: str,
    metrics: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
    error: str | None = None,
) -> Path:
    """Append one structured run record to output_dir/run_log.jsonl."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.jsonl"
    record = {
        "script_name": script_name,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_seconds": round(float(duration_seconds), 3),
        "status": status,
        "metrics": metrics or {},
        "details": details or {},
    }
    if error:
        record["error"] = error

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")

    _try_log_mlflow(record, output_dir)

    # Used by auto-logger to avoid duplicate writes
    global _EXPLICIT_LOG_WRITTEN
    _EXPLICIT_LOG_WRITTEN = True

    return log_path


def _coerce_float_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Keep only finite numeric metrics for MLflow."""
    clean: dict[str, float] = {}
    for key, value in (metrics or {}).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            # Exclude NaN and infinities.
            if value != value:  # NaN check
                continue
            if value in (float("inf"), float("-inf")):
                continue
            clean[key] = float(value)
    return clean


def _try_log_mlflow(record: dict[str, Any], output_dir: Path) -> None:
    """Best-effort MLflow logging; never raises."""
    if os.getenv("TS_DISABLE_MLFLOW", "").lower() in {"1", "true", "yes"}:
        return
    try:
        import mlflow
    except Exception:
        return

    try:
        repo_root = output_dir.parent.parent
        tracking_uri = os.getenv(
            "TS_MLFLOW_TRACKING_URI", f"file://{repo_root / 'mlruns'}"
        )
        experiment_name = os.getenv(
            "TS_MLFLOW_EXPERIMENT", "time_series_experiments"
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        run_name = f"{record.get('script_name', 'script')}:{record.get('started_at_utc', '')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "script_name": str(record.get("script_name", "")),
                    "status": str(record.get("status", "")),
                    "started_at_utc": str(record.get("started_at_utc", "")),
                    "ended_at_utc": str(record.get("ended_at_utc", "")),
                }
            )
            mlflow.log_metric(
                "duration_seconds", float(record.get("duration_seconds", 0.0))
            )

            for k, v in _coerce_float_metrics(record.get("metrics", {})).items():
                mlflow.log_metric(k, v)

            for k, v in (record.get("details", {}) or {}).items():
                mlflow.log_param(str(k), str(v))

            if record.get("error"):
                mlflow.set_tag("error", str(record["error"]))
    except Exception:
        # MLflow must not break model execution/logging.
        return


_AUTO_STARTED = False
_EXPLICIT_LOG_WRITTEN = False
_AUTO_START_ISO = ""
_AUTO_T0 = 0.0
_AUTO_STATUS = "success"
_AUTO_ERROR = None
_AUTO_SCRIPT = ""


def _default_output_dir_for_script(script_path: Path) -> Path:
    """Infer output directory from calling script path."""
    return script_path.parent / "outputs"


def _on_unhandled_exception(exc_type, exc_value, exc_traceback):
    """Capture uncaught exceptions so auto-log marks run as failed."""
    global _AUTO_STATUS, _AUTO_ERROR
    _AUTO_STATUS = "failed"
    _AUTO_ERROR = str(exc_value)
    _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_traceback)


def _flush_auto_log() -> None:
    """Write one auto run record if explicit logger didn't write one."""
    if not _AUTO_STARTED:
        return
    if _EXPLICIT_LOG_WRITTEN:
        return

    script_path = Path(_AUTO_SCRIPT)
    output_dir = _default_output_dir_for_script(script_path)
    append_run_log(
        output_dir=output_dir,
        script_name=script_path.parent.name,
        started_at_utc=_AUTO_START_ISO,
        ended_at_utc=utc_now_iso(),
        duration_seconds=time.perf_counter() - _AUTO_T0,
        status=_AUTO_STATUS,
        metrics={},
        details={
            "auto_logger": True,
            "script_path": str(script_path),
            "cwd": os.getcwd(),
        },
        error=_AUTO_ERROR,
    )


_ORIGINAL_EXCEPTHOOK = sys.excepthook


def start_auto_run_logger() -> None:
    """Start process-level auto run logging for scripts importing src."""
    global _AUTO_STARTED, _AUTO_START_ISO, _AUTO_T0, _AUTO_SCRIPT
    if _AUTO_STARTED:
        return
    # Only log direct script execution, not interactive usage/import.
    if not sys.argv or not sys.argv[0] or sys.argv[0].startswith("-"):
        return
    script_path = Path(sys.argv[0])
    if script_path.name != "main.py":
        return

    _AUTO_STARTED = True
    _AUTO_START_ISO = utc_now_iso()
    _AUTO_T0 = time.perf_counter()
    _AUTO_SCRIPT = str(script_path.resolve())

    sys.excepthook = _on_unhandled_exception
    atexit.register(_flush_auto_log)
