
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG_PATH = Path("logs/inference/inference_log.jsonl")

_LOCK = threading.Lock()


def log_inference(
    input_data: dict[str, Any],
    prediction: int | None = None,
    attrition_probability: float | None = None,
    latency_ms: float | None = None,
    status: str = "success",
    error: str | None = None,
    model_name: str | None = None,
    model_version: int | str | None = None,
    run_id: str | None = None,
    request_id: str | None = None,
    include_input: bool = False
) -> str:

    # Generate a request ID if one was not supplied.
    request_id = request_id or uuid.uuid4().hex[:12]

    # Prepare employee input data.
    safe_input = dict(input_data)

    # Remove employee identifier and target column.
    safe_input.pop("EmployeeNumber", None)
    safe_input.pop("Attrition", None)

    # Avoid storing raw employee information by default.
    logged_input = safe_input if include_input else None

    # Create the inference event.
    event = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": logged_input,
        "prediction": prediction,
        "attrition_probability": attrition_probability,
        "latency_ms": latency_ms,
        "status": status,
        "error": error,
        "model_name": model_name,
        "model_version": model_version,
        "run_id": run_id,
    }

    # Create the log directory if it does not exist.
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Append the event to the JSONL file.
    with _LOCK:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

    return request_id
