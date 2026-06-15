import os
import socket
import datetime
import yaml

RESULTS_DIR = os.environ.get("BENCH_RESULTS_DIR", "results")


def _host():
    """Real host name, passed in from run.sh (inside a container `hostname` is the container id)."""
    return os.environ.get("BENCH_HOST") or socket.gethostname()


def results_path():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f"{_host()}.yaml")


def gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "gpu": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "vram_gb_each": round(props.total_memory / 1e9, 1),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            }
    except Exception:
        pass
    return {"gpu": "cpu", "gpu_count": 0, "vram_gb_each": 0}


def vram_gb():
    return gpu_info().get("vram_gb_each", 0)


def _load():
    p = results_path()
    if os.path.exists(p):
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


def _save(data):
    with open(results_path(), "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _ensure_meta(data):
    if "Meta" not in data:
        meta = {
            "host": _host(),
            "timestamp": os.environ.get("BENCH_TIMESTAMP")
            or datetime.datetime.now().isoformat(timespec="seconds"),
        }
        meta.update(gpu_info())
        data["Meta"] = meta


def update_results(section, data):
    """Append one benchmark section to results/<host>.yaml, tagging host/GPU metadata once."""
    results = _load()
    _ensure_meta(results)
    results[section] = data
    _save(results)
