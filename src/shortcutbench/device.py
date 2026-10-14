"""Pick the best available accelerator and describe the hardware."""
from __future__ import annotations

import platform
import subprocess

import torch


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hardware_info(device: torch.device) -> dict:
    info = {"device": str(device), "torch": torch.__version__, "python": platform.python_version(),
            "platform": platform.platform()}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    elif platform.system() == "Darwin":
        try:
            info["chip"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                          capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    return info


def sync(device: torch.device) -> None:
    """Block until queued kernels finish (for honest timing)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
