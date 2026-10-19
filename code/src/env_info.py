"""Record machine specs and library versions to results/env.json."""
import json
import os
import platform
import subprocess
import sys
from importlib import metadata

from results_log import RESULTS_DIR

PACKAGES = ["faiss-cpu", "numpy", "h5py", "pandas", "matplotlib", "pytest"]


def cpu_name():
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def ram_gb():
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        return None


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect():
    return {
        "os": platform.platform(),
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "cpus_available_to_job": (len(os.sched_getaffinity(0))
                                  if hasattr(os, "sched_getaffinity") else None),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "ram_gb": ram_gb(),
        "python": platform.python_version(),
        "packages": {name: package_version(name) for name in PACKAGES},
    }


def main():
    out = RESULTS_DIR / "env.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    info = collect()
    out.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
