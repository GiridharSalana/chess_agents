#!/usr/bin/env python3
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"
TRAIN = ROOT / "train.py"
LOG = ROOT / "nohup_train.out"
MONITOR_LOG = ROOT / "monitor_train.log"

ERROR_PATTERNS = [
    "BrokenPipeError",
    "unable to mmap",
    "Cannot allocate memory",
    "resource_tracker",
    "Exception in thread",
    "Traceback (most recent call last)",
    "RuntimeError",
    "OSError",
]
SUCCESS_PATTERN = re.compile(r"DONE!|Best model:|win rate:.*" )


def write_monitor(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    print(line, flush=True)
    with open(MONITOR_LOG, "a") as fp:
        fp.write(line + "\n")


def train_process_lines() -> list[str]:
    p = subprocess.run(["pgrep", "-af", "train.py"], capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if "monitor_train.py" not in l]
    return lines


def is_train_running() -> bool:
    return len(train_process_lines()) > 0


def kill_train_processes() -> None:
    write_monitor("Stopping existing train.py processes.")
    subprocess.run(["pkill", "-f", "train.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)


def start_train() -> None:
    if not PYTHON.exists():
        raise FileNotFoundError(f"Python executable not found at {PYTHON}")
    cmd = f"cd {ROOT} && nohup {PYTHON} train.py > nohup_train.out 2>&1 &"
    subprocess.run(cmd, shell=True, executable="/bin/bash")
    write_monitor("Started train.py in nohup.")
    time.sleep(5)


def read_tail(lines: int = 80) -> str:
    if not LOG.exists():
        return ""
    with LOG.open("rb") as fp:
        fp.seek(0, os.SEEK_END)
        size = fp.tell()
        block_size = 1024
        data = b""
        while size > 0 and data.count(b"\n") <= lines:
            seek = max(0, size - block_size)
            fp.seek(seek)
            data = fp.read() + data
            size = seek
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return data.decode("utf-8", "ignore")


def has_fatal_error(content: str) -> bool:
    text = content.lower()
    return any(pattern.lower() in text for pattern in ERROR_PATTERNS)


def has_completed(content: str) -> bool:
    return "DONE!" in content or "Best model:" in content and "done" in content.lower()


def monitor_loop(interval: int = 30) -> None:
    write_monitor("Entering monitor loop.")
    while True:
        running = is_train_running()
        tail = read_tail(120)
        if has_completed(tail):
            write_monitor("Training completed successfully.")
            break

        if not running:
            write_monitor("No active training process detected.")
            start_train()
            continue

        if has_fatal_error(tail):
            write_monitor("Fatal error detected in log. Restarting training.")
            kill_train_processes()
            start_train()
            continue

        write_monitor("Training running normally. Continuing monitoring.")
        time.sleep(interval)


if __name__ == "__main__":
    monitor_loop()