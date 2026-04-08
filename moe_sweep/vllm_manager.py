import subprocess
import time
from pathlib import Path

import httpx


def start_vllm(
    variant_path: str | Path,
    port: int,
    gpu_ids: list[int] | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Start a vLLM server for a model variant.

    Args:
        variant_path: Path to model variant directory.
        port: Port to serve on.
        gpu_ids: GPU IDs to use (sets CUDA_VISIBLE_DEVICES).
        extra_args: Additional vLLM CLI arguments.

    Returns:
        The subprocess.Popen object for the vLLM server.
    """
    env = None
    if gpu_ids is not None:
        import os
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    cmd = [
        "vllm", "serve", str(variant_path),
        "--port", str(port),
        "--enable-prefix-caching",
    ]
    if gpu_ids and len(gpu_ids) > 1:
        cmd.extend(["--tensor-parallel-size", str(len(gpu_ids))])
    if extra_args:
        cmd.extend(extra_args)

    print(f"Starting vLLM: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env)

    # Wait for health check
    url = f"http://localhost:{port}/health"
    for attempt in range(120):  # up to 10 minutes
        try:
            resp = httpx.get(url, timeout=5)
            if resp.status_code == 200:
                print(f"vLLM ready on port {port}")
                return proc
        except httpx.ConnectError:
            pass
        time.sleep(5)

    proc.terminate()
    raise TimeoutError(f"vLLM failed to start on port {port} within 10 minutes")


def stop_vllm(proc: subprocess.Popen) -> None:
    """Stop a vLLM server."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    print("vLLM stopped")
