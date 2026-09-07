"""Launch Marker 1 or Marker 2 only after validating NVIDIA GPU support."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


RUNTIME_DIR = Path(__file__).resolve().parent / "marker1_runtime"


class GPUUnavailable(RuntimeError):
    pass


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise GPUUnavailable(f"{label} not found: {path}")
    return path


def nvidia_gpu_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GPUUnavailable("The NVIDIA driver could not be reached.") from exc
    name = result.stdout.strip()
    if result.returncode or not name:
        raise GPUUnavailable("No NVIDIA GPU was reported by the driver.")
    return name


def require_marker1_cuda(python: Path) -> None:
    python = require_file(python, "Python executable")
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import sys, torch; sys.exit(0 if torch.cuda.is_available() "
            "and torch.cuda.device_count() else 1)",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode:
        raise GPUUnavailable("Marker 1's PyTorch environment cannot initialize CUDA.")


def llama_cuda_details(llama_server: Path) -> str:
    llama_server = require_file(llama_server, "llama-server")
    version = subprocess.run(
        [str(llama_server), "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    devices = subprocess.run(
        [str(llama_server), "--list-devices"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    device_text = devices.stdout + devices.stderr
    if version.returncode:
        raise GPUUnavailable("llama-server could not report its version.")
    if devices.returncode or "CUDA" not in device_text.upper():
        raise GPUUnavailable("llama.cpp cannot find a CUDA device.")
    return version.stdout.strip() or version.stderr.strip()


def marker1_environment(base: dict[str, str], gpu_index: int) -> dict[str, str]:
    env = base.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["MARKER1_SEQUENTIAL_OFFLOAD"] = "1"
    env["MARKER1_DELAYED_OFFLOAD"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(RUNTIME_DIR), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return env


def marker2_environment(
    base: dict[str, str],
    gpu_index: int,
    llama_server: Path,
    model: Path,
    mmproj: Path,
    gpu_layers: int,
    llama_extra_args: str,
) -> dict[str, str]:
    if gpu_layers < 1:
        raise GPUUnavailable("Marker 2 requires at least one llama.cpp GPU layer.")
    env = base.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "TORCH_DEVICE": "cpu",
            "SURYA_INFERENCE_BACKEND": "llamacpp",
            "SURYA_INFERENCE_PARALLEL": "1",
            "SURYA_INFERENCE_KEEP_ALIVE": "false",
            "SURYA_GGUF_LOCAL_MODEL_PATH": str(require_file(model, "Surya GGUF")),
            "SURYA_GGUF_LOCAL_MMPROJ_PATH": str(
                require_file(mmproj, "Surya multimodal projector")
            ),
            "LLAMA_CPP_BINARY": str(require_file(llama_server, "llama-server")),
            "LLAMA_CPP_NGL": str(gpu_layers),
            "LLAMA_CPP_NO_MMPROJ_OFFLOAD": "false",
            "LLAMA_CPP_EXTRA_ARGS": llama_extra_args,
        }
    )
    return env


def marker_command(marker: str, marker_args: list[str]) -> list[str]:
    if not Path(marker).is_file() and shutil.which(marker) is None:
        raise GPUUnavailable(f"Marker executable not found: {marker}")
    if marker_args[:1] == ["--"]:
        marker_args = marker_args[1:]
    if not marker_args:
        raise GPUUnavailable("Pass the Marker input and options after a standalone --.")
    return [marker, *marker_args]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Configure and guard NVIDIA GPU execution for Marker 1 or Marker 2."
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    subparsers = parser.add_subparsers(dest="version", required=True)

    marker1 = subparsers.add_parser("marker1")
    marker1.add_argument("--marker", default="marker_single")
    marker1.add_argument("--python", type=Path, default=Path(sys.executable))
    marker1.add_argument("marker_args", nargs=argparse.REMAINDER)

    marker2 = subparsers.add_parser("marker2")
    marker2.add_argument("--marker", default="marker_single")
    marker2.add_argument("--llama-server", type=Path, required=True)
    marker2.add_argument("--model", type=Path, required=True)
    marker2.add_argument("--mmproj", type=Path, required=True)
    marker2.add_argument("--gpu-layers", type=int, default=30)
    marker2.add_argument(
        "--llama-extra-args",
        default="--threads 4 --threads-batch 4 --load-mode none",
    )
    marker2.add_argument("marker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    try:
        gpu = nvidia_gpu_name()
        command = marker_command(args.marker, args.marker_args)
        if args.version == "marker1":
            require_marker1_cuda(args.python)
            env = marker1_environment(os.environ, args.gpu_index)
            detail = "PyTorch CUDA with sequential model offload"
        else:
            version = llama_cuda_details(args.llama_server)
            env = marker2_environment(
                os.environ,
                args.gpu_index,
                args.llama_server,
                args.model,
                args.mmproj,
                args.gpu_layers,
                args.llama_extra_args,
            )
            detail = version
    except (GPUUnavailable, OSError, subprocess.TimeoutExpired) as exc:
        parser.error(f"GPU validation failed: {exc}")

    print(f"GPU: {gpu}")
    print(f"Backend: {detail}")
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

