import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gpu_launcher import (
    GPUUnavailable,
    llama_cuda_details,
    marker1_environment,
    marker2_environment,
    nvidia_gpu_name,
)


class GPULauncherTest(unittest.TestCase):
    def test_marker1_environment_enables_offload(self):
        env = marker1_environment({"PYTHONPATH": "existing"}, 1)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["MARKER1_SEQUENTIAL_OFFLOAD"], "1")
        self.assertEqual(env["MARKER1_DELAYED_OFFLOAD"], "1")
        self.assertTrue(env["PYTHONPATH"].endswith(os.pathsep + "existing"))

    def test_marker2_environment_uses_local_ggufs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            llama = root / "llama-server.exe"
            model = root / "surya-2.gguf"
            mmproj = root / "surya-2-mmproj.gguf"
            for path in (llama, model, mmproj):
                path.touch()

            env = marker2_environment({}, 0, llama, model, mmproj, 30, "--threads 4")

            self.assertEqual(env["SURYA_INFERENCE_BACKEND"], "llamacpp")
            self.assertEqual(env["SURYA_INFERENCE_KEEP_ALIVE"], "false")
            self.assertEqual(env["LLAMA_CPP_NGL"], "30")
            self.assertEqual(env["SURYA_GGUF_LOCAL_MODEL_PATH"], str(model.resolve()))
            self.assertEqual(env["SURYA_GGUF_LOCAL_MMPROJ_PATH"], str(mmproj.resolve()))

    @patch("gpu_launcher.subprocess.run")
    def test_gpu_checks_reject_missing_cuda(self, run):
        run.return_value = SimpleNamespace(returncode=1, stdout="", stderr="")
        with self.assertRaisesRegex(GPUUnavailable, "No NVIDIA GPU"):
            nvidia_gpu_name()

        with tempfile.TemporaryDirectory() as temporary:
            llama = Path(temporary) / "llama-server.exe"
            llama.touch()
            run.side_effect = [
                SimpleNamespace(returncode=0, stdout="build 10621", stderr=""),
                SimpleNamespace(returncode=0, stdout="CPU", stderr=""),
            ]
            with self.assertRaisesRegex(GPUUnavailable, "CUDA device"):
                llama_cuda_details(llama)

    @patch("gpu_launcher.subprocess.run")
    def test_llama_version_and_cuda_are_reported(self, run):
        run.side_effect = [
            SimpleNamespace(returncode=0, stdout="build 10621", stderr=""),
            SimpleNamespace(returncode=0, stdout="CUDA0: NVIDIA GPU", stderr=""),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            llama = Path(temporary) / "llama-server.exe"
            llama.touch()
            self.assertEqual(llama_cuda_details(llama), "build 10621")


if __name__ == "__main__":
    unittest.main()

