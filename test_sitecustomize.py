import os
import runpy
import sys
import types
import unittest
from pathlib import Path


class Tensor:
    def __init__(self):
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self


class Foundation:
    def __init__(self):
        self.device = "cpu"
        self.device_pad_token = Tensor()
        self.device_beacon_token = Tensor()
        self.kv_cache = object()
        self.prompt_queue = ["pending"]
        self.batch_prompt_mapping = {1: 1}

    def to(self, device):
        self.device = device


class Predictor:
    def __init__(self, foundation=False):
        self.device = "cpu"
        if foundation:
            self.foundation_predictor = Foundation()

    def to(self, device):
        self.device = device

    def __call__(self):
        return "ok"


class SiteCustomizeTest(unittest.TestCase):
    def test_rotates_predictors_and_clears_foundation_cache(self):
        torch = types.ModuleType("torch")
        torch.float16 = "float16"
        torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
        marker = types.ModuleType("marker")
        marker.__path__ = []
        models_module = types.ModuleType("marker.models")
        first = Predictor(foundation=True)
        second = Predictor()

        def create_model_dict(**kwargs):
            self.assertEqual(kwargs["device"], "cpu")
            self.assertEqual(kwargs["dtype"], "float16")
            return {"layout_model": first, "detection_model": second}

        models_module.create_model_dict = create_model_dict
        marker.models = models_module
        saved_modules = {
            name: sys.modules.get(name) for name in ("torch", "marker", "marker.models")
        }
        saved_environment = {
            name: os.environ.get(name)
            for name in ("MARKER1_SEQUENTIAL_OFFLOAD", "MARKER1_DELAYED_OFFLOAD")
        }
        try:
            sys.modules.update(
                {"torch": torch, "marker": marker, "marker.models": models_module}
            )
            os.environ["MARKER1_SEQUENTIAL_OFFLOAD"] = "1"
            os.environ["MARKER1_DELAYED_OFFLOAD"] = "1"
            runpy.run_path(
                Path(__file__).parent / "marker1_runtime" / "sitecustomize.py"
            )

            wrapped = models_module.create_model_dict()
            self.assertEqual(wrapped["layout_model"](), "ok")
            self.assertEqual(first.foundation_predictor.device, "cuda")
            self.assertEqual(wrapped["detection_model"](), "ok")
            foundation = first.foundation_predictor
            self.assertEqual(foundation.device, "cpu")
            self.assertEqual(foundation.device_pad_token.device, "cpu")
            self.assertIsNone(foundation.kv_cache)
            self.assertEqual(foundation.prompt_queue, [])
            self.assertIsNone(foundation.batch_prompt_mapping)
            self.assertEqual(second.device, "cuda")
        finally:
            for name, value in saved_modules.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
            for name, value in saved_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()

