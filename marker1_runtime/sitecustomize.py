"""Sequential CPU/GPU model offloading for marker-pdf 1.10.2."""

import os


if os.environ.get("MARKER1_SEQUENTIAL_OFFLOAD") == "1":
    import torch
    import marker.models as marker_models

    _original_create_model_dict = marker_models.create_model_dict
    _delayed_offload = os.environ.get("MARKER1_DELAYED_OFFLOAD") == "1"
    _active_predictor = None
    _active_name = None

    def _move_predictor(predictor, device):
        foundation = getattr(predictor, "foundation_predictor", None)
        if foundation is None:
            predictor.to(device)
            return

        foundation.to(device)
        foundation.device_pad_token = foundation.device_pad_token.to(device)
        foundation.device_beacon_token = foundation.device_beacon_token.to(device)
        if device == "cpu":
            foundation.kv_cache = None
            foundation.prompt_queue.clear()
            foundation.batch_prompt_mapping = None

    def _activate_predictor(name, predictor):
        global _active_name, _active_predictor
        if _active_predictor is predictor:
            return
        if _active_predictor is not None:
            _move_predictor(_active_predictor, "cpu")
            print(f"VRAM -> RAM: {_active_name}", flush=True)
            _active_name = None
            _active_predictor = None
            torch.cuda.empty_cache()
        print(f"RAM -> VRAM: {name}", flush=True)
        _move_predictor(predictor, "cuda")
        _active_name = name
        _active_predictor = predictor

    class _OffloadedPredictor:
        def __init__(self, name, predictor):
            object.__setattr__(self, "_name", name)
            object.__setattr__(self, "_predictor", predictor)

        def __getattr__(self, name):
            return getattr(self._predictor, name)

        def __setattr__(self, name, value):
            if name.startswith("_"):
                object.__setattr__(self, name, value)
            else:
                setattr(self._predictor, name, value)

        def __call__(self, *args, **kwargs):
            if _delayed_offload:
                _activate_predictor(self._name, self._predictor)
                return self._predictor(*args, **kwargs)

            print(f"RAM -> VRAM: {self._name}", flush=True)
            _move_predictor(self._predictor, "cuda")
            try:
                return self._predictor(*args, **kwargs)
            finally:
                _move_predictor(self._predictor, "cpu")
                torch.cuda.empty_cache()
                print(f"VRAM -> RAM: {self._name}", flush=True)

    def _create_offloaded_model_dict(
        device=None, dtype=None, attention_implementation=None
    ):
        models = _original_create_model_dict(
            device="cpu",
            dtype=torch.float16 if dtype is None else dtype,
            attention_implementation=attention_implementation,
        )
        print("Marker 1.10 sequential CPU/GPU model offloading enabled", flush=True)
        return {
            name: _OffloadedPredictor(name, predictor)
            for name, predictor in models.items()
        }

    marker_models.create_model_dict = _create_offloaded_model_dict

