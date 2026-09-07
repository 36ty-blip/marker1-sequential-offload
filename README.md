# Sequential CPU/GPU model offloading for Marker 1.10.2

Run Marker 1.10.2 on a low-VRAM NVIDIA GPU by rotating its five predictors through CUDA instead of keeping all five in VRAM simultaneously.

The technique is **sequential CPU↔GPU model offloading**, also known as just-in-time model swapping. It is independent of page-by-page processing and can be used with a whole-document Marker 1.10.2 conversion.

This implementation is only for the five-model Marker 1.x architecture. It does not apply directly to Marker 2 or current Marker `master`, which use a newer VLM architecture.

## Result

On an RTX 2050 with 4 GB VRAM, Marker 1.10.2 was observed at roughly 2.1–2.2 GB VRAM with fp16 weights and OCR enabled. The tradeoff is additional system RAM and model-transfer latency.

## How it works

Marker 1.10.x creates five predictors:

- `layout_model`
- `recognition_model`
- `table_rec_model`
- `detection_model`
- `ocr_error_model`

`marker1_runtime/sitecustomize.py` loads them on the CPU. Immediately before a predictor runs, the wrapper moves it to CUDA. When the next predictor is needed, the previous one moves back to CPU and its transient cache is cleared.

## Install

Pin the tested Marker version in an environment with CUDA-enabled PyTorch:

```console
python -m pip install "marker-pdf==1.10.2"
```

Add `marker1_runtime` to `PYTHONPATH` and enable the hook in the environment that launches `marker_single`:

```console
PYTHONPATH=/path/to/this-repo/marker1_runtime
MARKER1_SEQUENTIAL_OFFLOAD=1
MARKER1_DELAYED_OFFLOAD=1
```

PowerShell example:

```powershell
$env:PYTHONPATH = (Resolve-Path .\marker1_runtime)
$env:MARKER1_SEQUENTIAL_OFFLOAD = "1"
$env:MARKER1_DELAYED_OFFLOAD = "1"
marker_single input.pdf --output_dir output --layout_batch_size 1 --detection_batch_size 1 --ocr_error_batch_size 1 --recognition_batch_size 1 --equation_batch_size 1 --table_rec_batch_size 1
```

Python automatically imports `sitecustomize.py` from `PYTHONPATH` when each Marker process starts.

## Verify

Run the dependency-free wrapper check:

```console
python -m unittest -v test_sitecustomize.py
```

The final validation should be a real conversion with Marker 1.10.2 and an NVIDIA GPU.

## License

MIT. This is an independent reference implementation and is not affiliated with Datalab or the Marker project.

