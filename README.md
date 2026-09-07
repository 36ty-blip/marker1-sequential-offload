# Low-VRAM GPU setup for Marker 1.10 and Marker 2

This repository provides two separate NVIDIA GPU paths:

- **Marker 1.10.2:** rotate its five PyTorch predictors between RAM and CUDA
  instead of keeping every model in VRAM.
- **Marker 2:** run the Surya 2 GGUF model locally through a CUDA-enabled
  llama.cpp server.

`gpu_launcher.py` validates the selected backend before Marker starts. It stops
with an error when CUDA is unavailable instead of silently processing on the CPU.

The technique is **sequential CPU↔GPU model offloading**, also known as just-in-time model swapping. It is independent of page-by-page processing and can be used with a whole-document Marker 1.10.2 conversion.

The sequential offload hook is only for the five-model Marker 1.x architecture.
Marker 2 does **not** use that hook; it uses the separate llama.cpp configuration
documented below.

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

## Marker 1 installation

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

The guarded launcher can configure those variables and check PyTorch CUDA for
you. Run it with the same Python environment that contains Marker 1:

```console
python gpu_launcher.py marker1 --marker /path/to/marker_single --python /path/to/python -- input.pdf --output_dir output --layout_batch_size 1 --detection_batch_size 1 --ocr_error_batch_size 1 --recognition_batch_size 1 --equation_batch_size 1 --table_rec_batch_size 1
```

## Marker 2 with local llama.cpp and GGUF files

Marker 2 uses llama.cpp; Marker 1 does not. The configuration tested for this
project is:

- llama.cpp **b10621**, commit **c1d0e7a00**;
- Windows x64 CUDA 12.4 build;
- `surya-2.gguf` and `surya-2-mmproj.gguf` from Datalab's
  [`surya-ocr-2-gguf`](https://huggingface.co/datalab-to/surya-ocr-2-gguf)
  repository;
- 30 llama.cpp GPU layers and one inference slot.

Newer llama.cpp builds may work, but b10621 is the exact version used for the
reported setup.

### 1. Download llama.cpp on Windows

Open the [llama.cpp b10621 release](https://github.com/ggml-org/llama.cpp/releases/tag/b10621)
and download both CUDA 12.4 archives:

- `llama-b10621-bin-win-cuda-12.4-x64.zip`
- `cudart-b10621-cuda-12.4-x64.zip`

Extract both archives into the same folder. Confirm that the build and CUDA
device are visible:

```console
llama-server.exe --version
llama-server.exe --list-devices
```

The first command should report build 10621. The second must list a CUDA device.

### 2. Download the Surya GGUF files

Install the Hugging Face command-line client, then download both required files:

```console
python -m pip install --upgrade huggingface_hub
hf download datalab-to/surya-ocr-2-gguf surya-2.gguf surya-2-mmproj.gguf --local-dir models
```

Both files are required: `surya-2.gguf` contains the model and
`surya-2-mmproj.gguf` is its vision projector. They can also be downloaded
manually from the [official model files](https://huggingface.co/datalab-to/surya-ocr-2-gguf/tree/main).

### 3. Launch Marker 2 with GPU safeguards

```console
python gpu_launcher.py marker2 --marker /path/to/marker_single --llama-server /path/to/llama-server.exe --model models/surya-2.gguf --mmproj models/surya-2-mmproj.gguf --gpu-layers 30 -- input.pdf --output_dir output --mode balanced
```

The launcher supplies Marker 2's local Surya and llama.cpp environment,
verifies `nvidia-smi`, checks that llama.cpp can see CUDA, checks both GGUF
paths, and disables server keep-alive so the model releases its resources after
the conversion.

Use `--gpu-index N` before `marker1` or `marker2` to select another NVIDIA GPU.

## Verify

Run the dependency-free checks:

```console
python -m unittest -v test_sitecustomize.py test_gpu_launcher.py
```

The final validation should be a real conversion with the selected Marker
version and an NVIDIA GPU.

## License

MIT. This is an independent reference implementation and is not affiliated with Datalab or the Marker project.

