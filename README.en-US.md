

# Revoice ASR-TTS

[English](./README.en.md)

`Revoice ASR-TTS` is a real-time voice changer desktop application for Windows, built on `PySide6`, `FunASR Nano`, `IndexTTS-2`, and an experimental `emotion2vec+` emotion recognition pipeline.

## Project Overview

- Real-time microphone voice changing: Microphone input -> ASR -> TTS -> Playback output
- Text mode: Direct text input for speech synthesis
- Reference audio voice cloning: Supports selecting reference audio for runtime warm-up
- Experimental emotion recognition: Automatically recognizes microphone speech emotion and maps it to TTS emotion vectors
- Desktop UI: Features a dual-layer progress bar during startup, with a main interface optimized for real-time operation

## Core Features

- `FunASR Nano` for real-time speech recognition, defaulting to `AutoModel`
- `IndexTTS-2` for streaming speech synthesis
- `emotion2vec+` for experimental emotion recognition
- Microphone, playback, and monitor devices can be configured separately
- Supports reference audio warm-up to reduce first-sentence latency
- Built-in duplicate ASR text suppression to prevent feeding the same sentence repeatedly to TTS

## Tech Stack

- Python `3.13`
- `PySide6`
- `torch` / `torchaudio` / `torchvision`
- `sounddevice` / `soundfile` / `librosa`
- `funasr`
- `indextts`
- `modelscope`

## Project Structure

```text
.
|- app/                  # Core configuration, service layer, model wrappers
|- ui/                   # Qt UI, splash screen, worker threads
|- vendor/               # Local vendored dependencies, e.g., index-tts
|- bin/                  # Local runtime directory, must be prepared manually, not distributed with the repo
|- checkpoints/          # Model directory
`- main.py               # Python entry point
```

## Operating Principle

![Revoice ASR-TTS Principle](./asset/Principle.png)

## UI Preview

Main Interface (Initializing):

![Main Interface - Initializing](./asset/screenshots/live-main-loading.png)

Main Interface (Ready):

![Main Interface - Ready](./asset/screenshots/live-main-ready.png)

Quick Settings and Translation Node:

![Quick Settings and Translation Node](./asset/screenshots/quick-settings-translation.png)

## Environment Requirements

- OS: `Windows` recommended
- Python: `3.13.x`
- GPU: NVIDIA GPU with CUDA support recommended for better TTS performance
- Audio Environment: Prepare a working microphone, playback device, and virtual sound card or monitor device

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```


Additional runtime dependencies must be prepared manually:

- `ffmpeg`: Install to your system `PATH`, or manually place it at `bin/ffmpeg/bin/ffmpeg.exe`
  - The Windows packaging script defaults to **not** bundling the local `ffmpeg` into the distribution directory; if strictly needed, explicitly use `-IncludeBundledFfmpeg`
  - Note: If you distribute the `ffmpeg` binaries bundled with this project, handle them according to their own licenses, not the root `MIT` license of this repository
  - The current local `bin/ffmpeg` example is `FFmpeg 7.1.1 full_build (gyan.dev)`, with its accompanying `LICENSE` stating `GPL v3`
- `flash_attn`: Optional, used only for the experimental ultra-fast GPT path; to enable, download the version matching your environment from this project's GitHub Releases and install manually

### 2. Prepare Models

The following directories are used by default:

- `checkpoints/IndexTTS-2`
- `checkpoints/emotion2vec_plus_base`
- `checkpoints/Fun-ASR-Nano-2512`
- `checkpoints/speech_campplus_sv_zh-cn_16k-common`

The application will currently check and automatically prepare the following `IndexTTS-2` models at startup:

- `IndexTeam/IndexTTS-2`
- `xmccln/IndexTTS-2-Japanese`

The default repository in use is `IndexTeam/IndexTTS-2`. If you switch to another IndexTTS-2 repository without explicitly setting a directory, the program will automatically download the models to `checkpoints/<repo-name>`. For example, the Japanese model `xmccln/IndexTTS-2-Japanese` will default to `checkpoints/IndexTTS-2-Japanese`.

The ASR model will automatically prepare `FunAudioLLM/Fun-ASR-Nano-2512` during the startup phase, defaulting to:

- `checkpoints/Fun-ASR-Nano-2512`

Speaker filtering in Live mode now defaults to using `iic/speech_campplus_sv_zh-cn_16k-common` to extract speaker embeddings, and the model will also be automatically prepared to:

- `checkpoints/speech_campplus_sv_zh-cn_16k-common`



You can now also modify the following directly within the app via "Quick Settings" in the top-right corner:

- Current output language
- TTS model list to check/download at startup

Post-save behavior falls into two categories:

- If the change alters the currently used `TTS` repository or modifies the pre-download model list, the app will prompt for a restart. Upon exit, it will re-check and automatically prepare models according to the new configuration on the next launch.
- If only the current output language is changed without altering the underlying `TTS` repository, the app will not exit; if the microphone is active, it will automatically stop it to refresh the current settings.

To customize the download directory, it is recommended to specify it in `.env`:

```dotenv
INDEX_TTS_MODEL_REPO=xmccln/IndexTTS-2-Japanese
INDEX_TTS_MODEL_DIR=D:\models\IndexTTS-2-Japanese
```

Or temporarily specify:

```powershell
$env:INDEX_TTS_MODEL_REPO = "xmccln/IndexTTS-2-Japanese"
$env:INDEX_TTS_MODEL_DIR = "D:\models\IndexTTS-2-Japanese"
uv run python main.py
```

Device-related initialization is also recommended in `.env`:

```dotenv
TTS_DEVICE=cuda
ASR_DEVICE=cpu
EMOTION_DEVICE=cpu
```

For temporary overrides, you can still write:

```powershell
$env:TTS_DEVICE = "cuda"
$env:ASR_DEVICE = "cpu"
$env:EMOTION_DEVICE = "cpu"
```

### 3. Launch the Application

```bash
uv run python main.py
```

## Windows Packaging

The current Windows distribution no longer uses `Nuitka` or `PyInstaller`, but instead:

1. Uses [build.ps1](./packaging/windows/build.ps1) to compile local `app`, `ui`, and other code into `.pyd` files and assemble the runtime directory
2. Uses [build-installer.ps1](./packaging/windows/build-installer.ps1) to build an installer based on the `dist/` artifacts

The startup script in the runtime directory will automatically execute `uv sync --frozen` before launching `main.py`.

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\windows\build.ps1
powershell -ExecutionPolicy Bypass -File .\packaging\windows\build-installer.ps1
```

Notes:

- The packaging script defaults to not copying the local `bin/ffmpeg/`; users should install `ffmpeg` themselves, or explicitly use `-IncludeBundledFfmpeg` if absolutely necessary
- Packaged artifacts do not include sample audio from `asset/`; please prepare your own reference audio for use
- The repository also does not distribute local runtime contents like `bin/`, `checkpoints/`, etc.

## Usage Instructions

### Microphone Mode

1. Select reference audio
2. Select recording, playback, and monitor devices
3. Wait for models and reference voice warm-up to complete
4. Click "Start Voice Changing"

### Text Mode

1. Switch to text mode
2. Enter the text to synthesize
3. Wait for the button on the right to switch from "Preparing" to "Start Synthesis"
4. Begin streaming synthesis

## Model Notes

- Model existence is checked at startup
- At startup, the program uniformly checks and automatically prepares `Fun-ASR-Nano-2512`, `speech_campplus_sv_zh-cn_16k-common`, all built-in available `IndexTTS-2` models, and `emotion2vec+`
- The splash screen will detail the specific `IndexTTS-2` models and their associated resources currently being checked or downloaded
- `IndexTTS-2` currently supports switching between repositories via `INDEX_TTS_MODEL_REPO`, such as third-party Japanese models
- `Fun-ASR-Nano-2512` is automatically prepared via `ModelScope` by default, but can be switched to a local directory using environment variables
- Voiceprint filtering in Live mode has been switched to the `CAMPPlus speaker verification` model, no longer relying on custom voiceprint extraction on the TTS side
- The current public source repository does not include the `bin/` local runtime directory
- In packaged environments, `ModelScope`-related modules are explicitly imported in the code to avoid registry missing issues

## Known Notes

- This project is primarily designed for Windows desktop real-time use cases
- Title bar theme linking is only supported on Windows
- TTS default device allocation is `cuda`, ASR defaults to `cpu`, and emotion recognition defaults to `cpu`
- The first launch or initial warm-up may be slow
- The project has many dependencies; using `uv` to manage the environment is recommended

## Open Source Notice

This repository aggregates multiple third-party models and dependencies. Please verify the license terms of the relevant models, weights, datasets, and upstream projects before distribution or commercial use.

## Citations & Acknowledgments

### ASR

```bibtex
@inproceedings{gao2023funasr,
  author={Zhifu Gao and others},
  title={FunASR: A Fundamental End-to-End Speech Recognition Toolkit},
  booktitle={INTERSPEECH},
  year={2023}
}
```

### IndexTTS-2

```bibtex
@article{zhou2025indextts2,
  title={IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech},
  author={Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, Jingchen Shu},
  journal={arXiv preprint arXiv:2506.21619},
  year={2025}
}
```

### Emotion Recognition

```bibtex
@article{ma2023emotion2vec,
  title={emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation},
  author={Ma, Ziyang and Zheng, Zhisheng and Ye, Jiaxin and Li, Jinchao and Gao, Zhifu and Zhang, Shiliang and Chen, Xie},
  journal={arXiv preprint arXiv:2312.15185},
  year={2023}
}
```

## License

- Repository's own code defaults to the [MIT License](./LICENSE)
- Third-party vendored code, binaries, models, and other non-original content are not covered by the root MIT license
- If you distribute `bin/ffmpeg/` alongside the project, fulfill the corresponding obligations according to the license included in that directory; the current local example is a `GPL v3` build
- For details, see [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)
