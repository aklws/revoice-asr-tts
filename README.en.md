# Revoice ASR-TTS

[中文](./README.md)

`Revoice ASR-TTS` is a Windows-first desktop voice conversion app built with `PySide6`, `Qwen3 ASR`, `IndexTTS-2`, and an experimental `emotion2vec+` emotion recognition pipeline.

## Overview

- Real-time microphone voice conversion: microphone input -> ASR -> TTS -> playback
- Text mode: enter text directly and synthesize speech
- Reference-audio voice cloning with runtime warm-up
- Experimental emotion recognition that maps detected emotion to TTS emotion vectors
- Desktop UI with dual startup progress bars and controls optimized for live use

## Features

- `Qwen3 ASR` for real-time speech recognition
- `IndexTTS-2` for streaming speech synthesis
- `emotion2vec+` + `ModelScope`/`FunASR` for experimental emotion recognition
- Separate input, playback, and monitor device selection
- Reference-audio warm-up to reduce first-utterance latency
- Built-in duplicate ASR suppression to avoid sending the same sentence to TTS repeatedly

## Tech Stack

- Python `3.13`
- `PySide6`
- `torch` / `torchaudio` / `torchvision`
- `sounddevice` / `soundfile` / `librosa`
- `qwen-asr`
- `indextts`
- `modelscope`
- `funasr`

## Project Structure

```text
.
|- app/                  # Core config, services, and model wrappers
|- ui/                   # Qt UI, startup screen, worker threads
|- vendor/               # Vendored dependencies such as index-tts
|- bin/                  # Local runtime directory prepared manually, not distributed in this repo
|- checkpoints/          # Model directories
`- main.py               # Python entry point
```

## Requirements

- OS: `Windows` recommended
- Python: `3.13.x`
- GPU: NVIDIA GPU with CUDA is recommended for better TTS performance
- Audio setup: a working microphone, playback device, and optionally a virtual cable or monitor device

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

Prepare the extra runtime dependencies manually:

- `ffmpeg`: install it into your system `PATH`, or place it at `bin/ffmpeg/bin/ffmpeg.exe`
- `flash_attn`: download a build that matches your environment from this project's GitHub Releases and install it manually

### 2. Prepare Models

The app uses the following default directories:

- `checkpoints/IndexTTS-2`
- `checkpoints/emotion2vec_plus_base`
- `bin/Qwen3-ASR-Transcribe/model`

`Qwen3 ASR` runtime assets are not included in this public repository. Prepare the local directory manually:

1. Create `bin/Qwen3-ASR-Transcribe/model`
2. Download [Qwen3-ASR-0.6B-gguf.zip](https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/download/models/Qwen3-ASR-0.6B-gguf.zip)
3. Extract the archive contents into `bin/Qwen3-ASR-Transcribe/model`

You can also override them with environment variables:

```powershell
$env:INDEX_TTS_MODEL_DIR = "D:\path\to\IndexTTS-2"
$env:QWEN_ASR_MODEL_DIR = "D:\path\to\Qwen3-ASR-0.6B"
$env:EMOTION_MODEL_DIR = "D:\path\to\emotion2vec_plus_base"
```

Device-related environment variables:

```powershell
$env:TTS_DEVICE = "cuda"
$env:ASR_DEVICE = "cpu"
$env:EMOTION_DEVICE = "cpu"
```

### 3. Run The App

```bash
uv run python main.py
```

## Usage

### Microphone Mode

1. Select a reference audio
2. Choose input, playback, and monitor devices
3. Wait until model loading and reference warm-up are complete
4. Click `开始变声`

### Text Mode

1. Switch to text mode
2. Enter the text to synthesize
3. Wait until the button changes from `准备中` to `开始合成`
4. Start streaming synthesis

## Model Notes

- Models are checked during startup
- `IndexTTS-2` and `emotion2vec+` can be prepared automatically when missing
- `Qwen3 ASR` runtime assets must be placed manually in `bin/Qwen3-ASR-Transcribe/model`
- This public source repository does not include the local `bin/` runtime directory
- In packaged builds, `ModelScope` modules are explicitly imported in code to avoid registry loss issues

## Notes

- The project is primarily designed for Windows desktop real-time usage
- Title-bar theme syncing is only supported on Windows
- TTS defaults to `cuda`, while ASR and emotion recognition default to `cpu`
- First launch or first warm-up may take noticeable time
- The dependency stack is heavy, so using `uv` is recommended

## Open Source Note

This repository integrates multiple third-party models and dependencies. Before redistribution or commercial use, please review the licenses of the related models, weights, datasets, and upstream projects.

## Acknowledgements And Citations

### ASR

- Qwen3 ASR GGUF: [HaujetZhao/Qwen3-ASR-GGUF](https://github.com/HaujetZhao/Qwen3-ASR-GGUF)

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

- The original code authored for this repository is released under the [MIT License](./LICENSE)
- Vendored third-party code, binaries, models, and other non-original assets are not covered by the root MIT license
- See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) for details
