# Revoice ASR-TTS Usage

This project is organized for direct Python calls between modules and features a real-time UI. It uses `FunASR Nano` for speech-to-text, `IndexTTS-2` for text-to-speech, and `CAMPPlus speaker verification` for live voiceprint filtering.

## Directory Recommendation

Place the model bundle in one of these ways:

1. Recommended TTS path: `d:\work\pywork\checkpoints\IndexTTS-2`
2. Recommended ASR path: `d:\work\pywork\checkpoints\Fun-ASR-Nano-2512`
3. Recommended speaker path: `d:\work\pywork\checkpoints\speech_campplus_sv_zh-cn_16k-common`
4. Or set `INDEX_TTS_MODEL_DIR` / `FUNASR_ASR_MODEL_DIR` / `SPEAKER_MODEL_DIR` to the real model directories

## Project Structure

```text
d:\work\pywork
|- app
|  |- core
|  |  |- config.py
|  |  |- logging.py
|  |  `- runtime.py
|  |- models
|  |  |- asr.py
|  |  `- ...
|  |- services
|  |  |- asr.py
|  |  |- index_tts.py
|  |  `- ...
|  `- __init__.py
|- checkpoints
|  |- Fun-ASR-Nano-2512
|  |- IndexTTS-2
|  `- speech_campplus_sv_zh-cn_16k-common
|- ui
|  |- app.py
|  |- main_window.py
|  |- workers.py
|  `- ...
|- docs
|  `- USAGE.md
`- main.py
```

## 1. Install With uv

```bash
uv sync
```

This project resolves PyTorch packages from the PyTorch CUDA index via `uv`. The default ASR path uses `funasr.AutoModel`.

## 2. Configure The Model Path

PowerShell:

```powershell
$env:INDEX_TTS_MODEL_DIR = "d:\work\pywork\checkpoints\IndexTTS-2"
$env:FUNASR_ASR_MODEL_DIR = "d:\work\pywork\checkpoints\Fun-ASR-Nano-2512"
$env:SPEAKER_MODEL_DIR = "d:\work\pywork\checkpoints\speech_campplus_sv_zh-cn_16k-common"
```

If you keep the model in the recommended default path above, this step is optional.

## 3. Usage

Run the main application:

```bash
uv run python main.py
```

## 4. Optional Runtime Settings

Recommended `.env` entries:

```dotenv
TTS_DEVICE=cuda
ASR_DEVICE=cpu
EMOTION_DEVICE=cpu
```

The packaged Windows launcher also runs `uv sync --frozen` before starting the app, so keeping `uv.lock` in sync with the repository is recommended.

In the top-right quick settings:

- `当前输出语言` now supports `中文` / `英文` / `日语`
- `中文` and `英文` share the official `IndexTTS-2` base model
- `日语` uses the Japanese model and requires the corresponding model bundle to be prepared
- `翻译节点` supports OpenAI-compatible endpoints such as `OpenAI` and `DeepSeek`
- After configuring a translation node and enabling translation, the app will translate text to the selected output language before sending it to TTS

## Notes

- The generated audio is stereo at `48000 Hz`.
- The default low-memory device split is `FunASR -> cpu`, `IndexTTS-2 -> cuda`, and `emotion/speaker -> cpu`.
- The default project layout uses `d:\work\pywork\checkpoints\Fun-ASR-Nano-2512`, `d:\work\pywork\checkpoints\IndexTTS-2`, and `d:\work\pywork\checkpoints\speech_campplus_sv_zh-cn_16k-common`.
- Output language and model are now separated:
  - `中文` -> official model
  - `英文` -> official model
  - `日语` -> Japanese model
- If you only switch between `中文` and `英文`, a restart is typically not required.
- Translation nodes use OpenAI-compatible `/chat/completions` APIs. You need to provide `Base URL`, `model`, and `API key`.
- Live voiceprint filtering no longer depends on the TTS-side custom voiceprint extractor. It now uses the dedicated `iic/speech_campplus_sv_zh-cn_16k-common` speaker model.
- Packaged Windows builds use the current Cython bundle + installer flow described in [PACKAGING.md](./PACKAGING.md).
