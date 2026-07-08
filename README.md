# Revoice ASR-TTS

[English](./README.en.md)

`Revoice ASR-TTS` 是一个面向 Windows 的实时变声器桌面应用，基于 `PySide6`、`Qwen3 ASR`、`IndexTTS-2` 和实验性的 `emotion2vec+` 情感识别流程构建。

## 项目简介

- 实时麦克风变声：麦克风输入 -> ASR -> TTS -> 播放输出
- 文本模式：直接输入文本并进行语音合成
- 参考音频音色克隆：支持选择参考音频并进行运行时预热
- 实验性情感识别：自动识别麦克风语音情绪并映射到 TTS 情感向量
- 桌面 UI：启动阶段带有双层进度条，主界面适合实时操作

## 核心特性

- `Qwen3 ASR` 用于实时语音识别
- `IndexTTS-2` 用于流式语音合成
- `emotion2vec+` + `ModelScope`/`FunASR` 用于实验性情感识别
- 麦克风、播放、耳返设备可分别配置
- 支持参考音频预热，减少首句延迟
- 内置重复 ASR 文本抑制，避免同一句连续送入 TTS

## 技术栈

- Python `3.13`
- `PySide6`
- `torch` / `torchaudio` / `torchvision`
- `sounddevice` / `soundfile` / `librosa`
- `qwen-asr`
- `indextts`
- `modelscope`
- `funasr`

## 项目结构

```text
.
|- app/                  # 核心配置、服务层、模型封装
|- ui/                   # Qt 界面、启动页、工作线程
|- vendor/               # 本地 vendor 依赖，例如 index-tts
|- bin/                  # 本地运行时目录，需手动准备，不随仓库分发
|- checkpoints/          # 模型目录
`- main.py               # Python 入口
```

## 环境要求

- 操作系统：推荐 `Windows`
- Python：`3.13.x`
- GPU：推荐使用支持 CUDA 的 NVIDIA GPU 以获得更好的 TTS 性能
- 音频环境：建议准备可用的麦克风、播放设备，以及虚拟声卡或耳返设备

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

额外运行时依赖请手动准备：

- `ffmpeg`：请自行安装到系统 `PATH`，或手动放到 `bin/ffmpeg/bin/ffmpeg.exe`
- `flash_attn`：请从本项目 GitHub Releases 下载与你环境匹配的版本后手动安装

### 2. 准备模型

默认会使用以下目录：

- `checkpoints/IndexTTS-2`
- `checkpoints/emotion2vec_plus_base`
- `bin/Qwen3-ASR-Transcribe/model`

其中 `Qwen3 ASR` 运行时资源不会随本仓库一起上传，请手动准备本地目录：

1. 创建目录 `bin/Qwen3-ASR-Transcribe/model`
2. 下载 [Qwen3-ASR-0.6B-gguf.zip](https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/download/models/Qwen3-ASR-0.6B-gguf.zip)
3. 将压缩包内容解压到 `bin/Qwen3-ASR-Transcribe/model`

也可以通过环境变量覆盖：

```powershell
$env:INDEX_TTS_MODEL_DIR = "D:\path\to\IndexTTS-2"
$env:QWEN_ASR_MODEL_DIR = "D:\path\to\Qwen3-ASR-0.6B"
$env:EMOTION_MODEL_DIR = "D:\path\to\emotion2vec_plus_base"
```

设备相关环境变量：

```powershell
$env:TTS_DEVICE = "cuda"
$env:ASR_DEVICE = "cpu"
$env:EMOTION_DEVICE = "cpu"
```

### 3. 启动应用

```bash
uv run python main.py
```

## 使用说明

### 麦克风模式

1. 选择参考音频
2. 选择录音、播放和耳返设备
3. 等待模型和参考音色预热完成
4. 点击“开始变声”

### 文本模式

1. 切换到文本模式
2. 输入要合成的文本
3. 等待右侧按钮从“准备中”切换到“开始合成”
4. 开始流式合成

## 模型说明

- 启动时会检查模型是否存在
- `IndexTTS-2` 和 `emotion2vec+` 缺失时支持通过 `ModelScope` 自动准备
- `Qwen3 ASR` 运行时资源需手动放到 `bin/Qwen3-ASR-Transcribe/model`
- 当前公开源码仓库不包含 `bin/` 本地运行时目录
- 打包环境下，`ModelScope` 相关模块会在代码中显式导入，以避免注册表缺失问题

## 已知说明

- 当前项目以 Windows 桌面实时使用场景为主
- 标题栏主题联动仅在 Windows 下支持
- TTS 默认设备分配为 `cuda`，ASR 和情感识别默认使用 `cpu`
- 首次启动或首次预热可能较慢
- 项目依赖较多，建议使用 `uv` 管理环境

## 开源说明

本仓库聚合了多个第三方模型与依赖，请在分发或商用前自行核对相关模型、权重、数据集与上游项目的许可证条款。

## 引用与致谢

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

### 情感识别

```bibtex
@article{ma2023emotion2vec,
  title={emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation},
  author={Ma, Ziyang and Zheng, Zhisheng and Ye, Jiaxin and Li, Jinchao and Gao, Zhifu and Zhang, Shiliang and Chen, Xie},
  journal={arXiv preprint arXiv:2312.15185},
  year={2023}
}
```

## 许可证

- 仓库自有代码默认采用 [MIT License](./LICENSE)
- 第三方 vendored 代码、二进制、模型与其他非自有内容不受根 MIT 许可证覆盖
- 详细说明见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)
