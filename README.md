# Revoice ASR-TTS

[English](./README.en.md)

`Revoice ASR-TTS` 是一个面向 Windows 的实时变声器桌面应用，基于 `PySide6`、`FunASR Nano`、`IndexTTS-2` 和实验性的 `emotion2vec+` 情感识别流程构建。

## 项目简介

- 实时麦克风变声：麦克风输入 -> ASR -> TTS -> 播放输出
- 文本模式：直接输入文本并进行语音合成
- 参考音频音色克隆：支持选择参考音频并进行运行时预热
- 实验性情感识别：自动识别麦克风语音情绪并映射到 TTS 情感向量
- 桌面 UI：启动阶段带有双层进度条，主界面适合实时操作

## 核心特性

- `FunASR Nano` 用于实时语音识别，默认走 `AutoModel`
- `IndexTTS-2` 用于流式语音合成
- `emotion2vec+` 用于实验性情感识别
- 麦克风、播放、耳返设备可分别配置
- 支持参考音频预热，减少首句延迟
- 内置重复 ASR 文本抑制，避免同一句连续送入 TTS

## 技术栈

- Python `3.13`
- `PySide6`
- `torch` / `torchaudio` / `torchvision`
- `sounddevice` / `soundfile` / `librosa`
- `funasr`
- `indextts`
- `modelscope`

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

## 运行原理

![Revoice ASR-TTS Principle](./asset/Principle.png)

## 界面预览

主界面（初始化中）：

![主界面-初始化中](./asset/screenshots/live-main-loading.png)

主界面（就绪）：

![主界面-就绪](./asset/screenshots/live-main-ready.png)

快速设置与翻译节点：

![快速设置与翻译节点](./asset/screenshots/quick-settings-translation.png)

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
  - Windows 打包脚本默认**不会**把本地 `ffmpeg` 一起打进分发目录；如确有需要，可显式使用 `-IncludeBundledFfmpeg`
  - 注意：如果你分发自己附带的 `ffmpeg` 二进制，请按该二进制自己的许可证处理，而不是按本仓库根 `MIT` 许可证处理
  - 当前本地 `bin/ffmpeg` 示例为 `FFmpeg 7.1.1 full_build (gyan.dev)`，其随附 `LICENSE` 显示为 `GPL v3`
- `flash_attn`：可选，仅在实验性极速 GPT 路径下使用；如需启用，请从本项目 GitHub Releases 下载与你环境匹配的版本后手动安装

### 2. 准备模型

默认会使用以下目录：

- `checkpoints/IndexTTS-2`
- `checkpoints/emotion2vec_plus_base`
- `checkpoints/Fun-ASR-Nano-2512`
- `checkpoints/speech_campplus_sv_zh-cn_16k-common`

当前内置会在程序启动时统一检查并自动准备以下 `IndexTTS-2` 模型：

- `IndexTeam/IndexTTS-2`
- `xmccln/IndexTTS-2-Japanese`

默认当前使用的仓库为 `IndexTeam/IndexTTS-2`。如果切换到其他 IndexTTS-2 仓库而未显式设置目录，程序会自动把模型下载到 `checkpoints/<仓库名>`，例如日语模型 `xmccln/IndexTTS-2-Japanese` 会默认落到 `checkpoints/IndexTTS-2-Japanese`。

其中 ASR 默认会在启动阶段自动准备 `FunAudioLLM/Fun-ASR-Nano-2512`，默认落到：

- `checkpoints/Fun-ASR-Nano-2512`

Live 模式的说话人筛选现在默认使用 `iic/speech_campplus_sv_zh-cn_16k-common` 提取 speaker embedding，模型也会自动准备到：

- `checkpoints/speech_campplus_sv_zh-cn_16k-common`




现在也可以在程序内通过“右上角快速设置”直接修改：

- 当前输出语言
- 启动时要统一检查/下载的 `TTS` 模型清单

保存后的行为分两类：

- 如果修改导致当前使用的 `TTS` 仓库发生变化，或者修改了预下载模型清单，程序会提示需要重启，并在退出后于下次启动时按新配置重新检查和自动准备模型。
- 如果只是修改当前输出语言，但底层 `TTS` 仓库没有变化，程序不会直接退出；如果此时麦克风正在运行，会自动关闭麦克风以刷新当前设置。

如需自定义落盘目录，也推荐写到 `.env`：

```dotenv
INDEX_TTS_MODEL_REPO=xmccln/IndexTTS-2-Japanese
INDEX_TTS_MODEL_DIR=D:\models\IndexTTS-2-Japanese
```

或者临时指定：

```powershell
$env:INDEX_TTS_MODEL_REPO = "xmccln/IndexTTS-2-Japanese"
$env:INDEX_TTS_MODEL_DIR = "D:\models\IndexTTS-2-Japanese"
uv run python main.py
```

设备相关初始化也建议写在 `.env`：

```dotenv
TTS_DEVICE=cuda
ASR_DEVICE=cpu
EMOTION_DEVICE=cpu
```

临时覆盖时仍然可以这样写：

```powershell
$env:TTS_DEVICE = "cuda"
$env:ASR_DEVICE = "cpu"
$env:EMOTION_DEVICE = "cpu"
```

### 3. 启动应用

```bash
uv run python main.py
```

## Windows 打包

当前 Windows 分发方案不再使用 `Nuitka` 或 `PyInstaller`，而是：

1. 使用 [build.ps1](./packaging/windows/build.ps1) 将本地 `app`、`ui` 等代码编译为 `.pyd` 并组装运行目录
2. 使用 [build-installer.ps1](./packaging/windows/build-installer.ps1) 基于 `dist/` 产物构建安装包

运行目录中的启动脚本会自动执行 `uv sync --frozen`，然后再启动 `main.py`。

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\windows\build.ps1
powershell -ExecutionPolicy Bypass -File .\packaging\windows\build-installer.ps1
```

说明：

- 打包脚本默认不会复制本地 `bin/ffmpeg/`；请让用户自行安装 `ffmpeg`，或在确有需要时显式使用 `-IncludeBundledFfmpeg`
- 打包产物不会附带 `asset/` 下的示例音频，使用时请自行准备参考音频
- 仓库也不会分发 `bin/`、`checkpoints/` 等本地运行时内容

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
- 程序启动时会统一检查并自动准备 `Fun-ASR-Nano-2512`、`speech_campplus_sv_zh-cn_16k-common`、所有内置可用的 `IndexTTS-2` 模型，以及 `emotion2vec+`
- 启动页会细化显示当前正在检查或下载的具体 `IndexTTS-2` 模型及其附属资源
- `IndexTTS-2` 当前支持通过 `INDEX_TTS_MODEL_REPO` 切换不同仓库，例如第三方日语模型
- `Fun-ASR-Nano-2512` 默认通过 `ModelScope` 自动准备，也可以通过环境变量切到本地目录
- Live 模式的声纹过滤已切换到 `CAMPPlus speaker verification` 模型，不再依赖 TTS 侧的自定义声纹提取
- 当前公开源码仓库不包含 `bin/` 本地运行时目录
- 打包环境下，`ModelScope` 相关模块会在代码中显式导入，以避免注册表缺失问题

## 已知说明

- 当前项目以 Windows 桌面实时使用场景为主
- 标题栏主题联动仅在 Windows 下支持
- TTS 默认设备分配为 `cuda`，ASR 默认使用 `cpu`，情感识别默认使用 `cpu`
- 首次启动或首次预热可能较慢
- 项目依赖较多，建议使用 `uv` 管理环境

## 开源说明

本仓库聚合了多个第三方模型与依赖，请在分发或商用前自行核对相关模型、权重、数据集与上游项目的许可证条款。

## 引用与致谢

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
- 如果你随项目一起分发 `bin/ffmpeg/`，请按该目录内随附许可证履行相应义务；当前本地示例为 `GPL v3` build
- 详细说明见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)
