# Revoice ASR-TTS v0.0.2 更新公告

发布日期：2026-07-08

## 本次更新重点

- 调整实时引擎空闲自动卸载策略：ASR 与 TTS 至少 1 小时未使用后才会自动卸载，避免频繁回收后重复加载。
- 修复文本模式问题：文本模式下如果模型因空闲被卸载，重新预热后不再错误触发录音设备链路。
- 优化实时链路体验：补强 ASR 重复文本抑制，减少同一句内容被连续送入 TTS 的情况。
- 优化界面可读性：整体字体略微缩小，主界面“文本与波形”区域减少一层卡片嵌套，视觉更简洁。
- 统一模型下载来源：项目内自动下载模型的逻辑统一改为只使用 ModelScope。

## 模型下载策略调整

- `IndexTTS-2`、`emotion2vec+` 以及 `IndexTTS` 附属模型现在统一通过 `ModelScope` 下载。
- 不再自动切换到 `HuggingFace` 或 `hf-mirror`。
- `BigVGAN` 默认仓库已调整为：
  - `nv-community/bigvgan_v2_22khz_80band_256x`
- `campplus` 默认仓库已调整为：
  - `iic/speech_campplus_sv_zh-cn_16k-common`

## 打包与版本

- 版本号已提升到 `v0.0.2`
- Windows 打包脚本与安装器脚本已同步更新：
  - PyInstaller 输出目录名将使用 `Revoice-ASR-TTS-v0.0.2`
  - 安装包输出名将使用 `Revoice-ASR-TTS-v0.0.2-Setup.exe`

## 额外说明

- `Qwen3-ASR` 运行时资源仍按当前项目约定手动准备，不走自动下载。
- `ffmpeg` 与 `flash_attn` 仍按 README 中的说明手动准备。
