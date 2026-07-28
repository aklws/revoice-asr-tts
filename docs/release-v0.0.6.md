# Revoice ASR-TTS v0.0.6 更新公告

发布日期：2026-07-28

## 本次更新重点

- Windows 打包方案已切换为 `Cython bundle` 路线：本地 `app`、`ui` 等模块会编译为 `.pyd`，再由启动脚本和安装器组装分发。
- 启动器逻辑已调整为 `uv` 驱动：打包产物内的 `.bat` 会先检查 `uv`，再执行 `uv sync --frozen` 后启动程序。
- 修复实时链路中的空文本崩溃：当 `ASR` 输出为空或只包含标点时，不再继续送入 `TTS`，避免后续生成空语义 token 并触发底层越界错误。
- 补强 `IndexTTS-2` 与新版 `transformers` 的兼容：保持 `transformers >= 4.51` 以支持 `Fun-ASR-Nano-2512`，同时修复 `IndexTTS` 在新缓存接口下的生成兼容问题。
- 修复打包版资源定位异常：构建后的运行目录现在能正确解析 `checkpoints`、`vendor` 等本地资源，不再错误回退到解释器目录。

## 推理与稳定性修复

- `Live ASR -> TTS` 链路已增加双层保护：
  - 业务层会拦截空文本和纯标点结果
  - `IndexTTS` 引擎层会在生成空语义 token 时直接跳过当前分段
- 修复了 `IndexTTS` 在新版 `transformers` 下的若干兼容问题，包括：
  - `GPT2InferenceModel` 补回 `GenerationMixin` 继承
  - `past_key_values` 从旧 tuple 结构向新 `Cache` 接口过渡时的兼容处理
  - 常规 GPT 生成路径中对空 cache 的防御
- `vendor/index-tts` 已从 Cython 编译名单中排除，保留源码形式，避免 `torch.jit` / `inspect` / 反射链路因编译后对象类型变化而失效。

## 打包与分发调整

- 当前 Windows 打包流程为：
  - [build.ps1](../packaging/windows/build.ps1) 负责生成 Cython 运行目录
  - [build-installer.ps1](../packaging/windows/build-installer.ps1) 负责基于 `dist/` 目录制作安装包
- 打包目录中会自动创建：
  - `outputs/`
  - `checkpoints/`
  - `logs/`
- 打包产物不再附带 `asset/` 下的示例音频，使用时请自行准备参考音频。
- 本地 `bin/ffmpeg`、`.env.example`、README 等运行所需文件会继续按规则复制到分发目录。

## 依赖与构建调整

- 已移除不再使用的 `vendor/editdistance-shim` 及其构建链路依赖。
- 根项目 `uv.lock` 已重新生成，依赖声明与当前仓库状态保持一致。
- `flash_attn` 仍为可选依赖，仅在实验性极速 GPT 路径中使用；未启用时不会作为基础运行前提。

## 打包与版本

- 版本号已提升到 `v0.0.6`
- Windows 打包脚本中的应用版本号已同步更新：
  - `packaging/windows/build.ps1` -> `0.0.6`
  - `packaging/windows/build-installer.ps1` -> `0.0.6`

## 额外说明

- 当前公开仓库不分发 `bin/`、`checkpoints/` 和示例参考音频等本地运行时内容。
- 首次启动仍可能因为模型检查、自动下载或参考音频预热而耗时较长。
- `flash_attn` 和模型权重请继续按 README 与文档说明手动准备或按需下载。
