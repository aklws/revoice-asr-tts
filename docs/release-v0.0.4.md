# Revoice ASR-TTS v0.0.4 更新公告

发布日期：2026-07-11

## 本次更新重点

- 优化实时播放链路：播放线程改为共享常驻模式，主播放与耳返统一走同一套分发逻辑，减少每段语音反复创建线程和输出流带来的额外开销。
- 改进停止响应：关闭麦克风或停止任务时，会更积极地丢弃待播放音频块与未完成片段，减少“已经点停止但还在继续播一点”的情况。
- 提升波形显示效率：`WaveformWidget` 改为环形缓冲与有序缓存方案，避免高频 `np.concatenate` 带来的重复拷贝。
- 减少情感识别额外 I/O：情感识别已改为纯内存路径，不再先写临时 `wav` 再推理。
- 收紧线程与状态同步：Live 页面的运行状态统一改为信号 + 状态枚举，减少跨线程直读 worker 内部字段导致的竞态风险。
- 补充链路观测：新增录音分段队列、播放队列、TTS 首包与块间隔等统计日志，便于定位实时性能瓶颈。

## 细节调整

- 空闲模型卸载从零散 `threading.Thread + sleep` 方式改为统一调度器，ASR / TTS 的超时卸载管理更集中。
- 删除了旧的 `_asr_idle_generation` 与 `_tts_idle_generation` 历史字段，空闲卸载逻辑更简洁。
- `IndexTTSService.synthesize_stream()` 改为短锁登记 + 活跃操作计数，避免流式生成全过程长时间占住服务锁。
- Worker 与主窗口间补充了更明确的 Qt `@Slot` 槽函数声明，跨线程信号边界更清晰。
- Live 停止信号改为直接连接方式，避免在长时间运行的 worker 循环里出现停止请求无法及时执行的问题。

## IndexTTS 试验性优化

- 在 `IndexTTS` 的 GPT 主干与 `s2mel` 路径中增加了 `torch.compile` 试验支持，当前采用 `max-autotune-no-cudagraphs` 模式。
- GPT 推理路径已从 `torch.no_grad()` 改为 `torch.inference_mode()`，用于测试 PyTorch 2.11 下的推理开销收益。
- 由于 Windows 分发环境下难以保证 `flash_attention_2` 可用，本版本暂不依赖该方案。

## 打包与版本

- 版本号已提升到 `v0.0.4`
- 项目元数据与界面显示版本已同步更新：
  - `pyproject.toml` -> `0.0.4`
  - `ui/main_window.py` -> `v0.0.4`

## 额外说明

- 当前 `torch.compile` 仍属于实验性优化，首次推理可能会包含额外编译开销，建议结合第二次及后续推理日志评估实际收益。
- 若用户环境不适合 `torch.compile`，程序会自动回退到常规推理路径。
