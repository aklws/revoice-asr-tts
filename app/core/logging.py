from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from loguru import logger as _loguru_logger

# 移除默认的处理器
_loguru_logger.remove()

# 添加控制台输出 (异步)
_loguru_logger.add(sys.stderr, enqueue=True)

# 添加文件输出: 设置存放位置, 大小限制, 并启用异步
_loguru_logger.add("logs/app.log", rotation="10 MB", enqueue=True, encoding="utf-8")


def setup_logging(*args: Any, **kwargs: Any) -> None:
    """为了兼容旧代码保留的空函数"""
    pass


def add_log_sink(sink: Callable[[str], Any] | Any, **kwargs: Any) -> int:
    """添加自定义日志接收器 (例如 UI)"""
    return _loguru_logger.add(sink, enqueue=True)


def remove_log_sink(sink_id: int) -> None:
    """移除指定的日志接收器"""
    _loguru_logger.remove(sink_id)


def get_logger(name: str | None = None) -> Any:
    """获取 logger，loguru 会自动处理模块名"""
    return _loguru_logger
