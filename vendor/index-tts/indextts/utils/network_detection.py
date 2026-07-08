"""
Compatibility shim for legacy callers.

The project now always uses ModelScope for model downloads, so this helper
returns ``True`` unconditionally.
"""

import logging

logger = logging.getLogger(__name__)


def need_proxy(timeout: float = 3.0) -> bool:
    """
    Always indicate that ModelScope should be used.
    """
    _ = timeout
    logger.info("网络检测已固定为 ModelScope 模式。")
    return True
