"""
Network detection utility for determining whether the current network
environment needs a proxy to access HuggingFace, to decide whether to
use ModelScope for model downloads.
"""

import os
import socket
import time
import logging

logger = logging.getLogger(__name__)

# Cache the detection result so we only check once per process
_detection_cache = None


def _tcp_latency(host: str, port: int = 443, timeout: float = 3.0):
    """TCP handshake latency in seconds, or None if unreachable."""
    try:
        start = time.perf_counter()
        sock = socket.create_connection((host, port), timeout=timeout)
        latency = time.perf_counter() - start
        sock.close()
        return latency
    except (socket.timeout, socket.error, OSError):
        return None


def need_proxy(timeout: float = 3.0) -> bool:
    """
    Detect if the current network environment needs a proxy to access HF.

    Returns True if a proxy is needed (use ModelScope / hf-mirror),
    False otherwise.

    Detection methods (in order):
    1. Check environment variable ``USE_MODELSCOPE`` for manual override
    2. Try TCP connection to huggingface.co (if unreachable, need proxy)
    3. Compare latency between modelscope.cn and huggingface.co

    The result is cached after the first call so subsequent calls are instant.
    """
    global _detection_cache
    if _detection_cache is not None:
        return _detection_cache

    # Allow manual override via environment variable
    env_override = os.environ.get("USE_MODELSCOPE", "").lower()
    if env_override == "true":
        logger.info("网络检测: 已强制使用代理模式 (USE_MODELSCOPE=true)")
        _detection_cache = True
        return True
    if env_override == "false":
        logger.info("网络检测: 已强制使用直连模式 (USE_MODELSCOPE=false)")
        _detection_cache = False
        return False

    # Check if huggingface.co is accessible and measure latency
    hf_latency = _tcp_latency("huggingface.co", timeout=timeout)
    if hf_latency is None:
        logger.info("网络检测: huggingface.co 不可达，需要代理")
        _detection_cache = True
        return True

    # Compare: if modelscope is significantly faster, likely in China
    ms_latency = _tcp_latency("modelscope.cn", timeout=timeout)
    if ms_latency is not None and ms_latency < hf_latency * 0.5:
        logger.info(
            "网络检测: modelscope.cn ({:.2f}s) 明显快于 huggingface.co ({:.2f}s)，需要代理",
            ms_latency,
            hf_latency,
        )
        _detection_cache = True
        return True

    logger.info("网络检测: huggingface.co 可访问，使用直连模式")
    _detection_cache = False
    return False
