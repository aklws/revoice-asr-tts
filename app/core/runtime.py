from __future__ import annotations

import ctypes
import gc
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from ctypes import wintypes
from pathlib import Path

import torch

from app.core.logging import get_logger

logger = get_logger(__name__)
_MONITOR_WARNINGS: set[str] = set()
_WMI_SERVICES: dict[int, object] = {}
_GPU_TOTAL_MEMORY_CACHE_GB: float | None = None
_GPU_TOTAL_MEMORY_CACHE_AT: float = 0.0
_CPU_TIMES_LOCK = threading.Lock()
_CPU_LAST_TIMES: tuple[int, int, int] | None = None
_CPU_LAST_PERCENT: int | None = None

DXGI_ERROR_NOT_FOUND = ctypes.c_long(0x887A0002).value
DXGI_ADAPTER_FLAG_SOFTWARE = 0x2
HRESULT = ctypes.c_long
ULONG = ctypes.c_ulong


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _Guid(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_string(cls, value: str) -> _Guid:
        from uuid import UUID

        parsed = UUID(value)
        data4 = (ctypes.c_ubyte * 8)(*parsed.bytes[8:])
        return cls(
            parsed.time_low,
            parsed.time_mid,
            parsed.time_hi_version,
            data4,
        )


class _Luid(ctypes.Structure):
    _fields_ = [
        ("LowPart", wintypes.DWORD),
        ("HighPart", wintypes.LONG),
    ]


class _DxgiAdapterDesc1(ctypes.Structure):
    _fields_ = [
        ("Description", wintypes.WCHAR * 128),
        ("VendorId", ctypes.c_uint),
        ("DeviceId", ctypes.c_uint),
        ("SubSysId", ctypes.c_uint),
        ("Revision", ctypes.c_uint),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _Luid),
        ("Flags", ctypes.c_uint),
    ]


def _is_cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _warn_once(key: str, message: str, *args: object) -> None:
    if key in _MONITOR_WARNINGS:
        return
    _MONITOR_WARNINGS.add(key)
    logger.warning(message, *args)


def _normalize_percent(value: object) -> int | None:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return None


def _filetime_to_int64(value: wintypes.FILETIME) -> int:
    return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)


def _call_com_method(
    interface_ptr: ctypes.c_void_p,
    method_index: int,
    restype: object,
    *argtypes: object,
):
    vtable = ctypes.cast(interface_ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    method = prototype(vtable[method_index])
    return method


def _release_com(interface_ptr: ctypes.c_void_p | None) -> None:
    if not interface_ptr:
        return
    try:
        _call_com_method(interface_ptr, 2, ULONG)(interface_ptr)
    except Exception:
        pass


def _get_cpu_utilization_from_win32() -> int | None:
    if sys.platform != "win32":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        idle = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            raise ctypes.WinError(ctypes.get_last_error())
        current = (
            _filetime_to_int64(idle),
            _filetime_to_int64(kernel),
            _filetime_to_int64(user),
        )
    except Exception as exc:
        _warn_once("get_system_times", "通过 GetSystemTimes 读取 CPU 占用失败: {}", exc)
        return None

    global _CPU_LAST_TIMES, _CPU_LAST_PERCENT
    with _CPU_TIMES_LOCK:
        previous = _CPU_LAST_TIMES
        _CPU_LAST_TIMES = current
        if previous is None:
            return _CPU_LAST_PERCENT
        idle_delta = max(0, current[0] - previous[0])
        kernel_delta = max(0, current[1] - previous[1])
        user_delta = max(0, current[2] - previous[2])
        total_delta = kernel_delta + user_delta
        if total_delta <= 0:
            return _CPU_LAST_PERCENT
        busy_percent = (1.0 - (idle_delta / total_delta)) * 100.0
        _CPU_LAST_PERCENT = _normalize_percent(busy_percent)
        return _CPU_LAST_PERCENT


def _get_memory_utilization_from_win32() -> int | None:
    if sys.platform != "win32":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        return _normalize_percent(status.dwMemoryLoad)
    except Exception as exc:
        _warn_once("global_memory_status", "通过 GlobalMemoryStatusEx 读取内存占用失败: {}", exc)
        return None


def _get_gpu_total_memory_gb_from_dxgi() -> float | None:
    if sys.platform != "win32":
        return None
    factory = ctypes.c_void_p()
    dxgi = None
    try:
        dxgi = ctypes.WinDLL("dxgi", use_last_error=True)
        create_factory = dxgi.CreateDXGIFactory1
        create_factory.argtypes = [ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p)]
        create_factory.restype = HRESULT

        factory_iid = _Guid.from_string("770aae78-f26f-4dba-a829-253c83d1b387")
        hr = create_factory(ctypes.byref(factory_iid), ctypes.byref(factory))
        if hr != 0:
            raise OSError(f"CreateDXGIFactory1 failed: HRESULT=0x{hr & 0xFFFFFFFF:08X}")

        enum_adapters = _call_com_method(factory, 12, HRESULT, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))

        best_overall = 0.0
        best_nvidia = 0.0
        adapter_index = 0
        while True:
            adapter = ctypes.c_void_p()
            hr = enum_adapters(factory, adapter_index, ctypes.byref(adapter))
            if hr == DXGI_ERROR_NOT_FOUND:
                break
            if hr != 0:
                raise OSError(f"IDXGIFactory1::EnumAdapters1 failed: HRESULT=0x{hr & 0xFFFFFFFF:08X}")
            try:
                desc = _DxgiAdapterDesc1()
                adapter_get_desc1 = _call_com_method(adapter, 10, HRESULT, ctypes.POINTER(_DxgiAdapterDesc1))
                hr_desc = adapter_get_desc1(adapter, ctypes.byref(desc))
                if hr_desc != 0:
                    adapter_index += 1
                    continue
                if int(desc.Flags) & DXGI_ADAPTER_FLAG_SOFTWARE:
                    adapter_index += 1
                    continue
                dedicated_bytes = float(desc.DedicatedVideoMemory or 0)
                if dedicated_bytes <= 0:
                    adapter_index += 1
                    continue
                name = str(desc.Description or "").strip().lower()
                best_overall = max(best_overall, dedicated_bytes)
                if "nvidia" in name:
                    best_nvidia = max(best_nvidia, dedicated_bytes)
            finally:
                _release_com(adapter)
            adapter_index += 1

        chosen_bytes = best_nvidia or best_overall
        if chosen_bytes <= 0:
            return None
        return chosen_bytes / (1024 ** 3)
    except Exception as exc:
        _warn_once("dxgi_vram", "通过 DXGI 读取 GPU 总显存失败: {}", exc)
        return None
    finally:
        _release_com(factory)


def _get_wmi_service() -> object | None:
    if sys.platform != "win32":
        return None
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        _warn_once("pywin32_import", "系统监控依赖 pywin32 未安装: {}", exc)
        return None
    try:
        thread_id = threading.get_ident()
        cached = _WMI_SERVICES.get(thread_id)
        if cached is not None:
            return cached
        pythoncom.CoInitialize()
        locator = win32com.client.Dispatch("WbemScripting.SWbemLocator")
        service = locator.ConnectServer(".", "root\\cimv2")
        _WMI_SERVICES[thread_id] = service
        return service
    except Exception as exc:
        _warn_once("wmi_connect", "连接 Windows 系统监控服务失败: {}", exc)
        return None


def _query_win32_rows(query: str) -> list[dict[str, object]]:
    service = _get_wmi_service()
    if service is None:
        return []
    try:
        result = service.ExecQuery(query)
        rows: list[dict[str, object]] = []
        for row in result:
            item: dict[str, object] = {}
            for prop in row.Properties_:
                item[str(prop.Name)] = prop.Value
            rows.append(item)
        return rows
    except Exception as exc:
        _warn_once(f"wmi:{query}", "读取 Windows 系统监控失败: {}", exc)
        return []


def _read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-16", "utf-8", "mbcs", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except Exception:
            continue
    return path.read_text(errors="ignore")


def _get_gpu_total_memory_gb_from_dxdiag() -> float | None:
    global _GPU_TOTAL_MEMORY_CACHE_AT, _GPU_TOTAL_MEMORY_CACHE_GB

    now = time.monotonic()
    if _GPU_TOTAL_MEMORY_CACHE_AT and now - _GPU_TOTAL_MEMORY_CACHE_AT < 300:
        return _GPU_TOTAL_MEMORY_CACHE_GB

    report_path = Path(tempfile.gettempdir()) / "revoice_dxdiag_gpu.txt"
    try:
        dxdiag_path = Path(
            str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "dxdiag.exe")
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run(
            [str(dxdiag_path), "/dontskip", "/whql:off", "/t", str(report_path)],
            check=True,
            timeout=30,
            creationflags=creationflags,
        )
        text = _read_text_with_fallback(report_path)
    except Exception as exc:
        _warn_once("dxdiag_vram", "通过 dxdiag 读取 GPU 总显存失败: {}", exc)
        _GPU_TOTAL_MEMORY_CACHE_AT = now
        _GPU_TOTAL_MEMORY_CACHE_GB = None
        return None

    preferred_mb = 0.0
    fallback_mb = 0.0
    current_card_name = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower_line = line.lower()
        if lower_line.startswith("card name:"):
            current_card_name = line.split(":", 1)[1].strip().lower()
            continue
        if not lower_line.startswith("dedicated memory:"):
            continue
        match = re.search(r"([\d,]+)\s*mb", line, re.IGNORECASE)
        if match is None:
            continue
        memory_mb = float(match.group(1).replace(",", ""))
        fallback_mb = max(fallback_mb, memory_mb)
        if "nvidia" in current_card_name:
            preferred_mb = max(preferred_mb, memory_mb)

    chosen_mb = preferred_mb or fallback_mb
    _GPU_TOTAL_MEMORY_CACHE_AT = now
    _GPU_TOTAL_MEMORY_CACHE_GB = (chosen_mb / 1024.0) if chosen_mb > 0 else None
    return _GPU_TOTAL_MEMORY_CACHE_GB


def get_cpu_utilization() -> int | None:
    win32_value = _get_cpu_utilization_from_win32()
    if win32_value is not None:
        return win32_value
    rows = _query_win32_rows(
        "SELECT PercentProcessorTime FROM Win32_PerfFormattedData_PerfOS_Processor WHERE Name='_Total'"
    )
    if not rows:
        return None
    return _normalize_percent(rows[0].get("PercentProcessorTime"))


def get_memory_utilization() -> int | None:
    win32_value = _get_memory_utilization_from_win32()
    if win32_value is not None:
        return win32_value
    rows = _query_win32_rows(
        "SELECT TotalVisibleMemorySize,FreePhysicalMemory FROM Win32_OperatingSystem"
    )
    if not rows:
        return None
    total = rows[0].get("TotalVisibleMemorySize")
    free = rows[0].get("FreePhysicalMemory")
    try:
        total_kb = float(total)
        free_kb = float(free)
    except (TypeError, ValueError):
        return None
    if total_kb <= 0:
        return None
    used_percent = ((total_kb - free_kb) / total_kb) * 100.0
    return _normalize_percent(used_percent)


def resolve_device(preferred: str | None = None) -> str:
    normalized = str(preferred or "auto").strip().lower()
    if normalized == "auto":
        return "cuda" if _is_cuda_available() else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda"):
        return normalized if _is_cuda_available() else "cpu"
    raise ValueError(f"Unsupported device preference: {preferred!r}")


def resolve_dtype(device: str) -> torch.dtype:
    return torch.float16 if str(device).startswith("cuda") else torch.float32


def resolve_attn_implementation(device: str) -> str:
    if str(device).startswith("cuda"):
        return "sdpa"
    return "eager"


def resolve_generation_cache_implementation(preferred: str | None = None, *, device: str) -> str:
    normalized = str(preferred or "dynamic").strip().lower()
    allowed = {
        "dynamic",
        "static",
        "offloaded",
        "offloaded_static",
        "quantized",
        "sliding_window",
        "hybrid",
        "mamba",
    }
    if normalized == "auto":
        return "static" if str(device).startswith("cuda") else "dynamic"
    if normalized not in allowed:
        return "dynamic"
    if not str(device).startswith("cuda") and normalized in {"offloaded", "offloaded_static"}:
        return "dynamic"
    return normalized


def get_gpu_total_memory_gb() -> float | None:
    """Return total GPU memory in GB, preferring DXGI and falling back to dxdiag/WMI."""
    global _GPU_TOTAL_MEMORY_CACHE_AT, _GPU_TOTAL_MEMORY_CACHE_GB

    now = time.monotonic()
    if _GPU_TOTAL_MEMORY_CACHE_AT and now - _GPU_TOTAL_MEMORY_CACHE_AT < 300:
        return _GPU_TOTAL_MEMORY_CACHE_GB

    dxgi_value = _get_gpu_total_memory_gb_from_dxgi()
    if dxgi_value is not None:
        _GPU_TOTAL_MEMORY_CACHE_AT = now
        _GPU_TOTAL_MEMORY_CACHE_GB = dxgi_value
        return dxgi_value

    dxdiag_value = _get_gpu_total_memory_gb_from_dxdiag()
    if dxdiag_value is not None:
        _GPU_TOTAL_MEMORY_CACHE_AT = now
        _GPU_TOTAL_MEMORY_CACHE_GB = dxdiag_value
        return dxdiag_value

    rows = _query_win32_rows("SELECT Name,AdapterRAM FROM Win32_VideoController")
    if not rows:
        _GPU_TOTAL_MEMORY_CACHE_AT = now
        _GPU_TOTAL_MEMORY_CACHE_GB = None
        return None

    best_bytes = 0.0
    for row in rows:
        try:
            adapter_ram = float(row.get("AdapterRAM", 0) or 0)
        except (TypeError, ValueError):
            continue
        if adapter_ram < 0:
            adapter_ram += 2 ** 32
        if adapter_ram <= 0:
            continue
        name = str(row.get("Name", "") or "").lower()
        # 优先取独显/NVIDIA，其次回退到显存最大的适配器。
        if "nvidia" in name:
            _GPU_TOTAL_MEMORY_CACHE_AT = now
            _GPU_TOTAL_MEMORY_CACHE_GB = adapter_ram / (1024 ** 3)
            return _GPU_TOTAL_MEMORY_CACHE_GB
        best_bytes = max(best_bytes, adapter_ram)

    if best_bytes <= 0:
        _GPU_TOTAL_MEMORY_CACHE_AT = now
        _GPU_TOTAL_MEMORY_CACHE_GB = None
        return None
    _GPU_TOTAL_MEMORY_CACHE_AT = now
    _GPU_TOTAL_MEMORY_CACHE_GB = best_bytes / (1024 ** 3)
    return _GPU_TOTAL_MEMORY_CACHE_GB


def get_gpu_memory_used_gb() -> float | None:
    """Return current local GPU memory usage in GB from Windows counters."""
    rows = _query_win32_rows(
        "SELECT Name,LocalUsage FROM Win32_PerfFormattedData_GPUPerformanceCounters_GPULocalAdapterMemory"
    )
    if not rows:
        return None

    used_bytes = 0.0
    for row in rows:
        try:
            local_usage = float(row.get("LocalUsage", 0) or 0)
        except (TypeError, ValueError):
            continue
        used_bytes = max(used_bytes, local_usage)

    if used_bytes <= 0:
        return 0.0
    return used_bytes / (1024 ** 3)


def get_gpu_memory_utilization() -> int | None:
    """Return current dedicated GPU memory usage percent from Windows counters."""
    used_gb = get_gpu_memory_used_gb()
    total_gb = get_gpu_total_memory_gb()
    if used_gb is None or total_gb is None or total_gb <= 0:
        return None
    return _normalize_percent((used_gb / total_gb) * 100.0)


def get_gpu_utilization() -> int | None:
    """Return current GPU utilization percent from Windows performance counters."""
    rows = _query_win32_rows(
        "SELECT Name,UtilizationPercentage FROM Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine"
    )
    if not rows:
        return None

    preferred_values: list[int] = []
    fallback_values: list[int] = []
    for row in rows:
        value = _normalize_percent(row.get("UtilizationPercentage"))
        if value is None:
            continue
        fallback_values.append(value)
        name = str(row.get("Name", "") or "").lower()
        # 仅保留图形/计算引擎，更接近任务管理器里常见的整体 GPU 忙碌度。
        if any(tag in name for tag in ("engtype_3d", "engtype_compute", "engtype_cuda")):
            preferred_values.append(value)

    values = preferred_values or fallback_values
    if not values:
        return None
    return max(values)


def release_memory() -> None:
    gc.collect()
    if _is_cuda_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
