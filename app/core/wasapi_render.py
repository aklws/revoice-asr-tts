from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

if hasattr(ctypes, "WINFUNCTYPE"):
    _COM_METHOD = ctypes.WINFUNCTYPE
else:
    _COM_METHOD = ctypes.CFUNCTYPE

HRESULT = ctypes.c_long
REFERENCE_TIME = ctypes.c_longlong
CLSCTX_ALL = 0x17
COINIT_APARTMENTTHREADED = 0x2
S_OK = 0
S_FALSE = 1
RPC_E_CHANGED_MODE = ctypes.c_long(0x80010106).value
AUDCLNT_SHAREMODE_SHARED = 0
AUDCLNT_STREAMFLAGS_EVENTCALLBACK = 0x00040000
AUDCLNT_BUFFERFLAGS_SILENT = 0x00000002
AUDIO_CATEGORY_SPEECH = 9
AUDCLNT_STREAMOPTIONS_NONE = 0x00
WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE
E_RENDER = 0
E_MULTIMEDIA = 1
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102
_DEFAULT_BUFFER_DURATION = REFERENCE_TIME(500_000)


class _Guid(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_string(cls, value: str) -> "_Guid":
        from uuid import UUID

        parsed = UUID(value)
        return cls(
            parsed.time_low,
            parsed.time_mid,
            parsed.time_hi_version,
            (ctypes.c_ubyte * 8)(*parsed.bytes[8:]),
        )

    def to_bytes(self) -> bytes:
        return bytes(memoryview(self))


class _WaveFormatEx(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("wFormatTag", wintypes.WORD),
        ("nChannels", wintypes.WORD),
        ("nSamplesPerSec", wintypes.DWORD),
        ("nAvgBytesPerSec", wintypes.DWORD),
        ("nBlockAlign", wintypes.WORD),
        ("wBitsPerSample", wintypes.WORD),
        ("cbSize", wintypes.WORD),
    ]


class _WaveFormatSamples(ctypes.Union):
    _fields_ = [
        ("wValidBitsPerSample", wintypes.WORD),
        ("wSamplesPerBlock", wintypes.WORD),
        ("wReserved", wintypes.WORD),
    ]


class _WaveFormatExtensible(ctypes.Structure):
    _pack_ = 1
    _anonymous_ = ("Samples",)
    _fields_ = [
        ("Format", _WaveFormatEx),
        ("Samples", _WaveFormatSamples),
        ("dwChannelMask", wintypes.DWORD),
        ("SubFormat", _Guid),
    ]


class _AudioClientProperties(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.UINT),
        ("bIsOffload", wintypes.BOOL),
        ("eCategory", ctypes.c_int),
        ("Options", ctypes.c_int),
    ]


class _IUnknownVTable(ctypes.Structure):
    _fields_ = [
        ("QueryInterface", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p))),
        ("AddRef", _COM_METHOD(ctypes.c_ulong, ctypes.c_void_p)),
        ("Release", _COM_METHOD(ctypes.c_ulong, ctypes.c_void_p)),
    ]


class _IMMDeviceEnumeratorVTable(ctypes.Structure):
    _fields_ = [
        ("QueryInterface", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p))),
        ("AddRef", _COM_METHOD(ctypes.c_ulong, ctypes.c_void_p)),
        ("Release", _COM_METHOD(ctypes.c_ulong, ctypes.c_void_p)),
        ("EnumAudioEndpoints", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.c_int, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p))),
        ("GetDefaultAudioEndpoint", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))),
        ("GetDevice", _COM_METHOD(HRESULT, ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p))),
    ]


class _IMMDeviceVTable(ctypes.Structure):
    _fields_ = [
        ("QueryInterface", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p))),
        ("AddRef", _COM_METHOD(ctypes.c_ulong, ctypes.c_void_p)),
        ("Release", _COM_METHOD(ctypes.c_ulong, ctypes.c_void_p)),
        ("Activate", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(_Guid), wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))),
        ("OpenPropertyStore", _COM_METHOD(HRESULT, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p))),
        ("GetId", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR))),
        ("GetState", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD))),
    ]


class _IAudioClientVTable(ctypes.Structure):
    _fields_ = [
        ("QueryInterface", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p))),
        ("AddRef", _COM_METHOD(ctypes.c_ulong, ctypes.c_void_p)),
        ("Release", _COM_METHOD(ctypes.c_ulong, ctypes.c_void_p)),
        (
            "Initialize",
            _COM_METHOD(
                HRESULT,
                ctypes.c_void_p,
                wintypes.DWORD,
                wintypes.DWORD,
                REFERENCE_TIME,
                REFERENCE_TIME,
                ctypes.POINTER(_WaveFormatEx),
                ctypes.c_void_p,
            ),
        ),
        ("GetBufferSize", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(wintypes.UINT))),
        ("GetStreamLatency", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(REFERENCE_TIME))),
        ("GetCurrentPadding", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(wintypes.UINT))),
        (
            "IsFormatSupported",
            _COM_METHOD(
                HRESULT,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.POINTER(_WaveFormatEx),
                ctypes.POINTER(ctypes.c_void_p),
            ),
        ),
        ("GetMixFormat", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))),
        ("GetDevicePeriod", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(REFERENCE_TIME), ctypes.POINTER(REFERENCE_TIME))),
        ("Start", _COM_METHOD(HRESULT, ctypes.c_void_p)),
        ("Stop", _COM_METHOD(HRESULT, ctypes.c_void_p)),
        ("Reset", _COM_METHOD(HRESULT, ctypes.c_void_p)),
        ("SetEventHandle", _COM_METHOD(HRESULT, ctypes.c_void_p, wintypes.HANDLE)),
        ("GetService", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p))),
    ]


class _IAudioRenderClientVTable(ctypes.Structure):
    _fields_ = [
        ("QueryInterface", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p))),
        ("AddRef", _COM_METHOD(ctypes.c_ulong, ctypes.c_void_p)),
        ("Release", _COM_METHOD(ctypes.c_ulong, ctypes.c_void_p)),
        ("GetBuffer", _COM_METHOD(HRESULT, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))),
        ("ReleaseBuffer", _COM_METHOD(HRESULT, ctypes.c_void_p, wintypes.UINT, wintypes.DWORD)),
    ]


class _IAudioClient2VTable(ctypes.Structure):
    _fields_ = _IAudioClientVTable._fields_ + [
        ("IsOffloadCapable", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(wintypes.BOOL))),
        ("SetClientProperties", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(_AudioClientProperties))),
        (
            "GetBufferSizeLimits",
            _COM_METHOD(
                HRESULT,
                ctypes.c_void_p,
                ctypes.POINTER(_WaveFormatEx),
                wintypes.BOOL,
                ctypes.POINTER(REFERENCE_TIME),
                ctypes.POINTER(REFERENCE_TIME),
            ),
        ),
    ]


class _IAudioClient3VTable(ctypes.Structure):
    _fields_ = _IAudioClient2VTable._fields_ + [
        (
            "GetSharedModeEnginePeriod",
            _COM_METHOD(
                HRESULT,
                ctypes.c_void_p,
                ctypes.POINTER(_WaveFormatEx),
                ctypes.POINTER(wintypes.UINT),
                ctypes.POINTER(wintypes.UINT),
                ctypes.POINTER(wintypes.UINT),
                ctypes.POINTER(wintypes.UINT),
            ),
        ),
        (
            "GetCurrentSharedModeEnginePeriod",
            _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.UINT)),
        ),
        (
            "InitializeSharedAudioStream",
            _COM_METHOD(
                HRESULT,
                ctypes.c_void_p,
                wintypes.DWORD,
                wintypes.UINT,
                ctypes.POINTER(_WaveFormatEx),
                ctypes.c_void_p,
            ),
        ),
    ]


class _IUnknown(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_IUnknownVTable))]


class _IMMDeviceEnumerator(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_IMMDeviceEnumeratorVTable))]


class _IMMDevice(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_IMMDeviceVTable))]


class _IAudioClient(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_IAudioClientVTable))]


class _IAudioClient2(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_IAudioClient2VTable))]


class _IAudioClient3(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_IAudioClient3VTable))]


class _IAudioRenderClient(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_IAudioRenderClientVTable))]


@dataclass(slots=True)
class WasapiMixFormat:
    sample_rate: int
    channels: int
    bytes_per_frame: int
    bytes_per_sample: int
    is_float: bool
    bits_per_sample: int
    channel_mask: int


_CLSID_MMDEVICE_ENUMERATOR = _Guid.from_string("BCDE0395-E52F-467C-8E3D-C4579291692E")
_IID_IMMDEVICE_ENUMERATOR = _Guid.from_string("A95664D2-9614-4F35-A746-DE8DB63617E6")
_IID_IAUDIOCLIENT = _Guid.from_string("1CB9AD4C-DBFA-4C32-B178-C2F568A703B2")
_IID_IAUDIOCLIENT3 = _Guid.from_string("7ED4EE07-8E67-4CD4-8C1A-2B7A5987AD42")
_IID_IAUDIORENDERCLIENT = _Guid.from_string("F294ACFC-3146-4483-A7BF-ADDCA7C260E2")
_SUBTYPE_PCM = _Guid.from_string("00000001-0000-0010-8000-00AA00389B71")
_SUBTYPE_IEEE_FLOAT = _Guid.from_string("00000003-0000-0010-8000-00AA00389B71")

_ole32 = ctypes.OleDLL("ole32")
_ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
_ole32.CoInitializeEx.restype = HRESULT
_ole32.CoUninitialize.argtypes = []
_ole32.CoUninitialize.restype = None
_ole32.CoCreateInstance.argtypes = [
    ctypes.POINTER(_Guid),
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(_Guid),
    ctypes.POINTER(ctypes.c_void_p),
]
_ole32.CoCreateInstance.restype = HRESULT
_ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
_ole32.CoTaskMemFree.restype = None

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
_kernel32.CreateEventW.restype = wintypes.HANDLE
_kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_kernel32.WaitForSingleObject.restype = wintypes.DWORD
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.CloseHandle.restype = wintypes.BOOL


def _succeeded(hr: int) -> bool:
    return hr >= 0


def _check_hresult(hr: int, action: str) -> None:
    if _succeeded(hr):
        return
    raise OSError(f"{action} 失败: 0x{ctypes.c_uint32(hr).value:08X}")


def _release(ptr: ctypes.c_void_p | None) -> None:
    if ptr is None or not ptr:
        return
    unknown = ctypes.cast(ptr, ctypes.POINTER(_IUnknown))
    unknown.contents.lpVtbl.contents.Release(ptr)


def _query_interface(ptr: ctypes.c_void_p, iid: _Guid) -> ctypes.c_void_p | None:
    interface_ptr = ctypes.c_void_p()
    unknown = ctypes.cast(ptr, ctypes.POINTER(_IUnknown))
    hr = unknown.contents.lpVtbl.contents.QueryInterface(ptr, ctypes.byref(iid), ctypes.byref(interface_ptr))
    if not _succeeded(hr) or not interface_ptr:
        return None
    return interface_ptr


def _align_period_to_fundamental(period_frames: int, *, fundamental_frames: int, min_frames: int, max_frames: int) -> int:
    if fundamental_frames <= 0:
        return max(min_frames, min(period_frames, max_frames))
    aligned = max(min_frames, period_frames)
    remainder = aligned % fundamental_frames
    if remainder:
        aligned += fundamental_frames - remainder
    return max(min_frames, min(aligned, max_frames))


def _co_initialize() -> tuple[bool, bool]:
    hr = _ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    if hr in (S_OK, S_FALSE):
        return True, True
    if hr == RPC_E_CHANGED_MODE:
        return True, False
    return False, False


def _guid_equals(left: _Guid, right: _Guid) -> bool:
    return left.to_bytes() == right.to_bytes()


def _parse_mix_format(format_ptr: ctypes.c_void_p) -> WasapiMixFormat:
    wave_format = ctypes.cast(format_ptr, ctypes.POINTER(_WaveFormatEx)).contents
    is_float = wave_format.wFormatTag == WAVE_FORMAT_IEEE_FLOAT
    channel_mask = 0
    if wave_format.wFormatTag == WAVE_FORMAT_EXTENSIBLE and wave_format.cbSize >= 22:
        extensible = ctypes.cast(format_ptr, ctypes.POINTER(_WaveFormatExtensible)).contents
        channel_mask = int(extensible.dwChannelMask or 0)
        if _guid_equals(extensible.SubFormat, _SUBTYPE_IEEE_FLOAT):
            is_float = True
        elif _guid_equals(extensible.SubFormat, _SUBTYPE_PCM):
            is_float = False
        else:
            raise RuntimeError("当前 WASAPI 混音格式不是 PCM/IEEE_FLOAT，暂不支持。")
    elif wave_format.wFormatTag not in (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT):
        raise RuntimeError(f"当前 WASAPI 混音格式不支持: tag={int(wave_format.wFormatTag)}")

    channels = max(1, int(wave_format.nChannels or 1))
    bytes_per_frame = max(channels, int(wave_format.nBlockAlign or channels))
    bytes_per_sample = max(1, bytes_per_frame // channels)
    bits_per_sample = max(8, int(wave_format.wBitsPerSample or bytes_per_sample * 8))
    return WasapiMixFormat(
        sample_rate=max(1, int(wave_format.nSamplesPerSec or 48_000)),
        channels=channels,
        bytes_per_frame=bytes_per_frame,
        bytes_per_sample=bytes_per_sample,
        is_float=is_float,
        bits_per_sample=bits_per_sample,
        channel_mask=channel_mask,
    )


def _build_preferred_wave_format(
    source_format_ptr: ctypes.c_void_p,
    *,
    sample_rate: int,
) -> _WaveFormatEx | _WaveFormatExtensible:
    base_format = ctypes.cast(source_format_ptr, ctypes.POINTER(_WaveFormatEx)).contents
    target_rate = max(1, int(sample_rate))
    if base_format.wFormatTag == WAVE_FORMAT_EXTENSIBLE and base_format.cbSize >= 22:
        preferred = _WaveFormatExtensible()
        ctypes.pointer(preferred)[0] = ctypes.cast(source_format_ptr, ctypes.POINTER(_WaveFormatExtensible)).contents
        preferred.Format.nSamplesPerSec = target_rate
        preferred.Format.nAvgBytesPerSec = target_rate * int(preferred.Format.nBlockAlign or 1)
        return preferred

    preferred = _WaveFormatEx()
    ctypes.pointer(preferred)[0] = base_format
    preferred.nSamplesPerSec = target_rate
    preferred.nAvgBytesPerSec = target_rate * int(preferred.nBlockAlign or 1)
    return preferred


def _coerce_channel_count(audio_np: np.ndarray, target_channels: int) -> np.ndarray:
    if audio_np.ndim == 1:
        working = audio_np.reshape(-1, 1)
    else:
        working = audio_np
    source_channels = int(working.shape[1])
    if source_channels == target_channels:
        return np.ascontiguousarray(working, dtype=np.float32)
    if target_channels == 1:
        mono = np.mean(working, axis=1, dtype=np.float32)
        return np.ascontiguousarray(mono.reshape(-1, 1), dtype=np.float32)
    if source_channels == 1:
        repeated = np.repeat(working, target_channels, axis=1)
        return np.ascontiguousarray(repeated, dtype=np.float32)
    if source_channels > target_channels:
        return np.ascontiguousarray(working[:, :target_channels], dtype=np.float32)
    extra = np.repeat(working[:, -1:], target_channels - source_channels, axis=1)
    return np.ascontiguousarray(np.concatenate((working, extra), axis=1), dtype=np.float32)


def _float_to_pcm_bytes(audio_np: np.ndarray, *, bytes_per_sample: int) -> bytes:
    clipped = np.clip(audio_np, -1.0, 1.0)
    flattened = np.ascontiguousarray(clipped.reshape(-1), dtype=np.float32)
    if bytes_per_sample == 1:
        pcm_u8 = np.clip(np.rint((flattened + 1.0) * 127.5), 0, 255).astype(np.uint8)
        return pcm_u8.tobytes()
    if bytes_per_sample == 2:
        pcm_i16 = np.clip(np.rint(flattened * 32767.0), -32768, 32767).astype("<i2")
        return pcm_i16.tobytes()
    if bytes_per_sample == 3:
        pcm_i32 = np.clip(np.rint(flattened * 8388607.0), -8388608, 8388607).astype("<i4")
        packed = pcm_i32.view(np.uint8).reshape(-1, 4)[:, :3]
        return np.ascontiguousarray(packed).tobytes()
    if bytes_per_sample == 4:
        pcm_i32 = np.clip(np.rint(flattened * 2147483647.0), -2147483648, 2147483647).astype("<i4")
        return pcm_i32.tobytes()
    raise RuntimeError(f"不支持的 PCM 样本宽度: {bytes_per_sample} 字节")


class WasapiRenderClient:
    def __init__(self, *, endpoint_id: str | None, label: str, preferred_sample_rate: int | None = None) -> None:
        self.endpoint_id = endpoint_id
        self.label = label
        self.preferred_sample_rate = max(1, int(preferred_sample_rate)) if preferred_sample_rate else None
        self._should_uninitialize = False
        self._enumerator_ptr = ctypes.c_void_p()
        self._device_ptr = ctypes.c_void_p()
        self._audio_client_ptr = ctypes.c_void_p()
        self._render_client_ptr = ctypes.c_void_p()
        self._mix_format_ptr = ctypes.c_void_p()
        self._buffer_frames = 0
        self._mix_format: WasapiMixFormat | None = None
        self._buffer_duration_ms = 0.0
        self._period_frames = 0
        self._event_handle = wintypes.HANDLE()
        self._started = False
        self._open()

    @property
    def sample_rate(self) -> int:
        if self._mix_format is None:
            raise RuntimeError("WASAPI 渲染器尚未初始化。")
        return self._mix_format.sample_rate

    @property
    def channels(self) -> int:
        if self._mix_format is None:
            raise RuntimeError("WASAPI 渲染器尚未初始化。")
        return self._mix_format.channels

    def _try_initialize_shared_low_latency(self, audio_client_ptr: ctypes.c_void_p, wave_format_ptr: ctypes.c_void_p) -> bool:
        audio_client3_ptr = _query_interface(audio_client_ptr, _IID_IAUDIOCLIENT3)
        if not audio_client3_ptr:
            return False
        try:
            audio_client3 = ctypes.cast(audio_client3_ptr, ctypes.POINTER(_IAudioClient3))
            audio_client2 = ctypes.cast(audio_client3_ptr, ctypes.POINTER(_IAudioClient2))

            properties = _AudioClientProperties(
                cbSize=ctypes.sizeof(_AudioClientProperties),
                bIsOffload=False,
                eCategory=AUDIO_CATEGORY_SPEECH,
                Options=AUDCLNT_STREAMOPTIONS_NONE,
            )
            hr = audio_client2.contents.lpVtbl.contents.SetClientProperties(audio_client3_ptr, ctypes.byref(properties))
            if not _succeeded(hr):
                logger.info("IAudioClient3 设置流属性失败，继续尝试默认低延迟周期: 0x{:08X}", ctypes.c_uint32(hr).value)

            default_period = wintypes.UINT()
            fundamental_period = wintypes.UINT()
            min_period = wintypes.UINT()
            max_period = wintypes.UINT()
            hr = audio_client3.contents.lpVtbl.contents.GetSharedModeEnginePeriod(
                audio_client3_ptr,
                ctypes.cast(wave_format_ptr, ctypes.POINTER(_WaveFormatEx)),
                ctypes.byref(default_period),
                ctypes.byref(fundamental_period),
                ctypes.byref(min_period),
                ctypes.byref(max_period),
            )
            if not _succeeded(hr):
                logger.info("IAudioClient3 读取共享模式周期失败，回退常规事件驱动: 0x{:08X}", ctypes.c_uint32(hr).value)
                return False

            target_period = _align_period_to_fundamental(
                int(min_period.value or default_period.value or 0),
                fundamental_frames=int(fundamental_period.value or 0),
                min_frames=int(min_period.value or 0),
                max_frames=max(int(max_period.value or 0), int(min_period.value or 0), 1),
            )
            if target_period <= 0:
                return False

            hr = audio_client3.contents.lpVtbl.contents.InitializeSharedAudioStream(
                audio_client3_ptr,
                AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                target_period,
                ctypes.cast(wave_format_ptr, ctypes.POINTER(_WaveFormatEx)),
                None,
            )
            if not _succeeded(hr):
                logger.info(
                    "IAudioClient3 低延迟共享流初始化失败，回退常规事件驱动: 0x{:08X} target_period={} default={} min={} max={}",
                    ctypes.c_uint32(hr).value,
                    target_period,
                    int(default_period.value),
                    int(min_period.value),
                    int(max_period.value),
                )
                return False

            self._period_frames = target_period
            logger.info(
                "IAudioClient3 低延迟共享流已启用: label={} target_period={} default={} min={} max={} fundamental={}",
                self.label,
                int(target_period),
                int(default_period.value),
                int(min_period.value),
                int(max_period.value),
                int(fundamental_period.value),
            )
            return True
        finally:
            _release(audio_client3_ptr)

    def _open(self) -> None:
        initialized, should_uninitialize = _co_initialize()
        if not initialized:
            raise RuntimeError("初始化 COM 失败，无法创建 WASAPI 播放设备。")
        self._should_uninitialize = should_uninitialize
        hr = _ole32.CoCreateInstance(
            ctypes.byref(_CLSID_MMDEVICE_ENUMERATOR),
            None,
            CLSCTX_ALL,
            ctypes.byref(_IID_IMMDEVICE_ENUMERATOR),
            ctypes.byref(self._enumerator_ptr),
        )
        _check_hresult(hr, "创建 IMMDeviceEnumerator")

        enumerator = ctypes.cast(self._enumerator_ptr, ctypes.POINTER(_IMMDeviceEnumerator))
        if self.endpoint_id:
            hr = enumerator.contents.lpVtbl.contents.GetDevice(
                self._enumerator_ptr,
                self.endpoint_id,
                ctypes.byref(self._device_ptr),
            )
            _check_hresult(hr, "读取指定输出端点")
        else:
            hr = enumerator.contents.lpVtbl.contents.GetDefaultAudioEndpoint(
                self._enumerator_ptr,
                E_RENDER,
                E_MULTIMEDIA,
                ctypes.byref(self._device_ptr),
            )
            _check_hresult(hr, "读取默认输出端点")

        device = ctypes.cast(self._device_ptr, ctypes.POINTER(_IMMDevice))
        hr = device.contents.lpVtbl.contents.Activate(
            self._device_ptr,
            ctypes.byref(_IID_IAUDIOCLIENT),
            CLSCTX_ALL,
            None,
            ctypes.byref(self._audio_client_ptr),
        )
        _check_hresult(hr, "激活 IAudioClient")

        audio_client = ctypes.cast(self._audio_client_ptr, ctypes.POINTER(_IAudioClient))
        hr = audio_client.contents.lpVtbl.contents.GetMixFormat(self._audio_client_ptr, ctypes.byref(self._mix_format_ptr))
        _check_hresult(hr, "读取设备混音格式")
        active_wave_format_ptr = self._mix_format_ptr
        self._mix_format = _parse_mix_format(self._mix_format_ptr)
        preferred_wave_format: _WaveFormatEx | _WaveFormatExtensible | None = None
        if self.preferred_sample_rate and self._mix_format.sample_rate != self.preferred_sample_rate:
            preferred_wave_format = _build_preferred_wave_format(
                self._mix_format_ptr,
                sample_rate=self.preferred_sample_rate,
            )
            preferred_wave_format_ptr = ctypes.cast(ctypes.pointer(preferred_wave_format), ctypes.c_void_p)
            closest_match_ptr = ctypes.c_void_p()
            hr = audio_client.contents.lpVtbl.contents.IsFormatSupported(
                self._audio_client_ptr,
                AUDCLNT_SHAREMODE_SHARED,
                ctypes.cast(preferred_wave_format_ptr, ctypes.POINTER(_WaveFormatEx)),
                ctypes.byref(closest_match_ptr),
            )
            if closest_match_ptr:
                _ole32.CoTaskMemFree(closest_match_ptr)
            if hr == S_OK:
                active_wave_format_ptr = preferred_wave_format_ptr
                self._mix_format = _parse_mix_format(active_wave_format_ptr)
                logger.info(
                    "WASAPI 端点将优先使用指定采样率: label={} sample_rate={} endpoint={}",
                    self.label,
                    self._mix_format.sample_rate,
                    self.endpoint_id or "default",
                )
            else:
                logger.info(
                    "WASAPI 端点不支持首选采样率，回退设备混音格式: label={} preferred={} actual={} endpoint={}",
                    self.label,
                    int(self.preferred_sample_rate),
                    self._mix_format.sample_rate,
                    self.endpoint_id or "default",
                )
        self._event_handle = _kernel32.CreateEventW(None, False, False, None)
        if not self._event_handle:
            raise OSError(f"创建 WASAPI 事件句柄失败: {ctypes.get_last_error()}")

        low_latency_enabled = self._try_initialize_shared_low_latency(self._audio_client_ptr, active_wave_format_ptr)
        if not low_latency_enabled:
            hr = audio_client.contents.lpVtbl.contents.Initialize(
                self._audio_client_ptr,
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                _DEFAULT_BUFFER_DURATION,
                REFERENCE_TIME(0),
                ctypes.cast(active_wave_format_ptr, ctypes.POINTER(_WaveFormatEx)),
                None,
            )
            _check_hresult(hr, "初始化 IAudioClient")

        hr = audio_client.contents.lpVtbl.contents.SetEventHandle(
            self._audio_client_ptr,
            self._event_handle,
        )
        _check_hresult(hr, "绑定 WASAPI 事件句柄")

        buffer_frames = wintypes.UINT()
        hr = audio_client.contents.lpVtbl.contents.GetBufferSize(self._audio_client_ptr, ctypes.byref(buffer_frames))
        _check_hresult(hr, "读取 WASAPI 缓冲区大小")
        self._buffer_frames = max(1, int(buffer_frames.value))
        if self._period_frames <= 0:
            self._period_frames = self._buffer_frames
        self._buffer_duration_ms = (self._buffer_frames / max(1, self.sample_rate)) * 1000.0

        hr = audio_client.contents.lpVtbl.contents.GetService(
            self._audio_client_ptr,
            ctypes.byref(_IID_IAUDIORENDERCLIENT),
            ctypes.byref(self._render_client_ptr),
        )
        _check_hresult(hr, "获取 IAudioRenderClient")

        hr = audio_client.contents.lpVtbl.contents.Start(self._audio_client_ptr)
        _check_hresult(hr, "启动 WASAPI 播放流")
        self._started = True
        logger.info(
            "已创建纯 IAudioClient 播放端点: label={} sample_rate={} channels={} bytes_per_sample={} period_frames={} buffer_frames={} endpoint={}",
            self.label,
            self.sample_rate,
            self.channels,
            self._mix_format.bytes_per_sample if self._mix_format is not None else 0,
            self._period_frames,
            self._buffer_frames,
            self.endpoint_id or "default",
        )

    def _wait_until_buffer_ready(self) -> None:
        effective_period_ms = (self._period_frames / max(1, self.sample_rate)) * 1000.0 if self._period_frames > 0 else 0.0
        timeout_ms = max(3, min(60, int(round(max(effective_period_ms, 1.0) * 3.0))))
        result = _kernel32.WaitForSingleObject(self._event_handle, timeout_ms)
        if result in (WAIT_OBJECT_0, WAIT_TIMEOUT):
            return
        raise OSError(f"等待 WASAPI 事件失败: 0x{ctypes.get_last_error():08X}")

    def _current_padding(self) -> int:
        if not self._audio_client_ptr:
            return 0
        padding = wintypes.UINT()
        audio_client = ctypes.cast(self._audio_client_ptr, ctypes.POINTER(_IAudioClient))
        hr = audio_client.contents.lpVtbl.contents.GetCurrentPadding(self._audio_client_ptr, ctypes.byref(padding))
        _check_hresult(hr, "读取当前播放缓冲占用")
        return int(padding.value)

    def _convert_audio(self, audio_np: np.ndarray) -> bytes:
        if self._mix_format is None:
            raise RuntimeError("WASAPI 渲染器尚未初始化。")
        coerced = _coerce_channel_count(audio_np, self._mix_format.channels)
        clipped = np.ascontiguousarray(np.clip(coerced, -1.0, 1.0), dtype=np.float32)
        if self._mix_format.is_float:
            if self._mix_format.bytes_per_sample != 4:
                raise RuntimeError(f"不支持的 IEEE_FLOAT 样本宽度: {self._mix_format.bytes_per_sample} 字节")
            return clipped.astype("<f4", copy=False).tobytes()
        return _float_to_pcm_bytes(clipped, bytes_per_sample=self._mix_format.bytes_per_sample)

    def write(self, audio_np: np.ndarray) -> None:
        if audio_np.size == 0:
            return
        if self._mix_format is None:
            raise RuntimeError("WASAPI 渲染器尚未初始化。")
        audio_bytes = self._convert_audio(audio_np)
        total_frames = len(audio_bytes) // self._mix_format.bytes_per_frame
        if total_frames <= 0:
            return

        render_client = ctypes.cast(self._render_client_ptr, ctypes.POINTER(_IAudioRenderClient))
        frames_written = 0
        byte_offset = 0
        while frames_written < total_frames:
            available_frames = max(0, self._buffer_frames - self._current_padding())
            if available_frames <= 0:
                self._wait_until_buffer_ready()
                continue

            frames_to_write = min(available_frames, total_frames - frames_written)
            buffer_ptr = ctypes.c_void_p()
            hr = render_client.contents.lpVtbl.contents.GetBuffer(
                self._render_client_ptr,
                frames_to_write,
                ctypes.byref(buffer_ptr),
            )
            _check_hresult(hr, "申请 WASAPI 输出缓冲")

            byte_count = frames_to_write * self._mix_format.bytes_per_frame
            chunk = audio_bytes[byte_offset : byte_offset + byte_count]
            if chunk:
                ctypes.memmove(buffer_ptr, chunk, len(chunk))
            hr = render_client.contents.lpVtbl.contents.ReleaseBuffer(
                self._render_client_ptr,
                frames_to_write,
                0 if chunk else AUDCLNT_BUFFERFLAGS_SILENT,
            )
            _check_hresult(hr, "提交 WASAPI 输出缓冲")
            frames_written += frames_to_write
            byte_offset += byte_count

    def close(self) -> None:
        if self._audio_client_ptr and self._started:
            try:
                audio_client = ctypes.cast(self._audio_client_ptr, ctypes.POINTER(_IAudioClient))
                audio_client.contents.lpVtbl.contents.Stop(self._audio_client_ptr)
                audio_client.contents.lpVtbl.contents.Reset(self._audio_client_ptr)
            except Exception:
                pass
            self._started = False
        if self._mix_format_ptr:
            _ole32.CoTaskMemFree(self._mix_format_ptr)
            self._mix_format_ptr = ctypes.c_void_p()
        _release(self._render_client_ptr)
        _release(self._audio_client_ptr)
        _release(self._device_ptr)
        _release(self._enumerator_ptr)
        self._render_client_ptr = ctypes.c_void_p()
        self._audio_client_ptr = ctypes.c_void_p()
        self._device_ptr = ctypes.c_void_p()
        self._enumerator_ptr = ctypes.c_void_p()
        if self._event_handle:
            _kernel32.CloseHandle(self._event_handle)
            self._event_handle = wintypes.HANDLE()
        if self._should_uninitialize:
            _ole32.CoUninitialize()
            self._should_uninitialize = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
