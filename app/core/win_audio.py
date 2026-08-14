from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass
from ctypes import wintypes

from app.core.logging import get_logger

logger = get_logger(__name__)

if hasattr(ctypes, "WINFUNCTYPE"):
    _COM_METHOD = ctypes.WINFUNCTYPE
else:
    _COM_METHOD = ctypes.CFUNCTYPE

HRESULT = ctypes.c_long
CLSCTX_ALL = 0x17
COINIT_APARTMENTTHREADED = 0x2
DEVICE_STATE_ACTIVE = 0x00000001
STGM_READ = 0x00000000
VT_LPWSTR = 31
E_RENDER = 0
E_CAPTURE = 1
E_MULTIMEDIA = 1
S_OK = 0
S_FALSE = 1
RPC_E_CHANGED_MODE = ctypes.c_long(0x80010106).value

_ENUMERATION_LOCK = threading.Lock()
_ENUMERATION_CACHE: tuple[float, "WindowsAudioSnapshot"] | None = None
_ENUMERATION_CACHE_TTL_SECONDS = 1.0


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


class _PropertyKey(ctypes.Structure):
    _fields_ = [
        ("fmtid", _Guid),
        ("pid", wintypes.DWORD),
    ]


class _VariantUnion(ctypes.Union):
    _fields_ = [
        ("pwszVal", wintypes.LPWSTR),
        ("uhVal", ctypes.c_ulonglong),
        ("llVal", ctypes.c_longlong),
    ]


class _PropVariant(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [
        ("vt", wintypes.USHORT),
        ("wReserved1", wintypes.USHORT),
        ("wReserved2", wintypes.USHORT),
        ("wReserved3", wintypes.USHORT),
        ("data", _VariantUnion),
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
    ]


class _IMMDeviceCollectionVTable(ctypes.Structure):
    _fields_ = [
        ("QueryInterface", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p))),
        ("AddRef", _COM_METHOD(ctypes.c_ulong, ctypes.c_void_p)),
        ("Release", _COM_METHOD(ctypes.c_ulong, ctypes.c_void_p)),
        ("GetCount", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(wintypes.UINT))),
        ("Item", _COM_METHOD(HRESULT, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))),
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


class _IPropertyStoreVTable(ctypes.Structure):
    _fields_ = [
        ("QueryInterface", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p))),
        ("AddRef", _COM_METHOD(ctypes.c_ulong, ctypes.c_void_p)),
        ("Release", _COM_METHOD(ctypes.c_ulong, ctypes.c_void_p)),
        ("GetCount", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD))),
        ("GetAt", _COM_METHOD(HRESULT, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_PropertyKey))),
        ("GetValue", _COM_METHOD(HRESULT, ctypes.c_void_p, ctypes.POINTER(_PropertyKey), ctypes.POINTER(_PropVariant))),
    ]


class _IUnknown(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_IUnknownVTable))]


class _IMMDeviceEnumerator(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_IMMDeviceEnumeratorVTable))]


class _IMMDeviceCollection(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_IMMDeviceCollectionVTable))]


class _IMMDevice(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_IMMDeviceVTable))]


class _IPropertyStore(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_IPropertyStoreVTable))]


@dataclass(slots=True)
class WindowsAudioEndpoint:
    endpoint_id: str
    name: str
    flow: str
    is_default: bool


@dataclass(slots=True)
class WindowsAudioSnapshot:
    inputs: list[WindowsAudioEndpoint]
    outputs: list[WindowsAudioEndpoint]
    default_input_id: str | None
    default_output_id: str | None


_CLSID_MMDEVICE_ENUMERATOR = _Guid.from_string("BCDE0395-E52F-467C-8E3D-C4579291692E")
_IID_IMMDEVICE_ENUMERATOR = _Guid.from_string("A95664D2-9614-4F35-A746-DE8DB63617E6")
_PKEY_DEVICE_FRIENDLY_NAME = _PropertyKey(
    _Guid.from_string("A45C254E-DF1C-4EFD-8020-67D146A850E0"),
    14,
)

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

_propsys = ctypes.OleDLL("ole32")
_propsys.PropVariantClear.argtypes = [ctypes.POINTER(_PropVariant)]
_propsys.PropVariantClear.restype = HRESULT


def _succeeded(hr: int) -> bool:
    return hr >= 0


def _release(ptr: ctypes.c_void_p | None) -> None:
    if ptr is None or not ptr:
        return
    unknown = ctypes.cast(ptr, ctypes.POINTER(_IUnknown))
    unknown.contents.lpVtbl.contents.Release(ptr)


def _co_initialize() -> tuple[bool, bool]:
    hr = _ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    if hr in (S_OK, S_FALSE):
        return True, True
    if hr == RPC_E_CHANGED_MODE:
        return True, False
    return False, False


def _get_device_id(device_ptr: ctypes.c_void_p) -> str:
    device = ctypes.cast(device_ptr, ctypes.POINTER(_IMMDevice))
    value = wintypes.LPWSTR()
    hr = device.contents.lpVtbl.contents.GetId(device_ptr, ctypes.byref(value))
    if not _succeeded(hr) or not value:
        raise OSError(f"IMMDevice.GetId failed: 0x{ctypes.c_uint32(hr).value:08X}")
    try:
        return str(value.value or "").strip()
    finally:
        _ole32.CoTaskMemFree(value)


def _get_device_name(device_ptr: ctypes.c_void_p) -> str:
    device = ctypes.cast(device_ptr, ctypes.POINTER(_IMMDevice))
    store_ptr = ctypes.c_void_p()
    hr = device.contents.lpVtbl.contents.OpenPropertyStore(device_ptr, STGM_READ, ctypes.byref(store_ptr))
    if not _succeeded(hr) or not store_ptr:
        raise OSError(f"IMMDevice.OpenPropertyStore failed: 0x{ctypes.c_uint32(hr).value:08X}")
    prop = _PropVariant()
    try:
        store = ctypes.cast(store_ptr, ctypes.POINTER(_IPropertyStore))
        hr = store.contents.lpVtbl.contents.GetValue(store_ptr, ctypes.byref(_PKEY_DEVICE_FRIENDLY_NAME), ctypes.byref(prop))
        if not _succeeded(hr):
            raise OSError(f"IPropertyStore.GetValue failed: 0x{ctypes.c_uint32(hr).value:08X}")
        if prop.vt == VT_LPWSTR and prop.pwszVal:
            return str(prop.pwszVal).strip()
        return ""
    finally:
        _propsys.PropVariantClear(ctypes.byref(prop))
        _release(store_ptr)


def _enumerate_flow(flow: int, flow_name: str) -> tuple[list[WindowsAudioEndpoint], str | None]:
    enumerator_ptr = ctypes.c_void_p()
    hr = _ole32.CoCreateInstance(
        ctypes.byref(_CLSID_MMDEVICE_ENUMERATOR),
        None,
        CLSCTX_ALL,
        ctypes.byref(_IID_IMMDEVICE_ENUMERATOR),
        ctypes.byref(enumerator_ptr),
    )
    if not _succeeded(hr) or not enumerator_ptr:
        raise OSError(f"CoCreateInstance(IMMDeviceEnumerator) failed: 0x{ctypes.c_uint32(hr).value:08X}")

    default_id: str | None = None
    collection_ptr = ctypes.c_void_p()
    try:
        enumerator = ctypes.cast(enumerator_ptr, ctypes.POINTER(_IMMDeviceEnumerator))
        default_device_ptr = ctypes.c_void_p()
        hr = enumerator.contents.lpVtbl.contents.GetDefaultAudioEndpoint(
            enumerator_ptr,
            flow,
            E_MULTIMEDIA,
            ctypes.byref(default_device_ptr),
        )
        if _succeeded(hr) and default_device_ptr:
            try:
                default_id = _get_device_id(default_device_ptr)
            finally:
                _release(default_device_ptr)

        hr = enumerator.contents.lpVtbl.contents.EnumAudioEndpoints(
            enumerator_ptr,
            flow,
            DEVICE_STATE_ACTIVE,
            ctypes.byref(collection_ptr),
        )
        if not _succeeded(hr) or not collection_ptr:
            raise OSError(f"EnumAudioEndpoints failed: 0x{ctypes.c_uint32(hr).value:08X}")

        collection = ctypes.cast(collection_ptr, ctypes.POINTER(_IMMDeviceCollection))
        count = wintypes.UINT()
        hr = collection.contents.lpVtbl.contents.GetCount(collection_ptr, ctypes.byref(count))
        if not _succeeded(hr):
            raise OSError(f"IMMDeviceCollection.GetCount failed: 0x{ctypes.c_uint32(hr).value:08X}")

        devices: list[WindowsAudioEndpoint] = []
        for index in range(int(count.value)):
            device_ptr = ctypes.c_void_p()
            hr = collection.contents.lpVtbl.contents.Item(collection_ptr, index, ctypes.byref(device_ptr))
            if not _succeeded(hr) or not device_ptr:
                continue
            try:
                endpoint_id = _get_device_id(device_ptr)
                name = _get_device_name(device_ptr) or endpoint_id
                devices.append(
                    WindowsAudioEndpoint(
                        endpoint_id=endpoint_id,
                        name=name,
                        flow=flow_name,
                        is_default=endpoint_id == default_id,
                    )
                )
            finally:
                _release(device_ptr)
        return devices, default_id
    finally:
        _release(collection_ptr)
        _release(enumerator_ptr)


def list_windows_audio_endpoints() -> WindowsAudioSnapshot:
    global _ENUMERATION_CACHE
    with _ENUMERATION_LOCK:
        now = time.monotonic()
        if _ENUMERATION_CACHE is not None and now - _ENUMERATION_CACHE[0] < _ENUMERATION_CACHE_TTL_SECONDS:
            return _ENUMERATION_CACHE[1]
        initialized, should_uninitialize = _co_initialize()
        if not initialized:
            logger.warning("初始化 Windows 音频端点 COM 失败，将返回空设备列表")
            snapshot = WindowsAudioSnapshot(inputs=[], outputs=[], default_input_id=None, default_output_id=None)
            _ENUMERATION_CACHE = (now, snapshot)
            return snapshot
        try:
            outputs, default_output_id = _enumerate_flow(E_RENDER, "output")
            inputs, default_input_id = _enumerate_flow(E_CAPTURE, "input")
            snapshot = WindowsAudioSnapshot(
                inputs=inputs,
                outputs=outputs,
                default_input_id=default_input_id,
                default_output_id=default_output_id,
            )
        except Exception as exc:
            logger.warning("读取 Windows 音频端点失败: {}", exc)
            snapshot = WindowsAudioSnapshot(inputs=[], outputs=[], default_input_id=None, default_output_id=None)
        finally:
            if should_uninitialize:
                _ole32.CoUninitialize()
        _ENUMERATION_CACHE = (now, snapshot)
        return snapshot
