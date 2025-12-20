import ctypes
from ctypes import wintypes
import os

# Ensure the native extension is loaded (and thus pycauset_core.dll is loaded)
import pycauset  # noqa: F401
import pycauset._pycauset  # noqa: F401

psapi = ctypes.WinDLL("psapi")
kernel32 = ctypes.WinDLL("kernel32")

EnumProcessModules = psapi.EnumProcessModules
EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
EnumProcessModules.restype = wintypes.BOOL

GetModuleFileNameExW = psapi.GetModuleFileNameExW
GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
GetModuleFileNameExW.restype = wintypes.DWORD

hProcess = kernel32.GetCurrentProcess()
arr = (wintypes.HMODULE * 2048)()
needed = wintypes.DWORD()
if not EnumProcessModules(hProcess, arr, ctypes.sizeof(arr), ctypes.byref(needed)):
    raise ctypes.WinError()

count = needed.value // ctypes.sizeof(wintypes.HMODULE)
paths: list[str] = []
for i in range(count):
    buf = ctypes.create_unicode_buffer(260)
    if GetModuleFileNameExW(hProcess, arr[i], buf, 260):
        paths.append(buf.value)

matches = [p for p in paths if "pycauset_core" in os.path.basename(p).lower()]
print("pycauset_core matches:")
for p in matches:
    print(" ", p)
