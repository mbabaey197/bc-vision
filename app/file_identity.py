"""Stable cross-platform filesystem identities.

Older Python builds on Windows can report ``st_ino == 0`` for every file.
That value is not safe for ownership checks, hard-link detection, or quota
deduplication.  Windows therefore uses the volume serial and native file
index returned by ``GetFileInformationByHandle``.  POSIX keeps the normal
``(st_dev, st_ino)`` identity.
"""
from __future__ import annotations

import os
from pathlib import Path


FileIdentity = tuple[int, int]


if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("creation_time", wintypes.FILETIME),
            ("last_access_time", wintypes.FILETIME),
            ("last_write_time", wintypes.FILETIME),
            ("volume_serial_number", wintypes.DWORD),
            ("file_size_high", wintypes.DWORD),
            ("file_size_low", wintypes.DWORD),
            ("number_of_links", wintypes.DWORD),
            ("file_index_high", wintypes.DWORD),
            ("file_index_low", wintypes.DWORD),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _get_file_information = _kernel32.GetFileInformationByHandle
    _get_file_information.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ByHandleFileInformation),
    ]
    _get_file_information.restype = wintypes.BOOL
    _create_file = _kernel32.CreateFileW
    _create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _create_file.restype = wintypes.HANDLE
    _close_handle = _kernel32.CloseHandle
    _close_handle.argtypes = [wintypes.HANDLE]
    _close_handle.restype = wintypes.BOOL

    _FILE_READ_ATTRIBUTES = 0x0080
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _FILE_SHARE_DELETE = 0x00000004
    _OPEN_EXISTING = 3
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


def _windows_handle_identity(handle: int) -> FileIdentity:
    information = _ByHandleFileInformation()
    if not _get_file_information(
        wintypes.HANDLE(handle),
        ctypes.byref(information),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    file_index = (
        int(information.file_index_high) << 32
    ) | int(information.file_index_low)
    return int(information.volume_serial_number), file_index


def descriptor_file_identity(
    descriptor: int,
    *,
    details=None,
) -> FileIdentity:
    """Return the stable identity of an already-open descriptor."""

    if os.name == "nt":
        try:
            handle = msvcrt.get_osfhandle(descriptor)
        except OSError:
            raise
        return _windows_handle_identity(handle)
    current = details if details is not None else os.fstat(descriptor)
    return int(current.st_dev), int(current.st_ino)


def path_file_identity(path, *, details=None) -> FileIdentity:
    """Return a no-follow identity for a file or directory path."""

    target = Path(path)
    if os.name != "nt":
        current = details if details is not None else target.lstat()
        return int(current.st_dev), int(current.st_ino)

    handle = _create_file(
        str(target),
        _FILE_READ_ATTRIBUTES,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle == _INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return _windows_handle_identity(handle)
    finally:
        if not _close_handle(handle):
            raise ctypes.WinError(ctypes.get_last_error())
