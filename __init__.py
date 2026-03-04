import time
import torch
import safetensors.torch
import comfy
import comfy.utils

import mmap
import os
import re
import sys
import ctypes

# Save the original mmap constructor
_original_mmap = mmap.mmap

def get_filename_from_fd(fd):
    """Retrieves the absolute path of a file descriptor."""
    try:
        if os.name == 'nt':
            # On Windows, we can use the Win32 API via ctypes or a helper
            import msvcrt
            from ctypes import wintypes
            handle = msvcrt.get_osfhandle(fd)
            buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            ctypes.windll.kernel32.GetFinalPathNameByHandleW(handle, buf, wintypes.MAX_PATH, 0)
            return buf.value
        else:
            # On Linux/macOS, read from /proc/self/fd/
            return os.readlink(f"/proc/self/fd/{fd}")
    except Exception:
        return None

def prefetch_virtual_memory(mm_obj):
    """Applies OS-specific prefetch hints."""
    if os.name == 'nt':
        from ctypes import wintypes
        class WIN32_MEMORY_RANGE_ENTRY(ctypes.Structure):
            _fields_ = [("VirtualAddress", wintypes.LPVOID), ("NumberOfBytes", ctypes.c_size_t)]
        
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        # -1 is the pseudo-handle for current process
        entry = WIN32_MEMORY_RANGE_ENTRY()
        entry.VirtualAddress = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(mm_obj)), ctypes.c_void_p)
        entry.NumberOfBytes = len(mm_obj)
        kernel32.PrefetchVirtualMemory(kernel32.GetCurrentProcess(), 1, ctypes.byref(entry), 0)
    else:
        mm_obj.madvise(mmap.MADV_WILLNEED)

def patched_mmap(fileno, length, *args, **kwargs):
    """Wrapper that checks filename patterns before prefetching."""
    # 1. Call original mmap to create the object
    mm = _original_mmap(fileno, length, *args, **kwargs)
    
    # 2. Pattern to match (e.g., all .dat or .bin files)
    pattern = r".*\.(safetensors|sft|gguf|bin|pt)$"
    
    # 3. Check if fileno is a valid file (not -1 for anonymous memory)
    if fileno != -1:
        fname = get_filename_from_fd(fileno)
        if fname and re.match(pattern, fname, re.IGNORECASE):
            try:
                prefetch_virtual_memory(mm)
                if os.name == 'nt':
                    print(f"Applied PrefetchVirtualMemory to: {fname}")
                else:
                    print(f"Applied MADV_WILLNEED to: {fname}")
            except Exception as e:
                if os.name == 'nt':
                    print(f"PrefetchVirtualMemory failed for {fname}: {e}")
                else:
                    print(f"MADV_WILLNEED marking failed for {fname}: {e}")

    return mm

# Apply the monkeypatch
mmap.mmap = patched_mmap

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

