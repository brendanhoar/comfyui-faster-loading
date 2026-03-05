import hashlib
import time
import torch
import safetensors.torch
import comfy
import comfy.utils

from mmap import mmap, ACCESS_READ, ACCESS_COPY
import os
import re
import sys
import ctypes

import logging
import builtins

from typing import IO
import os
import asyncio

import sys
from unittest.mock import MagicMock
sys.modules["blake3"] = MagicMock()
import app.assets.hashing

prefetch_pattern = r".*\.(safetensors|sft|gguf|bin|pt|ckpt)$"

# Save the original mmap constructor
_original_mmap = mmap

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
    return

def patched_mmap(fileno, length, *args, **kwargs):
    """Wrapper that checks filename patterns before prefetching."""
    # 1. Call original mmap to create the object
    print(f"calling original mmap()")
    mm = _original_mmap(fileno, length, *args, **kwargs)
    
    # 2. Pattern to match (e.g., all .dat or .bin files)
        
    # 3. Check if fileno is a valid file (not -1 for anonymous memory)
    if fileno != -1:
        fname = get_filename_from_fd(fileno)
        if fname and re.match(prefetch_pattern, fname, re.IGNORECASE):
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
        else:
            print(f"mmap() was called on a file, but it did not match the faster-loading file pattern.")
    else:
        print(f"mmap() was called, but not on a file object.")
    return mm

# Apply the monkeypatch
mmap = patched_mmap

# I need to track the mode flags to be sure I don't apply any of this to writeable file descriptors.

# Array to store active file info
active_files = []

# Save the original open function
original_open = builtins.open

def tracked_open(file, mode='r', *args, **kwargs):
    # Call the real open
    f = original_open(file, mode, *args, **kwargs)
    
    # Get file descriptor and metadata
    fd = f.fileno()
    file_info = {
        "fd": fd,
        "filename": os.path.abspath(file) if isinstance(file, str) else str(file),
        "mode": mode
    }
    active_files.append(file_info)
    print(f"TRACKED: Opened FD {fd} ({file_info['mode']}) ({file_info['filename']})")

    # Wrap the close method to remove from array
    original_close = f.close
    def tracked_close():
        # Remove from array before closing
        global active_files
        active_files = [item for item in active_files if item['fd'] != fd]
        print(f"UNTRACKED: Closed FD {fd} ({file_info['mode']}) ({file_info['filename']})")
        return original_close()

    # Re-bind the close method on this specific instance
    f.close = tracked_close
    
    return f

# Apply the global monkeypatch
builtins.open = tracked_open

# --- Test Case ---
# print("Initial active files:", active_files)
# 
# with open("example.txt", "w") as my_file:
#     print("Active during 'with' block:", active_files)
#     # The file is automatically removed from active_files when the block exits
# 
# print("Active after 'with' block:", active_files)

_load_torch_file_org = comfy.utils.load_torch_file

def _load_torch_file_with_precache(ckpt, safe_load=False, device=None, return_metadata=False):
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        #start=time.time()
        #print("Starting to preload model {0}".format(ckpt))
        #with open(ckpt, "rb") as f:
        #	#sd_cache= f.read() # why store the value in RAM if we're not using it directly? f.read() can just...read the file without storing the data locally...that should be sufficient to load the OS cache if no memory pressure, yes?
        #	f.read() # in the future, perhaps replace with an mmap handle with, if available, MADV_SEQUENTIAL and MADV_WILLNEED to encourage large chunking and readahead before calling load_torch_file, which invokes the slow safetensors methods.
        #end=time.time()
        #print("Completed preload in {0} seconds. Preloaded model: {1}".format(end-start,ckpt))
        #we don't need to keep the sd_cache object, we just want to force the OS to cache the file, so that invoking the normal path below will avoid the actual drive IO.
        #this may incur a memory penalty during load.
        f = open(ckpt, "rb")
        print(f"Starting to mmap {ckpt}")
        if os.name == 'nt':
            m = mmap(f.fileno(), length=0, access=ACCESS_COPY)
        print(f"Passing {ckpt} to torch_load_file")
        t = _load_torch_file_org(ckpt, safe_load, device, return_metadata) 
        print(f"Returned from torch_load_file of {ckpt}")
        m.close()
        f.close()
    return t

comfy.utils.load_torch_file = _load_torch_file_with_precache

_load_file_org = safetensors.torch.load_file


def _load_file_for_wsl(filename, device="cpu", *args, **kwargs):
    try:
        if device == "cpu":
            with open(filename, "rb") as f:
                print(f"Calling torch.load and passing f.read() of {ckpt}")
                return safetensors.torch.load(f.read())
                print(f"Returned from torch.load and passing f.read() of {ckpt}")
    except Exception:
        pass
    return _load_file_org(filename, device, *args, **kwargs)


safetensors.torch.load_file = _load_file_for_wsl

#-----------------------


DEFAULT_CHUNK = 8 * 1024 *1024 # 8MB

_patched_hash_file_object = app.assets.hashing._hash_file_object

def _hash_file_obj_precache(file_obj, chunk_size) -> str:
    fileno=file_obj.fileno()       
    if fileno != -1:
        fname = get_filename_from_fd(fileno)
        if fname and re.match(prefetch_pattern, fname, re.IGNORECASE):
            try:
                mm = mmap(fileno, length=0, access=ACCESS_COPY)
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
        else:
            print(f"mmap() was called on a file, but it did not match the faster-loading file pattern.")
    else:
        print(f"mmap() was called, but not on a file object.")
    hfo = _hash_file_object(file_obj, chunk_size) 
    if mm:
        mm.close()
    return hfo

app.assets.hashing._hash_file_object = _hash_file_object_precache

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

