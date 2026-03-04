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

import builtins

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
    logging.debug(f"calling original mmap()")
    mm = _original_mmap(fileno, length, *args, **kwargs)
    
    # 2. Pattern to match (e.g., all .dat or .bin files)
    pattern = r".*\.(safetensors|sft|gguf|bin|pt|ckpt)$"
    
    # 3. Check if fileno is a valid file (not -1 for anonymous memory)
    if fileno != -1:
        fname = get_filename_from_fd(fileno)
        if fname and re.match(pattern, fname, re.IGNORECASE):
            try:
                prefetch_virtual_memory(mm)
                if os.name == 'nt':
                    logging.debug(f"Applied PrefetchVirtualMemory to: {fname}")
                else:
                    logging.debug(f"Applied MADV_WILLNEED to: {fname}")
            except Exception as e:
                if os.name == 'nt':
                    logging.debug(f"PrefetchVirtualMemory failed for {fname}: {e}")
                else:
                    logging.debug(f"MADV_WILLNEED marking failed for {fname}: {e}")
        else:
            logging.debug(f"mmap() was called on a file, but it did not match the faster-loading file pattern.")
    else:
        logging.debug(f"mmap() was called, but not on a file object.")
    return mm

# Apply the monkeypatch
mmap.mmap = patched_mmap

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
    logging.debug(f"TRACKED: Opened FD {fd} ({file_info['mode']}) ({file_info['filename']})")

    # Wrap the close method to remove from array
    original_close = f.close
    def tracked_close():
        # Remove from array before closing
        global active_files
        active_files = [item for item in active_files if item['fd'] != fd]
        logging.debug(f"UNTRACKED: Closed FD {fd} ({file_info['mode']}) ({file_info['filename']})")
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


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

