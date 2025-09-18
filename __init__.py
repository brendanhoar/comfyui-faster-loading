import torch
import safetensors.torch
import comfy
import comfy.utils
from comfy.utils import load_torch_file

_load_file_org = safetensors.torch.load_file


def _load_file_for_wsl(filename, device="cpu", *args, **kwargs):
    try:
        if device == "cpu":
            with open(filename, "rb") as f:
                return safetensors.torch.load(f.read())
    except Exception:
        pass
    return _load_file_org(filename, device, *args, **kwargs)


safetensors.torch.load_file = _load_file_for_wsl

# New workaround for slow loading

_load_torch_file_org = comfy.utils.load_torch_file

def _load_torch_file_with_precache(ckpt, safe_load=False, device=None, return_metadata=False):
	if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
		with open(ckpt, "rb") as f:
			sd_cache= f.read()

	#we don't need to keep the sd_cache object, we just want to force the OS to cache the file, so that invoking the normal path below will avoid the actual drive IO.
    #this may incur a memory penalty during load.

	return _load_torch_file_org(ckpt, safe_load, device, return_metadata)

'''
# this commented out patch fails in some circumstances, to be discarded
def load_torch_file_for_slow(ckpt, safe_load=False, device=None, return_metadata=False):
    if device is None:
        device = torch.device("cpu")
    metadata = None
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        sd = safetensors.torch.load(open(ckpt, 'rb').read())
    else:
        if safe_load or ALWAYS_SAFE_LOAD:
            pl_sd = safetensors.torch.load(ckpt, map_location=device, weights_only=True)
        else:
            pl_sd = safetensors.torch.load(ckpt, map_location=device, pickle_module=comfy.checkpoint_pickle)
        if "global_step" in pl_sd:
            logging.debug(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            if len(pl_sd) == 1:
                key = list(pl_sd.keys())[0]
                sd = pl_sd[key]
                if not isinstance(sd, dict):
                    sd = pl_sd
            else:
                sd = pl_sd
    return (sd, metadata) if return_metadata else sd
'''

comfy.utils.load_torch_file = _load_torch_file_with_precache

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
