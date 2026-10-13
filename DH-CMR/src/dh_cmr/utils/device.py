import torch


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def autocast_options(device, precision):
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 requires a CUDA device")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bf16")
    return {"device_type": device.type, "enabled": precision != "fp32",
            "dtype": torch.float16 if precision == "fp16" else torch.bfloat16}
