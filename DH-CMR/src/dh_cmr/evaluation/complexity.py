import time

import torch


@torch.inference_mode()
def profile_model(model, batch, device, repeats=100, warmup=10):
    if repeats < 1 or warmup < 0 or len(batch.ids) != 1:
        raise ValueError("Profiling requires one query, positive repeats, and nonnegative warmup")
    model.eval()
    batch = batch.to(device)
    for _ in range(warmup):
        model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(repeats):
        model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    milliseconds = (time.perf_counter() - start) * 1000 / repeats
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=True) as profile:
        model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "profiled_operator_flops": sum(event.flops or 0 for event in profile.key_averages()),
            "latency_ms": milliseconds, "batch_size": 1, "repeats": repeats,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
