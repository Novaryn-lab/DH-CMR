SWEEPS = {
    "local_anchors": ("reference.anchors", [16, 32, 48, 64, 128]),
    "global_clusters": ("reference.global_clusters", [2, 4, 8, 16, 32]),
    "local_neighbors": ("hrp.local_neighbors", [2, 3, 4, 6, 8]),
    "global_topk": ("hrp.global_topk", [1, 2, 3, 4, 6]),
    "layers": ("model.layers", [1, 2, 3, 4, 5]),
    "experts": ("rmoe.experts", [1, 2, 3, 4, 6]),
}
