import builtins
import pickle
from pathlib import Path


class NumpyUnpickler(pickle.Unpickler):
    allowed_builtins = {"bytearray", "bytes", "complex", "dict", "float", "frozenset", "int", "list", "set", "slice", "str", "tuple"}
    allowed_numpy = {("numpy", "dtype"), ("numpy", "ndarray")}
    for prefix in ("numpy.core", "numpy._core"):
        allowed_numpy.update({(prefix + ".multiarray", "_reconstruct"), (prefix + ".multiarray", "scalar"),
                              (prefix + ".numeric", "_frombuffer")})

    def find_class(self, module, name):
        if module == "builtins" and name in self.allowed_builtins:
            return getattr(builtins, name)
        if (module, name) in self.allowed_numpy:
            return getattr(__import__(module, fromlist=[name]), name)
        raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}")


def load_payload(path, label_key):
    with Path(path).open("rb") as handle:
        data = NumpyUnpickler(handle).load()
    if not isinstance(data, dict) or not {"train", "valid", "test"} <= set(data):
        raise ValueError("Expected train, valid, and test dictionaries")
    for name in ("train", "valid", "test"):
        split = data[name]
        required = {"raw_text", "audio", "vision", "id", label_key}
        if not isinstance(split, dict) or not required <= set(split):
            raise ValueError(f"{name} requires fields {sorted(required)}")
        n = len(split["id"])
        if not n or any(len(split[k]) != n for k in required):
            raise ValueError(f"Inconsistent or empty {name} split")
        for key in ("audio_lengths", "vision_lengths"):
            if key in split and len(split[key]) != n:
                raise ValueError(f"{name}.{key} has an inconsistent length")
        if len({str(i) for i in split["id"]}) != n:
            raise ValueError(f"Duplicate sample identifiers in {name}")
    return {k: data[k] for k in ("train", "valid", "test")}
