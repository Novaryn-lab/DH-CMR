import numpy as np


def weighted_f1(target, predicted):
    score = 0.0
    for label in (False, True):
        tp = np.sum((target == label) & (predicted == label))
        fp = np.sum((target != label) & (predicted == label))
        fn = np.sum((target == label) & (predicted != label))
        denominator = 2 * tp + fp + fn
        score += np.mean(target == label) * (2 * tp / denominator if denominator else 0.0)
    return float(score * 100)


def compute_metrics(labels, predictions, dataset="mosi", sims_boundaries=(-0.7, -0.1, 0.1, 0.7)):
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if not len(y) or y.shape != p.shape or not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("Metrics require equal-length, nonempty, finite labels and predictions")
    result = {"MAE": float(np.abs(y - p).mean()),
              "Corr": float(np.corrcoef(y, p)[0, 1]) if y.std() > 1e-12 and p.std() > 1e-12 else 0.0,
              "Acc_2_nonnegative": float(np.mean((y >= 0) == (p >= 0)) * 100),
              "F1_nonnegative": weighted_f1(y >= 0, p >= 0)}
    nonzero = y != 0
    result["Acc_2_nonzero"] = float(np.mean((y[nonzero] > 0) == (p[nonzero] > 0)) * 100) if nonzero.any() else None
    result["F1_nonzero"] = weighted_f1(y[nonzero] > 0, p[nonzero] > 0) if nonzero.any() else None
    if dataset == "ch_sims":
        boundaries = np.asarray(sims_boundaries, dtype=np.float32).astype(np.float64)
        result["Acc_5"] = float(np.mean(np.digitize(np.clip(y, -1, 1), boundaries, right=True) ==
                                        np.digitize(np.clip(p, -1, 1), boundaries, right=True)) * 100)
    else:
        result["Acc_7"] = float(np.mean(np.rint(np.clip(y, -3, 3)) == np.rint(np.clip(p, -3, 3))) * 100)
    return result
