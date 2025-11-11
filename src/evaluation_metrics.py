import numpy as np

def _to_label_sets(y_true):
    if isinstance(y_true, np.ndarray) and y_true.ndim == 2 and y_true.dtype != object:
        return [set(np.flatnonzero(row)) for row in y_true]
    sets = []
    for val in y_true:
        if isinstance(val, (int, np.integer, float)):
            sets.append({int(val)})
        elif isinstance(val, (list, set, tuple, np.ndarray)):
            sets.append(set(val))
        else:
            raise ValueError(f"Unsupported label type: {type(val)}")
    return sets

def rank_predictions(y_scores):
    return np.argsort(-y_scores, axis=1)

def topk_accuracy(y_true, y_scores, k):
    true_sets = _to_label_sets(y_true)
    ranks = rank_predictions(y_scores)
    topk = ranks[:, :k]
    return np.mean([
        float(any(lbl in true_sets[i] for lbl in topk[i]))
        for i in range(len(true_sets))
    ])

def precision_at_k(y_true, y_scores, k):
    true_sets = _to_label_sets(y_true)
    ranks = rank_predictions(y_scores)
    topk = ranks[:, :k]
    return np.mean([
        sum(lbl in true_sets[i] for lbl in topk[i]) / k
        for i in range(len(true_sets))
    ])