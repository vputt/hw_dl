from collections.abc import Sequence

import numpy as np

from calculate_eer import compute_eer


def compute_eer_percent(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[float, float]:
    """Считает EER в процентах для bonafide=1 и spoof=0"""
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    target_scores = score_array[label_array == 1]
    nontarget_scores = score_array[label_array == 0]
    eer, threshold = compute_eer(target_scores, nontarget_scores)
    return float(eer * 100.0), float(threshold)
