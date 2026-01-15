"""
Correct Evaluation Metrics for Video Scene Retrieval
- One-to-one timestamp matching
- Properly bounded MRR, MAP, NDCG, Precision, Recall
"""

import json
import numpy as np
from typing import List, Dict




def match_and_remove(
    obtained_time: float,
    remaining_expected: List[float],
    tolerance: float
):
    """
    Match obtained_time to the closest expected timestamp within tolerance.
    If matched, remove it from remaining_expected and return True.
    """
    for i, expected in enumerate(remaining_expected):
        if abs(obtained_time - expected) <= tolerance:
            remaining_expected.pop(i)
            return True
    return False



def reciprocal_rank(
    obtained_times: List[float],
    expected_times: List[float],
    tolerance: float
) -> float:
    """Reciprocal Rank (RR) for a single query."""
    remaining_expected = expected_times.copy()

    for rank, obtained in enumerate(obtained_times, start=1):
        if match_and_remove(obtained, remaining_expected, tolerance):
            return 1.0 / rank

    return 0.0


def average_precision(
    obtained_times: List[float],
    expected_times: List[float],
    tolerance: float
) -> float:
    """Average Precision (AP) for a single query."""
    if not expected_times:
        return 0.0

    remaining_expected = expected_times.copy()
    relevant_found = 0
    precision_sum = 0.0

    for k, obtained in enumerate(obtained_times, start=1):
        if match_and_remove(obtained, remaining_expected, tolerance):
            relevant_found += 1
            precision_sum += relevant_found / k

    return precision_sum / len(expected_times)


def dcg_at_k(
    obtained_times: List[float],
    expected_times: List[float],
    k: int,
    tolerance: float
) -> float:
    """Discounted Cumulative Gain (DCG@k)."""
    remaining_expected = expected_times.copy()
    dcg = 0.0

    for i, obtained in enumerate(obtained_times[:k], start=1):
        if match_and_remove(obtained, remaining_expected, tolerance):
            dcg += 1.0 / np.log2(i + 1)

    return dcg


def ideal_dcg_at_k(expected_times: List[float], k: int) -> float:
    """Ideal DCG assuming perfect ranking."""
    num_relevant = min(len(expected_times), k)
    return sum(1.0 / np.log2(i + 1) for i in range(1, num_relevant + 1))


def ndcg_at_k(
    obtained_times: List[float],
    expected_times: List[float],
    k: int,
    tolerance: float
) -> float:
    """Normalized DCG (NDCG@k)."""
    if not expected_times:
        return 0.0

    dcg = dcg_at_k(obtained_times, expected_times, k, tolerance)
    idcg = ideal_dcg_at_k(expected_times, k)

    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(
    obtained_times: List[float],
    expected_times: List[float],
    k: int,
    tolerance: float
) -> float:
    if not obtained_times:
        return 0.0

    remaining_expected = expected_times.copy()
    relevant = 0
    for obtained in obtained_times[:k]:
        if match_and_remove(obtained, remaining_expected, tolerance):
            relevant += 1

    return relevant / min(k, len(obtained_times))



def recall_at_k(
    obtained_times: List[float],
    expected_times: List[float],
    k: int,
    tolerance: float
) -> float:
    """Recall@k."""
    if not expected_times:
        return 0.0

    remaining_expected = expected_times.copy()
    relevant = 0

    for obtained in obtained_times[:k]:
        if match_and_remove(obtained, remaining_expected, tolerance):
            relevant += 1

    return relevant / len(expected_times)




def evaluate_results(
    results: List[Dict],
    k: int = 5,
    tolerance: float = 0.5
) -> Dict[str, float]:
    """Evaluate metrics over all queries in a dataset."""

    mrr, map_, ndcg, p_k, r_k = [], [], [], [], []

    for r in results:
        obtained = r["obtained_time"]
        expected = r["expected_time"]

        mrr.append(reciprocal_rank(obtained, expected, tolerance))
        map_.append(average_precision(obtained, expected, tolerance))
        ndcg.append(ndcg_at_k(obtained, expected, k, tolerance))
        p_k.append(precision_at_k(obtained, expected, k, tolerance))
        r_k.append(recall_at_k(obtained, expected, k, tolerance))

    return {
        "MRR": float(np.mean(mrr)),
        "MAP": float(np.mean(map_)),
        f"NDCG@{k}": float(np.mean(ndcg)),
        f"P@{k}": float(np.mean(p_k)),
        f"R@{k}": float(np.mean(r_k)),
        "Queries": len(results)
    }



if __name__ == "__main__":
    with open("scene_visual.json", "r") as f:
        data = json.load(f)

    metrics = evaluate_results(data, k=5, tolerance=0.5)

    print("\nEvaluation Results")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
