import numpy as np
from scipy.optimize import linear_sum_assignment
from sparrow_datums import FrameBoxes, pairwise_iou


class MODA:
    """A summable metric class to track the components of MODA"""

    def __init__(
        self, false_negatives: int = 0, false_positives: int = 0, n_truth: int = 0
    ) -> None:
        self.false_negatives = false_negatives
        self.false_positives = false_positives
        self.n_truth = n_truth

    def __add__(self, other: "MODA") -> "MODA":
        return MODA(
            false_negatives=self.false_negatives + other.false_negatives,
            false_positives=self.false_positives + other.false_positives,
            n_truth=self.n_truth + other.n_truth,
        )

    @property
    def value(self) -> float:
        n_errors = abs(self.false_negatives) + abs(self.false_positives)
        if self.n_truth == 0:
            return 0
        return 1 - n_errors / self.n_truth


def compute_moda(
    predicted_boxes: FrameBoxes,
    ground_truth_boxes: FrameBoxes,
    iou_threshold: float = 0.5,
) -> MODA:
    """
    Compute MODA for a set of predicted boxes

    Parameters
    ----------
    predicted_boxes : FrameBoxes
        Predicted boxes
    ground_truth_boxes : FrameBoxes
        Ground truth boxes
    iou_threshold : float
        The overlap threshold below which boxes are not considered the same
    """
    if len(predicted_boxes) == 0:
        n = len(ground_truth_boxes)
        return MODA(false_negatives=n, false_positives=0, n_truth=n)
    elif len(ground_truth_boxes) == 0:
        return MODA(false_negatives=0, false_positives=len(predicted_boxes), n_truth=0)
    cost = 1 - pairwise_iou(predicted_boxes, ground_truth_boxes)
    pred_indices, gt_indices = linear_sum_assignment(cost)

    false_positives = set(np.arange(len(predicted_boxes))) - set(pred_indices)
    false_negatives = set(np.arange(len(ground_truth_boxes))) - set(gt_indices)

    unmatched = cost[pred_indices, gt_indices] > iou_threshold
    false_positives |= set(pred_indices[unmatched])
    false_negatives |= set(gt_indices[unmatched])

    return MODA(
        false_negatives=len(false_negatives),
        false_positives=len(false_positives),
        n_truth=len(ground_truth_boxes),
    )
