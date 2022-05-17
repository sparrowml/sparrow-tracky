from __future__ import annotations

from collections import defaultdict
from typing import Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from sparrow_datums import FrameAugmentedBoxes, FrameBoxes, pairwise_iou


class MODA:
    """A summable metric class to track the components of MODA."""

    def __init__(
        self, false_negatives: int = 0, false_positives: int = 0, n_truth: int = 0
    ) -> None:
        self.false_negatives = false_negatives
        self.false_positives = false_positives
        self.n_truth = n_truth

    def __add__(self, other: Union[int, "MODA"]) -> "MODA":
        """Add two MODA objects."""
        if isinstance(other, int):
            return self
        return MODA(
            false_negatives=self.false_negatives + other.false_negatives,
            false_positives=self.false_positives + other.false_positives,
            n_truth=self.n_truth + other.n_truth,
        )

    def __radd__(self, other: Union[int, "MODA"]) -> "MODA":
        """Add two MODA objects."""
        return self + other

    def __repr__(self) -> str:
        """Create a string representation."""
        return (
            f"MODA(false_negatives={self.false_negatives}, "
            f"false_positives={self.false_positives}, "
            f"n_truth={self.n_truth})"
        )

    @property
    def value(self) -> float:
        """Compute the MODA metric."""
        n_errors = abs(self.false_negatives) + abs(self.false_positives)
        if self.n_truth == 0:
            return 1.0
        return 1 - n_errors / self.n_truth


def compute_moda(
    predicted_boxes: Union[FrameAugmentedBoxes, FrameBoxes],
    ground_truth_boxes: Union[FrameAugmentedBoxes, FrameBoxes],
    iou_threshold: float = 0.5,
) -> MODA:
    """
    Compute MODA for a set of predicted boxes.

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
    iou = pairwise_iou(predicted_boxes, ground_truth_boxes)
    cost = 1 - iou
    pred_indices, gt_indices = linear_sum_assignment(cost)

    false_positives = set(np.arange(len(predicted_boxes))) - set(pred_indices)
    false_negatives = set(np.arange(len(ground_truth_boxes))) - set(gt_indices)

    unmatched = iou[pred_indices, gt_indices] < iou_threshold
    false_positives |= set(pred_indices[unmatched])
    false_negatives |= set(gt_indices[unmatched])

    return MODA(
        false_negatives=len(false_negatives),
        false_positives=len(false_positives),
        n_truth=len(ground_truth_boxes),
    )


def compute_moda_by_class(
    predicted_boxes: FrameAugmentedBoxes,
    ground_truth_boxes: FrameAugmentedBoxes,
    iou_threshold: float = 0.5,
) -> defaultdict[int, MODA]:
    """Compute MODA separately for different classes."""
    moda_collector: defaultdict[int, MODA] = defaultdict(MODA)
    all_labels = set(predicted_boxes.labels) | set(ground_truth_boxes.labels)
    for label in all_labels:
        moda_collector[label] += compute_moda(
            predicted_boxes[predicted_boxes.labels == label],
            ground_truth_boxes[ground_truth_boxes.labels == label],
            iou_threshold=iou_threshold,
        )
    return moda_collector
