from typing import Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from sparrow_datums import AugmentedBoxTracking, BoxTracking, pairwise_iou


class MOTA:
    """A summable metric class to track the components of MOTA."""

    def __init__(
        self,
        false_negatives: int = 0,
        false_positives: int = 0,
        id_switches: int = 0,
        n_truth: int = 0,
    ) -> None:
        self.false_negatives = false_negatives
        self.false_positives = false_positives
        self.id_switches = id_switches
        self.n_truth = n_truth

    def __add__(self, other: Union[int, "MOTA"]) -> "MOTA":
        """Add two MODA objects."""
        if isinstance(other, int):
            return self
        return MOTA(
            false_negatives=self.false_negatives + other.false_negatives,
            false_positives=self.false_positives + other.false_positives,
            id_switches=self.id_switches + other.id_switches,
            n_truth=self.n_truth + other.n_truth,
        )

    def __radd__(self, other: Union[int, "MOTA"]) -> "MOTA":
        """Add two MOTA objects."""
        return self + other

    def __repr__(self) -> str:
        """Create a string representation."""
        return (
            f"MOTA(false_negatives={self.false_negatives}, "
            f"false_positives={self.false_positives}, "
            f"id_switches={self.id_switches}, "
            f"n_truth={self.n_truth})"
        )

    @property
    def value(self) -> float:
        """Compute the MODA metric."""
        n_errors = (
            abs(self.false_negatives)
            + abs(self.false_positives)
            + abs(self.id_switches)
        )
        if self.n_truth == 0:
            return 1.0
        return 1 - n_errors / self.n_truth


def compute_mota(
    predicted_tracking: Union[AugmentedBoxTracking, BoxTracking],
    ground_truth_tracking: Union[AugmentedBoxTracking, BoxTracking],
    iou_threshold: float = 0.5,
) -> MOTA:
    """
    Compute MOTA for a predicted box tracking chunk.

    Parameters
    ----------
    predicted_tracking : BoxTracking
        Predicted tracking
    ground_truth_tracking : BoxTracking
        Ground truth tracking
    iou_threshold : float
        The overlap threshold below which boxes are not considered the same
    """
    n_false_positives = 0
    n_false_negatives = 0
    n_id_switches = 0
    n_ground_truth = 0
    matches: dict[int, int] = {}
    for pred_frame, gt_frame in zip(predicted_tracking, ground_truth_tracking):
        finite_pred_frame = pred_frame[~np.isnan(pred_frame.x)]
        finite_gt_frame = gt_frame[~np.isnan(gt_frame.x)]
        n_ground_truth += len(finite_gt_frame)
        if len(finite_pred_frame) == 0:
            n_false_negatives += len(finite_gt_frame)
            continue
        elif len(finite_pred_frame) == 0:
            n_false_positives += len(finite_pred_frame)
            continue
        cost = 1 - pairwise_iou(finite_pred_frame, finite_gt_frame)
        pred_indices, gt_indices = linear_sum_assignment(cost)
        new_matches: dict[int, int] = {}
        for pred, gt in zip(pred_indices, gt_indices):
            new_matches[pred] = gt
            if pred in matches and matches[pred] != gt:
                n_id_switches += 1
        matches = new_matches

        false_positives = set(np.arange(len(finite_pred_frame))) - set(pred_indices)
        false_negatives = set(np.arange(len(finite_gt_frame))) - set(gt_indices)

        unmatched = cost[pred_indices, gt_indices] > iou_threshold
        false_positives |= set(pred_indices[unmatched])
        false_negatives |= set(gt_indices[unmatched])

        n_false_positives += len(false_positives)
        n_false_negatives += len(false_negatives)
    return MOTA(
        false_negatives=n_false_negatives,
        false_positives=n_false_positives,
        id_switches=n_id_switches,
        n_truth=n_ground_truth,
    )


# def compute_moda_by_class(
#     predicted_boxes: FrameAugmentedBoxes,
#     ground_truth_boxes: FrameAugmentedBoxes,
#     iou_threshold: float = 0.5,
# ) -> defaultdict[int, MODA]:
#     """Compute MODA separately for different classes."""
#     moda_collector: defaultdict[int, MODA] = defaultdict(MODA)
#     all_labels = set(predicted_boxes.labels) | set(ground_truth_boxes.labels)
#     for label in all_labels:
#         moda_collector[label] += compute_moda(
#             predicted_boxes[predicted_boxes.labels == label],
#             ground_truth_boxes[ground_truth_boxes.labels == label],
#             iou_threshold=iou_threshold,
#         )
#     return moda_collector
