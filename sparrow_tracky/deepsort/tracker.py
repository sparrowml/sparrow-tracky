import numpy as np
from scipy.optimize import linear_sum_assignment
from sparrow_datums import FrameBoxes, pairwise_iou

from .tracklet import Tracklet


class Tracker:
    """Maintain and update tracklets."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        """
        Maintain and update tracklets.

        Parameters
        ----------
        distance_threshold : float
            A cost score beyond which potential pairs are eliminated
        """
        self.active_tracklets: list[Tracklet] = []
        self.finished_tracklets: list[Tracklet] = []
        self.previous_boxes: FrameBoxes = FrameBoxes(np.zeros((0, 4)))
        self.iou_threshold: float = iou_threshold
        self.frame_index: int = 0

    def track(self, boxes: FrameBoxes) -> None:
        """
        Update tracklets with boxes from a new frame.

        Parameters
        ----------
        boxes : FrameBoxes
            A ``(n_boxes, 4)`` array of bounding boxes
        """
        prev_indices = boxes_indices = []
        if len(boxes) > 0 and len(self.previous_boxes) > 0:
            # Pairwise cost: euclidean distance between boxes
            ious = 1 - pairwise_iou(self.previous_boxes, boxes)
            # Object matching
            prev_indices, boxes_indices = linear_sum_assignment(1 - ious)
            mask = ious[prev_indices, boxes_indices] > self.iou_threshold
            prev_indices = prev_indices[mask]
            boxes_indices = boxes_indices[mask]
        # Add matches to active tracklets
        for prev_idx, box_idx in zip(prev_indices, boxes_indices):
            self.active_tracklets[prev_idx].add_box(boxes.get_single_box(box_idx))
        # Finalize lost tracklets
        lost_indices = set(range(len(self.active_tracklets))) - set(prev_indices)
        for lost_idx in sorted(lost_indices, reverse=True):
            self.finished_tracklets.append(self.active_tracklets.pop(lost_idx))
        # Activate new tracklets
        new_indices = set(range(len(boxes))) - set(boxes_indices)
        for new_idx in new_indices:
            self.active_tracklets.append(
                Tracklet(self.frame_index, boxes.get_single_box(new_idx))
            )
        # "Predict" next frame for comparison
        if len(self.active_tracklets):
            self.previous_boxes = FrameBoxes(
                np.stack(
                    [tracklet.previous_box.array for tracklet in self.active_tracklets]
                ),
                ptype=boxes.ptype,
                **boxes.metadata_kwargs,
            )
        else:
            self.previous_boxes = np.zeros((0, 4)).view(FrameBoxes)
        self.frame_index += 1

    @property
    def tracklets(self) -> list[Tracklet]:
        """Return the list of all tracklets."""
        return sorted(
            self.finished_tracklets + self.active_tracklets, key=lambda t: t.start_index
        )
