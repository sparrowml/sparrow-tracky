from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sparrow_datums import BoxTracking, FrameBoxes, PType, pairwise_iou

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
        self.previous_boxes: Optional[FrameBoxes] = None
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
        boxes = boxes[np.isfinite(boxes.x)]
        if self.previous_boxes is None:
            self.previous_boxes = self.empty_previous_boxes(boxes)
        prev_indices = boxes_indices = []
        if len(boxes) > 0 and len(self.previous_boxes) > 0:
            # Pairwise cost: euclidean distance between boxes
            ious = pairwise_iou(self.previous_boxes, boxes)
            ious = np.nan_to_num(ious, nan=-1)
            costs = 1 - ious
            # Object matching
            prev_indices, boxes_indices = linear_sum_assignment(costs)
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
            self.previous_boxes = FrameBoxes.from_single_boxes(
                [t.previous_box for t in self.active_tracklets],
                ptype=boxes.ptype,
                **boxes.metadata_kwargs,
            )
        else:
            self.previous_boxes = self.empty_previous_boxes(boxes)
        self.frame_index += 1

    @property
    def tracklets(self) -> list[Tracklet]:
        """Return the list of all tracklets."""
        return sorted(
            self.finished_tracklets + self.active_tracklets, key=lambda t: t.start_index
        )

    def empty_previous_boxes(self, boxes: FrameBoxes) -> FrameBoxes:
        """Initialize empty FrameBoxes for previous_boxes attribute."""
        return FrameBoxes(
            np.zeros((0, 4)),
            ptype=boxes.ptype,
            **boxes.metadata_kwargs,
        )

    def make_chunk(self, fps: float, min_tracklet_length: int = 1) -> BoxTracking:
        """Consolidate tracklets to BoxTracking chunk."""
        tracklets = [t for t in self.tracklets if len(t) >= min_tracklet_length]
        if len(tracklets) == 0:
            ptype = PType.unknown
            metadata = {"fps": fps}
            chunk_start = 0
            chunk_end = 0
        else:
            ptype = tracklets[0].boxes.ptype
            metadata = tracklets[0].boxes.metadata_kwargs
            metadata["fps"] = fps
            chunk_start = min(t.start_index for t in tracklets)
            chunk_end = max(t.start_index + len(t) for t in tracklets)
        n_frames = chunk_end - chunk_start
        n_objects = len(tracklets)
        data = np.zeros((n_frames, n_objects, 4)) * np.nan
        for object_idx, tracklet in enumerate(tracklets):
            start = tracklet.start_index - chunk_start
            end = tracklet.start_index + len(tracklet) - chunk_start
            data[start:end, object_idx] = tracklet.boxes.array
        return BoxTracking(
            data,
            ptype=ptype,
            **metadata,
        )
