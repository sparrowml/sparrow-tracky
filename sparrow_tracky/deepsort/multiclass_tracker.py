from typing import Callable

import numpy as np
import numpy.typing as npt
from sparrow_datums import (AugmentedBoxTracking, BoxTracking,
                            FrameAugmentedBoxes, FrameBoxes, PType)

from .distance import iou_distance
from .tracker import Tracker


class MultiClassTracker:
    """Maintain and update tracklets with separate classes."""

    def __init__(
        self,
        n_classes: int,
        distance_threshold: float = 0.5,
        distance_function: Callable[
            [FrameBoxes, FrameBoxes], npt.NDArray[np.float64]
        ] = iou_distance,
    ) -> None:
        """
        Maintain and update tracklets.

        Parameters
        ----------
        distance_threshold
            An IoU score below which potential pairs are eliminated
        distance_function
            Function for computing pairwise distances
        """
        self.n_classes = n_classes
        self.trackers: dict[int, Tracker] = {}
        for class_idx in range(n_classes):
            self.trackers[class_idx] = Tracker(distance_threshold, distance_function)

    def track(self, boxes: FrameAugmentedBoxes) -> None:
        """
        Update tracklets with boxes from a new frame.

        Parameters
        ----------
        boxes : FrameAugmentedBoxes
            A ``(n_boxes, 6)`` array of bounding boxes
        """
        for class_idx in range(self.n_classes):
            _boxes = boxes[boxes.labels == class_idx]
            self.trackers[class_idx].track(_boxes)

    def make_chunk(
        self, fps: float, min_tracklet_length: int = 1
    ) -> AugmentedBoxTracking:
        """Consolidate tracklets to AugmentedBoxTracking chunk."""
        n_frames = self.trackers[0].frame_index - self.trackers[0].start_frame
        chunks: list[BoxTracking] = []
        n_objects = 0
        for class_idx in range(self.n_classes):
            if len(self.trackers[class_idx].tracklets) == 0:
                continue
            chunk = self.trackers[class_idx].make_chunk(fps, min_tracklet_length)
            n_objects += chunk.shape[1]
            chunks.append(chunk)
        if len(chunks) == 0:
            return AugmentedBoxTracking(np.ones((n_frames, 0, 6)), ptype=PType.unknown)
        data = np.zeros((n_frames, n_objects, 6)) * np.nan
        object_idx = 0
        object_ids = []
        for chunk in chunks:
            _n_objects = chunk.shape[1]
            object_ids.extend(chunk.object_ids)
            data[:, object_idx : object_idx + _n_objects] = chunk.array
        metadata = {**chunk.metadata_kwargs}
        metadata["object_ids"] = object_ids
        return AugmentedBoxTracking(data, PType=chunk.ptype, **metadata)
