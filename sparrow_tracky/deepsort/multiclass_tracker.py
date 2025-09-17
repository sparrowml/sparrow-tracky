from typing import Callable

import numpy as np
import numpy.typing as npt
from sparrow_datums import (
    AugmentedBoxTracking,
    BoxTracking,
    FrameAugmentedBoxes,
    FrameBoxes,
    PType,
)

from .distance import iou_distance
from .tracker import Tracker


class MultiClassTracker:
    """Maintain and update tracklets with separate classes using ByteTrack."""

    def __init__(
        self,
        n_classes: int,
        distance_threshold: float = 0.5,
        distance_function: Callable[
            [FrameBoxes, FrameBoxes], npt.NDArray[np.float64]
        ] = iou_distance,
        missing_threshold: int = 30,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        new_track_thresh: float = 0.7,
        second_association_thresh: float = 0.5,
        preserve_history: bool = False,
    ) -> None:
        """
        Maintain and update tracklets using ByteTrack algorithm.

        Parameters
        ----------
        n_classes
            Number of classes to track
        distance_threshold
            An IoU score below which potential pairs are eliminated for high-confidence associations
        distance_function
            Function for computing pairwise distances
        missing_threshold
            Number of frames to wait before finalizing a tracklet
        high_thresh
            High confidence threshold for detections
        low_thresh
            Low confidence threshold for detections
        new_track_thresh
            Threshold for creating new tracks
        second_association_thresh
            Threshold for associating low-confidence detections with unmatched tracks.
            Typically more lenient than distance_threshold to recover tracks with poor detections.
        preserve_history
            Whether to preserve finished tracklets history for debugging
        """
        if n_classes < 1:
            raise ValueError(f"Invalid number of classes: {n_classes}")
        self.n_classes = n_classes
        self.trackers: dict[int, Tracker] = {}
        for class_idx in range(n_classes):
            self.trackers[class_idx] = Tracker(
                distance_threshold=distance_threshold,
                distance_function=distance_function,
                missing_threshold=missing_threshold,
                high_thresh=high_thresh,
                low_thresh=low_thresh,
                new_track_thresh=new_track_thresh,
                second_association_thresh=second_association_thresh,
                preserve_history=preserve_history,
            )

    @property
    def _first_tracker(self) -> Tracker:
        return self.trackers[0]

    @property
    def start_frame(self) -> int:
        return self._first_tracker.start_frame

    @property
    def frame_index(self) -> int:
        return self._first_tracker.frame_index

    def track(self, boxes: FrameAugmentedBoxes) -> None:
        """
        Update tracklets with boxes from a new frame.

        Parameters
        ----------
        boxes : FrameAugmentedBoxes
            A ``(n_boxes, 6)`` array of bounding boxes
        """
        for class_idx in range(self.n_classes):
            _boxes = boxes[boxes.labels == class_idx].to_frame_boxes()
            self.trackers[class_idx].track(_boxes)

    def export_history(self) -> dict[int, list]:
        """Export tracklet history for all classes."""
        return {class_idx: tracker.export_history() for class_idx, tracker in self.trackers.items()}

    def make_chunk(
        self, fps: float, min_tracklet_length: int = 1
    ) -> AugmentedBoxTracking:
        """Consolidate tracklets to AugmentedBoxTracking chunk."""
        n_frames = self.frame_index - self.start_frame
        start_time = self.start_frame / fps
        n_objects = 0
        metadata = {"fps": fps, "start_time": start_time}
        chunks: dict[int, BoxTracking] = {}
        for class_idx in range(self.n_classes):
            chunk = self.trackers[class_idx].make_chunk(fps, min_tracklet_length)
            n_objects += chunk.shape[1]
            chunks[class_idx] = chunk
        if n_objects == 0:
            return AugmentedBoxTracking(
                np.ones((n_frames, 0, 6)), ptype=PType.unknown, **metadata
            )
        data = np.zeros((n_frames, n_objects, 6)) * np.nan
        object_idx = 0
        object_ids = []
        ptype = PType.unknown

        for class_idx in range(self.n_classes):
            chunk = chunks[class_idx]
            if chunk.ptype != PType.unknown:
                ptype = chunk.ptype
                metadata.update(chunk.metadata_kwargs)
            _n_objects = chunk.shape[1]
            object_ids.extend(chunk.object_ids)
            data[:, object_idx : object_idx + _n_objects, :4] = chunk.array
            data[:, object_idx : object_idx + _n_objects, -2] = 1.0
            data[:, object_idx : object_idx + _n_objects, -1] = class_idx
            object_idx += _n_objects
        metadata["object_ids"] = object_ids
        return AugmentedBoxTracking(data, ptype=ptype, **metadata)