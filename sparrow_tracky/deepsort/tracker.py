from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from sparrow_datums import BoxTracking, FrameBoxes, PType

from .distance import iou_distance
from .tracklet import Tracklet


class Tracker:
    """Maintain and update tracklets."""

    def __init__(
        self,
        distance_threshold: float = 0.5,
        distance_function: Callable[
            [FrameBoxes, FrameBoxes], npt.NDArray[np.float64]
        ] = iou_distance,
        missing_threshold: int = 0,
    ) -> None:
        """
        Maintain and update tracklets.

        Parameters
        ----------
        distance_threshold
            An IoU score below which potential pairs are eliminated
        distance_function
            Function for computing pairwise distances
        missing_threshold
            Number of frames to wait before finalizing a tracklet
        """
        self.active_tracklets: list[Tracklet] = []
        self.missing_tracklets: list[Tracklet] = []
        self.finished_tracklets: list[Tracklet] = []
        self.previous_boxes: Optional[FrameBoxes] = None
        self.distance_threshold: float = distance_threshold
        self.distance_function = distance_function
        self.missing_threshold: int = missing_threshold
        self.frame_index: int = 0
        self.start_frame: int = 0

    @property
    def possible_tracklets(self) -> list[Tracklet]:
        """Return the list of possible tracklets."""
        return self.active_tracklets + self.missing_tracklets

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
        prev_indices: list[int] = []
        boxes_indices: list[int] = []
        if len(boxes) > 0 and len(self.previous_boxes) > 0:
            # Pairwise cost between boxes
            costs = self.distance_function(self.previous_boxes, boxes)
            costs = np.nan_to_num(costs, nan=-1)
            # Object matching
            prev_indices, boxes_indices = linear_sum_assignment(costs)
            mask = costs[prev_indices, boxes_indices] < self.distance_threshold
            prev_indices = prev_indices[mask]
            boxes_indices = boxes_indices[mask]

        tracklet_switches: dict[int, str] = {}
        # Add matches to active tracklets
        for prev_idx, box_idx in zip(prev_indices, boxes_indices):
            if prev_idx < len(self.active_tracklets):
                self.active_tracklets[prev_idx].add_box(boxes.get_single_box(box_idx))
            else:
                # missing -> active
                missing_tracklet_idx = prev_idx - len(self.active_tracklets)
                self.missing_tracklets[missing_tracklet_idx].finalize_missing_boxes()
                self.missing_tracklets[missing_tracklet_idx].add_box(
                    boxes.get_single_box(box_idx)
                )
                tracklet_switches[prev_idx] = "active"
        # Handle lost tracklets
        missing_indices = set(range(len(self.possible_tracklets))) - set(prev_indices)
        for missing_idx in sorted(missing_indices, reverse=True):
            if missing_idx < len(self.active_tracklets):
                n_missing = self.active_tracklets[missing_idx].n_missing
                if n_missing < self.missing_threshold:
                    # active -> missing
                    self.active_tracklets[missing_idx].add_missing_box()
                    tracklet_switches[missing_idx] = "missing"
                else:
                    # active -> finished
                    self.active_tracklets[missing_idx].finalize_missing_boxes()
                    tracklet_switches[missing_idx] = "finished"
            else:
                missing_tracklet_idx = missing_idx - len(self.active_tracklets)
                n_missing = self.missing_tracklets[missing_tracklet_idx].n_missing
                if n_missing < self.missing_threshold:
                    self.missing_tracklets[missing_tracklet_idx].add_missing_box()
                else:
                    # missing -> finished
                    self.missing_tracklets[missing_tracklet_idx].scratch_missing_boxes()
                    tracklet_switches[missing_idx] = "finished"

        new_active: list[Tracklet] = []
        new_missing: list[Tracklet] = []
        new_finished: list[Tracklet] = []
        for idx in sorted(tracklet_switches.keys(), reverse=True):
            if idx < len(self.active_tracklets):
                tracklet = self.active_tracklets.pop(idx)
            else:
                missing_tracklet_idx = idx - len(self.active_tracklets)
                tracklet = self.missing_tracklets.pop(missing_tracklet_idx)
            if tracklet_switches[idx] == "active":
                new_active.append(tracklet)
            elif tracklet_switches[idx] == "missing":
                new_missing.append(tracklet)
            elif tracklet_switches[idx] == "finished":
                new_finished.append(tracklet)

        self.active_tracklets.extend(new_active)
        self.missing_tracklets.extend(new_missing)
        self.finished_tracklets.extend(new_finished)

        # Activate new tracklets
        new_indices = set(range(len(boxes))) - set(boxes_indices)
        for new_idx in new_indices:
            self.active_tracklets.append(
                Tracklet(self.frame_index, boxes.get_single_box(new_idx))
            )
        # "Predict" next frame for comparison
        if len(self.possible_tracklets) > 0:
            self.previous_boxes = FrameBoxes.from_single_boxes(
                [t.previous_box for t in self.possible_tracklets],
                ptype=boxes.ptype,
                **boxes.metadata_kwargs,
            )
        else:
            self.previous_boxes = self.empty_previous_boxes(boxes)
        self.frame_index += 1

    @property
    def tracklets(self) -> list[Tracklet]:
        """Return the list of all tracklets."""
        all_tracklets = self.finished_tracklets + self.possible_tracklets
        return sorted(all_tracklets, key=lambda t: t.start_index)

    def empty_previous_boxes(self, boxes: FrameBoxes) -> FrameBoxes:
        """Initialize empty FrameBoxes for previous_boxes attribute."""
        return FrameBoxes(
            np.zeros((0, 4)),
            ptype=boxes.ptype,
            **boxes.metadata_kwargs,
        )

    def make_chunk(self, fps: float, min_tracklet_length: int = 1) -> BoxTracking:
        """Consolidate tracklets to BoxTracking chunk."""
        tracklets = [
            t
            for t in self.tracklets
            if len(t) >= min_tracklet_length
            and t.start_index + len(t) > self.start_frame
        ]
        n_objects = len(tracklets)
        metadata: dict[str, Any]
        n_frames = self.frame_index - self.start_frame
        if len(tracklets) == 0:
            ptype = PType.unknown
            metadata = {"fps": fps}
        else:
            ptype = tracklets[0].boxes.ptype
            metadata = tracklets[0].boxes.metadata_kwargs
            metadata["fps"] = fps
        metadata["object_ids"] = [t.object_id for t in tracklets]
        metadata["start_time"] = self.start_frame / fps
        data = np.zeros((n_frames, n_objects, 4)) * np.nan
        for object_idx, tracklet in enumerate(tracklets):
            start = max(tracklet.start_index - self.start_frame, 0)
            end = tracklet.start_index + len(tracklet) - self.start_frame
            n_tracklet_frames = end - start
            data[start:end, object_idx] = tracklet.boxes.array[-n_tracklet_frames:]
        chunk = BoxTracking(
            data,
            ptype=ptype,
            **metadata,
        )
        self.finished_tracklets = []
        self.start_frame += len(chunk)
        return chunk
