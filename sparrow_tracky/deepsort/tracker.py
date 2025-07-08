from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from sparrow_datums import BoxTracking, FrameBoxes, PType

from .distance import iou_distance
from .tracklet import Tracklet


class Tracker:
    """Maintain and update tracklets using ByteTrack algorithm."""

    def __init__(
        self,
        distance_threshold: float = 0.5,
        distance_function: Callable[
            [FrameBoxes, FrameBoxes], npt.NDArray[np.float64]
        ] = iou_distance,
        missing_threshold: int = 30,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        new_track_thresh: float = 0.7,
        second_association_thresh: float = 0.5,
    ) -> None:
        """
        Maintain and update tracklets using ByteTrack algorithm.

        Parameters
        ----------
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
        
        # ByteTrack specific parameters
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.second_association_thresh = second_association_thresh

    @property
    def possible_tracklets(self) -> list[Tracklet]:
        """Return the list of possible tracklets."""
        return self.active_tracklets + self.missing_tracklets

    def track(self, boxes: FrameBoxes) -> None:
        """
        Update tracklets with boxes from a new frame using ByteTrack algorithm.

        Parameters
        ----------
        boxes : FrameBoxes
            A ``(n_boxes, 4)`` array of bounding boxes
        """
        boxes = boxes[np.isfinite(boxes.x)]
        if self.previous_boxes is None:
            self.previous_boxes = self.empty_previous_boxes(boxes)
        
        # ByteTrack: Extract detection confidences, default to 1.0 if not available
        confidences = getattr(boxes, 'confidences', None)
        if confidences is None:
            confidences = boxes.metadata_kwargs.get('confidences', None)
        
        if confidences is None:
            confidences = np.ones(len(boxes))
        else:
            try:
                confidences = np.asarray(confidences)
                if confidences.shape != (len(boxes),):
                    confidences = np.ones(len(boxes))
            except (ValueError, TypeError):
                confidences = np.ones(len(boxes))
        
        high_conf_mask = confidences >= self.high_thresh
        low_conf_mask = (confidences >= self.low_thresh) & (confidences < self.high_thresh)
        
        high_conf_boxes = boxes[high_conf_mask] if np.any(high_conf_mask) else self.empty_previous_boxes(boxes)
        low_conf_boxes = boxes[low_conf_mask] if np.any(low_conf_mask) else self.empty_previous_boxes(boxes)
        
        # Step 1: Associate high confidence detections with active tracklets
        active_matches, active_unmatched_tracks, high_unmatched_dets = self._associate(
            self.active_tracklets, high_conf_boxes, self.distance_threshold
        )
        
        # Update matched active tracklets
        for track_idx, det_idx in active_matches:
            self.active_tracklets[track_idx].add_box(high_conf_boxes.get_single_box(det_idx))
        
        # Step 2: Associate unmatched active tracklets with low confidence detections
        # Use a more lenient threshold for low-confidence detections to recover tracks
        unmatched_active_tracklets = [self.active_tracklets[i] for i in active_unmatched_tracks]
        second_matches, second_unmatched_tracks, low_unmatched_dets = self._associate(
            unmatched_active_tracklets, low_conf_boxes, self.second_association_thresh
        )
        
        # Update second round matched tracklets
        for local_track_idx, det_idx in second_matches:
            global_track_idx = active_unmatched_tracks[local_track_idx]
            self.active_tracklets[global_track_idx].add_box(low_conf_boxes.get_single_box(det_idx))
        
        # Step 3: Associate missing tracklets with remaining high confidence detections
        remaining_high_dets = [high_conf_boxes.get_single_box(i) for i in high_unmatched_dets]
        if remaining_high_dets and self.missing_tracklets:
            remaining_high_boxes = FrameBoxes.from_single_boxes(
                remaining_high_dets, ptype=boxes.ptype, **boxes.metadata_kwargs
            )
            missing_matches, missing_unmatched_tracks, final_unmatched_dets = self._associate(
                self.missing_tracklets, remaining_high_boxes, self.distance_threshold
            )
            
            # Reactivate matched missing tracklets
            for track_idx, det_idx in missing_matches:
                self.missing_tracklets[track_idx].finalize_missing_boxes()
                self.missing_tracklets[track_idx].add_box(remaining_high_boxes.get_single_box(det_idx))
                # Move from missing to active
                self.active_tracklets.append(self.missing_tracklets[track_idx])
            
            # Remove matched missing tracklets
            for track_idx in sorted(set(match[0] for match in missing_matches), reverse=True):
                self.missing_tracklets.pop(track_idx)
            
            # Update unmatched detection indices
            high_unmatched_dets = [high_unmatched_dets[i] for i in final_unmatched_dets]
        
        # Step 4: Handle unmatched active tracklets
        final_unmatched_active = [active_unmatched_tracks[i] for i in second_unmatched_tracks]
        for track_idx in sorted(final_unmatched_active, reverse=True):
            tracklet = self.active_tracklets.pop(track_idx)
            if tracklet.n_missing < self.missing_threshold:
                tracklet.add_missing_box()
                self.missing_tracklets.append(tracklet)
            else:
                tracklet.finalize_missing_boxes()
                self.finished_tracklets.append(tracklet)
        
        # Step 5: Handle missing tracklets that weren't matched
        for i in range(len(self.missing_tracklets) - 1, -1, -1):
            tracklet = self.missing_tracklets[i]
            if tracklet.n_missing >= self.missing_threshold:
                tracklet.scratch_missing_boxes()
                self.finished_tracklets.append(self.missing_tracklets.pop(i))
            else:
                tracklet.add_missing_box()
        
        # Step 6: Create new tracklets from high confidence unmatched detections
        high_conf_indices = np.where(high_conf_mask)[0]
        for det_idx in high_unmatched_dets:
            original_idx = high_conf_indices[det_idx] if det_idx < len(high_conf_indices) else det_idx
            if original_idx < len(confidences) and confidences[original_idx] >= self.new_track_thresh:
                self.active_tracklets.append(
                    Tracklet(self.frame_index, high_conf_boxes.get_single_box(det_idx))
                )
        
        # Update previous boxes for next frame
        if len(self.possible_tracklets) > 0:
            self.previous_boxes = FrameBoxes.from_single_boxes(
                [t.previous_box for t in self.possible_tracklets],
                ptype=boxes.ptype,
                **boxes.metadata_kwargs,
            )
        else:
            self.previous_boxes = self.empty_previous_boxes(boxes)
        self.frame_index += 1

    def _associate(self, tracklets: list[Tracklet], boxes: FrameBoxes, threshold: float):
        """Associate tracklets with detections using Hungarian algorithm."""
        if len(tracklets) == 0 or len(boxes) == 0:
            return [], list(range(len(tracklets))), list(range(len(boxes)))
        
        # Create previous boxes for tracklets
        tracklet_boxes = FrameBoxes.from_single_boxes(
            [t.previous_box for t in tracklets],
            ptype=boxes.ptype,
            **boxes.metadata_kwargs,
        )
        
        # Calculate costs
        costs = self.distance_function(tracklet_boxes, boxes)
        costs = np.nan_to_num(costs, nan=1.0)
        
        # Apply threshold
        costs[costs > threshold] = 1.0
        
        # Hungarian assignment
        track_indices, det_indices = linear_sum_assignment(costs)
        
        # Filter out assignments with high cost
        valid_mask = costs[track_indices, det_indices] < threshold
        track_indices = track_indices[valid_mask]
        det_indices = det_indices[valid_mask]
        
        matches = list(zip(track_indices, det_indices))
        unmatched_tracks = [i for i in range(len(tracklets)) if i not in track_indices]
        unmatched_dets = [i for i in range(len(boxes)) if i not in det_indices]
        
        return matches, unmatched_tracks, unmatched_dets

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
        # First, move all active and missing tracklets to finished
        # This ensures that subsequent chunks don't share tracklets
        all_current_tracklets = self.active_tracklets + self.missing_tracklets
        for tracklet in all_current_tracklets:
            tracklet.finalize_missing_boxes()
            self.finished_tracklets.append(tracklet)
        
        # Clear active and missing tracklets
        self.active_tracklets = []
        self.missing_tracklets = []
        
        # Filter tracklets based on criteria
        tracklets = [
            t
            for t in self.finished_tracklets
            if len(t) >= min_tracklet_length
            and t.start_index + len(t) > self.start_frame
        ]
        
        n_objects = len(tracklets)
        n_frames = self.frame_index - self.start_frame
        
        if len(tracklets) == 0:
            ptype = PType.unknown
            metadata = {"fps": fps, "start_time": self.start_frame / fps, "object_ids": []}
        else:
            ptype = tracklets[0].boxes.ptype
            metadata = tracklets[0].boxes.metadata_kwargs.copy()
            metadata["fps"] = fps
            metadata["object_ids"] = [t.object_id for t in tracklets]
            metadata["start_time"] = self.start_frame / fps
        
        data = np.zeros((n_frames, n_objects, 4)) * np.nan
        
        for object_idx, tracklet in enumerate(tracklets):
            # Calculate the frame range for this tracklet
            tracklet_start = tracklet.start_index
            tracklet_end = tracklet.start_index + len(tracklet)
            
            # Calculate the overlap with the chunk range
            chunk_start = self.start_frame
            chunk_end = self.start_frame + n_frames
            
            # Find the actual overlap
            overlap_start = max(tracklet_start, chunk_start)
            overlap_end = min(tracklet_end, chunk_end)
            
            if overlap_start < overlap_end:
                # Calculate indices for the chunk array
                chunk_start_idx = overlap_start - chunk_start
                chunk_end_idx = overlap_end - chunk_start
                
                # Calculate indices for the tracklet array
                tracklet_start_idx = overlap_start - tracklet_start
                tracklet_end_idx = overlap_end - tracklet_start
                
                # Get the tracklet data for the overlapping frames
                tracklet_data = tracklet.boxes.array[tracklet_start_idx:tracklet_end_idx]
                
                # Assign to the chunk array
                data[chunk_start_idx:chunk_end_idx, object_idx] = tracklet_data
        
        chunk = BoxTracking(data, ptype=ptype, **metadata)
        
        # Clear finished tracklets and update start frame
        self.finished_tracklets = []
        self.start_frame += len(chunk)
        
        return chunk