from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from sparrow_datums import BoxTracking, FrameBoxes, PType, SingleBox

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
        preserve_history: bool = False,
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
        preserve_history
            Whether to preserve finished tracklets history for debugging
        """
        self.active_tracklets: list[Tracklet] = []
        self.missing_tracklets: list[Tracklet] = []
        self.finished_tracklets: list[Tracklet] = []
        self.tracklet_history: list[Tracklet] = []  # For debugging/analysis
        self.previous_boxes: Optional[FrameBoxes] = None
        self.distance_threshold: float = distance_threshold
        self.distance_function = distance_function
        self.missing_threshold: int = missing_threshold
        self.frame_index: int = 0
        self.start_frame: int = 0
        self.preserve_history: bool = preserve_history
        
        # ByteTrack specific parameters
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.second_association_thresh = second_association_thresh

    @property
    def possible_tracklets(self) -> list[Tracklet]:
        """Return the list of possible tracklets."""
        return self.active_tracklets + self.missing_tracklets

    def _extract_confidences(self, boxes: FrameBoxes) -> np.ndarray:
        """
        Extract confidence values from boxes in a standardized way.
        
        Parameters
        ----------
        boxes : FrameBoxes
            Input detection boxes
            
        Returns
        -------
        np.ndarray
            Array of confidence values, defaults to 1.0 if not available
        """
        # Try to get confidences from boxes attribute first
        confidences = getattr(boxes, 'confidences', None)
        
        # Fall back to metadata
        if confidences is None:
            confidences = boxes.metadata_kwargs.get('confidences', None)
        
        # Default to ones if still None
        if confidences is None:
            return np.ones(len(boxes))
        
        # Validate and convert to numpy array
        try:
            confidences = np.asarray(confidences)
            if confidences.shape != (len(boxes),):
                return np.ones(len(boxes))
        except (ValueError, TypeError):
            return np.ones(len(boxes))
            
        return confidences

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
        
        # Extract confidence values in a standardized way
        confidences = self._extract_confidences(boxes)
        
        # Create paired data structures: (box, confidence) for each confidence category
        high_conf_pairs = []
        low_conf_pairs = []
        
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            single_box = boxes.get_single_box(i)
            if conf >= self.high_thresh:
                high_conf_pairs.append((single_box, conf))
            elif conf >= self.low_thresh:
                low_conf_pairs.append((single_box, conf))
        
        # Create FrameBoxes for association (without confidence complexity)
        high_conf_boxes = FrameBoxes.from_single_boxes(
            [pair[0] for pair in high_conf_pairs], 
            ptype=boxes.ptype, **boxes.metadata_kwargs
        ) if high_conf_pairs else self.empty_previous_boxes(boxes)
        
        low_conf_boxes = FrameBoxes.from_single_boxes(
            [pair[0] for pair in low_conf_pairs], 
            ptype=boxes.ptype, **boxes.metadata_kwargs
        ) if low_conf_pairs else self.empty_previous_boxes(boxes)
        
        # Step 1: Associate high confidence detections with active tracklets
        active_matches, active_unmatched_tracks, high_unmatched_dets = self._associate(
            self.active_tracklets, high_conf_boxes, self.distance_threshold
        )
        
        # Update matched active tracklets with confidence
        for track_idx, det_idx in active_matches:
            box, detection_confidence = high_conf_pairs[det_idx]
            self.active_tracklets[track_idx].add_box(box, confidence=detection_confidence)
        
        # Step 2: Associate unmatched active tracklets with low confidence detections
        unmatched_active_tracklets = [self.active_tracklets[i] for i in active_unmatched_tracks]
        second_matches, second_unmatched_tracks, low_unmatched_dets = self._associate(
            unmatched_active_tracklets, low_conf_boxes, self.second_association_thresh
        )
        
        # Update second round matched tracklets with confidence
        for local_track_idx, det_idx in second_matches:
            global_track_idx = active_unmatched_tracks[local_track_idx]
            box, detection_confidence = low_conf_pairs[det_idx]
            self.active_tracklets[global_track_idx].add_box(box, confidence=detection_confidence)
        
        # Step 3: Associate missing tracklets with remaining high confidence detections
        remaining_high_pairs = [high_conf_pairs[i] for i in high_unmatched_dets]
        if remaining_high_pairs and self.missing_tracklets:
            remaining_high_boxes = FrameBoxes.from_single_boxes(
                [pair[0] for pair in remaining_high_pairs], 
                ptype=boxes.ptype, **boxes.metadata_kwargs
            )
            missing_matches, missing_unmatched_tracks, final_unmatched_dets = self._associate(
                self.missing_tracklets, remaining_high_boxes, self.distance_threshold
            )
            
            # Reactivate matched missing tracklets with confidence
            for track_idx, det_idx in missing_matches:
                self.missing_tracklets[track_idx].finalize_missing_boxes()
                box, detection_confidence = remaining_high_pairs[det_idx]
                self.missing_tracklets[track_idx].add_box(box, confidence=detection_confidence)
                # Move from missing to active
                self.active_tracklets.append(self.missing_tracklets[track_idx])
            
            # Remove matched missing tracklets
            for track_idx in sorted(set(match[0] for match in missing_matches), reverse=True):
                self.missing_tracklets.pop(track_idx)
            
            # Update remaining unmatched pairs
            remaining_high_pairs = [remaining_high_pairs[i] for i in final_unmatched_dets]
        
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
        for box, confidence in remaining_high_pairs:
            if confidence >= self.new_track_thresh:
                self.active_tracklets.append(
                    Tracklet(self.frame_index, box, confidence=confidence)
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

    def export_history(self) -> list[Tracklet]:
        """Export tracklet history for debugging/analysis."""
        return self.tracklet_history.copy()

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
        
        # Preserve history if requested before clearing
        if self.preserve_history:
            self.tracklet_history.extend(self.finished_tracklets)
        
        # Clear finished tracklets and update start frame
        self.finished_tracklets = []
        self.start_frame += len(chunk)
        
        return chunk