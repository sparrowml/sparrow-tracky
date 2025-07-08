from __future__ import annotations

import uuid
from typing import Optional, Union

import numpy as np
from sparrow_datums import FrameBoxes, SingleBox


class Tracklet:
    """Store the location history for an object."""

    def __init__(
        self,
        start_index: int,
        box: Union[SingleBox, FrameBoxes],
        object_id: Optional[str] = None,
        confidence: float = 1.0,
    ) -> None:
        """
        Store the location history for an object.

        Parameters
        ----------
        start_index
            The frame index that starts the tracklet
        box
            A NumPy array with shape (4,)
        object_id
            An ID for the tracklet
        confidence
            Initial confidence score
        """
        self.start_index = start_index
        self.boxes = FrameBoxes.from_single_box(box)
        self.missing_boxes = FrameBoxes(
            np.zeros((0, 4)),
            ptype=self.boxes.ptype,
            **self.boxes.metadata_kwargs,
        )
        self.object_id = object_id if object_id else str(uuid.uuid4())
        self.confidence = confidence
        self.time_since_update = 0
        self.confidence_history = [confidence]  # Track confidence over time

    def __len__(self) -> int:
        """Check number of boxes in the tracklet."""
        return len(self.boxes)

    def add_box(self, box: SingleBox, confidence: Optional[float] = None) -> None:
        """
        Append a box to the end of the array and update confidence.
        
        Parameters
        ----------
        box
            The detection box to add
        confidence
            Confidence score for this detection. If provided, updates tracklet confidence
            using exponential moving average.
        """
        self.boxes = self.boxes.add_box(box)
        self.time_since_update = 0
        
        # Update confidence using exponential moving average if provided
        if confidence is not None:
            alpha = 0.3  # Smoothing factor for moving average
            self.confidence = alpha * confidence + (1 - alpha) * self.confidence
            self.confidence_history.append(self.confidence)

    def add_missing_box(self) -> None:
        """Append a box to the missing box list."""
        self.missing_boxes = self.missing_boxes.add_box(self.previous_box)
        self.time_since_update += 1

    def scratch_missing_boxes(self) -> None:
        """Clear the missing box list."""
        self.missing_boxes = FrameBoxes(
            np.zeros((0, 4)),
            ptype=self.boxes.ptype,
            **self.boxes.metadata_kwargs,
        )

    def finalize_missing_boxes(self) -> None:
        """
        Finish the missing box list by adding all missing boxes to the main tracklet.
        
        Note: Missing boxes don't have associated confidence values, so we skip 
        confidence updates to avoid artificially inflating or deflating the tracklet's 
        confidence based on interpolated positions.
        """
        for box in self.missing_boxes:
            # Skip confidence update for missing boxes as they are interpolated
            self.add_box(box, confidence=None)
        self.missing_boxes = FrameBoxes(
            np.zeros((0, 4)),
            ptype=self.boxes.ptype,
            **self.boxes.metadata_kwargs,
        )

    @property
    def possible_boxes(self) -> FrameBoxes:
        """Return the list of all possible boxes."""
        return FrameBoxes(
            np.concatenate([self.boxes.array, self.missing_boxes.array]),
            ptype=self.boxes.ptype,
            **self.boxes.metadata_kwargs,
        )

    @property
    def previous_box(self) -> SingleBox:
        """Return the most recent addition."""
        if len(self.missing_boxes) > 0:
            return self.missing_boxes.get_single_box(-1)
        return self.boxes.get_single_box(-1)

    @property
    def n_missing(self) -> int:
        """Return the number of missing boxes."""
        return len(self.missing_boxes)

    @property
    def is_activated(self) -> bool:
        """Check if tracklet is activated (has at least one detection)."""
        return len(self.boxes) > 0

    @property
    def mean_confidence(self) -> float:
        """Return the mean confidence score over the tracklet's lifetime."""
        return np.mean(self.confidence_history)