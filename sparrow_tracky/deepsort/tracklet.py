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
        """
        self.start_index = start_index
        self.boxes = FrameBoxes.from_single_box(box)
        self.missing_boxes = FrameBoxes(
            np.zeros((0, 4)),
            ptype=self.boxes.ptype,
            **self.boxes.metadata_kwargs,
        )
        self.object_id = object_id if object_id else str(uuid.uuid4())

    def __len__(self) -> int:
        """Check number of boxes in the tracklet."""
        return len(self.boxes)

    def add_box(self, box: SingleBox) -> None:
        """Append a box to the end of the array."""
        self.boxes = self.boxes.add_box(box)

    def add_missing_box(self) -> None:
        """Append a box to the missing box list."""
        self.missing_boxes = self.missing_boxes.add_box(self.previous_box)

    def scratch_missing_boxes(self) -> None:
        """Clear the missing box list."""
        self.missing_boxes = FrameBoxes(
            np.zeros((0, 4)),
            ptype=self.boxes.ptype,
            **self.boxes.metadata_kwargs,
        )

    def finalize_missing_boxes(self) -> None:
        """Finish the missing box list."""
        for box in self.missing_boxes:
            self.add_box(box)
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
