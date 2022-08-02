from __future__ import annotations

import uuid
from typing import Optional, Union

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
        self.object_id = object_id if object_id else str(uuid.uuid4())

    def __len__(self) -> int:
        """Check number of boxes in the tracklet."""
        return len(self.boxes)

    def add_box(self, box: SingleBox) -> None:
        """Append a box to the end of the array."""
        self.boxes = self.boxes.add_box(box)

    @property
    def previous_box(self) -> SingleBox:
        """Return the most recent addition."""
        return self.boxes.get_single_box(-1)
