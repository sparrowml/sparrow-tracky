from __future__ import annotations

from typing import Optional, Union

import numpy as np
from sparrow_datums import FrameBoxes, SingleBox


class Tracklet:
    """Store the location history for an object."""

    def __init__(
        self,
        start_index: int,
        box: Union[SingleBox, FrameBoxes],
        observed: Optional[list[bool]] = None,
    ) -> None:
        """
        Store the location history for an object.

        Parameters
        ----------
        start_index
            The frame index that starts the tracklet
        box
            A NumPy array with shape (4,)
        """
        self.start_index = start_index
        if isinstance(box, SingleBox):
            self.boxes = FrameBoxes.from_single_box(box)
            self.observed = [True]
        elif isinstance(box, FrameBoxes) and isinstance(observed, list):
            self.boxes = box
            self.observed = observed

    def __len__(self) -> int:
        """Check number of boxes in the tracklet."""
        return len(self.boxes)

    def add_box(self, box: SingleBox, observed: bool = True) -> None:
        """Append a box to the end of the array."""
        self.boxes = self.boxes.add_box(box)
        self.observed.append(observed)

    @property
    def previous_box(self) -> SingleBox:
        """Return the most recent addition."""
        return self.boxes.get_single_box(-1)

    @property
    def finalized(self) -> "Tracklet":
        """Remove predicted boxes."""
        mask = np.array(self.observed)
        return Tracklet(
            self.start_index,
            self.boxes[mask],
            mask[mask].tolist(),
        )
