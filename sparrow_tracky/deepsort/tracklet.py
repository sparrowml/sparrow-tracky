from sparrow_datums import FrameBoxes, SingleBox


class Tracklet:
    """Store the location history for an object."""

    def __init__(self, start_index: int, box: SingleBox) -> None:
        """
        Store the location history for an object.

        Parameters
        ----------
        start_index : int
            The frame index that starts the tracklet
        box : SingleBox
            A NumPy array with shape (4,)
        """
        self.start_index = start_index
        self.boxes = FrameBoxes.from_single_box(box)
        self.observed = [True]

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
    def n_predicted(self) -> int:
        """Return the number of predicted boxes at the end of the tracklet."""
        n_predicted = 0
        for observation in self.observed[::-1]:
            if observation:
                return n_predicted
            n_predicted += 1
        return n_predicted

    def predict_next_box(self) -> SingleBox:
        """Return the prediction for the box at the next time step."""
        return self.previous_box
