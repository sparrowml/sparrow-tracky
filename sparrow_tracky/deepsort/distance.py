import numpy as np
import numpy.typing as npt
from sparrow_datums import FrameBoxes, pairwise_iou


def iou_distance(a: FrameBoxes, b: FrameBoxes) -> npt.NDArray[np.float64]:
    """Pairwise IoU as a distance."""
    ious: npt.NDArray[np.float64] = pairwise_iou(a, b)
    return 1 - ious


def euclidean_distance(a: FrameBoxes, b: FrameBoxes) -> npt.NDArray[np.float64]:
    """Pairwise Euclidean distance."""
    distances: npt.NDArray[np.float64] = np.linalg.norm(
        a.array[:, None] - b.array[None],
        axis=-1,
    ).astype("float64")
    return distances
