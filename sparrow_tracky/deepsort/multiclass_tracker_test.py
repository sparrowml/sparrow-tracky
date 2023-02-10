from functools import partial

import numpy as np
import numpy.typing as npt
from sparrow_datums import AugmentedBoxTracking, FrameAugmentedBoxes, PType

from .multiclass_tracker import MultiClassTracker

tlwh_boxes = partial(FrameAugmentedBoxes, ptype=PType.absolute_tlwh)
NDArray = npt.NDArray[np.float64]


def test_multiclass_tracks_multiple_classes():
    tracker = MultiClassTracker(n_classes=2)
    class_0 = np.zeros((1, 6))
    class_0[:, :4] = 1
    class_1 = np.ones((1, 6))
    tracker.track(tlwh_boxes(class_0))
    tracker.track(tlwh_boxes(class_0))
    tracker.track(tlwh_boxes(class_1))
    tracker.track(tlwh_boxes(class_1))
    assert tracker.frame_index == 4
    assert len(tracker.trackers.keys()) == 2
    assert len(tracker.trackers[0].tracklets) == 1
    assert len(tracker.trackers[1].tracklets) == 1


def test_make_chunk_makes_augmented_box_tracking_chunk():
    tracker = MultiClassTracker(2)
    class_0 = np.zeros((1, 6))
    class_0[:, :4] = 1
    class_1 = np.ones((1, 6))
    tracker.track(tlwh_boxes(class_0))
    tracker.track(tlwh_boxes(class_0))
    tracker.track(tlwh_boxes(class_1))
    tracker.track(tlwh_boxes(class_1))
    chunk = tracker.make_chunk(fps=1, min_tracklet_length=2)
    assert isinstance(chunk, AugmentedBoxTracking)
    assert chunk.fps == 1
    assert isinstance(chunk.object_ids, list)
