import os
import tempfile
from functools import partial

import numpy as np
from sparrow_datums import BoxTracking, FrameBoxes, PType

from .tracker import Tracker

tlwh_boxes = partial(FrameBoxes, ptype=PType.absolute_tlwh)


def test_negative_iou_threshold_always_matches():
    tracker = Tracker(iou_threshold=-10)
    tracker.track(tlwh_boxes(np.zeros((1, 4))))
    tracker.track(tlwh_boxes(np.ones((1, 4))))
    assert len(tracker.tracklets) == 1, "Boxes should get matched"


def test_track_recovers_from_no_box_frames():
    tracker = Tracker()
    tracker.track(tlwh_boxes(np.ones((1, 4))))
    tracker.track(tlwh_boxes(np.zeros((0, 4))))
    tracker.track(tlwh_boxes(np.ones((3, 4))))
    tracker.track(tlwh_boxes(np.ones((3, 4))))
    assert len(tracker.active_tracklets) == 3, "Should have 3 active tracklets"


def test_inf_threshold_never_matches():
    tracker = Tracker(iou_threshold=np.inf)
    tracker.track(tlwh_boxes(np.ones((1, 4))))
    tracker.track(tlwh_boxes(np.ones((1, 4))))
    assert len(tracker.tracklets) == 2, "Boxes shouldn't get matched"


def test_make_chunk_makes_a_box_tracking_chunk():
    tracker = Tracker()
    tracker.track(tlwh_boxes(np.ones((1, 4))))
    tracker.track(tlwh_boxes(np.zeros((0, 4))))
    tracker.track(tlwh_boxes(np.ones((3, 4))))
    tracker.track(tlwh_boxes(np.ones((3, 4))))
    chunk = tracker.make_chunk(fps=1, min_tracklet_length=2)
    assert isinstance(chunk, BoxTracking)
    assert chunk.fps == 1
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.json.gz")
        chunk.to_file(path)
        new_chunk = BoxTracking.from_file(path)
    assert isinstance(new_chunk, BoxTracking)
    assert chunk.shape == new_chunk.shape
