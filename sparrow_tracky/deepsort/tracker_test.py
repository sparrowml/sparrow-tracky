from functools import partial

import numpy as np
from sparrow_datums import FrameBoxes, PType

from .tracker import Tracker

tlwh_boxes = partial(FrameBoxes, ptype=PType.absolute_tlwh)


def test_0_iou_threshold_always_matches():
    tracker = Tracker(iou_threshold=0)
    tracker.track(tlwh_boxes(np.zeros((1, 4))))
    tracker.track(tlwh_boxes(np.ones((1, 4))))
    assert len(tracker.tracklets) == 1, "Boxes should get matched"


def test_track_recovers_from_no_box_frames():
    tracker = Tracker()
    tracker.track(tlwh_boxes(np.ones((1, 4))))
    tracker.track(tlwh_boxes(np.zeros((0, 4))))
    tracker.track(tlwh_boxes(np.ones((2, 4))))
    tracker.track(tlwh_boxes(np.ones((2, 4))))
    assert len(tracker.active_tracklets) == 2, "Should have 2 active tracklets"


def test_inf_threshold_never_matches():
    tracker = Tracker(iou_threshold=np.inf)
    tracker.track(tlwh_boxes(np.zeros((1, 4))))
    tracker.track(tlwh_boxes(np.ones((1, 4))))
    assert len(tracker.tracklets) == 2, "Boxes shouldn't get matched"
