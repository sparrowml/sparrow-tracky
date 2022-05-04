import json
import os
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from sparrow_datums import AugmentedBoxTracking, BoxTracking, FrameBoxes, PType

from ..metrics import compute_mota
from .tracker import Tracker

tlwh_boxes = partial(FrameBoxes, ptype=PType.absolute_tlwh)
CLASS_MAP = {
    1: "person",
    2: "bicycle",
    3: "car",
    62: "chair",
}


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


@pytest.mark.skipif(os.getenv("FAST") == "1", reason="Skip slow tests")
def test_non_kalman_filter_mota():
    detections = AugmentedBoxTracking.from_file("data/pred-detections.json.gz")
    tracking = AugmentedBoxTracking.from_file("data/gt-tracking.json.gz")
    if detections.fps != tracking.fps:
        detections = detections.resample(tracking.fps)
    all_classes = sorted([c for c in set(detections.labels.ravel()) if c >= 0])
    class_mota = {}
    for class_idx in all_classes:
        class_detections = detections.filter_by_class(class_idx)
        tracker = Tracker()
        for frame in class_detections:
            tracker.track(frame)
        class_pred_tracking = tracker.make_chunk(
            detections.fps, min_tracklet_length=round(detections.fps)
        )
        class_gt_tracking = tracking.filter_by_class(class_idx)
        mota = compute_mota(class_pred_tracking, class_gt_tracking)
        class_mota[CLASS_MAP[class_idx]] = mota.to_dict()
    folder = Path("data/no-kalman-filter")
    folder.mkdir(exist_ok=True, parents=True)
    for key, metrics in class_mota.items():
        with open(folder / f"{key}.json", "w") as f:
            f.write(json.dumps(metrics, indent=4, sort_keys=True))
