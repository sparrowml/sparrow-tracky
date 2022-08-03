from functools import partial

import numpy as np
from sparrow_datums import FrameBoxes, PType

from .distance import euclidean_distance, iou_distance

tlwh_boxes = partial(FrameBoxes, ptype=PType.absolute_tlwh)


def test_iou_distance_make_sense():
    a = np.random.uniform(size=(10, 4))
    b = np.random.uniform(size=(5, 4))
    costs = iou_distance(tlwh_boxes(a), tlwh_boxes(b))
    assert costs.shape == (10, 5)
    assert (costs <= 1).all()
    assert (costs >= 0).all()


def test_euclidean_distance_makes_sense():
    a = np.random.uniform(size=(10, 4))
    b = np.random.uniform(size=(5, 4))
    costs = euclidean_distance(tlwh_boxes(a), tlwh_boxes(b))
    assert costs.shape == (10, 5)
    assert (costs >= 0).all()
