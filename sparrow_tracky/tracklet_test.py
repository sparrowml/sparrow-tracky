import numpy as np
from sparrow_datums import SingleBox

from .tracklet import Tracklet


def test_tracklet_accepts_1d_array():
    box = SingleBox(np.random.randn(4))
    tracklet = Tracklet(0, box)
    assert tracklet.boxes.shape == (1, 4)


def test_tracklet_add_box_concatenates():
    box = SingleBox(np.random.randn(4))
    tracklet = Tracklet(0, box)
    tracklet.add_box(box)
    assert tracklet.boxes.shape == (2, 4)


def test_tracklet_previous_box_is_last_box_added():
    box_a = SingleBox(np.random.randn(4))
    box_b = SingleBox(np.zeros(4))
    box_c = SingleBox(np.ones(4))
    tracklet = Tracklet(0, box_a)
    tracklet.add_box(box_b)
    tracklet.add_box(box_c)
    np.testing.assert_equal(tracklet.previous_box.array, np.ones(4))


def test_len_of_tracklet():
    box_a = SingleBox(np.random.randn(4))
    box_b = SingleBox(np.zeros(4))
    box_c = SingleBox(np.ones(4))
    tracklet = Tracklet(0, box_a)
    tracklet.add_box(box_b)
    tracklet.add_box(box_c)
    assert len(tracklet) == 3
