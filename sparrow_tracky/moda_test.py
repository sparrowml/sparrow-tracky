from tkinter import Frame

import numpy as np
from sparrow_datums import BoxType, FrameBoxes

from .moda import compute_moda


def test_perfect_box_accuracy():
    a = np.zeros((2, 4))
    a[:, 2:] += 1
    boxes_a = FrameBoxes(a, BoxType.absolute_tlbr)
    boxes_b = boxes_a.copy()
    moda = compute_moda(boxes_a, boxes_b)
    assert moda.value == 1


def test_no_predictions():
    boxes_a = FrameBoxes(np.zeros((0, 4)), BoxType.absolute_tlbr)
    b = np.zeros((2, 4))
    b[:, 2:] += 1
    boxes_b = FrameBoxes(b, BoxType.absolute_tlbr)
    moda = compute_moda(boxes_a, boxes_b)
    assert moda.value == 0


def test_no_ground_truth():
    a = np.zeros((2, 4))
    a[:, 2:] += 1
    boxes_a = FrameBoxes(a, BoxType.absolute_tlbr)
    boxes_b = FrameBoxes(np.zeros((0, 4)), BoxType.absolute_tlbr)
    moda = compute_moda(boxes_a, boxes_b)
    assert moda.value == 0
