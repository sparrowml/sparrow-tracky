import numpy as np
from sparrow_datums import FrameAugmentedBoxes, FrameBoxes, PType

from .moda import MODA, compute_moda, compute_moda_by_class


def test_perfect_box_accuracy():
    a = np.zeros((2, 4))
    a[:, 2:] += 1
    boxes_a = FrameBoxes(a, PType.absolute_tlbr)
    boxes_b = boxes_a.copy()
    moda = compute_moda(boxes_a, boxes_b)
    assert moda.value == 1


def test_no_predictions():
    boxes_a = FrameBoxes(np.zeros((0, 4)), PType.absolute_tlbr)
    b = np.zeros((2, 4))
    b[:, 2:] += 1
    boxes_b = FrameBoxes(b, PType.absolute_tlbr)
    moda = compute_moda(boxes_a, boxes_b)
    assert moda.value == 0


def test_no_ground_truth():
    a = np.zeros((2, 4))
    a[:, 2:] += 1
    boxes_a = FrameBoxes(a, PType.absolute_tlbr)
    boxes_b = FrameBoxes(np.zeros((0, 4)), PType.absolute_tlbr)
    moda = compute_moda(boxes_a, boxes_b)
    assert moda.value == 1.0


def test_moda_by_class():
    a = np.zeros((2, 6))
    a[:, -2] += 1
    a[:, -1] = np.array([0, 1])
    boxes_a = FrameAugmentedBoxes(a, PType.absolute_tlwh)
    boxes_b = FrameAugmentedBoxes(a, PType.absolute_tlwh)
    moda_dict = compute_moda_by_class(boxes_a, boxes_b)
    for label in (0, 1):
        assert label in moda_dict
        assert isinstance(moda_dict[label], MODA)


def test_moda_sum_function_works():
    moda_a = MODA(n_truth=1)
    moda_b = MODA(n_truth=2)
    moda_c = sum([moda_a, moda_b], MODA())
    assert isinstance(moda_c, MODA)
    assert moda_c.n_truth == 3


def test_string_representation():
    assert "n_truth=0" in str(MODA())
