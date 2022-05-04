import numpy as np
import numpy.typing as npt
from sparrow_datums import AugmentedBoxTracking, BoxTracking, PType

from .mota import MOTA, compute_mota


def test_perfect_box_accuracy():
    a = np.zeros((5, 2, 4))
    a[..., 2:] += 1
    boxes_a = BoxTracking(a, PType.absolute_tlbr)
    boxes_b = boxes_a.copy()
    mota = compute_mota(boxes_a, boxes_b)
    assert mota.value == 1


def test_no_predictions():
    boxes_a = BoxTracking(np.zeros((2, 2, 4)) * np.nan, PType.absolute_tlbr)
    b = np.zeros((5, 2, 4))
    b[:, 2:] += 1
    boxes_b = BoxTracking(b, PType.absolute_tlbr)
    moda = compute_mota(boxes_a, boxes_b)
    assert moda.value == 0


def test_no_ground_truth():
    a = np.zeros((5, 2, 4))
    a[:, 2:] += 1
    boxes_a = BoxTracking(a, PType.absolute_tlbr)
    boxes_b = BoxTracking(np.zeros((2, 2, 4)) * np.nan, PType.absolute_tlbr)
    moda = compute_mota(boxes_a, boxes_b)
    assert moda.value == 0.0


def test_mota_sum_function_works():
    moda_a = MOTA(id_switches=1)
    moda_b = MOTA(id_switches=2)
    moda_c = sum([moda_a, moda_b], MOTA())
    assert isinstance(moda_c, MOTA)
    assert moda_c.id_switches == 3


def test_string_representation():
    assert "id_switches=0" in str(MOTA())


def test_id_switch_is_counted():
    b1 = np.zeros(4)
    b1[2:] += 1
    b2 = np.ones(4)
    b2[2:] += 1
    empty = np.ones(4) * np.nan
    pred1: npt.NDArray[np.float64] = np.stack([b1[None], b1[None], b1[None], b1[None]])
    pred2: npt.NDArray[np.float64] = np.stack(
        [b2[None], b2[None], empty[None], empty[None]]
    )
    pred: npt.NDArray[np.float64] = np.concatenate([pred1, pred2], axis=1)
    gt1: npt.NDArray[np.float64] = np.stack([b1[None], b1[None], b2[None], b2[None]])
    gt2: npt.NDArray[np.float64] = np.stack([b2[None], b2[None], b1[None], b1[None]])
    gt: npt.NDArray[np.float64] = np.concatenate([gt1, gt2], axis=1)
    pred_tracking = BoxTracking(pred, PType.absolute_tlbr)
    gt_tracking = BoxTracking(gt, PType.absolute_tlbr)
    mota = compute_mota(pred_tracking, gt_tracking)
    assert mota.id_switches == 1
    assert mota.value == 0.625


def test_identical_tracking_chunks_are_perfect():
    chunk = AugmentedBoxTracking.from_file("data/gt-tracking.json.gz")
    mota = compute_mota(chunk, chunk)
    assert mota.n_truth == 23698
    assert mota.value == 1.0


def test_to_dict():
    mota = MOTA()
    mota_dict = mota.to_dict()
    assert mota_dict["value"] == 1.0
