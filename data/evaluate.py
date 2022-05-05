import json
from pathlib import Path

import fire
from sparrow_datums import AugmentedBoxTracking

from sparrow_tracky import Tracker, compute_mota

CLASS_MAP = {
    1: "person",
    2: "bicycle",
    3: "car",
    62: "chair",
}


def evaluate(
    detection_path: str = "data/pred-detections.json.gz",
    tracking_path: str = "data/gt-tracking.json.gz",
    output_path: str = "data/metrics",
) -> None:
    """Evaluate the tracking algorithm."""
    detections = AugmentedBoxTracking.from_file(detection_path)
    tracking = AugmentedBoxTracking.from_file(tracking_path)
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
        class_mota[CLASS_MAP[class_idx]]["_n_tracklets"] = len(tracker.tracklets)
    folder = Path(output_path)
    folder.mkdir(exist_ok=True, parents=True)
    for key, metrics in class_mota.items():
        with open(folder / f"{key}.json", "w") as f:
            f.write(json.dumps(metrics, indent=4, sort_keys=True))


if __name__ == "__main__":
    fire.Fire(evaluate)
