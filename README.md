# Sparrow Tracky

Sparrow Tracky is a Python package that implements basic object tracking and related metrics. The object tracking algorithm is a simplification of SORT and is designed for prototyping in Python -- not for production. The metrics Multi-Object Detection Accuracy (MODA) and Multi-Object Tracking Accuracy (MOTA) are useful for measuring the quality of box predictions.

# Quick Start Example

## Installation

```bash
pip install -U sparrow-tracky
```

## Measuring MODA on frame boxes

```python
import numpy as np
from sparrow_datums import FrameBoxes, PType
from sparrow_tracky import compute_moda

boxes = FrameBoxes(np.ones((4, 4)), PType.absolute_tlwh)
moda = compute_moda(boxes, boxes + 0.1)
moda

# Expected result
# MODA(false_negatives=0, false_positives=0, n_truth=4)

moda.value

# Expected result
# 1.0
```

## Adding MODA objects

```python
moda + moda

# Expected result
# MODA(false_negatives=0, false_positives=0, n_truth=8)

(moda + moda).value

# Expected result
# 1.0
```