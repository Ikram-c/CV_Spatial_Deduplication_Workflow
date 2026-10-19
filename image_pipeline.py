"""Optional original-style UAV image pipeline: ODM -> AKAZE -> float boxes -> graph.

No object detector is invoked. COCO annotations must already exist.
OpenCV/NumPy are imported lazily; the default Snakemake demo needs neither.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from deduplication import deduplicated_counts, overlapping_annotation_pairs


# Accept original DJI names as well as other common UAV image basenames.
MATCHED_IMAGE_LINE = re.compile(
    r"\bMatching\s+(\S+?\.(?:jpe?g|png))\s+and\s+"
    r"(\S+?\.(?:jpe?g|png))\b.*?\bSuccess:\s*True\b",
    re.IGNORECASE,
)


def read_matched_image_pairs(log_path: str | Path) -> list[tuple[str, str]]:
    """Read unique successful ODM matching pairs, preserving their first order."""
    pairs = []
    seen = set()
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = MATCHED_IMAGE_LINE.search(line)
        if not match:
            continue
        first, second = (Path(s).name for s in match.groups())
        if first == second:
            continue
        key = tuple(sorted((first, second)))
        if key not in seen:
            seen.add(key)
            pairs.append((first, second))
    return pairs


def load_coco_annotations(coco_path: str | Path):
    """Index COCO detections by source image without loading any images.

    Returns (filenames_to_ids, boxes_by_image_id, categories_by_annotation_id).
    Boxes are x_min, y_min, x_max, y_max, category_id, annotation_id.
    """
    data = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    names = {}
    for frame in data['images']:
        name = Path(frame['file_name']).name
        if name in names:
            raise ValueError(f"Duplicate COCO image filename: {name}")
        names[name] = int(frame['id'])

    image_ids = set(names.values())
    boxes = defaultdict(list)
    categories = {}
    for ann in data['annotations']:
        ann_id = int(ann['id'])
        frame_id = int(ann['image_id'])
        category_id = int(ann['category_id'])
        if ann_id in categories:
            raise ValueError(f"Duplicate COCO annotation ID: {ann_id}")
        if frame_id not in image_ids:
            raise ValueError(f"Annotation {ann_id} references missing image ID {frame_id}")
        if len(ann['bbox']) != 4:
            raise ValueError(f"Annotation {ann_id} does not have a COCO xywh bbox")
        x, y, w, h = map(float, ann['bbox'])
        if not all(map(math.isfinite, (x, y, w, h))) or w <= 0 or h <= 0:
            raise ValueError(f"Annotation {ann_id} has invalid bbox")
        categories[ann_id] = category_id
        boxes[frame_id].append((x, y, x + w, y + h, category_id, ann_id))
    return names, boxes, categories


def find_cpu_homography(first_file: Path, second_file: Path):
    """Estimate first->second image registration using CPU-only AKAZE/RANSAC.

    Returns None for pairs without sufficient good feature matches.
    """
    import cv2
    import numpy as np

    gray_a = cv2.imread(str(first_file), cv2.IMREAD_GRAYSCALE)
    gray_b = cv2.imread(str(second_file), cv2.IMREAD_GRAYSCALE)
    if gray_a is None or gray_b is None:
        raise FileNotFoundError(f"Cannot read image pair: {first_file}, {second_file}")

    akaze = cv2.AKAZE_create()
    key_a, descriptors_a = akaze.detectAndCompute(gray_a, None)
    key_b, descriptors_b = akaze.detectAndCompute(gray_b, None)
    if descriptors_a is None or descriptors_b is None or len(descriptors_a) < 2 or len(descriptors_b) < 2:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(good) < 11:  # Preserve original code's >10 high-confidence match threshold.
        return None

    src = np.float32([key_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([key_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if (homography is None or homography.shape != (3, 3)
            or not np.all(np.isfinite(homography))
            or inliers is None or int(inliers.sum()) < 4):
        return None
    return homography


def transform_boxes_float(rectangles: list[tuple], homography) -> list[tuple]:
    """Project every bbox corner; return fractional axis-aligned enclosing bboxes."""
    import cv2
    import numpy as np

    if not rectangles:
        return []
    result = []
    for x0, y0, x1, y1, category, ann_id in rectangles:
        corners = np.array([
            [x0, y0], [x1, y0], [x1, y1], [x0, y1]
        ], dtype=np.float64).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        if not np.all(np.isfinite(transformed)):
            continue
        lo = transformed.min(axis=0)
        hi = transformed.max(axis=0)
        if hi[0] <= lo[0] or hi[1] <= lo[1]:
            continue
        result.append((float(lo[0]), float(lo[1]), float(hi[0]),
                       float(hi[1]), category, ann_id))
    return result


def count_uav_detections(odm_log: str | Path, image_dir: str | Path,
                         annotations: str | Path) -> dict[str, Any]:
    """Deduplicate prior COCO detections using selected aerial-image pairs.

    Unmatched/unalignable annotations stay as singleton graph components.
    """
    image_dir = Path(image_dir)
    names, per_image_boxes, categories = load_coco_annotations(annotations)
    matched_pairs = read_matched_image_pairs(odm_log)
    edges = set()
    skipped = []
    aligned = 0
    for first_name, second_name in matched_pairs:
        if first_name not in names or second_name not in names:
            skipped.append({'pair': [first_name, second_name], 'reason': 'not_in_coco'})
            continue
        first_file, second_file = image_dir / first_name, image_dir / second_name
        transform = find_cpu_homography(first_file, second_file)
        if transform is None:
            skipped.append({'pair': [first_name, second_name], 'reason': 'insufficient_features'})
            continue
        aligned += 1
        first_boxes = transform_boxes_float(per_image_boxes[names[first_name]], transform)
        second_boxes = per_image_boxes[names[second_name]]
        edges.update(overlapping_annotation_pairs(first_boxes, second_boxes))

    raw = Counter(categories.values())
    counts = deduplicated_counts(categories, edges)
    return {
        'raw_counts': {str(k): n for k, n in sorted(raw.items())},
        'unique_counts': {str(k): n for k, n in sorted(counts.items())},
        'raw_total': len(categories),
        'unique_total': sum(counts.values()),
        'duplicates_removed': len(categories) - sum(counts.values()),
        'overlap_edges': [list(edge) for edge in sorted(edges)],
        'odm_pairs': len(matched_pairs),
        'aligned_pairs': aligned,
        'skipped_pairs': skipped,
    }
