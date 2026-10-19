"""Run the WorkflowHub demonstration on precomputed aerial-frame detections.

The JSON contains COCO-format detections already aligned in a common reference
coordinate system. This stage calls no image detector or feature matcher.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from deduplication import deduplicated_counts, overlapping_annotation_pairs


def run(data: dict) -> dict:
    per_frame = defaultdict(list)
    categories = {}
    for ann in data['annotations']:
        ann_id = int(ann['id'])
        category = int(ann['category_id'])
        x, y, width, height = map(float, ann['bbox'])
        if width <= 0 or height <= 0:
            raise ValueError(f'Annotation {ann_id} has a non-positive bounding box')
        if ann_id in categories:
            raise ValueError(f'Duplicate annotation ID: {ann_id}')
        categories[ann_id] = category
        per_frame[int(ann['image_id'])].append(
            (x, y, x + width, y + height, category, ann_id))

    edges = set()
    for first_id, second_id in data['frame_pairs']:
        edges.update(overlapping_annotation_pairs(
            per_frame[int(first_id)], per_frame[int(second_id)]))

    raw = Counter(categories.values())
    counted = deduplicated_counts(categories, edges)
    return {
        'raw_counts': {str(k): v for k, v in sorted(raw.items())},
        'unique_counts': {str(k): v for k, v in counted.items()},
        'raw_total': len(categories),
        'unique_total': sum(counted.values()),
        'duplicates_removed': len(categories) - sum(counted.values()),
        'overlap_edges': [list(edge) for edge in sorted(edges)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.input.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"{result['raw_total']} detections -> {result['unique_total']} unique objects")


if __name__ == '__main__':
    main()
