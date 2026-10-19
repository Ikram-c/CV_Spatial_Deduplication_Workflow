"""CPU-only geometry and graph-based deduplication of existing detections.

Rectangles use floating-point (x_min, y_min, x_max, y_max, category_id,
annotation_id) coordinates in a shared frame reference system.
"""
from collections import Counter
from typing import Iterable, Mapping, Sequence

import networkx as nx


def intersection_area(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Return the exact axis-aligned intersection area; touching is not overlap."""
    width = max(0.0, min(float(box_a[2]), float(box_b[2])) -
                max(float(box_a[0]), float(box_b[0])))
    height = max(0.0, min(float(box_a[3]), float(box_b[3])) -
                 max(float(box_a[1]), float(box_b[1])))
    return width * height


def overlapping_annotation_pairs(
    first: Iterable[Sequence[float]], second: Iterable[Sequence[float]]
) -> list[tuple[int, int]]:
    """Find category-consistent overlaps between detections in two aligned frames.

    The inputs already contain bounding boxes transformed into the same frame.
    No image processing, model inference or GPU work occurs here.
    """
    result = set()
    for a in first:
        for b in second:
            if (a[4] == b[4] and a[5] != b[5]
                    and intersection_area(a, b) > 0.0):
                result.add((int(a[5]), int(b[5])))
    return sorted(result)


def deduplicated_counts(
    categories_by_annotation: Mapping[int, int],
    overlapping_pairs: Iterable[tuple[int, int]],
) -> dict[int, int]:
    """Count each category once per NetworkX connected component.

    A--B and B--C implies one physical object even without a direct A--C
    comparison. Unknown IDs, cross-category edges and self-edges are ignored.
    Isolated annotations count as singletons.
    """
    category_lookup = {int(k): int(v) for k, v in categories_by_annotation.items()}
    graph = nx.Graph()
    graph.add_nodes_from(category_lookup)
    for a, b in overlapping_pairs:
        a, b = int(a), int(b)
        if (a != b and a in category_lookup and b in category_lookup
                and category_lookup[a] == category_lookup[b]):
            graph.add_edge(a, b)
    counts = Counter(category_lookup[min(component)]
                     for component in nx.connected_components(graph))
    return dict(sorted(counts.items()))
