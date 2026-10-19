# WorkflowHub registration notes

**Title:** CPU-Only Spatial Object Deduplication for UAV Aerial Imagery

**Description:** Reproducible Snakemake workflow for counting unique objects
across overlapping UAV frames from previously produced COCO detections. It uses
floating-point bounding-box intersection geometry and NetworkX connected
components to resolve duplicate detections transitively without adding any
GPU inference. The self-contained synthetic demo requires only Python and
NetworkX; real UAV-frame mode optionally aligns frames with OpenCV/AKAZE.

**Keywords:** UAV, aerial imagery, litter, CPU, Snakemake, NetworkX,
computer vision, floating-point geometry, bounding boxes, graph transitivity,
object counting, deduplication.

**Main workflow file:** `workflow.snake` (**Snakemake**)

**Source repository:** https://github.com/Ikram-c/CV_Spatial_Deduplication_Workflow

**Licence:** Apache-2.0

**Run/test:** See README and GitHub Actions.

## Reproducibility and claims

| Property | What is actually demonstrated |
| --- | --- |
| No GPU inference | The deduplication stage uses no object detector or GPU |
| Floating-point geometry | Positive rectangle intersection without pixel rounding |
| NetworkX transitivity | A–B and B–C are grouped as one connected component |
| Correctness | Five synthetic detections become three unique objects |
| Real-image accuracy | **Not measured**; requires labelled ground truth |
| CPU cost | Additional geometry/graph work; **no additional model calls** |

In real-image mode, OpenCV feature matching and homography still consume CPU
and require sufficiently consistent views. Intersections are candidate duplicate
relations and can over-merge touching/adjacent objects when registration is wrong.

## Import

1. Upload all extracted repository files to GitHub, preserving directories.
2. Register the public GitHub URL in WorkflowHub's Git repository importer.
3. Select branch `main`, main workflow `workflow.snake`, type `Snakemake`.
4. Add the title and metadata above; make the workflow public.

Official reference: https://about.workflowhub.eu/docs/registering_workflows/adding-files/
