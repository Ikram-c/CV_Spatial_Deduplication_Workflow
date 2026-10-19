# CPU-Only Spatial Object Deduplication for UAV Aerial Imagery

[![CPU-only tests](https://github.com/Ikram-c/CV_Spatial_Deduplication_Workflow/actions/workflows/cpu-deduplication.yml/badge.svg)](https://github.com/Ikram-c/CV_Spatial_Deduplication_Workflow/actions/workflows/cpu-deduplication.yml)

A small, **WorkflowHub-ready Snakemake workflow** adapted from
[UAV_unsupervised_litter_counter](https://github.com/Ikram-c/UAV_unsupervised_litter_counter).
It deduplicates **existing** COCO object detections across overlapping UAV frames
using floating-point rectangle intersection and NetworkX transitive graph
components. Deduplication runs on CPU and **does not call a detection model**.

## Try the reproducible demo

Requires Python 3.11+ (use `python3` on macOS):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
snakemake --snakefile workflow.snake --cores 1
cat output/demo_counts.json
python -m unittest discover -s unit_tests -p 'test_*.py' -v
```

The example has **5 input detections**, **3 unique objects**, and **2 duplicate
detections removed**. Its graph has edges `101–102` and `102–103`: NetworkX
correctly groups all three even though `101–103` is not a direct overlap pair.
No images, detector weights, GPU, ODM output, or internet access are needed once
Python dependencies are installed.

Run without Snakemake if desired:

```bash
python -m examples.run_demo --input examples/overlapping_frames.json --output output/demo_counts.json
```

## Use actual aerial image frames (optional)

This mode retains the original high-level processing flow:

**ODM match log → CPU OpenCV/AKAZE homography → floating-point projected COCO
boxes → cross-frame overlap edges → NetworkX connected components → JSON counts.**

You supply prior detections and the ODM log, not a detector model. Install CPU
image-matching dependencies:

```bash
python -m pip install -r requirements-images.txt
snakemake --snakefile workflow.snake --cores 1 \
  --config mode=images \
    odm_log=/absolute/path/to/odm_match_log.txt \
    images=/absolute/path/to/aerial_images \
    annotations=/absolute/path/to/detections.coco.json
```

Or use the compatible direct CLI:

```bash
python main.py \
  --txt_file_path /path/to/odm_match_log.txt \
  --img_sub_dir /path/to/aerial_images \
  --json_file_path /path/to/detections.coco.json
```

Output: `output/<image_directory_name>_counts.json`. Frames must have image
names matching COCO `images[*].file_name`. The matcher skips image pairs that
cannot be reliably aligned and records them in a summary; skipped pairs can
lead to under-deduplication. Bounding-box overlap indicates a **candidate**
duplicate, not a guaranteed instance identity.

## How it works

1. Transform existing `[x, y, width, height]` COCO boxes into a common frame
   using a homography (or provide boxes already aligned for the demo).
2. Intersect rectangles as **floating-point geometry**, with positive area
   and matching category required; touching edges are not duplicates.
3. Add edges between IDs of overlapping cross-frame detections.
4. Count one instance per **NetworkX connected component**; isolated
   detections remain singleton objects.
5. Export raw/unique counts and the overlap edges as JSON.

This adds **zero new GPU inference calls**; it does not claim zero CPU work.
The synthetic example verifies deterministic behavior, **not accuracy on a
labelled real-world UAV dataset**.

## Publish to WorkflowHub

1. Extract the ZIP from ChatGPT.
2. Upload the **extracted contents** of this repository to
   [CV_Spatial_Deduplication_Workflow](https://github.com/Ikram-c/CV_Spatial_Deduplication_Workflow)
   (GitHub → Add file → Upload files; preserve folder paths).
3. On [WorkflowHub](https://workflowhub.eu/), choose to register/import a
   Git repository, enter the repository URL and select `main` and `workflow.snake`.
4. Set workflow type to **Snakemake**, supply a title/description and publish.

See [WORKFLOWHUB.md](WORKFLOWHUB.md) for metadata and limitations.

## Source and licence

Adapted for reproducible CPU-only demonstrations from
[Ikram-c/UAV_unsupervised_litter_counter](https://github.com/Ikram-c/UAV_unsupervised_litter_counter),
licensed Apache-2.0. `LICENSE` retains the upstream licence. This focused
workflow retains the algorithmic image-alignment/deduplication flow, while
removing unused plotting and packaging code.
