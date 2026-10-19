"""Original-style CLI entrypoint for CPU spatial deduplication on UAV frames.

Consumes prior COCO detections and successful ODM image-pair registrations.
No detector inference is performed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from image_pipeline import count_uav_detections


def process_single(odm_log: Path, images: Path, coco: Path) -> Path:
    if not odm_log.is_file() or not images.is_dir() or not coco.is_file():
        raise FileNotFoundError(f'Expected ODM log, image directory and COCO JSON: {odm_log}, {images}, {coco}')
    results = count_uav_detections(odm_log, images, coco)
    result_path = Path('output') / f'{images.name}_counts.json'
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(f"{results['raw_total']} detections -> {results['unique_total']} unique objects: {result_path}")
    return result_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--txt_file_path', type=Path, help='ODM successful image matching log')
    parser.add_argument('--img_sub_dir', type=Path, help='UAV image directory')
    parser.add_argument('--json_file_path', type=Path, help='COCO annotations JSON')
    parser.add_argument('--batch_processing', action='store_true')
    parser.add_argument('--img_super_dir', type=Path)
    parser.add_argument('--txt_super_dir', type=Path)
    parser.add_argument('--json_super_dir', type=Path)
    args = parser.parse_args()
    if args.batch_processing:
        if not all((args.img_super_dir, args.txt_super_dir, args.json_super_dir)):
            parser.error('--batch_processing requires all three --*_super_dir arguments')
        folders = sorted(p for p in args.img_super_dir.iterdir() if p.is_dir())
        logs = sorted(p for p in args.txt_super_dir.iterdir() if p.is_file())
        jsons = sorted(p for p in args.json_super_dir.iterdir() if p.suffix.lower() == '.json')
        if len(folders) != len(logs) or len(logs) != len(jsons):
            parser.error('Batch inputs have unequal lengths; pair files/folders by their sorted names')
        for log, images, coco in zip(logs, folders, jsons):
            process_single(log, images, coco)
    else:
        if not all((args.txt_file_path, args.img_sub_dir, args.json_file_path)):
            parser.error('Provide --txt_file_path, --img_sub_dir and --json_file_path')
        process_single(args.txt_file_path, args.img_sub_dir, args.json_file_path)


if __name__ == '__main__':
    main()
