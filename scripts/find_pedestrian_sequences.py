# find_pedestrian_sequences.py

from pathlib import Path
import argparse
import csv
import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan dataset sequences and find frames/sequences containing pedestrians."
    )

    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to dataset sequences folder, e.g. /path/to/dataset/sequences"
    )

    parser.add_argument(
        "--start_sequence",
        default=None,
        help="Start sequence ID, e.g. 000001"
    )

    parser.add_argument(
        "--end_sequence",
        default=None,
        help="End sequence ID, e.g. 001472"
    )

    parser.add_argument(
        "--person_conf",
        type=float,
        default=0.70,
        help="Confidence threshold for person detection"
    )

    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model name/path, e.g. yolov8n.pt"
    )

    parser.add_argument(
        "--output_prefix",
        default="matched",
        help="Prefix for output CSV files"
    )

    return parser.parse_args()


def get_front_frames(sequence_dir):
    """
    Expected structure:
    sequence_id/rgb/front/*.jpg
    """
    front_dir = sequence_dir / "rgb" / "front"

    if not front_dir.exists():
        return []

    frames = sorted(
        list(front_dir.glob("*.jpg")) +
        list(front_dir.glob("*.png")) +
        list(front_dir.glob("*.jpeg"))
    )

    return frames


def sequence_in_range(seq_id, start_seq, end_seq):
    if start_seq is not None and seq_id < start_seq:
        return False
    if end_seq is not None and seq_id > end_seq:
        return False
    return True


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    model = YOLO(args.model)

    sequence_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])

    matched_sequences = []
    matched_frames = []

    for sequence_dir in sequence_dirs:
        seq_id = sequence_dir.name

        if not sequence_in_range(seq_id, args.start_sequence, args.end_sequence):
            continue

        frames = get_front_frames(sequence_dir)

        if not frames:
            print(f"[SKIP] No frames found for sequence {seq_id}")
            continue

        print(f"[PROCESSING] Sequence {seq_id} | Frames: {len(frames)}")

        sequence_person_frames = 0
        max_person_count = 0
        max_confidence = 0.0

        for frame_path in frames:
            image = cv2.imread(str(frame_path))

            if image is None:
                print(f"[WARNING] Could not read frame: {frame_path}")
                continue

            results = model(image, verbose=False)[0]

            person_count = 0
            frame_max_conf = 0.0

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # COCO class 0 = person
                if cls_id == 0 and conf >= args.person_conf:
                    person_count += 1
                    frame_max_conf = max(frame_max_conf, conf)

            if person_count > 0:
                sequence_person_frames += 1
                max_person_count = max(max_person_count, person_count)
                max_confidence = max(max_confidence, frame_max_conf)

                matched_frames.append({
                    "sequence_id": seq_id,
                    "frame_name": frame_path.name,
                    "frame_path": str(frame_path),
                    "person_count": person_count,
                    "max_confidence": round(frame_max_conf, 4)
                })

        if sequence_person_frames > 0:
            matched_sequences.append({
                "sequence_id": seq_id,
                "total_frames": len(frames),
                "person_frames": sequence_person_frames,
                "max_person_count": max_person_count,
                "max_confidence": round(max_confidence, 4)
            })

            print(f"[MATCH] {seq_id} | Person frames: {sequence_person_frames}")
        else:
            print(f"[NO MATCH] {seq_id}")

    seq_csv = f"{args.output_prefix}_sequences_report.csv"
    frame_csv = f"{args.output_prefix}_frames_report.csv"

    with open(seq_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence_id",
                "total_frames",
                "person_frames",
                "max_person_count",
                "max_confidence"
            ]
        )
        writer.writeheader()
        writer.writerows(matched_sequences)

    with open(frame_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence_id",
                "frame_name",
                "frame_path",
                "person_count",
                "max_confidence"
            ]
        )
        writer.writeheader()
        writer.writerows(matched_frames)

    print("\nDone.")
    print(f"Saved sequence report: {seq_csv}")
    print(f"Saved frame report: {frame_csv}")


if __name__ == "__main__":
    main()