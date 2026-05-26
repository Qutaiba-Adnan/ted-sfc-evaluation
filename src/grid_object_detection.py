import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.transforms import functional as F

import helper


PERSON_LABEL = 1


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "data_path",
        help="Path to the folder containing the dataset, e.g. ./data/{dataset_name}",
    )
    parser.add_argument(
        "output_path",
        help="Path to the folder where the output will be saved, e.g. ./output/{dataset_name}",
    )
    parser.add_argument(
        "config_path",
        help="Path to the config yml file.",
    )
    parser.add_argument(
        "--method",
        default="faster-rcnn",
        choices=["faster-rcnn", "detr"],
        help="Object detection model to use.",
    )
    parser.add_argument(
        "--cpu", help="Use CPU instead of GPU.", action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--display_results", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def load_model(method: str, device: torch.device):
    if method == "detr":
        model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True)
        model.eval()
        model.to(device)
        return model

    if method != "faster-rcnn":
        raise ValueError(f"Unsupported object detection method: {method}")

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    model.to(device)
    return model


def calculate_overlap(box, cell):
    box_x1, box_y1, box_x2, box_y2 = box
    (cell_x1, cell_y1), (cell_x2, cell_y2) = cell

    overlap_x1 = max(box_x1, cell_x1)
    overlap_y1 = max(box_y1, cell_y1)
    overlap_x2 = min(box_x2, cell_x2)
    overlap_y2 = min(box_y2, cell_y2)

    width = max(0, overlap_x2 - overlap_x1)
    height = max(0, overlap_y2 - overlap_y1)
    return width * height


def point_in_cell(point, cell):
    x, y = point
    (cell_x1, cell_y1), (cell_x2, cell_y2) = cell
    return cell_x1 <= x <= cell_x2 and cell_y1 <= y <= cell_y2


def get_cell_index_for_box(box, cell_positions):
    overlaps = [calculate_overlap(box, cell) for cell in cell_positions]
    max_overlap = max(overlaps) if overlaps else 0

    if max_overlap > 0:
        return int(np.argmax(overlaps))

    x1, y1, x2, y2 = box
    fallback_points = [
        ((x1 + x2) / 2, y2),  # Feet/road contact point.
        ((x1 + x2) / 2, (y1 + y2) / 2),
    ]

    for point in fallback_points:
        for cell_index, cell in enumerate(cell_positions):
            if point_in_cell(point, cell):
                return cell_index

    return None


def detect_person_cell(prediction, cell_positions, score_threshold):
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    boxes = prediction["boxes"].detach().cpu().numpy()

    person_indices = np.where((labels == PERSON_LABEL) & (scores >= score_threshold))[0]
    if len(person_indices) == 0:
        return None, None, None

    best_index = person_indices[np.argmax(scores[person_indices])]
    box = boxes[best_index]
    score = float(scores[best_index])
    cell_index = get_cell_index_for_box(box, cell_positions)

    return cell_index, box, score


def frame_to_tensor(frame, device, method):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(frame_rgb)

    if method == "detr":
        tensor = F.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    return tensor.to(device)


def box_cxcywh_to_xyxy(boxes):
    x_center, y_center, width, height = boxes.unbind(-1)
    return torch.stack(
        [
            x_center - 0.5 * width,
            y_center - 0.5 * height,
            x_center + 0.5 * width,
            y_center + 0.5 * height,
        ],
        dim=-1,
    )


def convert_detr_predictions(outputs, frames):
    probabilities = outputs["pred_logits"].softmax(-1)
    scores, labels = probabilities[:, :, :-1].max(-1)
    boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"])

    predictions = []
    for frame, frame_scores, frame_labels, frame_boxes in zip(
        frames, scores, labels, boxes
    ):
        height, width = frame.shape[:2]
        scale = torch.tensor(
            [width, height, width, height],
            dtype=frame_boxes.dtype,
            device=frame_boxes.device,
        )
        predictions.append(
            {
                "labels": frame_labels,
                "scores": frame_scores,
                "boxes": frame_boxes * scale,
            }
        )

    return predictions


def draw_detection(frame, cell_positions, cell_index, box, score):
    helper.draw_grid(frame, cell_positions)

    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        helper.annotate_frame(
            frame,
            f"person: {score:.2f}",
            (max(10, x1), max(30, y1 - 10)),
            font_scale=0.8,
            font_color=(0, 255, 255),
        )

    if cell_index is not None:
        top_left, bottom_right = cell_positions[cell_index]
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 4)


def process_batch(frames, model, method, device, cell_positions, score_threshold):
    tensors = [frame_to_tensor(frame, device, method) for frame in frames]

    with torch.inference_mode():
        model_output = model(tensors)

    if method == "detr":
        predictions = convert_detr_predictions(model_output, frames)
    else:
        predictions = model_output

    batch_cell_values = []
    batch_detections = []
    total_cells = len(cell_positions)

    for prediction in predictions:
        cell_index, box, score = detect_person_cell(
            prediction, cell_positions, score_threshold
        )

        values = [0] * total_cells
        if cell_index is not None:
            values[cell_index] = score

        batch_cell_values.append(values)
        batch_detections.append((cell_index, box, score))

    return batch_cell_values, batch_detections


def process_video(
    video_path,
    target_path,
    video_id,
    config,
    model,
    method,
    device,
    display_results,
):
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()

    if not ret:
        print("Unable to read video")
        exit(1)

    out = cv2.VideoWriter(
        os.path.join(target_path, f"{video_id}_object_detection_grid.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        config["fps"],
        (first_frame.shape[1], first_frame.shape[0]),
    )

    object_detection_config = config.get("object_detection", {})
    score_threshold = object_detection_config.get("score_threshold", 0.5)
    batch_size = object_detection_config.get("batch_size", 1 if device.type == "cpu" else 2)

    cell_positions = helper.calculate_grid_cell_positions(first_frame, config)
    cell_values = {}
    frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pending_frames = []
    pending_frame_numbers = []

    def flush_batch():
        if not pending_frames:
            return

        batch_values, batch_detections = process_batch(
            pending_frames, model, method, device, cell_positions, score_threshold
        )

        for frame, current_frame_number, values, detection in zip(
            pending_frames, pending_frame_numbers, batch_values, batch_detections
        ):
            cell_index, box, score = detection
            cell_values[current_frame_number] = values

            draw_detection(frame, cell_positions, cell_index, box, score)
            helper.annotate_frame(
                frame,
                f"frame: {current_frame_number}. score_threshold: {score_threshold}",
                (10, 30),
            )

            out.write(frame)

            if display_results:
                cv2.imshow("Object detection grid", frame)

        pending_frames.clear()
        pending_frame_numbers.clear()

    with tqdm(total=total_frames, desc="Frame progress", leave=False) as pbar_frames:
        while ret:
            frame_number += 1
            pending_frames.append(first_frame)
            pending_frame_numbers.append(frame_number)

            if len(pending_frames) >= batch_size:
                flush_batch()

            pbar_frames.update(1)

            if cv2.waitKey(30) & 0xFF == ord("q"):
                print("Interrupted by user")
                break

            ret, first_frame = cap.read()

        flush_batch()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return cell_values


def main(
    data_path: str,
    output_path: str,
    config_path: str,
    method: str = "faster-rcnn",
    display_results: bool = False,
    use_cpu: bool = False,
):
    config = helper.load_yml(config_path)
    grid_config = config["grid_config"]

    os.makedirs(output_path, exist_ok=True)

    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = load_model(method, device)

    for video_dir, video_id, tqdm_obj in helper.traverse_videos(data_path):
        video_path = os.path.join(video_dir, f"{video_id}.avi")
        target_path = os.path.join(output_path, video_id)

        os.makedirs(target_path, exist_ok=True)

        if not os.path.exists(video_path):
            tqdm_obj.write(f"Skipping {video_id}: Video does not exist")
            continue

        output_cell_value_map = process_video(
            video_path,
            target_path,
            video_id,
            grid_config,
            model,
            method,
            device,
            display_results,
        )

        helper.save_cell_value_csv(output_cell_value_map, target_path, grid_config)

        plot_path = os.path.join(target_path, "plots")
        os.makedirs(plot_path, exist_ok=True)

        helper.save_cell_value_subplots(
            output_cell_value_map, plot_path, display_results, "Cell value"
        )
        helper.save_combined_plot(
            output_cell_value_map, plot_path, display_results, "Cell value"
        )

    helper.save_config(config, output_path, "config.yml")


if __name__ == "__main__":
    args = parse_arguments()
    main(
        args.data_path,
        args.output_path,
        args.config_path,
        args.method,
        args.display_results,
        args.cpu,
    )
