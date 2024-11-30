import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch

from pathlib import Path
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

device = "mps"
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

def extract_frames(video: Path, max_frames: int=1000) -> Path:
    output_dir: Path = video.parent / video.stem
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    # Open the video file
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        logger.error(f"Error: Cannot open video file {video}.")
        return
    logger.info(f"Opened {video} with OpenCV.")

    # Save each frame to .jpg file
    frame_count = 0
    while True:
        ret, frame = cap.read()

        if frame_count>=max_frames:
            break

        if not ret:
            logger.info("No more frames found.")
            break

        output_file = output_dir / f"{frame_count:06d}.jpg"
        cv2.imwrite(output_file, frame)
        logger.info(f"Saved: {output_file}")

        frame_count += 1

    # Release the video capture object
    cap.release()
    logger.info(f"Finished: {frame_count} frames saved to {output_dir}")

    return output_dir

def show_mask(mask, ax, obj_id=None, random_color=False) -> None:
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def plot_frame_with_masks(masks, object_ids: list[int], frame: int, video_dir: Path, output_dir: Path) -> None:
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {frame}")
    plt.imshow(Image.open(video_dir / f"{frame:06d}.jpg"))
    for i_mask, object_id in enumerate(object_ids):
        show_mask((masks[i_mask] > 0.0).cpu().numpy(), plt.gca(), obj_id=object_id)
    plt.savefig(output_dir / f"{frame:06d}.jpg")
    plt.close()

def track_from_frame(
    video_dir: Path,
    output_dir: Path,
    points: list[tuple[int, int]],
    init_frame: int=0,
) -> None:

    labels = np.array([1 for _ in range(len(points))], np.int32)
    points = np.array([[x,y] for x,y in points], dtype=np.float32)
    logger.debug("before context manager.")
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        logger.debug("before initializer.")
        state = predictor.init_state(video_path=str(video_dir))# THIS LINE KILLS THE PROCESS.
        logger.info("Predictor initialized.")

        # add new prompts and instantly get the output on the same frame
        frame_idx, object_ids, masks = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=init_frame,
            obj_id=0,
            points=points,
            labels=labels,
        )
        logger.debug(f"Frame {init_frame} segmented.")

        plot_frame_with_masks(
            masks=masks,
            object_ids=object_ids,
            frame=0,
            video_dir=video_dir,
            output_dir=output_dir,
        )

        # video_segments = {}  # video_segments contains the per-frame segmentation results
        # propagate the prompts to get masklets throughout the video
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            # video_segments[out_frame_idx] = {
            #     out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            #     for i, out_obj_id in enumerate(out_obj_ids)
            # }
            plot_frame_with_masks(
                masks=out_mask_logits,
                object_ids=out_obj_ids,
                frame=out_frame_idx,
                video_dir=video_dir,
                output_dir=output_dir,
            )
            logging.info(f"Propagated to frame {out_frame_idx} with {len(out_obj_ids)} masks.")

def main() -> None:
    video: Path = Path("/Users/danielmillman/Desktop/data/lab_0_720p.mov")
    video_dir = extract_frames(video=video)
    output_dir: Path = Path("/Users/danielmillman/Desktop/data/segmented")
    output_dir.mkdir(exist_ok=True)
    points: list[tuple[int, int]] = [
        (400, 460),
    ]
    track_from_frame(
        video_dir=video_dir,
        output_dir=output_dir,
        points=points,
    )

if __name__ == "__main__":
    main()