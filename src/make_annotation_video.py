# SET JP FONT
FONT_FAMILY = "HackGen"

# Following code print installed fonts
# from matplotlib import font_manager
# [print(f.name) for f in font_manager.fontManager.ttflist];

# SET YOUR DIR PATH
INPUT_DIR = "input/train/"  # contains input mp4, json files
OUTPUT_DIR = "input/annotated/"  # annotated movies will be saved here


import json
import pathlib
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm

logger.remove()
logger.add(sys.stderr, colorize=True, level="INFO")

plt.rcParams["font.family"] = FONT_FAMILY
INPUT_DIR = pathlib.Path(INPUT_DIR)
OUTPUT_DIR = pathlib.Path(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CATEGORY2COLOR = {
    "要補修-1.区画線": "cyan",
    "要補修-2.道路標識": "magenta",
    "要補修-3.照明": "yellow",
    "補修不要-1.区画線": "darkcyan",
    "補修不要-2.道路標識": "darkmagenta",
    "補修不要-3.照明": "olive",
}

CATEGORIES = list(CATEGORY2COLOR.keys())


def main():
    annot_paths = sorted(list(INPUT_DIR.glob("*.json")))
    video_paths = sorted(list(INPUT_DIR.glob("*.mp4")))

    i_scene = 0
    i_frame = 0
    video_path = video_paths[i_scene]
    annot_path = annot_paths[i_scene]

    frames, _, _, _ = load_frames(video_path)
    annots = load_annots(annot_path)

    frame = frames[i_frame]
    annot = annots[i_frame]

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.imshow(frame)
    ax.set_title("original")
    plt.show()

    scene = video_path.stem
    frame = draw_annotation(scene, i_frame, frame, annot)
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_title("annotated")
    ax.imshow(frame)
    plt.show()

    n_scenes = len(video_paths)
    for i_scene in range(n_scenes):
        video_path = video_paths[i_scene]
        annot_path = annot_paths[i_scene]
        assert video_path.stem == annot_path.stem
        create_annotated_video(video_path, annot_path, output_dir=OUTPUT_DIR)
        break


def need_repair(cat):
    return cat.startswith("要")


def get_category_weight(cat):
    return 3 if need_repair(cat) else 2


def get_category_line_style(cat):
    return "-" if need_repair(cat) else "--"


def load_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"failed to open {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("opened {} ({}, {}) {} fps {} frames", video_path, w, h, fps, n_frames)
    frames = []
    for i in tqdm(range(n_frames)):
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"failed to read {i}-th frame")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames, w, h, fps


def load_annots(annot_path):
    with open(annot_path, "r", encoding="utf-8") as f:
        annot = json.load(f)
    return annot


def draw_annotation(scene, i_frame, frame, annot, return_type="ndarray"):
    text_color = "white"
    font_size = 12
    h, w = frame.shape[:2]
    info_x = 100
    info_y = h - 30
    label_x0 = w - 650
    label_y0 = h - 100
    label_w = 300
    label_h = 25
    legend_x_offset = 220
    legend_y_offset = 10
    legend_length = 30

    # 1920 x 1080
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=120)

    # draw image
    ax.imshow(frame)
    ax.set_axis_off()

    # draw annot
    for cat, bboxes in annot["labels"].items():
        lc = CATEGORY2COLOR[cat]
        lw = get_category_weight(cat)
        ls = get_category_line_style(cat)
        for ((x1, y1), (x2, y2)) in bboxes:
            bw = x2 - x1
            bh = y2 - y1
            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    width=bw,
                    height=bh,
                    color=lc,
                    fill=False,
                    alpha=0.7,
                    lw=lw,
                    ls=ls,
                )
            )

    # draw additional info
    ax.text(
        info_x, info_y, f"{scene}/{i_frame:03d}", color=text_color, fontsize=font_size
    )

    # draw legend
    for i, cat in enumerate(CATEGORIES):
        iy = i % 3
        ix = i // 3
        label_xi = label_x0 + label_w * ix
        label_yi = label_y0 + label_h * iy
        ax.text(label_xi, label_yi, cat, color=text_color, fontsize=font_size)
        legend_line_x = [
            label_xi + legend_x_offset,
            label_xi + legend_x_offset + legend_length,
        ]
        legend_line_y = [label_yi - legend_y_offset, label_yi - legend_y_offset]
        lc = CATEGORY2COLOR[cat]
        lw = get_category_weight(cat)
        ls = get_category_line_style(cat)
        ax.plot(legend_line_x, legend_line_y, color=lc, lw=lw, ls=ls)

    if return_type == "ndarray":
        # remove margin
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.margins(0, 0)

        # draw into np.array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape((h, w, 3))
        plt.clf()
        plt.close()
        return img
    else:
        return fig, ax


def create_annotated_video(video_path, annot_path, output_dir):
    frames, w, h, fps = load_frames(video_path)
    annots = load_annots(annot_path)
    assert len(annots) == len(frames), "num of annot and frame mismatched."

    output_path = output_dir / video_path.name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    if not video.isOpened():
        raise ValueError(f"failed to initialize video file: {output_path}")

    n_frames = len(frames)
    scene = video_path.stem

    for i in tqdm(range(n_frames)):
        frame = frames[i]
        annot = annots[i]
        frame = draw_annotation(scene, i, frame, annot)
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    logger.info("wrote {}", output_path)
    video.release()
