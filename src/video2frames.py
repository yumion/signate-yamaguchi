from pathlib import Path
import cv2
from tqdm import tqdm

root_dir = Path('../input/train')
save_dir = Path('../input/train_image')

with tqdm(root_dir.glob('*.mp4')) as pbar:
    for video_path in pbar:
        video = cv2.VideoCapture(str(video_path))
        save_path = save_dir / video_path.stem / 'images'
        save_path.mkdir(parents=True, exist_ok=True)
        num_frame = 0
        while video.isOpened():
            pbar.set_description(f'{video_path.stem}/frame{num_frame:06d}')
            ret, image = video.read()
            if not ret:
                break
            cv2.imwrite(str(save_path / f'{video_path.stem}_{num_frame:06d}.png'), image)
            num_frame += 1
        video.release()
