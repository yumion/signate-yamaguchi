from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import imageio


def main():
    result_dir = Path('../work_dirs/retinanet_r50_fpn_1x_coco/20230102_031851/prediction')
    save_dir = result_dir.parent / 'prediction_video'

    save_dir.mkdir(parents=True, exist_ok=True)
    pre_scene = None
    for img_p in tqdm(result_dir.glob('*.png')):
        scene = '_'.join(img_p.name.split('_')[:-1])
        if pre_scene is None:
            # 新しい動画ならwriterを新しく立てる
            tqdm.write(f'first video: {scene}')
            writer = VideoWriter(save_dir / f'{scene}.mp4')
        elif pre_scene != scene:
            # 次の動画ならwriterを立て直す
            writer.release()
            tqdm.write(f'next video: {scene}')
            writer = VideoWriter(save_dir / f'{scene}.mp4')
        else:
            # 同じ動画ならフレームを追加していく
            image = np.array(Image.open(img_p))
            writer.write(image)
        pre_scene = scene


class VideoWriter:
    def __init__(self, video_path, fps=10):
        self.writer = imageio.get_writer(
            video_path, format='ffmpeg', fps=fps)

    def release(self):
        self.writer.close()

    def write(self, image):
        self.writer.append_data(image)


if __name__ == '__main__':
    main()
