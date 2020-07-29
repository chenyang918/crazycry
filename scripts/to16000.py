#ffmpeg -i $DATA_DIR/$FILE -acodec pcm_s16le -ac 1 -ar 16000 $DATA_DIR/$FILENAME.wav

import os
import glob
from tqdm import tqdm


def maybe_makedir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)

rootdir1 = "../datasets/one/train/*/*.wav"
videos = glob.glob(rootdir1)
rootdir2 = "../datasets/one/test/*.wav"
videos.extend(glob.glob(rootdir2))
for video_path in tqdm(videos):
    wav_path = video_path.replace('one', 'one16000')
    folder = os.path.dirname(wav_path)
    maybe_makedir(folder)
    os.system(f'ffmpeg -i {video_path} -ac 1 -ar 16000 {wav_path} >/dev/null 2>&1')

