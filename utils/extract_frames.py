import os
import threading
import argparse

parser = argparse.ArgumentParser(description='directories')
parser.add_argument('video_root', default='20bn-something-something-v2', type=str)
parser.add_argument('frame_root', default='20bn-something-something-v2-frames', type=str)
parser.add_argument('num_threds', default=100, type=int)
args = parser.parse_args()

NUM_THREADS = args.num_threds
VIDEO_ROOT = args.video_root        # Downloaded webm videos
FRAME_ROOT = args.frame_root  # Directory for extracted frames
# VIDEO_ROOT = '20bn-something-something-v2'         # Downloaded webm videos
# FRAME_ROOT = '20bn-something-something-v2-frames'  # Directory for extracted frames


def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def extract(video, tmpl='%06d.jpg'):
    os.system(f'ffmpeg -loglevel panic -i {VIDEO_ROOT}/{video} -vf scale=256:256 '
              f'{FRAME_ROOT}/{video[:-5]}/{tmpl}')


def target(video_list):
    for video in video_list:
        os.makedirs(os.path.join(FRAME_ROOT, video[:-5]))
        extract(video)


if not os.path.exists(VIDEO_ROOT):
    raise ValueError('Please download videos and set VIDEO_ROOT variable.')
if not os.path.exists(FRAME_ROOT):
    os.makedirs(FRAME_ROOT)

video_list = os.listdir(VIDEO_ROOT)
splits = list(split(video_list, NUM_THREADS))

threads = []
for i, split in enumerate(splits):
    thread = threading.Thread(target=target, args=(split,))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()
