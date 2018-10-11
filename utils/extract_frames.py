import os
import threading
import argparse
import subprocess
import shutil
from tqdm import tqdm
import skvideo.io

# multithreading
from joblib import delayed
from joblib import Parallel
parser = argparse.ArgumentParser(description='directories')
parser.add_argument('--video_root', default='20bn-something-something-v2', type=str)
parser.add_argument('--frame_root', default='20bn-something-something-v2-frames', type=str)
parser.add_argument('--num_threads', default=100, type=int)
args = parser.parse_args()

NUM_THREADS = args.num_threads
VIDEO_ROOT = args.video_root        # Downloaded webm videos
FRAME_ROOT = args.frame_root  # Directory for extracted frames
# VIDEO_ROOT = '20bn-something-something-v2'         # Downloaded webm videos
# FRAME_ROOT = '20bn-something-something-v2-frames'  # Directory for extracted frames


def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def extract(video, tmpl='%06d.jpg'):
    # read first frame of the video and get the SIZE
    video_file_path = os.path.join(VIDEO_ROOT, video)
    try:
        reader = skvideo.io.FFmpegReader(video_file_path)
        for frame in reader.nextFrame():
            # print (frame.shape)
            width, height = frame.shape[1], frame.shape[0]
            break
        reader.close()
    except:
        print (video_file_path)
        return

    os.makedirs(os.path.join(FRAME_ROOT, video[:-5]))

    if width>height:
        cmd = 'ffmpeg -loglevel panic -i \"{}/{}\" -vf scale=-1:320 -sws_flags bilinear -q:v 5 \"{}/{}/{}\"'.\
        format(VIDEO_ROOT, video, FRAME_ROOT, video[:-5], tmpl)
    else:
        cmd = 'ffmpeg -loglevel panic -i \"{}/{}\" -vf scale=320:-1 -sws_flags bilinear -q:v 5 {} \"{}/{}/{}\"'.\
        format(VIDEO_ROOT, video, FRAME_ROOT, video[:-5], tmpl)

    # cmd = 'ffmpeg -loglevel panic -i \"{}/{}\" -vf scale=256:256 \"{}/{}/{}\"'.\
    # format(VIDEO_ROOT, video, FRAME_ROOT, video[:-5], tmpl)

    # print (cmd)
    # asdf
    subprocess.call(cmd, shell=True)
    # os.system(f'ffmpeg -loglevel panic -i {VIDEO_ROOT}/{video} -vf scale=256:256 '
    #           f'{FRAME_ROOT}/{video[:-5]}/{tmpl}')

    # cmd = 'ffmpeg -loglevel panic -i \"{}\" -vf scale=-1:{} -r {} -sws_flags bilinear -q:v {} \"{}/image_%05d.jpg\"'.\
    # format(video_file_path, smaller_size, fps, jpg_quality, dst_directory_path)


def target(video_list):
    for video in video_list:
        os.makedirs(os.path.join(FRAME_ROOT, video[:-5]))
        extract(video)


if not os.path.exists(VIDEO_ROOT):
    raise ValueError('Please download videos and set VIDEO_ROOT variable.')
if not os.path.exists(FRAME_ROOT):
    os.makedirs(FRAME_ROOT)

video_list = os.listdir(VIDEO_ROOT)

Parallel(n_jobs=NUM_THREADS)(delayed(extract)(each_video) for each_video in tqdm(video_list, ascii='#'))
# splits = list(split(video_list, NUM_THREADS))

# threads = []
# for i, split in enumerate(splits):
#     thread = threading.Thread(target=target, args=(split,))
#     thread.start()
#     threads.append(thread)

# for thread in threads:
#     thread.join()
