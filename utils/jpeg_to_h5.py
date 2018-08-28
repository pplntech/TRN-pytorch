import os
import h5py
import numpy as np
import time
import argparse
from tqdm import tqdm

# multithreading
from joblib import delayed
from joblib import Parallel

def save_h5(videoname_ind, videoname):
    start_time = time.time()
    input_frame_dir = os.path.join(input_jpegs_dir, videoname)
    files = os.listdir(input_frame_dir)
    files = [os.path.join(input_frame_dir, f) for f in files if not f.startswith('.') and f.lower().endswith(img_ext.lower())]
    files = sorted(files)

    outfile_path = os.path.join(output_dir, videoname + '_jpegs.h5')
    outfile = h5py.File(outfile_path, 'w')
    dset = outfile.create_dataset('jpegs', (len(files),), 
      maxshape=(len(files),), chunks=True, dtype=dt)

    for f_ind, f in enumerate(tqdm(files)):
      # read jpeg as binary and put into h5
      jpeg = open(f, 'rb')
      binary_data = jpeg.read()
      dset[f_ind] = np.fromstring(binary_data, dtype=np.uint8)
      jpeg.close()

    outfile.close()
    end_time = time.time()
    time_delta = end_time - start_time
    print('{}/{}. converting jpegs of {} to h5 done. ({} secs)'.format(
      videoname_ind+1, length, videoname, time_delta) )

def main():
  global input_jpegs_dir, output_dir, length, img_ext, dt

  args = get_args()
  
  # Path
  input_jpegs_dir = args.input_dir
  output_dir = args.target_dir
  if not os.path.isdir(output_dir): os.makedirs(output_dir)
  img_ext = args.frame_extension
  n_jobs = args.n_jobs


  dt = h5py.special_dtype(vlen=np.uint8)

  videonames = os.listdir(input_jpegs_dir)
  videonames = [f for f in videonames if not f.startswith('.')]
  videonames = sorted(videonames)

  length = len(videonames)
  Parallel(n_jobs=n_jobs)(delayed(save_h5)(videoname_ind, videoname) for videoname_ind, videoname in enumerate(tqdm(videonames)))
  # for videoname_ind, videoname in enumerate(videonames):
  #   start_time = time.time()
  #   input_frame_dir = os.path.join(input_jpegs_dir, videoname)
  #   files = os.listdir(input_frame_dir)
  #   files = [os.path.join(input_frame_dir, f) for f in files if not f.startswith('.') and f.lower().endswith(img_ext.lower())]
  #   files = sorted(files)

  #   outfile_path = os.path.join(output_dir, videoname + '_jpegs.h5')
  #   outfile = h5py.File(outfile_path, 'w')
  #   dset = outfile.create_dataset('jpegs', (len(files),), 
  #     maxshape=(len(files),), chunks=True, dtype=dt)

  #   for f_ind, f in enumerate(tqdm(files)):
  #     # read jpeg as binary and put into h5
  #     jpeg = open(f, 'rb')
  #     binary_data = jpeg.read()
  #     dset[f_ind] = np.fromstring(binary_data, dtype=np.uint8)
  #     jpeg.close()

  #   outfile.close()
  #   end_time = time.time()
  #   time_delta = end_time - start_time
  #   print('{}/{}. converting jpegs of {} to h5 done. ({} secs)'.format(
  #     videoname_ind+1, length, videoname, time_delta) )



def get_args():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-t', '--target-dir', dest='target_dir', default='./h5s', 
    help="Target directory to save hdf5 files")
  parser.add_argument('-i', '--input-dir', dest='input_dir', 
    default='./frames', help='Input directory with video frames in each directory')
  parser.add_argument('-e', '--frame-extension', dest='frame_extension', 
    default='jpg', help='Extension of frame')
  parser.add_argument('-j', '--n_jobs', dest='n_jobs', type=int, 
    default=4, help='Number of jobs')
  args = parser.parse_args()

  return args

if __name__ == "__main__":
  main()
