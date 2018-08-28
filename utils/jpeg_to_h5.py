import os
import h5py
import numpy as np
import time
import argparse
from tqdm import tqdm

def main():
  args = get_args()
  
  # Path
  input_jpegs_dir = args.input_dir
  output_dir = args.target_dir
  if not os.path.isdir(output_dir): os.makedirs(output_dir)

  dt = h5py.special_dtype(vlen=np.uint8)

  videonames = os.listdir(input_jpegs_dir)
  videonames = [f in videonames if not f.startswith('.')]
  print (len(videonames))



def get_args():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-t', '--target-dir', dest='target_dir', default='./h5s', 
    help="Target directory to save hdf5 files")
  parser.add_argument('-i', '--input-dir', dest='input_dir', 
    default='./frames', help='Input directory with video frames in each directory')
  args = parser.parse_args()

  return args

if __name__ == "__main__":
  main()
