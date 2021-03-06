import io, os
import h5py
import numpy as np
from PIL import Image



######################################## per video ########################################
input_h5file = '/hdd/tmp_1_jpegs.h5'
input_h5 = h5py.File(input_h5file,'r')

img_ext = 'jpg'
input_jpgfolder = '/hdd/1/'
files = os.listdir(input_jpgfolder)
files = [os.path.join(input_jpgfolder, f) for f in files if not f.startswith('.') and f.lower().endswith(img_ext.lower())]
files = sorted(files)

for p in range(len(files)):
    h5_seg_imgs = Image.open(io.BytesIO(input_h5['jpegs'][p])).convert('RGB')
    h5_file = np.array(h5_seg_imgs)
    
    jpg_seg_imgs = Image.open(files[p]).convert('RGB')
    jpg_file = np.array(jpg_seg_imgs)
    
    print (np.array_equal(h5_file, jpg_file) )
######################################## per video ########################################


######################################## all in one ########################################
input_h5file_allinone = 'AllInOne.h5'
input_h5 = h5py.File(input_h5file_allinone,'r')
# len(input_h5) # 220847
# len(input_h5['1']) # 47
# input_h5['1'][0] # array([255, 216, 255, ...,   3, 255, 217], dtype=uint8)
h5_file = np.array(Image.open(io.BytesIO(input_h5['1'][0])).convert('RGB'))
jpg_file = np.array(Image.open('/raid/km/SthSth/20bn-something-something-v2-frames/1/000001.jpg').convert('RGB'))
np.array_equal(h5_file, jpg_file)

for p in range(len(input_h5['1'])):
    h5_file = np.array(Image.open(io.BytesIO(input_h5['1'][p])).convert('RGB'))
    jpg_file = np.array(Image.open('/raid/km/SthSth/20bn-something-something-v2-frames/1/{:06d}.jpg'.format(p+1)).convert('RGB'))
    np.array_equal(h5_file, jpg_file)
######################################## all in one ########################################