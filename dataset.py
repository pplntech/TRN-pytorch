import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import h5py
import io

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file, file_type,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.file_type = file_type
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

        # if self.file_type == 'h5':
        #     input_h5file = os.path.join(self.root_path, 'AllInOne.h5')
        #     self.input_h5 = h5py.File(input_h5file, 'r')

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            try:
                idx_skip = 1 + (idx-1)*5
                flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip))).convert('RGB')
            except Exception:
                print('error loading flow file:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip)))
                flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
            # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
            flow_x, flow_y, _ = flow.split()
            x_img = flow_x.convert('L')
            y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        # usualy it is [video_id, num_frames, class_idx]
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1])>=3]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d'%(len(self.video_list)))

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        while not os.path.exists(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))) and self.file_type != 'h5':
            print(os.path.join(self.root_path, record.path, self.image_tmpl.format(1)))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        if self.file_type == 'h5':
            input_h5file = os.path.join(self.root_path, 'AllInOne.h5')
            input_h5 = h5py.File(input_h5file, 'r')

        # print (indices)
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                if self.file_type == 'h5':
                    # n5 save data from idx 0 !
                    # so, [0] stores information of 000001.jpg
                    # print (record.path, type(record.path)) # 190641 <class 'str'>
                    # print (np.array(Image.open(io.BytesIO(input_h5[str(record.path)][p-1])).convert('RGB')))
                    seg_imgs = [Image.open(io.BytesIO(input_h5[str(record.path)][p-1])).convert('RGB')]
                    # asdf
                else:
                    seg_imgs = self._load_image(record.path, p)

                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        if self.file_type == 'h5':
            input_h5.close()


        process_data = self.transform(images)
        # print (indices, record.num_frames, record.path, process_data)
        return process_data, record.label, record.path, indices

    def __len__(self):
        return len(self.video_list)
