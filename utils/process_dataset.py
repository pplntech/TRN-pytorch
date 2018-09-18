# processing the raw data of the video datasets (Something-something and jester)
# generate the meta files:
#   category.txt:               the list of categories.
#   train_videofolder.txt:      each row contains [videoname num_frames classIDX]
#   val_videofolder.txt:        same as above
#
# Bolei Zhou, Dec.2 2017
#
#
import os
import pdb
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "root_dir",
    type=str,
    help='Root directory path of data')
parser.add_argument(
    "dataset",
    type=str, choices=['something','jester','moments', 'somethingv2', 'charades'],
    help='name of dataset')
parser.add_argument(
    "format",
    type=str,default='20bn-%s',
    help='format of image containing folder name')
args = parser.parse_args()

if args.dataset=='something':
    dataset_name = 'something-something-v1' # 'jester-v1'
    extension = 'csv'
elif args.dataset=='somethingv2':
    dataset_name = 'something-something-v2' # 'jester-v1'
    extension = 'json'

categories = []
if extension=='csv':
    with open(os.path.join(args.root_dir,'%s-labels.%s'% (dataset_name, extension))) as f:
        lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        categories.append(line)

elif extension=='json':
    with open(os.path.join(args.root_dir,'%s-labels.%s'% (dataset_name, extension))) as f:
        data = json.load(f)
    # print (data)
    # print (type(data))
    categories = data.keys()
    # print (categories)
    # asdf
    pass

categories = sorted(categories)

with open(os.path.join(args.root_dir,'category_%s.txt'%(dataset_name)),'w') as f:
    f.write('\n'.join(categories))

dict_categories = {}
for i, category in enumerate(categories):
    dict_categories[category] = i

files_input = ['%s-validation.%s'% (dataset_name, extension),'%s-train.%s'% (dataset_name, extension)]
files_output = ['val_videofolder_%s.txt'%(dataset_name),'train_videofolder_%s.txt'%(dataset_name)]
for (filename_input, filename_output) in zip(files_input, files_output):
    folders = []
    idx_categories = []

    if extension=='csv':
        with open(os.path.join(args.root_dir,filename_input)) as f:
            lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            items = line.split(';')
            folders.append(items[0])
            # print (dict_categories[items[1]])
            # asdf
            # idx_categories.append(os.path.join(dict_categories[items[1]]))
            idx_categories.append(dict_categories[items[1]])
    elif extension=='json':
        with open(os.path.join(args.root_dir,filename_input)) as f:
            data = json.load(f)
        for each_data in data:
            print (each_data['id'], dict_categories[each_data['template'].replace('[','').replace(']','')])
            folders.append(each_data['id'])
            idx_categories.append(dict_categories[each_data['template'].replace('[','').replace(']','')])

    output = []
    for i in range(len(folders)):
        curFolder = folders[i]
        curIDX = idx_categories[i]
        # counting the number of frames in each video folders
        dir_files = os.listdir(os.path.join(args.root_dir, args.format%dataset_name, curFolder))
        output.append('%s %d %d'%(curFolder, len(dir_files), curIDX))
        print('%d/%d'%(i, len(folders)))
    with open(os.path.join(args.root_dir,filename_output),'w') as f:
        f.write('\n'.join(output))
