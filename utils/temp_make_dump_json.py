import os
import json
import argparse
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str, required=True,
                    help='root directory that contains images')
parser.add_argument('--result_root', type=str, required=True,
                    help='directory of result.. We will use this root to read json and write html')
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--category_path', type=str, required=True, help='path of category.txt.. We will use this to creat idx2class dictionary')
args = parser.parse_args()

def _makedirs(path):
	if os.path.exists(path)==False:
		os.makedirs(path)

if __name__ == '__main__':

    with open(args.category_path) as f:
        lines = f.readlines()
    idx2class = [item.rstrip() for item in lines]

    # print (idx2class)
    json_file_path = os.path.join(args.result_root, 'results_epoch%d.json'%args.epoch)
    dicts = {}
    for idx, classstr in enumerate(idx2class):
    	dicts[str(idx)] = []

    each_dict = {}
    each_dict['id'] = 218
    each_dict['framenums'] = [1, 10, 20, 30, 32, 40]
    each_dict['hop_probabilities'] = [[0.8, 0.05, 0.05, 0.05, 0.05, 0], [0.1, 0.2, 0.2, 0.4, 0.05, 0.05]]
    each_dict['GT'] = 5
    each_dict['Predict'] = 6
    dicts[str(each_dict['GT'])].append(each_dict)
    each_dict = {}
    each_dict['id'] = 126
    each_dict['framenums'] = [1, 10, 20, 30, 32, 40]
    each_dict['hop_probabilities'] = [[0.8, 0.05, 0.05, 0.05, 0.05, 0], [0.1, 0.2, 0.2, 0.4, 0.05, 0.05]]
    each_dict['GT'] = 1
    each_dict['Predict'] = 8
    dicts[str(each_dict['GT'])].append(each_dict)
    each_dict = {}
    each_dict['id'] = 127
    each_dict['framenums'] = [1, 10, 20, 30, 32, 40]
    each_dict['hop_probabilities'] = [[0.8, 0.05, 0.05, 0.05, 0.05, 0], [0.1, 0.2, 0.2, 0.4, 0.05, 0.05]]
    each_dict['GT'] = 5
    each_dict['Predict'] = 8
    dicts[str(each_dict['GT'])].append(each_dict)

    # dicts.append(each_dict)

    with open(os.path.join(args.result_root, 'results_epoch%d.json'%args.epoch), 'w') as f:
        json.dump(dicts,f)

    # for each_class in data['class']:
    # 	html_file_name = os.path.join(html_root_folder,'%s.html'%(idx2class[each_class].replace(' ','_')))
