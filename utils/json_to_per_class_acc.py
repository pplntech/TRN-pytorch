import os
import json
import argparse
import numpy as np
import operator
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str, required=True,
                    help='root directory that contains images')
parser.add_argument('--result_root', type=str, required=True,
                    help='directory of result.. We will use this root to read json and write html')
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--category_path', type=str, required=True, help='path of category.txt.. We will use this to creat idx2class dictionary')
parser.add_argument('--confusion_topk', type=int, required=True)
args = parser.parse_args()

def _makedirs(path):
    if os.path.exists(path)==False:
        os.makedirs(path)

def _takeid(elem):
    return elem['id']

def _take2nd(elem):
    return elem[1]

def save(txt_filename, gt, pred, idx2class):
    n_classes = len(idx2class)
    ####################### per class accuracy #######################
    true, total = [0 for i in range(n_classes)], [0 for i in range(n_classes)]
    # TP, FN, FP, TN = [0 for i in range(n_classes)], [0 for i in range(n_classes)], [0 for i in range(n_classes)], [0 for i in range(n_classes)]
    for each_gt, each_pred in zip(gt, pred):
        # print (each_gt, each_pred)
        total[each_gt] += 1
        if(each_gt==each_pred):
            true[each_gt] += 1
        # else:
        #     FN[each_gt] += 1
        #     FP[each_pred] += 1
    print ('Accuracy')
    dic_acc = {}
    for each_class_int in range(n_classes):
        dic_acc[each_class_int] = float(true[each_class_int])/total[each_class_int]
    dic_acc = sorted( dic_acc.items(), key=operator.itemgetter(1), reverse = True)

    # dic_acc.sort(key=_take2nd)
    with open(txt_filename, 'a+') as f:
        f.write('%-100s / NUM_OF_DATA / ACC\n' % 'CLASS')
        for each_class_int, value in dic_acc:
            f.write('%-100s / %-4d / %f\n' %
             (idx2class[each_class_int], total[each_class_int], value))

    ####################### Confusion Matrix #######################
    cm = confusion_matrix(gt, pred, labels=list(range(n_classes)))

    # VV make diagonal elements zero VV
    cm[range(n_classes), range(n_classes)] = 0

    non_zero = np.count_nonzero(cm)
    topk = min(non_zero, args.confusion_topk)
    
    # indices : indices of high confusion value (tuple)
    indices = np.unravel_index(\
    np.argsort(cm.ravel())[-topk:], \
    cm.shape)
    indices = np.array(indices)
    indices = np.transpose(indices)

    with open(txt_filename, 'a+') as f:
        f.write('\n\n\nConfusion Matrix\n')
        f.write('Nth : (GT, PRED : HOW_MANY/TOTALNUM)\n')
        # f.write('The number of validation dataset : %d\n', len(self.gt))
        for number, idx in enumerate(indices[::-1]):
            gt_label = idx[0]
            pred_label = idx[1]
            f.write('%-2dth : (%-100s, %-100s : %-5d/%-5d)\n' %
             (number+1, idx2class[gt_label], idx2class[pred_label],\
             cm[gt_label][pred_label], len(gt)))
    ####################### Confusion Matrix #######################

if __name__ == '__main__':
    # load class name
    with open(args.category_path) as f:
        lines = f.readlines()
    idx2class = [item.rstrip() for item in lines]

    # load json file
    with open(os.path.join(args.result_root, 'results_epoch%d.json'%args.epoch)) as f:
        data = json.load(f)

    # make GT and PRED list
    gt = []
    pred = []
    for each_class in sorted(data.iterkeys()):
        class_int = int(each_class)
        class_data = data[each_class]
        class_data.sort(key=_takeid)

        # per each data, create html files
        for idx, each_data in enumerate(class_data):
            gt.append(class_int)
            pred.append(each_data['Predict'])

    acc_txt_filename = 'results_epoch%d_Acc.txt'%args.epoch
    acc_txt_filename = os.path.join(args.result_root, acc_txt_filename)
    save(acc_txt_filename, gt, pred, idx2class)
