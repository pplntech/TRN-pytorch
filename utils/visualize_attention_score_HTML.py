# Visualization of attention score and corresponding frames
'''
    run on Validation Set

    hops : multiple hops is possible
    which to show : frame images + hop possibilities + GT class + predicted class


    When store the file,
        gather the same class
        result_root
            -class1_name
                -1.png
                -143.png
                -...
            -class2_name

    Function Arguments
        0. root_dir
        1. img
        2. hop_probabilities
        3. # of frames
        4. # of hops
        5. GT_class (idx)
        6. predicted class (idx)
        7. idx2class dictionary
'''
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
parser.add_argument('--prefix', type=str, default='{:05d}.jpg', required=True, help='path of category.txt.. We will use this to creat idx2class dictionary')
# ^ refer to datasets_video.py # {:06d}.jpg for sth-v2
args = parser.parse_args()


def _makedirs(path):
    if os.path.exists(path)==False:
        os.makedirs(path)

def _takeid(elem):
    return elem['id']

if __name__ == '__main__':
    html_root_folder = os.path.join(args.result_root,'html_epoch%d'%args.epoch)
    _makedirs(html_root_folder)

    with open(args.category_path) as f:
        lines = f.readlines()
    idx2class = [item.rstrip() for item in lines]

    with open(os.path.join(args.result_root, 'results_epoch%d.json'%args.epoch)) as f:
        data = json.load(f)

    colors_bg = ['#DCDCDC', '#D3D3D3', '#A9A9A9', '#696969', '#000000']
    colors_txt = ['#000000', '#000000', '#000000', '#FFFFFF', '#FFFFFF']
    # per each class, create folders
    for each_class in sorted(data.iterkeys()):
        class_int = int(each_class)
        class_data = data[each_class]
        class_data.sort(key=_takeid)
        html_folder = os.path.join(html_root_folder,'%s'%(idx2class[class_int].replace(' ','_').replace(',','')))
        _makedirs(html_folder)

        # per each data, create html files
        for idx, each_data in enumerate(class_data):
            html_file_name = os.path.join(html_folder,'%d.html' % each_data['id'])
            num_of_frames = len(each_data['framenums'])
            num_of_hops = len(each_data['hop_probabilities'])

            # create color tables // keys : 'hop%d_frame%d'%(each_hop,frame_idx) // values : colors
            colortables_bg = {}
            colortables_txt = {}
            for each_frame in range(num_of_frames):
                for each_hop in range(num_of_hops):
                    prob = each_data['hop_probabilities'][each_hop][each_frame]
                    # print (prob, int(prob*len(colors_bg)))
                    # print (min(int(prob*len(colors_bg)),len(colors_bg)-1))
                    colortables_bg['hop%d_frame%d'%(each_hop,each_frame)] = colors_bg[min(int(prob*len(colors_bg)),len(colors_bg)-1)]
                    colortables_txt['hop%d_frame%d'%(each_hop,each_frame)] = colors_txt[min(int(prob*len(colors_txt)),len(colors_txt)-1)]
                    if prob>0.9:
                        print (each_data['id'])

            # styles
            html = '<html><head>'
            html += '<style>'
            html += 'th, td {text-align: right;}'
            # for k, b in colortables_bg.items():
            #     html += '%s { background-color : %s; color: black;}' %(k, b)
            html += '</style>'
            # html += '<style>{width: 100%;border: 1px solid #444444;}th, td {border: 1px solid #444444;}</style>'
            html += '</head>'

            # titles
            html += '<body><h1>{}</h1>'.format('%s [ID : %s]'%(idx2class[class_int], each_data['id']))
            html += '\n<h2>Predicted : {}</h2>'.format(idx2class[each_data['Predict']])


            # create the first row
            html += '<table border="1px solid gray" style="width=100%">'
            html += '\n<thead><tr><td><b>images</b></td>'
            for each_hop in range(num_of_hops):
                html += '<td><b>{}</b></td>'.format('hop%d'%(each_hop+1))
            html +='</tr></thead><tbody>'

            for frame_idx, each_frame in enumerate(each_data['framenums']):
                html += '\n<tr><td><img src="{}"></td>'.format(os.path.join(args.img_root,str(each_data['id']),args.prefix.format(each_frame)))

                for each_hop in range(num_of_hops):
                    # html += '<td class="{}">{}</td>'.format('hop%d_frame%d'%(each_hop,frame_idx), each_data['hop_probabilities'][each_hop][frame_idx])
                    # html += '<td bgcolor="{}" color="{}">{}</td>'.format(colortables_bg['hop%d_frame%d'%(each_hop,frame_idx)], colortables_txt['hop%d_frame%d'%(each_hop,frame_idx)],\
                    html += '<td style="background-color:{};color:{};">{}</td>'.format(colortables_bg['hop%d_frame%d'%(each_hop,frame_idx)], colortables_txt['hop%d_frame%d'%(each_hop,frame_idx)], each_data['hop_probabilities'][each_hop][frame_idx])
                html +='</tr>'
            html +='</tbody></table>'
            # create the last row
            # print(each_data['id'], each_data['framenums'], each_data['hop_probabilities'], each_data['Predict'], each_data['GT'])
            with open(html_file_name, 'w') as f:
                f.write(html)