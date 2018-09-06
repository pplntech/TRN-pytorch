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
import numpy as np

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

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

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
                    # print (each_data['hop_probabilities'][each_hop])
                    # print (normalize(each_data['hop_probabilities'][each_hop]))
                    # asfd
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
            html += '#tblOne th, td {text-align: center; vertical-align: middle;}'
            html += '#tblTwo th, td {text-align: center; vertical-align: middle;}'
            html += '#upper {margin:10px;}'
            html += '#bottom {margin:10px;}'
            html += '#left {margin:10px;}'
            html += '#right {margin:10px;}'
            # html += 'div {display: block;}'

            # for k, b in colortables_bg.items():
            #     html += '%s { background-color : %s; color: black;}' %(k, b)
            html += '</style>'
            # html += '<style>{width: 100%;border: 1px solid #444444;}th, td {border: 1px solid #444444;}</style>'
            html += '</head>'
            # html +='</tbody></table>'

            # titles
            html += '<body>'
            html +=  '<h2>{}</h2>'.format('ID : %s'%(each_data['id']))
            html +=  '<h2>[Groundtruth Label]&ensp;&ensp; {}</h2>'.format('%s'%(idx2class[class_int]))
            html += '\n<h2>[Predicted Label]&ensp;&ensp;&ensp;&ensp;&ensp; {}</h2>'.format(idx2class[each_data['Predict']])


            # configs
            with open (os.path.join(args.result_root,'opts.json'),'r') as f:
                configs = json.load(f)


            model_configs_to_print = ['consensus_type', 'hop', 'num_segments', 'how_to_get_query', 'hop_method', \
            'query_update_method', 'sorting', 'query_dim', 'value_dim', 'key_dim', 'no_softmax_on_p']
            model_num_column = 3
            model_num_row = len(model_configs_to_print) // model_num_column
            if len(model_configs_to_print) % model_num_column !=0:
                model_num_row += 1

            count = 0
            html += '<div id="upper">'
            html += '<span style="width:50%" id="left"><table id="tblOne" style="width:50%; float:left" border="1px solid gray">'
            html += '<caption align="top" style="text-align:left"><h3>Model Configs</h3></caption>'
            html += '\n<thead><tr>'
            for i in range(model_num_column):
                html += '<td>configs</td><td>values</td>'
            html += '</tr></thead>'

            html +='<tbody>'
            for i in range(model_num_row):
                html += '\n<tr>'
                for j in range(model_num_column):
                    # print (model_configs_to_print, count)
                    # print (configs.keys())
                    try:
                        if model_configs_to_print[count] in configs.keys():
                            value = str(configs[model_configs_to_print[count]])
                        else:
                            value = '-'
                        # print (model_configs_to_print[count])
                        html += '<td><b>{}</b></td><td>{}</td>'.format(model_configs_to_print[count], value)
                        count += 1
                    except:
                        html += '<td>{}</td><td>{}</td>'.format('-', '-')
                    # if count >= len(model_configs_to_print):
                    #     break
                html += '</tr>'
            html +='</tbody></table></span>'

            learning_configs_to_print = ['optimizer', 'lr', 'lr_steps', 'freezeBN', 'batch_size', 'no_clip']
            learning_num_column = 3
            learning_num_row = len(learning_configs_to_print) // learning_num_column
            if len(learning_configs_to_print) % learning_num_column !=0:
                learning_num_row += 1

            count = 0
            html += '<span style="width:50%" id="right"><table id="tblTwo" style="width:50%; float:left" border="1px solid gray">'
            html += '<caption align="top" style="text-align:left"><h3>Learning Configs</h3></caption>'
            html += '\n<thead><tr>'
            for i in range(learning_num_column):
                html += '<td>configs</td><td>values</td>'
            html += '</tr></thead>'

            html +='<tbody>'
            for i in range(model_num_row):
                html += '\n<tr>'
                for j in range(learning_num_column):
                    # print (learning_configs_to_print, count)
                    # print (configs.keys())
                    try:
                        if learning_configs_to_print[count] in configs.keys():
                            value = str(configs[learning_configs_to_print[count]])
                        else:
                            value = '-'
                        # print (learning_configs_to_print[count])
                        html += '<td><b>{}</b></td><td>{}</td>'.format(learning_configs_to_print[count], value)
                        count += 1
                    except:
                        html += '<td>{}</td><td>{}</td>'.format('-', '-')
                    # if count >= len(learning_configs_to_print):
                    #     break
                html += '</tr>'
            html +='</tbody></table></span>'
            html += '</div>'
            html +='<br/>'
            html +='<br/>'
            html +='<br/>'
            # html +='<div><table></table></div>'


            # visualization
            # create the first row
            html += '<div id="bottom">'
            html += '\n<div style="width:100%"><table border="1px solid gray" style="width=100%">'
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
            html +='</tbody></table></div>'
            html += '</div>'
            # html +='</p>'
            html += '</body></html>'
            # create the last row
            # print(each_data['id'], each_data['framenums'], each_data['hop_probabilities'], each_data['Predict'], each_data['GT'])
            with open(html_file_name, 'w') as f:
                f.write(html)