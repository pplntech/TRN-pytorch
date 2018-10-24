import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('dataset', type=str, choices=['something','jester','moments', 'somethingv2', 'charades'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow'])
parser.add_argument('--train_list', type=str,default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="", required=True)
parser.add_argument('--result_path', type=str, default="", required=True)
parser.add_argument('--file_type', type=str, default="jpg", choices=['jpg', 'h5'], required=True)
parser.add_argument('--store_name', type=str, default="")

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--channel', default=1024, type=int, help="dimension of 2D architecture's last fc")    ##### important args #####
# parser.add_argument('--arch_query', type=str, default="BNInception")
parser.add_argument('--num_segments', type=int, default=3)    ##### important args #####
parser.add_argument('--consensus_type', type=str, default='avg',\
     choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn', 'TRN', 'TRNmultiscale', 'MemNN'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', '--do', default=0.8, type=float, metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll", choices=['nll'])
parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")
parser.add_argument('--key_dim', type=int, help="the feature dimension of memory key")
parser.add_argument('--value_dim', type=int, help="the feature dimension of memory value")
parser.add_argument('--query_dim', type=int, help="the feature dimension of query")
parser.add_argument('--query_update_method', type=str, default="concat", choices=['sum', 'concat'])  ##### newly added #####
parser.add_argument('--memory_dim', default=1, type=int, help="dimension of feature stored on memory")    ##### important args #####
parser.add_argument('--hop', default=5, type=int, help="number of hops")    ##### important args #####
parser.add_argument('--hop_method', type=str, default="iterative", choices=['parallel', 'iterative'])    ##### newly added #####
parser.add_argument('--num_CNNs', default=1, type=int, help="number of CNNs to use")
parser.add_argument('--no_softmax_on_p', action='store_true', help='If true, not use softmax layer calculating p.')    ##### newly added #####
parser.set_defaults(no_softmax_on_p=False)    ##### newly added #####
parser.add_argument('--sorting', action='store_true', help='If true, sort multihop results')    ##### newly added #####
parser.set_defaults(sorting=False)    ##### newly added #####
parser.add_argument('--freezeBN_Eval', action='store_true', help='If true, freezeBN_Eval')    ##### newly added #####
parser.set_defaults(freezeBN_Eval=False)    ##### newly added #####
parser.add_argument('--freezeBN_Require_Grad_True', action='store_true', help='If true, freezeBN but require_grad True for weight and bias')    ##### newly added #####
parser.set_defaults(freezeBN_Require_Grad_True=False)    ##### newly added #####
parser.add_argument('--freezeBackbone', action='store_true', help='If true, freezeBackbone')    ##### newly added #####
parser.set_defaults(freezeBackbone=False)    ##### newly added #####
parser.add_argument('--CustomPolicy', action='store_true', help='If true, customize freezing')    ##### newly added #####
parser.set_defaults(CustomPolicy=False)    ##### newly added #####
parser.add_argument('--CC', action='store_true', help='If true, add CC')    ##### newly added #####
parser.set_defaults(CC=False)    ##### newly added #####
parser.add_argument('--MultiStageLoss', action='store_true', help='If true, use additional loss every hop')    ##### newly added #####
parser.set_defaults(MultiStageLoss=False)    ##### newly added #####
parser.add_argument('--MultiStageLoss_MLP', action='store_true', help='If true, use MLP for lstm estimation')    ##### newly added #####
parser.set_defaults(MultiStageLoss_MLP=False)    ##### newly added #####
parser.add_argument('--how_to_get_query', type=str, default="mean", choices=['mean', 'lstm'])  ##### newly added #####
parser.add_argument('--only_query', action='store_true', help='If true, use only query to predict the final')    ##### newly added #####
parser.set_defaults(only_query=False)    ##### newly added #####
parser.add_argument('--MoreAug_Rotation', action='store_true', help='If true, apply rotation as data augmentation')    ##### newly added #####
parser.set_defaults(MoreAug_Rotation=False)    ##### newly added #####
parser.add_argument('--MoreAug_ColorJitter', action='store_true', help='If true, apply color jittering as data augmentation')    ##### newly added #####
parser.set_defaults(MoreAug_ColorJitter=False)    ##### newly added #####
parser.add_argument('--image_resolution', default=256, type=int, help="resolution of image")    ##### newly added #####
parser.add_argument('--how_many_objects', default=1, type=int, help="decide how many heads each hop has to have")    ##### newly added #####
parser.add_argument('--Each_Embedding', action='store_true', help='Use per-hop key/value embedding on iterative version')    ##### newly added #####
parser.set_defaults(Each_Embedding=False)    ##### newly added #####
parser.add_argument('--Curriculum', action='store_true', help='If true, do curriculum learning')    ##### newly added #####
parser.set_defaults(Curriculum=False)    ##### newly added #####
parser.add_argument('--Curriculum_dim', default=512, type=int, help="intermediate dim of curriculum learning")    ##### newly added #####

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'], type=str)
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[50, 100, 150, 200], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_clip', action='store_true', help='If true, not clipping gradients.')    ##### newly added #####
parser.set_defaults(no_clip=False)    ##### newly added #####
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--break_while_train', action='store_true', help='If true, break train after 1st iteration.')    ##### newly added #####
parser.set_defaults(break_while_train=False)    ##### newly added #####
parser.add_argument('--break_while_val', action='store_true', help='If true, break val after 1st iteration.')    ##### newly added #####
parser.set_defaults(break_while_val=False)    ##### newly added #####


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',help='evaluate model on validation set')
parser.add_argument('--evaluation_epoch', default=0, type=int, metavar='N',help='evaluation epoch')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='model')
parser.add_argument('--root_output',type=str, default='output')
parser.add_argument('--no_cudnn', action='store_true', help='If true, not using cudnn_benchmark.')    ##### newly added #####
parser.set_defaults(no_cudnn=False)    ##### newly added #####
# parser.add_argument('--test', action='store_true', help='If true, test is performed.')
# parser.set_defaults(test=False)