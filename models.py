from torch import nn

from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant

import TRNmodule
import MemNNmodule


# rmed : query_base_model, img_feature_dim
# added : key_dim, value_dim, query_dim, query_update_method, hop_method, no_softmax_on_p
class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,key_dim=256,value_dim=256,query_dim=256,query_update_method=None,
                 crop_num=1, partial_bn=True, freezeBN_Eval=False, freezeBN_Require_Grad_True=False, print_spec=True, num_hop=1, hop_method=None, 
                 num_CNNs=1, no_softmax_on_p=False, freezeBackbone=False, CustomPolicy=False, sorting=False, MultiStageLoss=False, MultiStageLoss_MLP=False, \
                 how_to_get_query='mean', only_query=False, CC=False, channel=1024, memory_dim=1, image_resolution=256,how_many_objects=1):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = key_dim  # the dimension of the CNN feature to represent each frame
        self.freezeBN_Eval = freezeBN_Eval
        self.freezeBN_Require_Grad_True = freezeBN_Require_Grad_True
        self.freezeBackbone = freezeBackbone
        self.CustomPolicy = CustomPolicy
        self.MultiStageLoss = MultiStageLoss
        self.CC = CC
        self.memory_dim = memory_dim
        self.image_resolution = image_resolution
        self.how_many_objects = how_many_objects
        # self.sorting = sorting

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec == True:
            print(("""
                Initializing TSN with base model: {}.
                TSN Configurations:
                input_modality:     {}
                num_segments:       {}
                new_length:         {}
                consensus_module:   {}
                dropout_ratio:      {}
                img_feature_dim:    {}
                """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model) # assign 'self.base_model'


        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")


        if consensus_type in ['TRN', 'TRNmultiscale']:
            # plug in the Temporal Relation Network Module
            self.consensus = TRNmodule.return_TRN(consensus_type, self.img_feature_dim, self.num_segments, num_class) # (relation_type, img_feature_dim, num_frames, num_class)
        elif consensus_type in ['MemNN']:
            self.consensus = MemNNmodule.return_MemNN(consensus_type, self.num_segments, num_class, \
                key_dim=key_dim, value_dim=value_dim, query_dim=query_dim, memory_dim=memory_dim, query_update_method=query_update_method, \
                no_softmax_on_p=no_softmax_on_p, channel=channel, num_hop=num_hop, hop_method=hop_method, num_CNNs=num_CNNs, \
                sorting=sorting, MultiStageLoss=MultiStageLoss, MultiStageLoss_MLP=MultiStageLoss_MLP, how_to_get_query=how_to_get_query, only_query=only_query, CC=CC, how_many_objects=how_many_objects)
        else: # agv or something else
            self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        try:
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features # 1024
        except:
            feature_dim = None
            if self.base_model.last_layer_name is not None:
                raise ValueError("Something Wrong. Check")

        if self.dropout == 0:
            if self.consensus_type in ['TRN','TRNmultiscale']:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, self.img_feature_dim))

            elif self.consensus_type in ['MemNN']:
                self.base_model = nn.Sequential(*list(self.base_model.children())[:-1]) # remove final FC layer
                # setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, self.img_feature_dim))

            else:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None

        else: # dropout not ZERO
            try:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            except:
                if self.base_model.last_layer_name is not None:
                    raise ValueError("Something Wrong. Check")

            if self.consensus_type in ['TRN','TRNmultiscale']:
                # create a new linear layer as the frame feature
                self.new_fc = nn.Linear(feature_dim, self.img_feature_dim)

            elif self.consensus_type in ['MemNN']:
                self.new_fc = None

            else:
                # the default consensus types in TSN
                self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.consensus_type not in ['MemNN']:
            if self.new_fc is None: # dropout 0
                normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
                constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
            else:
                normal(self.new_fc.weight, 0, std)
                constant(self.new_fc.bias, 0)

        return feature_dim # 1024 on BNInception

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            # print (self.base_model)# (conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc)

            if self.CC: # CC
                self.base_model_first_conv = self.base_model.conv1
                self.base_model_cc = nn.Conv2d(3, 64, 1, 2)
                self.base_model = nn.Sequential(*list(self.base_model.children())[1:])

                self.base_model.last_layer_name = '8'

            if self.memory_dim == 2:
                self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])
                self.base_model.last_layer_name = None

            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'BNInception':
            import model_zoo
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

            if self.CC:
                raise ValueError("CC is not supported on BNInception architecture.")

            if self.memory_dim == 2:
                raise ValueError("memory_dim larger than 1 is not supported on BNInception architecture.")
                # self.base_model_first_conv = nn.Sequential(*list(self.base_model.children())[:1])
                # self.base_model_cc = nn.Conv2d(3, 64, 1, 2)

                # self.base_model = self.base_model.children()
                # self.base_model.last_layer_name = '218'

        elif base_model == 'InceptionV3':
            import model_zoo
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 299
            self.input_mean = [104,117,128]
            self.input_std = [1]
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1+self.new_length)
        elif 'inception' in base_model:
            import model_zoo
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)

        count = 0
        if self._enable_pbn: # partial batch norm
            print("[Partial Batcn Norm] Freezing BatchNorm2D except the first one in base_model.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

        if self.freezeBN_Eval:
            print("[Freezing BN] Make ALL BatchNorm2D eval mode in base_model.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        for m in self.base_model.modules():
            if isinstance(m, nn.BatchNorm2d):

                # shutdown update in frozen mode
                if self.freezeBN_Require_Grad_True:
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True
                else:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        # print (self.freezeBackbone)
        # asdf
        if self.freezeBackbone is False and self.CustomPolicy is False:
            first_conv_weight = []
            first_conv_bias = []
            normal_weight = []
            normal_bias = []
            bn = []

            conv_cnt = 0
            bn_cnt = 0
            for name, m in self.named_modules():
                # print (name, type(m))
                # if(name=='base_model'):
                #     conv_cnt = 0
                #     bn_cnt = 0
                print (name, m)
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                    ps = list(m.parameters())
                    conv_cnt += 1
                    if conv_cnt == 1:
                        first_conv_weight.append(ps[0])
                        if len(ps) == 2:
                            first_conv_bias.append(ps[1])
                    else:
                        normal_weight.append(ps[0])
                        if len(ps) == 2:
                            normal_bias.append(ps[1])
                elif isinstance(m, torch.nn.Linear):
                    ps = list(m.parameters())
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
                elif isinstance(m, torch.nn.modules.rnn.LSTM):
                    # print (m)
                    # print (list(m.parameters()))
                    # print (len(list(m.parameters())))
                    ps = list(m.parameters())
                    normal_weight.append(ps[0])
                    normal_weight.append(ps[1])
                    if len(ps) == 4:
                        normal_bias.append(ps[2])
                        normal_bias.append(ps[3])

                elif isinstance(m, torch.nn.BatchNorm1d):
                    bn.extend(list(m.parameters()))
                elif isinstance(m, torch.nn.BatchNorm2d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1 or (self.freezeBN_Require_Grad_True is True):
                        bn.extend(list(m.parameters()))
                elif len(m._modules) == 0:
                    if len(list(m.parameters())) > 0:
                        raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))


            # if self.consensus_type in ['MemNN']:
            return [
                {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
                 'name': "first_conv_weight"},
                {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
                 'name': "first_conv_bias"},
                {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
                 'name': "normal_weight"},
                {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
                 'name': "normal_bias"},
                {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
                 'name': "BN scale/shift"},
            ]

        elif self.freezeBackbone is False and self.CustomPolicy:
            backbone_weight = []
            backbone_bias = []
            consensus_weight = []
            consensus_bias = []
            bn = []
            bn_cnt = 0

            for name, m in self.named_modules():
                # print (name, type(m))
                print (name, m)
                if('consensus' in name):
                    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                        ps = list(m.parameters())
                        consensus_weight.append(ps[0])
                        if len(ps) == 2: consensus_bias.append(ps[1])
                    elif isinstance(m, torch.nn.Linear):
                        ps = list(m.parameters())
                        consensus_weight.append(ps[0])
                        if len(ps) == 2: consensus_bias.append(ps[1])
                    elif isinstance(m, torch.nn.modules.rnn.LSTM):
                        ps = list(m.parameters())
                        consensus_weight.append(ps[0])
                        consensus_weight.append(ps[1])
                        if len(ps) == 4:
                            consensus_bias.append(ps[2])
                            consensus_bias.append(ps[3])

                    elif isinstance(m, torch.nn.BatchNorm1d):
                        bn.extend(list(m.parameters()))
                    elif isinstance(m, torch.nn.BatchNorm2d):
                        bn_cnt += 1
                        # later BN's are frozen
                        # if not self._enable_pbn or bn_cnt == 1 and self.freezeBN is False:
                        if not self._enable_pbn or bn_cnt == 1 or (self.freezeBN_Require_Grad_True is True):
                            bn.extend(list(m.parameters()))
                    elif len(m._modules) == 0:
                        if len(list(m.parameters())) > 0:
                            raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

                elif('base_model' in name):
                    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                        ps = list(m.parameters())
                        backbone_weight.append(ps[0])
                        if len(ps) == 2: backbone_bias.append(ps[1])
                    elif isinstance(m, torch.nn.Linear):
                        ps = list(m.parameters())
                        backbone_weight.append(ps[0])
                        if len(ps) == 2: backbone_bias.append(ps[1])
                    elif isinstance(m, torch.nn.modules.rnn.LSTM):
                        ps = list(m.parameters())
                        backbone_weight.append(ps[0])
                        backbone_weight.append(ps[1])
                        if len(ps) == 4:
                            backbone_bias.append(ps[2])
                            backbone_bias.append(ps[3])

                    elif isinstance(m, torch.nn.BatchNorm1d):
                        bn.extend(list(m.parameters()))
                    elif isinstance(m, torch.nn.BatchNorm2d):
                        bn_cnt += 1
                        # later BN's are frozen
                        # if not self._enable_pbn or bn_cnt == 1 and self.freezeBN is False:
                        if not self._enable_pbn or bn_cnt == 1 or (self.freezeBN_Require_Grad_True is True):
                            bn.extend(list(m.parameters()))
                    elif len(m._modules) == 0:
                        if len(list(m.parameters())) > 0:
                            raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
                # asdf

            backbone_lr_mul = 0.1
            return [
                {'params': backbone_weight, 'lr_mult': 5 if self.modality == 'Flow' else backbone_lr_mul, 'decay_mult': 1,
                 'name': "backbone_weight"},
                {'params': backbone_bias, 'lr_mult': 10 if self.modality == 'Flow' else backbone_lr_mul*2, 'decay_mult': 0,
                 'name': "backbone_bias"},
                {'params': consensus_weight, 'lr_mult': 1, 'decay_mult': 1,
                 'name': "consensus_weight"},
                {'params': consensus_bias, 'lr_mult': 2, 'decay_mult': 0,
                 'name': "consensus_bias"},
                {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
                 'name': "BN scale/shift"},
            ]

        elif self.freezeBackbone:
            normal_weight = []
            normal_bias = []
            bn = []
            # normal_weight_name = []
            # normal_bias_name = []
            # bn_name = []

            for name, m in self.named_modules():
                print (name, type(m))
                if('consensus' in name) or (isinstance(m, torch.nn.Linear)):
                    print ('--------------------------------------------', name, m, '--------------------------------------------')
                    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                        ps = list(m.parameters())
                        normal_weight.append(ps[0])
                        # normal_weight_name.append(name)
                        if len(ps) == 2:
                            normal_bias.append(ps[1])
                            # normal_bias_name.append(name)
                    elif isinstance(m, torch.nn.Linear):
                        ps = list(m.parameters())
                        normal_weight.append(ps[0])
                        # normal_weight_name.append(name)
                        if len(ps) == 2:
                            normal_bias.append(ps[1])
                            # normal_bias_name.append(name)
                    elif isinstance(m, torch.nn.modules.rnn.LSTM):
                        ps = list(m.parameters())
                        normal_weight.append(ps[0])
                        normal_weight.append(ps[1])
                        if len(ps) == 4:
                            normal_bias.append(ps[2])
                            normal_bias.append(ps[3])

                    elif isinstance(m, torch.nn.BatchNorm1d):
                        bn.extend(list(m.parameters()))
                        # bn_name.extend(name)
                    elif isinstance(m, torch.nn.BatchNorm2d):
                        # if not self._enable_pbn and self.freezeBN is False:
                        if not self._enable_pbn and self.freezeBN_Require_Grad_True is True:
                            bn.extend(list(m.parameters()))
                            # bn_name.extend(name)
                    elif len(m._modules) == 0:
                        if len(list(m.parameters())) > 0:
                            raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
            # print ('------------------------')
            # print (normal_weight_name)
            # print (normal_bias_name)
            # print (bn_name)
            # asdf

            return [
                {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
                 'name': "normal_weight"},
                {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
                 'name': "normal_bias"},
                {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
                 'name': "BN scale/shift"},
            ]

    def Generate_grid(self, bs, temporal_length, size):
        oneset = []
        for t in range(temporal_length):
            tmap = torch.zeros(size) + t
            tmap = tmap / float(temporal_length)
            y = 0
            hmap = torch.arange(start=y, end=y+size[0]).view(-1,1).repeat(1,size[1]).float()
            hmap = hmap / 224.
            x = 0
            wmap = torch.arange(start=x, end=x+size[1]).view(1,-1).repeat(size[0], 1).float()
            wmap = wmap / 224.
            # print (torch.stack([tmap, hmap, wmap], dim=0))
            oneset.append(torch.stack([tmap, hmap, wmap], dim=0)) # 3, 224, 224
            # grid = torch.stack([tmap, hmap, wmap], dim=0)
        oneset = torch.stack(oneset,dim=0) # 8, 3, 224, 224
        allgrids = oneset.repeat(bs, 1, 1, 1) # bs*num_seg, 3, 224, 224
        # print (allgrids.view((bs, -1, 3) + size)) # bs, num_seg, 3, 224, 224
        # asdf

        return torch.autograd.Variable(allgrids.cuda())

    def forward(self, input, criterion, phase='eval', target=None, eval=False):
        # print (input.size()) # [72, 6, 224, 224] # [BS, num_seg * num_channel, h, w]

        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        # new_length is 1 when RGB, otherwise 5

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        # print (input.view((-1, sample_len) + input.size()[-2:]).size()) # (BS * num_seg, num_channel, h, w)
        if self.CC == False:
            base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:])) # BS * num_seg, num_channel, h, w
            # print (self.base_model)
            # print ('input size : ', input.view((-1, sample_len) + input.size()[-2:]).size()) # [120, 3, 224, 224]
            # print ('output size : ', base_out.size()) # [120, 2048]
            if self.consensus_type in ['MemNN'] and self.memory_dim==1:
                base_out = base_out.unsqueeze(-1)
                base_out = base_out.unsqueeze(-1)
        elif self.CC:
            # print (input.size()) # [bs, channel(3)*num_seg(8), 224, 224]
            # print (input.view((-1, sample_len) + input.size()[-2:]).size()) # [bs*num_seg(8), 3, 224, 224]
            first_conv_out = self.base_model_first_conv(input.view((-1, sample_len) + input.size()[-2:])) # 240, 64, 112, 112

            cc_grid = self.Generate_grid(input.size()[0], self.num_segments, (input.size()[2], input.size()[3])) # bs*num_seg(8), 3, 224, 224
            cc_out = self.base_model_cc(cc_grid) # 240, 64, 112, 112

            summation = first_conv_out + cc_out # 240, 64, 112, 112)

            base_out = self.base_model(summation) # 120(bs*num_seg), last_fc_dim, 1, 1 (1024 for BNInception, 2048 for ResNet50) # (bs*num_seg, last_feature_dim, 7, 7)
            # if self.consensus_type not in ['MemNN']:
            # base_out = base_out.squeeze(2)
            # base_out = base_out.squeeze(2)

            # print (self.base_model)
            # print ('input size : ', summation.size()) # [120, 64, 112, 112]
            # print ('output size : ', base_out.size()) # [120, 2048, 1, 1]


        if self.dropout > 0 and self.new_fc is not None:
            base_out = self.new_fc(base_out) # img_feature_dim
        # print (base_out.size()) # (BS * num_seg, img_feature_dim_OR_final_class_num)
        # base_out is class_logit when TSN, otherwise img_feature_dim when TRN

        # print (self.before_softmax) # True
        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            # ^^ bf : (BS * NUM_SEG, img_feature_dim_OR_final_class_num), (BS * NUM_SEG, img_feature_dim_OR_final_class_num, H, W)
            # ^^ af : (BS, NUM_SEG, img_feature_dim_OR_final_class_num), (BS, NUM_SEG, img_feature_dim_OR_final_class_num, H, W)


        # outputs : list of outputs of each prediction_branch (logits)
        if self.consensus_type in ['MemNN']:
            if eval:
                if self.how_many_objects == 2:
                    outputs, attentions, attentions_2 = self.consensus(base_out, eval=eval) # output : logit
                else:
                    outputs, attentions = self.consensus(base_out, eval=eval) # output : logit
            else:
                outputs = self.consensus(base_out, eval=eval) # output : logit
        else:
            outputs = [self.consensus(base_out).squeeze(1)]


        # Calculate Loss (Avg MultiStage Loss)
        total_loss = None
        total_output = None
        for idx, output in enumerate(outputs): # outputs : list of logits
            if total_loss is None:
                # output.size() : 
                # target.size() : 
                total_loss = criterion(output, target)
                total_output = nn.functional.softmax(output,1) # BS x 174
            else:
                total_loss += criterion(output, target)
                total_output += nn.functional.softmax(output,1) # BS x 174
                # print (idx, criterion(output, target))

        total_output = total_output / len(outputs)
        total_loss = total_loss / len(outputs)

        # total_loss = total_loss.mean()
        if eval and self.consensus_type in ['MemNN']:
            if self.how_many_objects == 2:
                return total_output, attentions, attentions_2, total_loss
            return total_output, attentions, total_loss
        else:
            return total_output, total_loss

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:] # change number of channels
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            scales = [1, .875, .75, .66]
            if self.image_resolution==320: scales = [1, .9, .85, .8]
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, scales, fix_crop=False)])
                # , GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75], fix_crop=False)])
                # ,GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75], fix_crop=False)])
                # ,GroupRandomHorizontalFlip(is_flow=False)])