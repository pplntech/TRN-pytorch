from torch import nn

from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant

import TRNmodule
import MemNNmodule

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', query_base_model='resnet18', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, num_hop=1, num_CNNs=2):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
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
        if consensus_type in ['MemNN']:
            self._prepare_query_base_model(query_base_model) # assign 'self.base_model'


        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            if consensus_type in ['MemNN']:
                asdf
                # todo : construct query specific function
                self.query_base_model = self._construct_flow_model(self.query_base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            if consensus_type in ['MemNN']:
                asdf
                # todo : construct query specific function
                self.query_base_model = self._construct_diff_model(self.query_base_model)
            print("Done. RGBDiff model ready.")


        if consensus_type in ['TRN', 'TRNmultiscale']:
            # plug in the Temporal Relation Network Module
            self.consensus = TRNmodule.return_TRN(consensus_type, self.img_feature_dim, self.num_segments, num_class)
            # (relation_type, img_feature_dim, num_frames, num_class)
        elif consensus_type in ['MemNN']:
            # plug in the Temporal Relation Network Module
            self.consensus = MemNNmodule.return_MemNN(consensus_type, self.img_feature_dim, self.num_segments, num_class, channel=1024, num_hop=num_hop, num_CNNs=num_CNNs)
        else:
            self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features # 1024
        # print (self.base_model)
        if self.dropout == 0:
            if self.consensus_type in ['TRN','TRNmultiscale']:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, self.img_feature_dim))

            elif self.consensus_type in ['MemNN']:
                self.base_model = nn.Sequential(*list(self.base_model.children())[:-1]) # feature_dim
                print ('2nd query related function')
                self.query_base_model = nn.Sequential(*list(self.query_base_model.children())[:-1]) # feature_dim
                # setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, self.img_feature_dim))

            else:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
            self.query_new_fc = None
        else: # dropout not ZERO
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))

            if self.consensus_type in ['TRN','TRNmultiscale']:
                # create a new linear layer as the frame feature
                self.new_fc = nn.Linear(feature_dim, self.img_feature_dim)

            elif self.consensus_type in ['MemNN']:
                print ('2nd query related function')
                setattr(self.query_base_model, self.query_base_model.last_layer_name, nn.Dropout(p=self.dropout))
                self.new_fc = None
                self.query_new_fc = None

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
        # print (self.base_model)
        # asdf
        return feature_dim

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
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

    def _prepare_query_base_model(self, base_model):
        print ('1st query related function')
        if 'resnet' in base_model or 'vgg' in base_model:
            self.query_base_model = getattr(torchvision.models, base_model)(True)
            self.query_base_model.last_layer_name = 'fc'
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
            self.query_base_model = getattr(model_zoo, base_model)()
            self.query_base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
        elif base_model == 'InceptionV3':
            import model_zoo
            self.query_base_model = getattr(model_zoo, base_model)()
            self.query_base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 299
            self.input_mean = [104,117,128]
            self.input_std = [1]
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1+self.new_length)

        elif 'inception' in base_model:
            import model_zoo
            self.query_base_model = getattr(model_zoo, base_model)()
            self.query_base_model.last_layer_name = 'classif'
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
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one in base_model.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

            if self.consensus_type in ['MemNN']:
                print("Freezing BatchNorm2D except the first one in query_model.")
                for m in self.query_base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= (2 if self._enable_pbn else 1):
                            m.eval()

                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for name, m in self.named_modules():
            # print (name, type(m))
            if(name=='base_model'):
                conv_cnt = 0
                bn_cnt = 0
            if(name=='query_base_model'):
                conv_cnt = 0
                bn_cnt = 0
            # print (name, m)
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
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

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
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

    def forward(self, input, eval=False):
        # print (input.size()) # [72, 6, 224, 224] # [BS, num_seg * num_channel, h, w]

        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        # new_length is 1 when RGB, otherwise 5

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        # print (input.view((-1, sample_len) + input.size()[-2:]).size()) # (BS * num_seg, num_channel, h, w)
        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        # print (base_out.size()) # (BS * num_seg, 1024) # 1024 is number of channels
        if self.consensus_type in ['MemNN']:
            query_out = self.query_base_model(input.view((-1, sample_len) + input.size()[-2:]))


        if self.dropout > 0 and self.consensus_type!='MemNN':
            base_out = self.new_fc(base_out) # img_feature_dim
        # print (base_out.size()) # (BS * num_seg, img_feature_dim_OR_final_class_num)
        # base_out is class_logit when TSN, otherwise img_feature_dim when TRN

        # print (self.before_softmax) # True
        if not self.before_softmax:
            base_out = self.softmax(base_out)
            if self.consensus_type in ['MemNN']:
                query_out = self.softmax(query_out)
        # print (base_out.size()) # (BS * num_seg, img_feature_dim_OR_final_class_num)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            if self.consensus_type in ['MemNN']:
                query_out = query_out.view((-1, self.num_segments) + query_out.size()[1:])
        # print (base_out.size()) # (BS, NUM_SEG, img_feature_dim_OR_final_class_num)
        if self.consensus_type in ['MemNN']:
            if eval:
                output, attentions = self.consensus(base_out, query_out, eval=eval)
            else:
                output = self.consensus(base_out, query_out, eval=eval)
        else:
            output = self.consensus(base_out)


        if eval:
            return output.squeeze(1), attentions
        else:
            return output.squeeze(1)

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
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
