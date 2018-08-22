# the MemNN consensus module by km
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pdb

class RelationModuleMultiScale(torch.nn.Module):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations # -1 for reverse order

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale) # all combinations
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Linear(num_bottleneck, self.num_class),
                        )

            self.fc_fusion_scales += [fc_fusion]

        print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0] , :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))


class MemNNModule(torch.nn.Module):
    # this is the naive implementation of the n-frame relation module, as num_frames == num_frames_relation
    def __init__(self, embedding_dim, num_frames, num_class, channel, num_hop):
        super(MemNNModule, self).__init__()

        self.channel = channel # 1024
        self.embedding_dim = embedding_dim # 256
        self.num_frames = num_frames # num of segments
        self.num_class = num_class
        self.hops = num_hop

        # if embedding_dim is None: embedding_dim = channel // 2 # 128

        # self.MemoryEmbedding = nn.Linear(self.num_frames * self.embedding_dim, num_bottleneck)
        # self.QueryEmbedding = ResNet502D(spatial_pooling=True, temporal_pooling=glimpse_pooling, dropout_or_not=dropout_or_not, dropout=dropout)

        self.additional_QueryEmbedding = nn.Linear(self.channel, self.embedding_dim)
        self.KeyEmbedding1 = nn.Linear(self.channel, self.embedding_dim) # conv_pi in Non-Local
        self.ValueEmbedding1 = nn.Linear(self.channel, self.embedding_dim) # conv_g in Non-Local
        if self.hops >= 2:
            self.ValueEmbedding2 = nn.Linear(self.channel, self.embedding_dim) # conv_g in Non-Local
        if self.hops >= 3:
            self.ValueEmbedding3 = nn.Linear(self.channel, self.embedding_dim) # conv_g in Non-Local
        self.classifier = self.fc_fusion()

    def fc_fusion(self):
        nums = 1 # self.hops
        num_bottleneck = 512
        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(nums * self.embedding_dim, (nums * self.embedding_dim)//2),
                nn.ReLU(),
                nn.Linear((nums * self.embedding_dim)//2, self.num_class),
                )
        return classifier

    def forward(self, memory_input, query_input): # (BS, num_frames, 1024), (BS, num_frames, 1024)
        bs = memory_input.size()[0]
        assert (memory_input.size()[1]==self.num_frames)
        
        queries_emb = torch.mean(query_input, 1) # (BS, 1024)
        # queries_emb = self.KeyEmbedding1(queries_emb) # (BS, 256)
        queries_emb = self.additional_QueryEmbedding(queries_emb) # (BS, 256)

        accumulated_output = []
        w_u1, w_u1_plus_query = self.hop(memory_input, queries_emb, self.KeyEmbedding1, self.ValueEmbedding1)
        accumulated_output.append(w_u1)

        if self.hops >= 2:
            w_u2, w_u2_plus_query = self.hop(memory_input, w_u1_plus_query, self.ValueEmbedding1, self.ValueEmbedding2)
            # w_u2, w_u2_plus_query = self.hop(memory_input, w_u1_plus_query, self.KeyEmbedding1, self.ValueEmbedding1)
            accumulated_output.append(w_u2)

        if self.hops >= 3:
            w_u3, w_u3_plus_query = self.hop(memory_input, w_u2_plus_query, self.ValueEmbedding2, self.ValueEmbedding3)
            # w_u3, w_u3_plus_query = self.hop(memory_input, w_u2_plus_query, self.KeyEmbedding1, self.ValueEmbedding1)
            # print (w_u3.size()) # (BS, 256)
            accumulated_output.append(w_u3)

        accumulated_output = torch.stack(accumulated_output, -1)
        print (accumulated_output.size())
        asdf
        accumulated_output = accumulated_output.view(bs, -1)

        output = self.classifier(accumulated_output)
        '''
        memory_input = memory_input.view(memory_input.size(0)*self.num_frames, -1)
        memory_input = self.KeyEmbedding1(memory_input)
        memory_input = memory_input.view(bs, self.num_frames*self.embedding_dim)

        output = self.classifier(memory_input)
        '''
        # output = self.classifier(w_u)
        return output

    def hop(self, mem_emb, queries_emb, KeyEmbedding, ValueEmbedding):
        bs = mem_emb.size(0)

        query = queries_emb.unsqueeze(1) # (BS, 1, 256)

        mem_emb = mem_emb.view(mem_emb.size(0)*self.num_frames, -1) # (BS * NUM_SEG, 1024)

        key = KeyEmbedding(mem_emb) # (BS * NUM_SEG, 256)
        key = key.view(bs, self.num_frames, -1) # (BS, NUM_SEG, 256)
        key = torch.transpose(key, 1, 2) # (BS, 256, NUM_SEG)

        p = torch.bmm(query, key) # (BS, 1, NUM_SEG)
        p = F.softmax(p.view(-1, p.size()[1]*p.size()[2]), dim=1).view(-1, p.size()[1], p.size()[2]) # (BS, 1, NUM_SEG)
        # print (p)

        value = ValueEmbedding(mem_emb) # (BS * NUM_SEG, 256)
        value = value.view(bs, self.num_frames, -1) # (BS, NUM_SEG, 256)

        out = torch.bmm(p, value) # (BS, 1, 256)
        out = torch.squeeze(out, 1).contiguous() # (BS, 256)

        # return out
        return out, (out + queries_emb)

# (consensus_type, self.img_feature_dim, self.num_segments, num_class)
def return_MemNN(relation_type, img_feature_dim, num_frames, num_class, channel, num_hop):
    if relation_type == 'MemNN':
        MemNNmodel = MemNNModule(img_feature_dim, num_frames, num_class, channel, num_hop)
    else:
        raise ValueError('Unknown TRN' + relation_type)

    return MemNNmodel

if __name__ == "__main__":
    batch_size = 10
    num_frames = 5
    num_class = 174
    img_feature_dim = 512
    input_var = Variable(torch.randn(batch_size, num_frames, img_feature_dim))
    model = RelationModuleMultiScale(img_feature_dim, num_frames, num_class)
    output = model(input_var)
    print(output)
