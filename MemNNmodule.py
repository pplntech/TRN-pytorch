# the MemNN consensus module by km
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pdb

# this is the naive implementation of the n-frame relation module, as num_frames == num_frames_relation
class MemNNModule(torch.nn.Module):
    def __init__(self, num_frames, num_class, channel, \
        key_dim, value_dim, query_dim, query_update_method, no_softmax_on_p, \
        num_hop, hop_method, num_CNNs, sorting):
        super(MemNNModule, self).__init__()

        self.num_frames = num_frames # num of segments
        self.num_class = num_class
        self.channel = channel # 1024
        self.sorting = sorting

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_dim = query_dim
        assert (self.key_dim==self.query_dim)

        self.query_update_method = query_update_method
        if query_update_method=='sum':
            assert (channel==self.value_dim)
        self.no_softmax_on_p = no_softmax_on_p

        self.hops = num_hop
        self.hop_method = hop_method
        self.num_CNNs = num_CNNs


        # define layers
        self.query_embedding1 = nn.Linear(self.channel, self.query_dim)
        # query_embedding1
        # self.query_embedding1 = nn.Linear(self.channel, self.query_dim)

        if self.hops >= 2:
            if self.hop_method=='iterative' and self.query_update_method=='concat': self.query_embedding2 = nn.Linear(self.channel + self.value_dim, self.query_dim)
            if self.hop_method=='iterative' and self.query_update_method=='sum': self.query_embedding2 = nn.Linear(self.channel, self.query_dim)
            # if self.query_update_method=='concat': self.QueryEmbedding2 = nn.Linear(self.channel + self.value_dim, self.query_dim)
            # if self.query_update_method=='sum': self.QueryEmbedding2 = nn.Linear(self.channel, self.query_dim)
            if self.hop_method=='parallel': self.query_embedding2 = nn.Linear(self.channel, self.query_dim)

        if self.hops >= 3:
            if self.hop_method=='iterative' and self.query_update_method=='concat': self.query_embedding3 = nn.Linear(self.channel + self.value_dim*2, self.query_dim)
            if self.hop_method=='iterative' and self.query_update_method=='sum': self.query_embedding3 = nn.Linear(self.channel, self.query_dim)
            # if self.query_update_method=='concat': self.QueryEmbedding3 = nn.Linear(self.channel + self.value_dim*2, self.query_dim)
            # if self.query_update_method=='sum': self.QueryEmbedding3 = nn.Linear(self.channel, self.query_dim)
            if self.hop_method=='parallel': self.query_embedding3 = nn.Linear(self.channel, self.query_dim)

        self.KeyEmbedding1 = nn.Linear(self.channel, self.key_dim)
        self.ValueEmbedding1 = nn.Linear(self.channel, self.value_dim)

        if self.hops >= 2:
            if self.hop_method=='parallel': self.KeyEmbedding2 = nn.Linear(self.channel, self.key_dim)
            if self.hop_method=='parallel': self.ValueEmbedding2 = nn.Linear(self.channel, self.value_dim)

        if self.hops >= 3:
            if self.hop_method=='parallel': self.KeyEmbedding3 = nn.Linear(self.channel, self.key_dim)
            if self.hop_method=='parallel': self.ValueEmbedding3 = nn.Linear(self.channel, self.value_dim)

        self.classifier = self.fc_fusion()

    def fc_fusion(self):
        nums = self.hops
        input_dim = (nums * self.value_dim)
        num_bottleneck = 512
        # num_bottleneck = input_dim // 2 # originally, 512

        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(input_dim, num_bottleneck),
                nn.ReLU(),
                nn.Linear(num_bottleneck, self.num_class),
                )
        return classifier

    def forward(self, memory_input, eval): # (BS, num_frames, 1024), (BS, num_frames, 1024)
        bs = memory_input.size()[0]
        assert (memory_input.size()[1]==self.num_frames)

        if self.num_CNNs==1: query_value = torch.mean(memory_input, 1) # (BS, 1024)
        elif self.num_CNNs>1: raise ValueError('not supporting more than one CNNs')

        accumulated_output = []
        attentions = []

        # first hop
        retrieved_value1, p1 = self.hop(memory_input, query_value, self.KeyEmbedding1, self.ValueEmbedding1, self.query_embedding1)
        accumulated_output.append(retrieved_value1)
        attentions.append(p1.cpu())

        if self.hops >= 2:
            if self.hop_method=='iterative':
                KeyEmbedding = self.KeyEmbedding1
                ValueEmbedding = self.ValueEmbedding1

                if self.query_update_method=='sum':
                    updated_query_value2 = query_value + retrieved_value1 # (bs, 1024), (bs, value_dim)
                    QueryEmbedding = self.query_embedding2
                    # QueryEmbedding = self.query_embedding1

                if self.query_update_method=='concat':
                    updated_query_value2 = torch.cat((query_value,retrieved_value1), dim=1) # (bs, 1024 + value_dim)
                    QueryEmbedding = self.query_embedding2

            elif self.hop_method=='parallel':
                KeyEmbedding = self.KeyEmbedding2
                ValueEmbedding = self.ValueEmbedding2

                updated_query_value2 = query_value
                QueryEmbedding = self.query_embedding2

            retrieved_value2, p2 = self.hop(memory_input, updated_query_value2, KeyEmbedding, ValueEmbedding, QueryEmbedding)
            accumulated_output.append(retrieved_value2)
            attentions.append(p2.cpu())

        if self.hops >= 3:
            if self.hop_method=='iterative':
                KeyEmbedding = self.KeyEmbedding1
                ValueEmbedding = self.ValueEmbedding1

                if self.query_update_method=='sum':
                    updated_query_value3 = updated_query_value2 + retrieved_value2
                    QueryEmbedding = self.query_embedding3
                    # QueryEmbedding = self.query_embedding1

                if self.query_update_method=='concat':
                    updated_query_value3 = torch.cat((updated_query_value2, retrieved_value2), dim=1)
                    QueryEmbedding = self.query_embedding3
                    
            elif self.hop_method=='parallel':
                KeyEmbedding = self.KeyEmbedding3
                ValueEmbedding = self.ValueEmbedding3

                updated_query_value3 = query_value
                QueryEmbedding = self.query_embedding3

            retrieved_value3, p3 = self.hop(memory_input, updated_query_value3, KeyEmbedding, ValueEmbedding, QueryEmbedding)
            accumulated_output.append(retrieved_value3)
            attentions.append(p3.cpu())

        accumulated_output = torch.stack(accumulated_output, -1)
        if self.sorting:
            bs = p1.size()[0]
            accumulated_time_weight = []

            # get weighted timestamp
            standard = np.array(list(range(1,self.num_frames+1)))
            time1 =  np.dot(p1.cpu().data.numpy(), standard) # (30, 1)
            accumulated_time_weight.append(time1)
            if self.hops >= 2:
                time2 =  np.dot(p2.cpu().data.numpy(), standard) # (30, 1)
                accumulated_time_weight.append(time2)
            if self.hops >= 3:
                time3 =  np.dot(p3.cpu().data.numpy(), standard) # (30, 1)
                accumulated_time_weight.append(time3)
            accumulated_time_weight = np.squeeze(np.stack(accumulated_time_weight, 1),2)
            arg_time = np.argsort(accumulated_time_weight)
            # print (accumulated_time_weight)
            # print (np.argsort(accumulated_time_weight))
            # print (accumulated_time_weight.shape) # (30,2)

            # permutate according to timestamp
            # print (accumulated_output.size()) # (30, 512, 2)
            # for inner_i in range(bs):
            for inner_i in range(bs):
                print (accumulated_output[inner_i].cpu().data.numpy()) # (512, 2)
                accumulated_output[inner_i].permute(tuple(arg_time[inner_i,:].tolist()))
                print (accumulated_output[inner_i].cpu().data.numpy(), arg_time[inner_i,:]) # (512, 2)
            print ('-----------------')
            print (accumulated_output[0].cpu().data.numpy(), arg_time[inner_i,:])
            print (accumulated_output[1].cpu().data.numpy(), arg_time[inner_i,:])
            print (accumulated_output[2].cpu().data.numpy(), arg_time[inner_i,:])
            print (accumulated_output[3].cpu().data.numpy(), arg_time[inner_i,:])
            print (accumulated_output[4].cpu().data.numpy(), arg_time[inner_i,:])
            print (accumulated_output[5].cpu().data.numpy(), arg_time[inner_i,:])
            print (accumulated_output[6].cpu().data.numpy(), arg_time[inner_i,:])
            print (accumulated_output[7].cpu().data.numpy(), arg_time[inner_i,:])
            print (accumulated_output[8].cpu().data.numpy(), arg_time[inner_i,:])
            print (accumulated_output[9].cpu().data.numpy(), arg_time[inner_i,:])
            print (accumulated_output[10].cpu().data.numpy(), arg_time[inner_i,:])

        asdf
        accumulated_output = accumulated_output.view(bs, -1)
        output = self.classifier(accumulated_output)

        attentions = torch.stack(attentions,-1)
        attentions = attentions.permute(0, 1, 3, 2)
        attentions = attentions.squeeze(1)
        attentions = attentions.data.numpy().tolist()

        if eval:
            return output, attentions
        else:
            return output

    def hop(self, memory, query_value, KeyEmbedding, ValueEmbedding, QueryEmbedding):
        bs = memory.size(0)

        query_key = QueryEmbedding(query_value) # (BS, query_dim)
        query = query_key.unsqueeze(1) # (BS, 1, query_dim)

        memory = memory.view(memory.size(0)*self.num_frames, -1) # (BS * NUM_SEG, channel)

        key = KeyEmbedding(memory) # (BS * NUM_SEG, key_dim)
        key = key.view(bs, self.num_frames, -1) # (BS, NUM_SEG, key_dim)
        key = torch.transpose(key, 1, 2) # (BS, key_dim, NUM_SEG)

        p = torch.bmm(query, key) # (BS, 1, NUM_SEG)
        if self.no_softmax_on_p is False:
            p = F.softmax(p.view(-1, p.size()[1]*p.size()[2]), dim=1).view(-1, p.size()[1], p.size()[2]) # (BS, 1, NUM_SEG)
        # print (p)

        value = ValueEmbedding(memory) # (BS * NUM_SEG, value_dim)
        value = value.view(bs, self.num_frames, -1) # (BS, NUM_SEG, value_dim)

        retrieved_value = torch.bmm(p, value) # (BS, 1, value_dim)
        retrieved_value = torch.squeeze(retrieved_value, 1).contiguous() # (BS, value_dim)

        return (retrieved_value), p
        # return (retrieved_value + query_key), p

def return_MemNN(
    relation_type, num_frames, num_class, \
    key_dim, value_dim, query_dim, query_update_method, no_softmax_on_p,
    channel, num_hop, hop_method, num_CNNs, sorting):

    if relation_type == 'MemNN':
        MemNNmodel = MemNNModule(num_frames, num_class, channel, \
            key_dim, value_dim, query_dim, query_update_method, no_softmax_on_p, \
            num_hop, hop_method, num_CNNs, sorting)
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
