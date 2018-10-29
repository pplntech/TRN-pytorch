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
        key_dim, value_dim, query_dim, memory_dim, query_update_method, no_softmax_on_p, \
        num_hop, hop_method, num_CNNs, sorting, MultiStageLoss, MultiStageLoss_MLP, how_to_get_query, only_query, CC, how_many_objects, Each_Embedding, Curriculum, Curriculum_dim, lr_steps):
        super(MemNNModule, self).__init__()

        self.num_frames = num_frames # num of segments
        self.num_class = num_class
        self.channel = channel # 1024
        self.sorting = sorting
        self.MultiStageLoss = MultiStageLoss
        self.MultiStageLoss_MLP = MultiStageLoss_MLP
        self.how_to_get_query = how_to_get_query
        self.only_query = only_query

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        assert (self.key_dim==self.query_dim)

        self.query_update_method = query_update_method
        if query_update_method=='sum':
            assert (channel==self.value_dim)
        self.no_softmax_on_p = no_softmax_on_p

        self.hops = num_hop
        self.hop_method = hop_method
        self.num_CNNs = num_CNNs

        self.CC = CC
        self.how_many_objects = how_many_objects
        self.Each_Embedding = Each_Embedding
        self.Curriculum = Curriculum
        self.Curriculum_dim = Curriculum_dim
        self.lr_steps = lr_steps

        if self.how_to_get_query=='lstm':
            '''
            input : 1024
            output : 1024
            '''
            self.query_lstm = nn.LSTM(self.channel, self.channel)

        # define layers
        if not self.only_query:

            # Query Embedding
            self.query_embedding1 = nn.Linear(self.channel, self.query_dim)
            if self.hop_method=='iterative' and self.query_update_method=='concat' and self.how_many_objects == 2: self.query_embedding1_2 = nn.Linear(self.channel, self.query_dim)

            if self.hops >= 2:
                if self.hop_method=='iterative' and self.query_update_method=='concat': self.query_embedding2 = nn.Linear(self.channel + self.value_dim, self.query_dim)
                if self.hop_method=='iterative' and self.query_update_method=='concat' and self.how_many_objects == 2: self.query_embedding2_2 = nn.Linear(self.channel + self.value_dim, self.query_dim)

                if self.hop_method=='iterative' and self.query_update_method=='sum': self.query_embedding2 = nn.Linear(self.channel, self.query_dim)
                # if self.query_update_method=='concat': self.QueryEmbedding2 = nn.Linear(self.channel + self.value_dim, self.query_dim)
                # if self.query_update_method=='sum': self.QueryEmbedding2 = nn.Linear(self.channel, self.query_dim)
                if self.hop_method=='parallel': self.query_embedding2 = nn.Linear(self.channel, self.query_dim)

            if self.hops >= 3:
                if self.hop_method=='iterative' and self.query_update_method=='concat': self.query_embedding3 = nn.Linear(self.channel + self.value_dim*2, self.query_dim)
                if self.hop_method=='iterative' and self.query_update_method=='concat' and self.how_many_objects == 2: self.query_embedding3_2 = nn.Linear(self.channel + self.value_dim*2, self.query_dim)

                if self.hop_method=='iterative' and self.query_update_method=='sum': self.query_embedding3 = nn.Linear(self.channel, self.query_dim)
                # if self.query_update_method=='concat': self.QueryEmbedding3 = nn.Linear(self.channel + self.value_dim*2, self.query_dim)
                # if self.query_update_method=='sum': self.QueryEmbedding3 = nn.Linear(self.channel, self.query_dim)
                if self.hop_method=='parallel': self.query_embedding3 = nn.Linear(self.channel, self.query_dim)


            # Key / Value Embedding
            self.KeyEmbedding1 = nn.Conv2d(self.channel, self.key_dim, kernel_size=1) # input : (N,Cin,H,W)
            self.ValueEmbedding1 = nn.Conv2d(self.channel, self.value_dim, kernel_size=1) # output :  (N,Cout,Hout,Wout)
            if self.hop_method=='iterative' and self.query_update_method=='concat' and self.how_many_objects == 2: self.KeyEmbedding1_2 = nn.Conv2d(self.channel, self.key_dim, kernel_size=1) # nn.Linear(self.channel, self.key_dim)
            if self.hop_method=='iterative' and self.query_update_method=='concat' and self.how_many_objects == 2: self.ValueEmbedding1_2 = nn.Conv2d(self.channel, self.value_dim, kernel_size=1) # nn.Linear(self.channel, self.value_dim)
            # self.KeyEmbedding1 = nn.Linear(self.channel, self.key_dim)
            # self.ValueEmbedding1 = nn.Linear(self.channel, self.value_dim)

            if self.hops >= 2:
                if self.hop_method=='parallel': self.KeyEmbedding2 = nn.Conv2d(self.channel, self.key_dim, kernel_size=1) # nn.Linear(self.channel, self.key_dim)
                if self.hop_method=='parallel': self.ValueEmbedding2 = nn.Conv2d(self.channel, self.value_dim, kernel_size=1) # nn.Linear(self.channel, self.value_dim)
                if self.hop_method=='iterative' and self.query_update_method=='concat' and self.Each_Embedding: self.KeyEmbedding2 = nn.Conv2d(self.channel, self.key_dim, kernel_size=1) # nn.Linear(self.channel, self.key_dim)
                if self.hop_method=='iterative' and self.query_update_method=='concat' and self.Each_Embedding: self.ValueEmbedding2 = nn.Conv2d(self.channel, self.value_dim, kernel_size=1) # nn.Linear(self.channel, self.value_dim)

            if self.hops >= 3:
                if self.hop_method=='parallel': self.KeyEmbedding3 = nn.Conv2d(self.channel, self.key_dim, kernel_size=1) # nn.Linear(self.channel, self.key_dim)
                if self.hop_method=='parallel': self.ValueEmbedding3 = nn.Conv2d(self.channel, self.value_dim, kernel_size=1) # nn.Linear(self.channel, self.value_dim)
                if self.hop_method=='iterative' and self.query_update_method=='concat' and self.Each_Embedding: self.KeyEmbedding3 = nn.Conv2d(self.channel, self.key_dim, kernel_size=1) # nn.Linear(self.channel, self.key_dim)
                if self.hop_method=='iterative' and self.query_update_method=='concat' and self.Each_Embedding: self.ValueEmbedding3 = nn.Conv2d(self.channel, self.value_dim, kernel_size=1) # nn.Linear(self.channel, self.value_dim)

        if self.MultiStageLoss:
            if self.MultiStageLoss_MLP:
                self.query_prediction = self.fc_fusion(self.channel)
            else:
                self.query_prediction = nn.Linear(self.channel, self.num_class)

            if self.hop_method=='iterative':
                if self.hops >= 2:
                    if self.MultiStageLoss_MLP:
                        if self.query_update_method=='concat': self.hop1_prediction = self.fc_fusion(self.channel + self.value_dim)
                        if self.query_update_method=='sum': self.hop1_prediction = self.fc_fusion(self.channel)
                    else:
                        if self.query_update_method=='concat': self.hop1_prediction = nn.Linear(self.channel + self.value_dim, self.num_class)
                        if self.query_update_method=='sum': self.hop1_prediction = nn.Linear(self.channel, self.num_class)
                if self.hops >= 3:
                    if self.MultiStageLoss_MLP:
                        if self.query_update_method=='concat': self.hop2_prediction = self.fc_fusion(self.channel + self.value_dim*2)
                        if self.query_update_method=='sum': self.hop2_prediction = self.fc_fusion(self.channel)
                    else:
                        if self.query_update_method=='concat': self.hop2_prediction = nn.Linear(self.channel + self.value_dim*2, self.num_class)
                        if self.query_update_method=='sum': self.hop2_prediction = nn.Linear(self.channel, self.num_class)

        if self.Curriculum:
            self.Curriculum_query = nn.Linear(self.channel, Curriculum_dim)
            self.Curriculum_hop1 = nn.Linear(self.channel + self.value_dim, Curriculum_dim)

            if self.hop_method=='iterative':
                if self.hops >= 2:
                    self.Curriculum_hop2 = nn.Linear(self.channel + self.value_dim*2, Curriculum_dim)
                if self.hops >= 3:
                    self.Curriculum_hop3 = nn.Linear(self.channel + self.value_dim*3, Curriculum_dim)


        if self.Curriculum: self.classifier = self.fc_fusion(Curriculum_dim)
        else: self.classifier = self.fc_fusion()

    def fc_fusion(self, given_input_dim=None):
        nums = self.hops
        input_dim = (self.channel + nums * self.value_dim)
        if self.only_query:
            input_dim = self.channel
        if given_input_dim is not None:
            input_dim = given_input_dim
        num_bottleneck = 512
        # num_bottleneck = input_dim // 2 # originally, 512

        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(input_dim, num_bottleneck),
                nn.ReLU(),
                nn.Linear(num_bottleneck, self.num_class),
                )
        return classifier

    def forward(self, memory_input, eval, epoch): # (BS, num_frames, channel, H, W)
        outputs = []
        bs = memory_input.size()[0]
        assert (memory_input.size()[1]==self.num_frames)


        # Calculate query_value
        if self.how_to_get_query=='lstm': # inputs : (temporal, batch, dim)
            # print (torch.mean(torch.mean(memory_input, -1),-1).size()) # torch.Size([4, 8, 2048])
            _, hidden = self.query_lstm(torch.mean(torch.mean(memory_input, -1),-1).permute(1,0,2)) # out : (8, 30, 1024), hidden : tuple # (out[7,0,:]) == (hidden[0].squeeze(0)[0,:])
            # out, hidden = self.query_lstm(memory_input.permute(1,0,2)) # out : (8, 30, 1024), hidden : tuple # (out[7,0,:]) == (hidden[0].squeeze(0)[0,:])
            query_value = hidden[0].squeeze(0) # (BS, 1024)

        elif self.how_to_get_query=='mean':
            if self.num_CNNs==1: query_value = torch.mean(torch.mean(torch.mean(memory_input, -1),-1), 1) # (BS, channel, H, W)
            elif self.num_CNNs>1: raise ValueError('not supporting more than one CNNs')

        # print (query_value.size()) # [4, 2048]
        accumulated_output = []
        attentions = []
        if self.hop_method=='iterative' and self.query_update_method=='concat' and self.how_many_objects == 2: attentions_2 = []
        if self.only_query:
            output = self.classifier(query_value)
            outputs.append(output.squeeze(1))
            if eval:
                attentions = Variable(torch.ones((int(bs), 1, int(self.num_frames), int(memory_input.size()[3]), int(memory_input.size()[4]), int(self.hops))))
                attentions = attentions.permute(0, 1, 5, 2, 3, 4) # (bs, 1, hop, num_seg, h, w)
                attentions = attentions.squeeze(1).cuda() # (bs, hop, num_seg, h, w)
                return outputs, attentions
            else:
                return outputs

        if self.MultiStageLoss:
            outputs.append(self.query_prediction(query_value).squeeze(1))

        if self.Curriculum:
            Curriculum_query_results = self.Curriculum_query(query_value)
            if epoch < self.lr_steps[0]: # only lstm
                output = self.classifier(Curriculum_query_results)
                outputs.append(output.squeeze(1))
                if eval:
                    attentions = Variable(torch.ones((int(bs), 1, int(self.num_frames), int(memory_input.size()[3]), int(memory_input.size()[4]), int(self.hops))))
                    attentions = attentions.permute(0, 1, 5, 2, 3, 4) # (bs, 1, hop, num_seg, h, w)
                    attentions = attentions.squeeze(1).cuda() # (bs, hop, num_seg, h, w)
                    return outputs, attentions
                else:
                    return outputs

        # first hop
        retrieved_value1, p1 = self.hop(memory_input, query_value, self.KeyEmbedding1, self.ValueEmbedding1, self.query_embedding1)

        if self.hop_method=='iterative' and self.query_update_method=='concat' and self.how_many_objects == 2:
            retrieved_value1_2, p1_2 = self.hop(memory_input, query_value, self.KeyEmbedding1_2, self.ValueEmbedding1_2, self.query_embedding1_2)
            retrieved_value1 = retrieved_value1 + retrieved_value1_2
            attentions_2.append(p1_2)

        accumulated_output.append(retrieved_value1)
        attentions.append(p1)
        
        if self.Curriculum:
            Curriculum_hop1_results = self.Curriculum_hop1(torch.cat((query_value,retrieved_value1), dim=1))
            if epoch < self.lr_steps[1]: # lstm + hop1
                output = self.classifier(Curriculum_query_results + Curriculum_hop1_results)
                outputs.append(output.squeeze(1))
                if eval:
                    attentions = torch.stack(attentions,-1) # (bs, 1, num_seg, h, w, hop)
                    attentions = attentions.permute(0, 1, 5, 2, 3, 4) # (bs, 1, hop, num_seg, h, w)
                    attentions = attentions.squeeze(1) # (bs, hop, num_seg, h, w)
                    return outputs, attentions
                else:
                    return outputs
        # attentions.append(p1.cpu())

        if self.hops >= 2:
            if self.hop_method=='iterative':

                KeyEmbedding = self.KeyEmbedding1
                ValueEmbedding = self.ValueEmbedding1

                if self.Each_Embedding:
                    KeyEmbedding = self.KeyEmbedding2
                    ValueEmbedding = self.ValueEmbedding2

                if self.query_update_method=='sum':
                    updated_query_value2 = query_value + retrieved_value1 # (bs, 1024), (bs, value_dim)
                    QueryEmbedding = self.query_embedding2
                    # QueryEmbedding = self.query_embedding1

                if self.query_update_method=='concat':
                    updated_query_value2 = torch.cat((query_value,retrieved_value1), dim=1) # (bs, 1024 + value_dim)
                    QueryEmbedding = self.query_embedding2

                if self.MultiStageLoss:
                    outputs.append(self.hop1_prediction(updated_query_value2).squeeze(1))

            elif self.hop_method=='parallel':
                KeyEmbedding = self.KeyEmbedding2
                ValueEmbedding = self.ValueEmbedding2

                updated_query_value2 = query_value
                QueryEmbedding = self.query_embedding2

            retrieved_value2, p2 = self.hop(memory_input, updated_query_value2, KeyEmbedding, ValueEmbedding, QueryEmbedding)
            if self.hop_method=='iterative' and self.query_update_method=='concat' and self.how_many_objects == 2:
                retrieved_value2_2, p2_2 = self.hop(memory_input, updated_query_value2, self.KeyEmbedding1_2, self.ValueEmbedding1_2, self.query_embedding2_2)
                retrieved_value2 = retrieved_value2 + retrieved_value2_2
                attentions_2.append(p2_2)

            accumulated_output.append(retrieved_value2)
            attentions.append(p2)

            if self.Curriculum:
                Curriculum_hop2_results = self.Curriculum_hop2(torch.cat((updated_query_value2, retrieved_value2), dim=1))
                if epoch < self.lr_steps[2]: # lstm + hop1 + hop2
                    output = self.classifier(Curriculum_query_results + Curriculum_hop1_results + Curriculum_hop2_results)
                    outputs.append(output.squeeze(1))
                    if eval:
                        attentions = torch.stack(attentions,-1) # (bs, 1, num_seg, h, w, hop)
                        attentions = attentions.permute(0, 1, 5, 2, 3, 4) # (bs, 1, hop, num_seg, h, w)
                        attentions = attentions.squeeze(1) # (bs, hop, num_seg, h, w)
                        return outputs, attentions
                    else:
                        return outputs
            # attentions.append(p2.cpu())

        if self.hops >= 3:
            if self.hop_method=='iterative':

                KeyEmbedding = self.KeyEmbedding1
                ValueEmbedding = self.ValueEmbedding1

                if self.Each_Embedding:
                    KeyEmbedding = self.KeyEmbedding3
                    ValueEmbedding = self.ValueEmbedding3

                if self.query_update_method=='sum':
                    updated_query_value3 = updated_query_value2 + retrieved_value2
                    QueryEmbedding = self.query_embedding3
                    # QueryEmbedding = self.query_embedding1

                if self.query_update_method=='concat':
                    updated_query_value3 = torch.cat((updated_query_value2, retrieved_value2), dim=1)
                    QueryEmbedding = self.query_embedding3
                    
                if self.MultiStageLoss:
                    outputs.append(self.hop2_prediction(updated_query_value3).squeeze(1))

            elif self.hop_method=='parallel':
                KeyEmbedding = self.KeyEmbedding3
                ValueEmbedding = self.ValueEmbedding3

                updated_query_value3 = query_value
                QueryEmbedding = self.query_embedding3

            retrieved_value3, p3 = self.hop(memory_input, updated_query_value3, KeyEmbedding, ValueEmbedding, QueryEmbedding)
            if self.hop_method=='iterative' and self.query_update_method=='concat' and self.how_many_objects == 2:
                retrieved_value3_2, p3_2 = self.hop(memory_input, updated_query_value3, self.KeyEmbedding1_2, self.ValueEmbedding1_2, self.query_embedding3_2)
                retrieved_value3 = retrieved_value3 + retrieved_value3_2
                attentions_2.append(p3_2)

            accumulated_output.append(retrieved_value3)
            attentions.append(p3)

            if self.Curriculum:
                Curriculum_hop3_results = self.Curriculum_hop3(torch.cat((updated_query_value3, retrieved_value3), dim=1))
                output = self.classifier(Curriculum_query_results + Curriculum_hop1_results + Curriculum_hop2_results + Curriculum_hop3_results)
                outputs.append(output.squeeze(1))
                if eval:
                    attentions = torch.stack(attentions,-1) # (bs, 1, num_seg, h, w, hop)
                    attentions = attentions.permute(0, 1, 5, 2, 3, 4) # (bs, 1, hop, num_seg, h, w)
                    attentions = attentions.squeeze(1) # (bs, hop, num_seg, h, w)
                    return outputs, attentions
                else:
                    return outputs
            # attentions.append(p3.cpu())

        # print (len(accumulated_output)) # 3
        # print (accumulated_output[0].size()) # [4, 512]
        accumulated_output = torch.stack(accumulated_output, -1)
        # print (accumulated_output.size()) # [4, 512, 3]
        if self.sorting:
            bs = p1.size()[0]
            accumulated_time_weight = []

            # get weighted timestamp
            standard = np.array(list(range(1,self.num_frames+1)))
            time1 =  np.dot(p1.mean(-1).mean(-1).cpu().data.numpy(), standard) # (30, 1)
            accumulated_time_weight.append(time1)
            if self.hops >= 2:
                time2 =  np.dot(p2.mean(-1).mean(-1).cpu().data.numpy(), standard) # (30, 1)
                accumulated_time_weight.append(time2)
            if self.hops >= 3:
                time3 =  np.dot(p3.mean(-1).mean(-1).cpu().data.numpy(), standard) # (30, 1)
                accumulated_time_weight.append(time3)
            accumulated_time_weight = np.squeeze(np.stack(accumulated_time_weight, 1),2)
            arg_time = np.argsort(accumulated_time_weight)

            # print (accumulated_time_weight)
            # print (np.argsort(accumulated_time_weight))
            # print (accumulated_time_weight.shape) # (30,2)

            # permute according to timestamp
            # print (accumulated_output.size()) # (30, 512, 2)
            # for inner_i in range(bs):
            for inner_i in range(bs):
                # print (accumulated_output[inner_i]) # (512, 2)
                # print (accumulated_output[inner_i].cpu().data.numpy()) # (512, 2)
                accumulated_output[inner_i] = accumulated_output[inner_i][:,tuple(arg_time[inner_i,:].tolist())]
                # .permute(tuple(arg_time[inner_i,:].tolist()))
                # print (accumulated_output[inner_i].cpu().data.numpy(), arg_time[inner_i,:]) # (512, 2)
        # print (query_value, query_value.unsqueeze(2)) # 4 x 512, 4 x 512 x 1
        # print (accumulated_output) # 4 x 512 x 3
        accumulated_output = accumulated_output.view(bs, -1)
        print ('--------------------------')
        print (accumulated_output) # 9x1536
        print ('--------------------------')
        print (query_value) # 9x2048x1
        print ('--------------------------')
        print (query_value.unsqueeze(2)) # 9x2048
        accumulated_output = torch.cat((query_value.unsqueeze(2), accumulated_output), 2)
        output = self.classifier(accumulated_output)

        # p1, p2, p3 : (4, 1, num_seg, h, w)
        attentions = torch.stack(attentions,-1) # (bs, 1, num_seg, h, w, hop)
        attentions = attentions.permute(0, 1, 5, 2, 3, 4) # (bs, 1, hop, num_seg, h, w)
        attentions = attentions.squeeze(1) # (bs, hop, num_seg, h, w)
        # attentions = attentions.data.numpy().tolist()
        # print (len(attentions)) # bs
        if self.hop_method=='iterative' and self.query_update_method=='concat' and self.how_many_objects == 2:
            attentions_2 = torch.stack(attentions_2,-1) # (bs, 1, num_seg, h, w, hop)
            attentions_2 = attentions_2.permute(0, 1, 5, 2, 3, 4) # (bs, 1, hop, num_seg, h, w)
            attentions_2 = attentions_2.squeeze(1) # (bs, hop, num_seg, h, w)
        outputs.append(output.squeeze(1))

        # print  (outputs) # list of (bsx174)
        # print (len(outputs)) # 2 for parallel

        if eval:
            if self.hop_method=='iterative' and self.query_update_method=='concat' and self.how_many_objects == 2:
                return outputs, attentions, attentions_2
            return outputs, attentions
        else:
            return outputs

    def hop(self, memory, query_value, KeyEmbedding, ValueEmbedding, QueryEmbedding):
        # memory : (BS, num_frames, channel, H, W)
        bs, T, H, W = memory.size(0), memory.size(1), memory.size(3), memory.size(4)

        memory = memory.view(bs*T, -1, H, W) # (BS*num_frames, channel, H, W)

        query_key = QueryEmbedding(query_value) # (BS, query_dim)
        query = query_key.unsqueeze(1) # (BS, 1, query_dim)

        # print (memory.size()) # [32, 2048, 1, 1]
        # print (KeyEmbedding) # Conv2d (2048, 256, kernel_size=(1, 1), stride=(1, 1))

        key = KeyEmbedding(memory) # ([BS*num_frames, img_dim, h, w]) >> ([BS*num_frames, key_dim, h, w])
        key = key.view(bs, T, self.key_dim, H, W) # (BS, num_frames, key_dim, h, w)
        key = torch.transpose(key,1,2).contiguous() # (BS, key_dim, num_frames, h, w)
        key = key.view(bs, self.key_dim, -1) # (B, key_dim, num_frames*h*w)

        p = torch.bmm(query, key) # (BS, 1, num_frames*h*w)
        if self.no_softmax_on_p is False:
            p = F.softmax(p.view(-1, p.size()[1]*p.size()[2]), dim=1).view(-1, p.size()[1], p.size()[2]) # (BS, 1, NUM_SEG) (BS, 1, num_frames*h*w)

        value = ValueEmbedding(memory) # ([BS*num_frames, img_dim, h, w]) >> ([BS*num_frames, value_dim, h, w])
        value = value.view(bs, T, self.value_dim, H, W) # (BS, num_frames, value_dim, h, w)
        value = value.permute(0, 1, 3, 4, 2).contiguous() # (BS, num_frames, h, w, value_dim)
        value = value.view(bs, -1, self.value_dim) # (BS, num_frames*hw, value_dim)

        # value = torch.transpose(value, 1, 2).contiguous() # (BS, value_dim, num_frames, h, w)
        # value = value.view(bs, self.value_dim, -1) # (BS, value_dim, num_frames*hw)
        # value = torch.transpose(value, 1, 2) # (BS, num_frames*h*w, value_dim)

        retrieved_value = torch.bmm(p, value) # (BS, 1, value_dim)
        retrieved_value = torch.squeeze(retrieved_value, 1).contiguous() # (BS, value_dim)

        return (retrieved_value), p.view(bs, 1, T, H, W)
        # return (retrieved_value + query_key), p

def return_MemNN(
    relation_type, num_frames, num_class, \
    key_dim, value_dim, query_dim, memory_dim, query_update_method, no_softmax_on_p,
    channel, num_hop, hop_method, num_CNNs, sorting, MultiStageLoss, MultiStageLoss_MLP, how_to_get_query, only_query, CC, how_many_objects, Each_Embedding, Curriculum, Curriculum_dim, lr_steps):

    if relation_type == 'MemNN':
        MemNNmodel = MemNNModule(num_frames, num_class, channel, \
            key_dim, value_dim, query_dim, memory_dim, query_update_method, no_softmax_on_p, \
            num_hop, hop_method, num_CNNs, sorting, MultiStageLoss, MultiStageLoss_MLP, how_to_get_query, only_query, CC, how_many_objects, Each_Embedding, Curriculum, Curriculum_dim, lr_steps)
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
