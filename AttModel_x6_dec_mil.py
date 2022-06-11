# -*- coding: utf-8 -*-

# this is from v2_dec
# just change the diag

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np
from modules import *
import pdb


class AttModel_self_vis(nn.Module):
    def __init__(self, hidden_size, num_classes, num_blocks, num_heads, dropout_rate, maxlen_v):
        '''Attention is all you nedd. https://arxiv.org/abs/1706.03762
        Args:
            enc_voc: vocabulary size of encoder language
            dec_voc: vacabulary size of decoder language
        '''
        super(AttModel_self_vis, self).__init__()

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.v_mlp = nn.Sequential( nn.Linear(2048, self.hidden_size), 
                                      nn.ReLU(inplace = True), 
                                      nn.Linear(self.hidden_size,self.hidden_size))


        self.v_positional_encoding = nn.Sequential(embedding(maxlen_v, self.hidden_size, zeros_pad=False, scale=False),
                                            nn.Dropout(self.dropout_rate))

        for i in range(self.num_blocks):
            self.__setattr__('enc_self_attention_%d' % i, new_multihead_attention_with_graph_mask(num_units=self.hidden_size,
                                                                              num_heads=self.num_heads,
                                                                              dropout_rate=self.dropout_rate,
                                                                              causality=False))
            self.__setattr__('enc_feed_forward_%d' % i, feedforward(self.hidden_size,
                                                                    [4 * self.hidden_size,
                                                                     self.hidden_size]))

    def forward(self, vis_fea, vis_mask):
        batch_size = vis_fea.size(0)
        if len(vis_fea.size()) == 4:
            vis_fea = vis_fea.reshape(-1, vis_fea.size(1) * vis_fea.size(2), vis_fea.size(3))

        vis_fea = self.v_mlp(vis_fea) 
        vis_enc_ipt = torch.unsqueeze(torch.arange(0, vis_fea.shape[1]), 0).repeat(vis_fea.shape[0], 1).long().cuda()
        vis_fea += self.v_positional_encoding(vis_enc_ipt)

        if vis_mask.numel() == 0:
            vis_mask = torch.ones((vis_fea.size(0), vis_fea.size(1))).cuda()

        fea_att = vis_fea
        
        # only consider each individual one.
        for i in range(self.num_blocks):
            fea_att = self.__getattr__('enc_self_attention_%d' % i)(fea_att, fea_att, fea_att, vis_mask, None)
            fea_att = self.__getattr__('enc_feed_forward_%d' % i)(fea_att)

        return fea_att

class AttModel_self_syb(nn.Module):
    def __init__(self, glove, hidden_size, maxlen, num_blocks, num_heads, dropout_rate):
        super(AttModel_self_syb, self).__init__()

        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        new_glove_voc = Parameter(torch.rand(401000, 300))
        nn.init.xavier_normal(new_glove_voc)
        new_glove_voc[:glove.vectors.shape[0],:] = glove.vectors
        
        self.syb_emb = nn.Embedding.from_pretrained(new_glove_voc, freeze=False)
        self.syb_mlp = nn.Sequential(nn.Linear(300, 2048),
                                     nn.ReLU(inplace = True),
                                     nn.Linear(2048, hidden_size))
        
        self.syb_positional_encoding = nn.Sequential(embedding(maxlen, hidden_size, zeros_pad=False, scale=False),
                                    nn.Dropout(dropout_rate))

        for i in range(self.num_blocks):
            self.__setattr__('enc_self_attention_%d' % i, new_multihead_attention_with_graph_mask(num_units=hidden_size,
                                                                              num_heads=self.num_heads,
                                                                              dropout_rate=dropout_rate,
                                                                              causality=False))
            self.__setattr__('enc_feed_forward_%d' % i, feedforward(hidden_size,
                                                                    [4 * hidden_size,
                                                                     hidden_size]))

    def forward(self, syb_ipt, syb_mask, syb_graph):
        batch_size = syb_ipt.shape[0]

        syb_fea = self.syb_emb(syb_ipt)
        syb_fea = self.syb_mlp(syb_fea)
        pos_ipt = torch.unsqueeze(torch.arange(0, syb_ipt.size(1)), 0).repeat(syb_ipt.size(0), 1).long().cuda()
        pos_fea = self.syb_positional_encoding(pos_ipt)

        syb_fea += pos_fea
        # Now the qeustion.

        fea_att = syb_fea
        # Blocks

        for i in range(self.num_blocks):
            fea_att = self.__getattr__('enc_self_attention_%d' % i)(fea_att, fea_att, fea_att, syb_mask, syb_graph)
            fea_att = self.__getattr__('enc_feed_forward_%d' % i)(fea_att)

        return fea_att

class AttModel_MIL_align(nn.Module):
    def __init__(self, hidden_size, num_blocks, num_heads, dropout_rate):
        super(AttModel_MIL_align, self).__init__()

        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        for i in range(self.num_blocks):
            self.__setattr__('enc_self_attention_%d' % i, new_multihead_attention_with_graph_mask(num_units=hidden_size,
                                                                              num_heads=self.num_heads,
                                                                              dropout_rate=dropout_rate,
                                                                              causality=False, return_att = True))
            self.__setattr__('enc_feed_forward_%d' % i, feedforward(hidden_size,
                                                                    [4 * hidden_size,
                                                                     hidden_size]))

    def forward(self, vis_fea, syb_fea, syb_mask):
        batch_size = vis_fea.shape[0]

        att_weights = []

        for i in range(self.num_blocks):
            vis_fea, att_w = self.__getattr__('enc_self_attention_%d' % i)(vis_fea, syb_fea, syb_fea, syb_mask, None)
            vis_fea = self.__getattr__('enc_feed_forward_%d' % i)(vis_fea)
            att_weights.append(att_w)

        return vis_fea, [att_weights[-1]]

class AttModel(nn.Module):
    def __init__(self, glove, hidden_size, num_classes, maxlen_q, maxlen, maxlen_v, num_blocks, num_heads, dropout_rate ):
        super(AttModel, self).__init__()
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.maxlen_q = maxlen_q

        self.att_vis = AttModel_self_vis(hidden_size, num_classes, 2, num_heads, dropout_rate, maxlen_v)
        self.att_syb = AttModel_self_syb(glove, hidden_size, maxlen, 2, num_heads, dropout_rate)
        self.MIL_align = AttModel_MIL_align(hidden_size, 2, num_heads, dropout_rate)


        for i in range(self.num_blocks):
            self.__setattr__('vq_dec_vanilla_attention_%d' % i,
                             new_multihead_attention_with_graph_mask(num_units=self.hidden_size,
                                                 num_heads=self.num_heads,
                                                 dropout_rate=self.dropout_rate,
                                                 causality=False))
            self.__setattr__('vq_dec_feed_forward_%d' % i, feedforward(self.hidden_size,
                                                                    [4 * self.hidden_size,
                                                                     self.hidden_size]))

            self.__setattr__('sq_dec_vanilla_attention_%d' % i,
                             new_multihead_attention_with_graph_mask(num_units=self.hidden_size,
                                                 num_heads=self.num_heads,
                                                 dropout_rate=self.dropout_rate,
                                                 causality=False))
            self.__setattr__('sq_dec_feed_forward_%d' % i, feedforward(self.hidden_size,
                                                                    [4 * self.hidden_size,
                                                                     self.hidden_size]))

        for i in range(self.num_blocks):
            self.__setattr__('vq_enc_self_attention_%d' % i, new_multihead_attention_with_graph_mask(num_units=self.hidden_size,
                                                                              num_heads=self.num_heads,
                                                                              dropout_rate=self.dropout_rate,
                                                                              causality=False))
            self.__setattr__('vq_enc_feed_forward_%d' % i, feedforward(self.hidden_size,
                                                                    [4 * self.hidden_size,
                                                                     self.hidden_size]))

            self.__setattr__('sq_enc_self_attention_%d' % i, new_multihead_attention_with_graph_mask(num_units=self.hidden_size,
                                                                              num_heads=self.num_heads,
                                                                              dropout_rate=self.dropout_rate,
                                                                              causality=False))
            self.__setattr__('sq_enc_feed_forward_%d' % i, feedforward(self.hidden_size,
                                                                    [4 * self.hidden_size,
                                                                     self.hidden_size]))


        self.cls = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size),
                                nn.ReLU(),
                                nn.Dropout(dropout_rate, inplace = True),
                                nn.Linear(hidden_size, num_classes))

        self.label_smoothing = label_smoothing()
        
        new_glove_voc = Parameter(torch.rand(401000, 300))
        nn.init.xavier_normal(new_glove_voc)
        new_glove_voc[:glove.vectors.shape[0],:] = glove.vectors

        self.q_emb = nn.Embedding.from_pretrained(new_glove_voc, freeze=False)
        self.q_mlp = nn.Sequential( nn.Linear(300, hidden_size), 
                                      nn.ReLU(inplace = True), 
                                      nn.Linear(hidden_size,hidden_size))


        self.q_positional_encoding = nn.Sequential(embedding(self.maxlen_q, self.hidden_size, zeros_pad=False, scale=False),
                                            nn.Dropout(self.dropout_rate))

        self.__setattr__('q_self_attention',
                             new_multihead_attention_with_graph_mask(num_units=self.hidden_size,
                                                 num_heads=self.num_heads,
                                                 dropout_rate=self.dropout_rate,
                                                 causality=False))
        self.__setattr__('q_self_feed_forward', feedforward(self.hidden_size,
                                                                    [4 * self.hidden_size,
                                                                     self.hidden_size]))


        self.dec_emb = embedding(2, self.hidden_size, scale=True)
        self.dec_positional_encoding = nn.Sequential(embedding(2, self.hidden_size, zeros_pad=False, scale=False),
                                            nn.Dropout(self.dropout_rate))


    def forward(self, vis_fea, vis_mask, q_ipt, q_mask, q_graph, syb_ipt, syb_mask, syb_graph, idx_of_obj):
        batch_size = vis_fea.shape[0]
        fea_vis = self.att_vis(vis_fea, vis_mask)
        fea_syb = self.att_syb(syb_ipt, syb_mask, syb_graph)

        fea_vis_align, att_weights = self.MIL_align(fea_vis, fea_syb, syb_mask)

        fea_vis_align *= vis_mask.unsqueeze(2)

        fea_syb_vis = fea_syb.clone()
        for i in range(batch_size):
            fea_syb_vis[i, idx_of_obj[i], :] =  fea_vis_align[i,0:idx_of_obj[i].shape[0],:]

        mask = torch.cat([vis_mask, syb_mask], dim = 1)

        #now individually attend each modality
        q_fea = self.q_emb(q_ipt)
        q_fea = self.q_mlp(q_fea)
        q_fea += self.q_positional_encoding(
                Variable(torch.unsqueeze(torch.arange(0, q_fea.shape[1]), 0).repeat(q_fea.shape[0], 1).long().cuda()))

        q_fea = self.__getattr__('q_self_attention')(q_fea,q_fea,q_fea,q_mask,None)
        q_fea = self.__getattr__('q_self_feed_forward')(q_fea)
        q_fea *= q_mask.unsqueeze(2)
        
        # process visual part first.
        fea_vis_q = torch.cat([fea_syb_vis, q_fea], dim = 1)
        size_graph = fea_syb_vis.shape[1] + q_graph.shape[1]

        mask_vq = torch.cat([syb_mask, q_mask], dim = 1)

        graph_diag= torch.zeros((batch_size, size_graph, size_graph)).cuda()
        graph_diag[:,0:fea_syb_vis.shape[1], 0:fea_syb_vis.shape[1]] = syb_mask.unsqueeze(1) # should be the same with syb now.
        graph_diag[:,fea_syb_vis.shape[1]:, fea_syb_vis.shape[1]:] = q_mask.unsqueeze(1)

        for i in range(2):
            fea_vis_q = self.__getattr__('vq_enc_self_attention_%d' % i)(fea_vis_q, fea_vis_q, fea_vis_q, mask_vq, graph_diag)
            fea_vis_q = self.__getattr__('vq_enc_feed_forward_%d' % i)(fea_vis_q)

        graph_cross= torch.zeros((batch_size, size_graph, size_graph)).cuda()
        graph_cross[:,0:syb_graph.size(1), syb_graph.size(1):] = q_mask.unsqueeze(1)
        graph_cross[:,syb_graph.size(1):, 0:syb_graph.size(1)] = syb_mask.unsqueeze(1)
        for i in range(2, 4):
            fea_vis_q = self.__getattr__('vq_enc_self_attention_%d' % i)(fea_vis_q, fea_vis_q, fea_vis_q, mask_vq, graph_cross)
            fea_vis_q = self.__getattr__('vq_enc_feed_forward_%d' % i)(fea_vis_q)

        graph = torch.ones((batch_size, size_graph, size_graph)).cuda()
        graph[:, 0:syb_graph.size(2), 0:syb_graph.size(2)] = syb_graph
        graph[:, syb_graph.size(2):, syb_graph.size(2):] = q_graph

        for i in range(4, self.num_blocks):
            fea_vis_q = self.__getattr__('vq_enc_self_attention_%d' % i)(fea_vis_q, fea_vis_q, fea_vis_q, mask_vq, graph)
            fea_vis_q = self.__getattr__('vq_enc_feed_forward_%d' % i)(fea_vis_q)


        decoder_inputs = torch.ones((batch_size,1)).cuda().long()
        dec_ipt = self.dec_emb(decoder_inputs)
        dec_vis_q = dec_ipt

        for i in range(self.num_blocks):
            # vanilla attention
            dec_vis_q = self.__getattr__('vq_dec_vanilla_attention_%d' % i)(dec_vis_q, fea_vis_q, fea_vis_q, mask_vq, None)
            dec_vis_q = self.__getattr__('vq_dec_feed_forward_%d' % i)(dec_vis_q)

        #process syb
        fea_syb_q = torch.cat([fea_syb, q_fea], dim = 1)

        for i in range(2):
            fea_syb_q = self.__getattr__('sq_enc_self_attention_%d' % i)(fea_syb_q, fea_syb_q, fea_syb_q, mask_vq, graph_diag)
            fea_syb_q = self.__getattr__('sq_enc_feed_forward_%d' % i)(fea_syb_q)

        for i in range(2, 4):
            fea_syb_q = self.__getattr__('sq_enc_self_attention_%d' % i)(fea_syb_q, fea_syb_q, fea_syb_q, mask_vq, graph_cross)
            fea_syb_q = self.__getattr__('sq_enc_feed_forward_%d' % i)(fea_syb_q)

        for i in range(4, self.num_blocks):
            fea_syb_q = self.__getattr__('sq_enc_self_attention_%d' % i)(fea_syb_q, fea_syb_q, fea_syb_q, mask_vq, graph)
            fea_syb_q = self.__getattr__('sq_enc_feed_forward_%d' % i)(fea_syb_q)

        dec_syb_q = dec_ipt
        for i in range(self.num_blocks):
            # vanilla attention
            dec_syb_q = self.__getattr__('sq_dec_vanilla_attention_%d' % i)(dec_syb_q, fea_syb_q, fea_syb_q, mask_vq, None)
            dec_syb_q = self.__getattr__('sq_dec_feed_forward_%d' % i)(dec_syb_q)

        dec_fea = torch.cat([dec_vis_q.squeeze(), dec_syb_q.squeeze()],dim=1)
        logits = self.cls(dec_fea)

        return logits, att_weights
    
