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


class AttModel_vis_grid(nn.Module):
    def __init__(self, hidden_size, num_classes, maxlen_q, num_blocks, num_heads, dropout_rate, maxlen_v):
        '''Attention is all you nedd. https://arxiv.org/abs/1706.03762
        Args:
            enc_voc: vocabulary size of encoder language
            dec_voc: vacabulary size of decoder language
        '''
        super(AttModel_vis_grid, self).__init__()

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.maxlen_q = maxlen_q
        self.dropout_rate = dropout_rate

        self.v_mlp = nn.Sequential( nn.Linear(2048, self.hidden_size), 
                                      nn.ReLU(inplace = True), 
                                      nn.Linear(self.hidden_size,self.hidden_size))


        self.v_positional_encoding = nn.Sequential(embedding(maxlen_v, self.hidden_size, zeros_pad=False, scale=False),
                                            nn.Dropout(self.dropout_rate))

        self.input_proj = nn.Linear(2048, self.hidden_size)

        for i in range(self.num_blocks):
            self.__setattr__('enc_self_attention_%d' % i, new_multihead_attention_with_graph_mask(num_units=self.hidden_size,
                                                                              num_heads=self.num_heads,
                                                                              dropout_rate=self.dropout_rate,
                                                                              causality=False))
            self.__setattr__('enc_feed_forward_%d' % i, feedforward(self.hidden_size,
                                                                    [4 * self.hidden_size,
                                                                     self.hidden_size]))

        self.q_mlp = nn.Sequential( nn.Linear(300, self.hidden_size), 
                                      nn.ReLU(inplace = True), 
                                      nn.Linear(self.hidden_size,self.hidden_size))

        self.q_positional_encoding = nn.Sequential(embedding(self.maxlen_q, self.hidden_size, zeros_pad=False, scale=False),
                                            nn.Dropout(self.dropout_rate))

        self.dec_emb = embedding(2, self.hidden_size, scale=True)
        self.dec_positional_encoding = nn.Sequential(embedding(2, self.hidden_size, zeros_pad=False, scale=False),
                                            nn.Dropout(self.dropout_rate))

        for i in range(self.num_blocks):
            self.__setattr__('dec_vanilla_attention_%d' % i,
                             new_multihead_attention_with_graph_mask(num_units=self.hidden_size,
                                                 num_heads=self.num_heads,
                                                 dropout_rate=self.dropout_rate,
                                                 causality=False))
            self.__setattr__('dec_feed_forward_%d' % i, feedforward(self.hidden_size,
                                                                    [4 * self.hidden_size,
                                                                     self.hidden_size]))

    def forward(self, vis_fea, vis_mask, q_fea, q_graph, q_mask):
        # vis_fea: bs x gridx x gridy x fea_size
        batch_size = vis_fea.size(0)
        if len(vis_fea.size()) == 4:
            vis_fea = vis_fea.reshape(-1, vis_fea.size(1) * vis_fea.size(2), vis_fea.size(3))

        vis_fea = self.v_mlp(vis_fea) 
        vis_enc_ipt = torch.unsqueeze(torch.arange(0, vis_fea.shape[1]), 0).repeat(vis_fea.shape[0], 1).long().cuda()
        vis_fea += self.v_positional_encoding(vis_enc_ipt)

        q_fea = self.q_mlp(q_fea)
        q_fea += self.q_positional_encoding(
                Variable(torch.unsqueeze(torch.arange(0, q_fea.shape[1]), 0).repeat(q_fea.shape[0], 1).long().cuda()))

        fea_ipt = torch.cat([vis_fea, q_fea], dim = 1)
        # Now transformer
        if vis_mask.numel() == 0:
            vis_mask = torch.ones((vis_fea.size(0), vis_fea.size(1))).cuda()

        mask = torch.cat([vis_mask, q_mask], dim = 1)

        size_graph = vis_fea.shape[1] + q_graph.shape[1]

        fea_att = fea_ipt
        
        # only consider each individual one.
        for i in range(2):
            fea_att = self.__getattr__('enc_self_attention_%d' % i)(fea_att, fea_att, fea_att, mask, None)
            fea_att = self.__getattr__('enc_feed_forward_%d' % i)(fea_att)

        graph_cross = torch.zeros((vis_fea.shape[0], size_graph, size_graph)).cuda()
        graph_cross[:,0:vis_fea.shape[1], vis_fea.shape[1]:] = q_mask.unsqueeze(1)

        graph_cross[:,vis_fea.shape[1]:, 0:vis_fea.shape[1]] = vis_mask.unsqueeze(1)

        for i in range(2, 4):
            fea_att = self.__getattr__('enc_self_attention_%d' % i)(fea_att, fea_att, fea_att, mask, graph_cross)
            fea_att = self.__getattr__('enc_feed_forward_%d' % i)(fea_att)

        fea_graph = fea_att 
        graph = torch.ones((vis_fea.shape[0], size_graph, size_graph)).cuda()
        graph[:,vis_fea.shape[1]:, vis_fea.shape[1]:] = q_graph
        for i in range(4, self.num_blocks):
            fea_graph = self.__getattr__('enc_self_attention_%d' % i)(fea_graph, fea_graph, fea_graph, mask, graph)
            fea_graph = self.__getattr__('enc_feed_forward_%d' % i)(fea_graph)

        fea_graph = fea_graph * mask.unsqueeze(2)

        decoder_inputs = torch.ones((batch_size,1)).cuda().long()
        dec = self.dec_emb(decoder_inputs)

        for i in range(self.num_blocks):
            dec = self.__getattr__('dec_vanilla_attention_%d' % i)(dec, fea_graph, fea_graph, mask, None)
            # feed forward
            dec = self.__getattr__('dec_feed_forward_%d' % i)(dec)

        return dec

class AttModel_syb(nn.Module):
    def __init__(self, glove, hidden_size, maxlen, maxlen_q, num_blocks, num_heads, dropout_rate):
        super(AttModel_syb, self).__init__()


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

        self.q_mlp = nn.Sequential( nn.Linear(300, hidden_size), 
                                      nn.ReLU(inplace = True), 
                                      nn.Linear(hidden_size,hidden_size))

        self.q_positional_encoding = nn.Sequential(embedding(maxlen_q, hidden_size, zeros_pad=False, scale=False),
                                            nn.Dropout(dropout_rate))


        self.dec_emb = embedding(2, self.hidden_size, scale=True)
        self.dec_positional_encoding = nn.Sequential(embedding(2, self.hidden_size, zeros_pad=False, scale=False),
                                            nn.Dropout(dropout_rate))

        for i in range(self.num_blocks):
            self.__setattr__('dec_vanilla_attention_%d' % i,
                             new_multihead_attention_with_graph_mask(num_units=self.hidden_size,
                                                 num_heads=self.num_heads,
                                                 dropout_rate=self.dropout_rate,
                                                 causality=False))
            self.__setattr__('dec_feed_forward_%d' % i, feedforward(self.hidden_size,
                                                                    [4 * self.hidden_size,
                                                                     self.hidden_size]))

        for i in range(self.num_blocks):
            self.__setattr__('enc_self_attention_%d' % i, new_multihead_attention_with_graph_mask(num_units=hidden_size,
                                                                              num_heads=self.num_heads,
                                                                              dropout_rate=dropout_rate,
                                                                              causality=False))
            self.__setattr__('enc_feed_forward_%d' % i, feedforward(hidden_size,
                                                                    [4 * hidden_size,
                                                                     hidden_size]))

    def forward(self, syb_ipt, syb_mask, syb_graph, q_fea, q_graph, q_mask):
        batch_size = syb_ipt.shape[0]

        syb_fea = self.syb_emb(syb_ipt)
        syb_fea = self.syb_mlp(syb_fea)
        pos_ipt = torch.unsqueeze(torch.arange(0, syb_ipt.size(1)), 0).repeat(syb_ipt.size(0), 1).long().cuda()
        pos_fea = self.syb_positional_encoding(pos_ipt)

        syb_fea += pos_fea
        # Now the qeustion.
        q_fea = self.q_mlp(q_fea)
        q_fea += self.q_positional_encoding(
                Variable(torch.unsqueeze(torch.arange(0, q_fea.shape[1]), 0).repeat(q_fea.shape[0], 1).long().cuda()))
        
        fea = torch.cat([syb_fea, q_fea], dim = 1) 
        mask = torch.cat([syb_mask, q_mask], dim = 1)

        fea_att = fea
        # Blocks
        size_graph = syb_graph.size(2) + q_graph.size(2)  

        for i in range(2):
            fea_att = self.__getattr__('enc_self_attention_%d' % i)(fea_att, fea_att, fea_att, mask, None)
            fea_att = self.__getattr__('enc_feed_forward_%d' % i)(fea_att)

        graph_cross = torch.zeros((batch_size, size_graph, size_graph)).cuda()
        graph_cross[:,0:syb_graph.size(1), syb_graph.size(1):] = q_mask.unsqueeze(1)
        graph_cross[:,syb_graph.size(1):, 0:syb_graph.size(1)] = syb_mask.unsqueeze(1)
        for i in range(2, 4):
            fea_att = self.__getattr__('enc_self_attention_%d' % i)(fea_att, fea_att, fea_att, mask, graph_cross)
            fea_att = self.__getattr__('enc_feed_forward_%d' % i)(fea_att)

        graph = torch.ones((batch_size, size_graph, size_graph)).cuda()
        graph[:, 0:syb_graph.size(2), 0:syb_graph.size(2)] = syb_graph
        graph[:, syb_graph.size(2):, syb_graph.size(2):] = q_graph
        fea_graph = fea_att

        for i in range(4, self.num_blocks):
            fea_graph = self.__getattr__('enc_self_attention_%d' % i)(fea_graph, fea_graph, fea_graph, mask, graph)
            fea_graph = self.__getattr__('enc_feed_forward_%d' % i)(fea_graph)

        fea = fea_graph * mask.unsqueeze(2)

        decoder_inputs = torch.ones((batch_size,1)).cuda().long()
        dec = self.dec_emb(decoder_inputs)

        for i in range(self.num_blocks):
            # vanilla attention
            dec = self.__getattr__('dec_vanilla_attention_%d' % i)(dec, fea_graph, fea_graph, mask, None)
            dec = self.__getattr__('dec_feed_forward_%d' % i)(dec)

        return dec

class AttModel(nn.Module):
    def __init__(self, glove, hidden_size, num_classes, maxlen_q, maxlen, maxlen_v, num_blocks, num_heads, dropout_rate ):
        super(AttModel, self).__init__()
        self.att_vis_grid = AttModel_vis_grid(hidden_size, num_classes, maxlen_q, num_blocks, num_heads, dropout_rate, maxlen_v)

        self.att_syb = AttModel_syb(glove, hidden_size, maxlen, maxlen_q, num_blocks, num_heads, dropout_rate)
        self.cls = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size),
                                nn.ReLU(),
                                nn.Dropout(dropout_rate, inplace = True),
                                nn.Linear(hidden_size, num_classes))

        self.label_smoothing = label_smoothing()
        
        new_glove_voc = Parameter(torch.rand(401000, 300))
        nn.init.xavier_normal(new_glove_voc)
        new_glove_voc[:glove.vectors.shape[0],:] = glove.vectors
        
        self.q_emb = nn.Embedding.from_pretrained(new_glove_voc, freeze=False)

    def forward(self, vis_fea, vis_mask, q_ipt, q_mask, q_graph, syb_ipt, syb_mask, syb_graph):
        
        q_fea = self.q_emb(q_ipt)

        fea_syb = self.att_syb(syb_ipt, syb_mask, syb_graph, q_fea, q_graph, q_mask)
        fea_vis_grid = self.att_vis_grid(vis_fea, vis_mask, q_fea, q_graph, q_mask)
        
        fea = torch.cat((fea_syb.squeeze(), fea_vis_grid.squeeze()), 1)
        logits = self.cls(fea)

        return logits
    
