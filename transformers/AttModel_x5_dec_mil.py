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

    def forward(self, vis_fea, vis_mask, syb_fea, syb_mask):
        batch_size = vis_fea.shape[0]

        mask = torch.cat([vis_mask, syb_mask], dim = 1)
        fea_att = torch.cat([vis_fea, syb_fea], dim = 1)

        att_weights = []

        for i in range(self.num_blocks):
            fea_att, att_w = self.__getattr__('enc_self_attention_%d' % i)(fea_att, fea_att, fea_att, mask, None)
            fea_att = self.__getattr__('enc_feed_forward_%d' % i)(fea_att)
            att_weights.append(att_w)

        return fea_att, att_weights

class AttModel(nn.Module):
    def __init__(self, glove, hidden_size, num_classes, maxlen_q, maxlen, maxlen_v, num_blocks, num_heads, dropout_rate ):
        super(AttModel, self).__init__()
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.maxlen_q = maxlen_q

        self.att_vis = AttModel_self_vis(hidden_size, num_classes, num_blocks, num_heads, dropout_rate, maxlen_v)
        self.att_syb = AttModel_self_syb(glove, hidden_size, maxlen, num_blocks, num_heads, dropout_rate)
        self.MIL_align = AttModel_MIL_align(hidden_size, num_blocks, num_heads, dropout_rate)


        for i in range(self.num_blocks):
            self.__setattr__('dec_vanilla_attention_%d' % i,
                             new_multihead_attention_with_graph_mask(num_units=self.hidden_size,
                                                 num_heads=self.num_heads,
                                                 dropout_rate=self.dropout_rate,
                                                 causality=False))
            self.__setattr__('dec_feed_forward_%d' % i, feedforward(self.hidden_size,
                                                                    [4 * self.hidden_size,
                                                                     self.hidden_size]))


        self.cls = nn.Sequential(nn.Linear(hidden_size, hidden_size),
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

    def forward(self, vis_fea, vis_mask, q_ipt, q_mask, q_graph, syb_ipt, syb_mask, syb_graph):
        
        dec = self.q_emb(q_ipt)
        dec = self.q_mlp(dec)
        dec += self.q_positional_encoding(
                Variable(torch.unsqueeze(torch.arange(0, dec.shape[1]), 0).repeat(dec.shape[0], 1).long().cuda()))

        fea_vis = self.att_vis(vis_fea, vis_mask)
        fea_syb = self.att_syb(syb_ipt, syb_mask, syb_graph)

        mask = torch.cat([vis_mask, syb_mask], dim = 1)
        fea, att_weights = self.MIL_align(fea_vis, vis_mask, fea_syb, syb_mask)
        mask = torch.cat([vis_mask, syb_mask], dim = 1)
        fea = fea * mask.unsqueeze(2)

        for i in range(self.num_blocks):
            dec = self.__getattr__('dec_vanilla_attention_%d' % i)(dec, fea, fea, mask, None)
            dec = self.__getattr__('dec_feed_forward_%d' % i)(dec)
        
        dec = dec * q_mask.unsqueeze(2) 
        dec, _ = torch.max(dec, dim = 1)
        logits = self.cls(dec)

        return logits, att_weights
    
