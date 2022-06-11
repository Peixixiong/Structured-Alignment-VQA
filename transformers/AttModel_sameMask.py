# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np
from modules import *
from hyperparams import Hyperparams as hp


class AttModel_vis(nn.Module):
    def __init__(self, glove, hp_, enc_voc, dec_voc):
        '''Attention is all you nedd. https://arxiv.org/abs/1706.03762
        Args:
            hp: Hyper Parameters
            enc_voc: vocabulary size of encoder language
            dec_voc: vacabulary size of decoder language
        '''
        super(AttModel_vis, self).__init__()
        self.hp = hp_

        self.enc_voc = enc_voc
        self.dec_voc = dec_voc

        # encoder
#         self.enc_emb = embedding(self.enc_voc, self.hp.hidden_units, scale=True)
        new_glove_voc = Parameter(torch.rand(400450, 300)) #bua 400189
        nn.init.xavier_normal(new_glove_voc)
        new_glove_voc[:glove.vectors.shape[0],:] = glove.vectors
        
        self.enc_emb = nn.Embedding.from_pretrained(new_glove_voc, freeze=False)
        self.glove_proj = nn.Linear(300, 2048)
        self.input_proj = nn.Linear(2048, self.hp.hidden_units) #nn.Sequential(nn.Linear(2048, self.hp.hidden_units), nn.ReLU())
        
        if self.hp.sinusoid:
            self.enc_positional_encoding = positional_encoding(num_units=self.hp.hidden_units,
                                                               zeros_pad=False,
                                                               scale=False)
        else:
            self.enc_positional_encoding = embedding(self.hp.maxlen, self.hp.hidden_units, zeros_pad=False, scale=False)

        self.enc_dropout = nn.Dropout(self.hp.dropout_rate)
        for i in range(self.hp.num_blocks):
            self.__setattr__('enc_self_attention_%d' % i, new_multihead_attention(num_units=self.hp.hidden_units,
                                                                              num_heads=self.hp.num_heads,
                                                                              dropout_rate=self.hp.dropout_rate,
                                                                              causality=False))
            self.__setattr__('enc_feed_forward_%d' % i, feedforward(self.hp.hidden_units,
                                                                    [4 * self.hp.hidden_units,
                                                                     self.hp.hidden_units]))

        # decoder
        self.dec_emb = embedding(self.dec_voc, self.hp.hidden_units, scale=True)
        if self.hp.sinusoid:
            self.dec_positional_encoding = positional_encoding(num_units=self.hp.hidden_units,
                                                               zeros_pad=False,
                                                               scale=False)
        else:
            self.dec_positional_encoding = embedding(self.hp.maxlen, self.hp.hidden_units, zeros_pad=False, scale=False)

        self.dec_dropout = nn.Dropout(self.hp.dropout_rate)
        for i in range(self.hp.num_blocks):
            self.__setattr__('dec_self_attention_%d' % i,
                             multihead_attention(num_units=self.hp.hidden_units,
                                                 num_heads=self.hp.num_heads,
                                                 dropout_rate=self.hp.dropout_rate,
                                                 causality=True))
            self.__setattr__('dec_vanilla_attention_%d' % i,
                             multihead_attention(num_units=self.hp.hidden_units,
                                                 num_heads=self.hp.num_heads,
                                                 dropout_rate=self.hp.dropout_rate,
                                                 causality=False))
            self.__setattr__('dec_feed_forward_%d' % i, feedforward(self.hp.hidden_units,
                                                                    [4 * self.hp.hidden_units,
                                                                     self.hp.hidden_units]))

    def forward(self, x, y, graph1, x0):
        self.decoder_inputs = Variable(torch.ones(y[:, :1].size()).cuda() * 2).long()

        # Encoder
        self.enc = self.enc_emb(x)
        self.enc = self.glove_proj(self.enc)

        condition = x0.eq(-1.).float()
        self.enc = self.enc*condition + x0 * (1. - condition)
        self.enc = self.input_proj(self.enc)

        if self.hp.sinusoid:
            self.enc += self.enc_positional_encoding(x).cuda()
        else:
            self.enc += self.enc_positional_encoding(
                Variable(torch.unsqueeze(torch.arange(0, x.size()[1]), 0).repeat(x.size(0), 1).long().cuda()))
        self.enc = self.enc_dropout(self.enc)

        # Blocks
        for i in range(self.hp.num_blocks):
            self.enc = self.__getattr__('enc_self_attention_%d' % i)(self.enc, self.enc, self.enc, graph1)
           
            # Feed Forward
            self.enc = self.__getattr__('enc_feed_forward_%d' % i)(self.enc)
        # Decoder
        self.dec = self.dec_emb(self.decoder_inputs)

        # Positional Encoding
        if self.hp.sinusoid:
            self.dec += self.dec_positional_encoding(self.decoder_inputs).cuda()
        else:
            self.dec += self.dec_positional_encoding(
                Variable(torch.unsqueeze(torch.arange(0, self.decoder_inputs.size()[1]), 0).repeat(self.decoder_inputs.size(0), 1).long().cuda()))
        # Dropout
        self.dec = self.dec_dropout(self.dec)
        # Blocks
        for i in range(self.hp.num_blocks):
            # self-attention
            self.dec = self.__getattr__('dec_self_attention_%d' % i)(self.dec, self.dec, self.dec)
            # vanilla attention
            self.dec = self.__getattr__('dec_vanilla_attention_%d' % i)(self.dec, self.enc, self.enc)
            # feed forward
            self.dec = self.__getattr__('dec_feed_forward_%d' % i)(self.dec)

        return self.dec
    
class AttModel_syb(nn.Module):
    def __init__(self, glove, hp_, enc_voc, dec_voc):
        '''Attention is all you nedd. https://arxiv.org/abs/1706.03762
        Args:
            hp: Hyper Parameters
            enc_voc: vocabulary size of encoder language
            dec_voc: vacabulary size of decoder language
        '''
        super(AttModel_syb, self).__init__()
        self.hp = hp_

        self.enc_voc = enc_voc
        self.dec_voc = dec_voc
        
        # encoder
        new_glove_voc = Parameter(torch.rand(400450, 300))
        nn.init.xavier_normal(new_glove_voc)
        new_glove_voc[:glove.vectors.shape[0],:] = glove.vectors
        
        self.enc_emb = nn.Embedding.from_pretrained(new_glove_voc, freeze=False)
        self.glove_proj = nn.Linear(300, 2048)
        self.input_proj = nn.Linear(2048, self.hp.hidden_units)
        
        if self.hp.sinusoid:
            self.enc_positional_encoding = positional_encoding(num_units=self.hp.hidden_units,
                                                               zeros_pad=False,
                                                               scale=False)
        else:
            self.enc_positional_encoding = embedding(self.hp.maxlen, self.hp.hidden_units, zeros_pad=False, scale=False)

        self.enc_dropout = nn.Dropout(self.hp.dropout_rate)
        for i in range(self.hp.num_blocks):
            self.__setattr__('enc_self_attention_%d' % i, new_multihead_attention(num_units=self.hp.hidden_units,
                                                                              num_heads=self.hp.num_heads,
                                                                              dropout_rate=self.hp.dropout_rate,
                                                                              causality=False))
            self.__setattr__('enc_feed_forward_%d' % i, feedforward(self.hp.hidden_units,
                                                                    [4 * self.hp.hidden_units,
                                                                     self.hp.hidden_units]))

        # decoder
        self.dec_emb = embedding(self.dec_voc, self.hp.hidden_units, scale=True)
        if self.hp.sinusoid:
            self.dec_positional_encoding = positional_encoding(num_units=self.hp.hidden_units,
                                                               zeros_pad=False,
                                                               scale=False)
        else:
            self.dec_positional_encoding = embedding(self.hp.maxlen, self.hp.hidden_units, zeros_pad=False, scale=False)

        self.dec_dropout = nn.Dropout(self.hp.dropout_rate)
        for i in range(self.hp.num_blocks):
            self.__setattr__('dec_self_attention_%d' % i,
                             multihead_attention(num_units=self.hp.hidden_units,
                                                 num_heads=self.hp.num_heads,
                                                 dropout_rate=self.hp.dropout_rate,
                                                 causality=True))
            self.__setattr__('dec_vanilla_attention_%d' % i,
                             multihead_attention(num_units=self.hp.hidden_units,
                                                 num_heads=self.hp.num_heads,
                                                 dropout_rate=self.hp.dropout_rate,
                                                 causality=False))
            self.__setattr__('dec_feed_forward_%d' % i, feedforward(self.hp.hidden_units,
                                                                    [4 * self.hp.hidden_units,
                                                                     self.hp.hidden_units]))

    def forward(self, x, y, graph1):
        self.decoder_inputs = Variable(torch.ones(y[:, :1].size()).cuda() * 2).long()

        # Encoder
        self.enc = self.enc_emb(x)
        self.enc = self.glove_proj(self.enc)        
        self.enc = self.input_proj(self.enc)

        if self.hp.sinusoid:
            self.enc += self.enc_positional_encoding(x).cuda()
        else:
            self.enc += self.enc_positional_encoding(
                Variable(torch.unsqueeze(torch.arange(0, x.size()[1]), 0).repeat(x.size(0), 1).long().cuda()))
        self.enc = self.enc_dropout(self.enc)

        # Blocks
        for i in range(self.hp.num_blocks):
            self.enc = self.__getattr__('enc_self_attention_%d' % i)(self.enc, self.enc, self.enc, graph1)
            
            # Feed Forward
            self.enc = self.__getattr__('enc_feed_forward_%d' % i)(self.enc)
        # Decoder
        self.dec = self.dec_emb(self.decoder_inputs)

        # Positional Encoding
        if self.hp.sinusoid:
            self.dec += self.dec_positional_encoding(self.decoder_inputs).cuda()
        else:
            self.dec += self.dec_positional_encoding(
                Variable(torch.unsqueeze(torch.arange(0, self.decoder_inputs.size()[1]), 0).repeat(self.decoder_inputs.size(0), 1).long().cuda()))
        # Dropout
        self.dec = self.dec_dropout(self.dec)
        # Blocks
        for i in range(self.hp.num_blocks):
            # self-attention
            self.dec = self.__getattr__('dec_self_attention_%d' % i)(self.dec, self.dec, self.dec)
            # vanilla attention
            self.dec = self.__getattr__('dec_vanilla_attention_%d' % i)(self.dec, self.enc, self.enc)
            # feed forward
            self.dec = self.__getattr__('dec_feed_forward_%d' % i)(self.dec)

        return self.dec

class AttModel(nn.Module):
    def __init__(self, glove, hp_, enc_voc, dec_voc):
        super(AttModel, self).__init__()
        self.hp = hp_
        self.enc_voc = enc_voc
        self.dec_voc = dec_voc
        
        self.att_syb = AttModel_syb(glove, self.hp, self.enc_voc, self.dec_voc)
        self.att_vis = AttModel_vis(glove, self.hp, self.enc_voc, self.dec_voc)
        self.logits_layer = nn.Linear(self.hp.hidden_units*2, self.dec_voc)
        self.label_smoothing = label_smoothing()

    def forward(self, x_syb, x_vis, y, graph1_syb1, graph1_vis1, x0):
        #x_syb_batch, x_vis_batch, y_batch, g_syb_batch, g_vis_batch, x0_batch
        self.dec_syb = self.att_syb(x_syb, y, graph1_syb1)
        self.dec_vis = self.att_vis(x_vis, y, graph1_vis1, x0)
        
        self.dec = torch.cat((self.dec_syb, self.dec_vis), 2)
        
        # Final linear projection
        self.logits = self.logits_layer(self.dec)

        self.probs = F.softmax(self.logits, dim=-1).view(-1, self.dec_voc)

        _, self.preds = torch.max(self.logits, -1) 
        
        #self.istarget = (1. - y.eq(0.).float()).view(-1)
        self.istarget = torch.ones(y.size()).view(-1).cuda()
        self.acc = torch.sum(self.preds.eq(y).float().view(-1) * self.istarget) / torch.sum(self.istarget)
        #print (self.preds, y, '\n')

        # Loss
        self.y_onehot = torch.zeros(self.logits.size()[0] * self.logits.size()[1], self.dec_voc).cuda()
        self.y_onehot = Variable(self.y_onehot.scatter_(1, y.view(-1, 1).data, 1))

        self.y_smoothed = self.label_smoothing(self.y_onehot)

        self.loss = - torch.sum(self.y_smoothed * torch.log(self.probs), dim=-1)


        self.mean_loss = torch.sum(self.loss * self.istarget) / torch.sum(self.istarget)

        return self.mean_loss, self.preds, self.acc
