# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np
from modules import *
import pdb


class AttModel_vis_grid(nn.Module):
    def __init__(self, glove, hidden_size, maxlen, maxlen_q, num_blocks, num_heads, dropout_rate, maxlen_v,
                 num_classes):
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

        new_glove_voc = Parameter(torch.rand(407000, 300))
        nn.init.xavier_normal(new_glove_voc)
        new_glove_voc[:glove.vectors.shape[0], :] = glove.vectors
        self.enc_dropout = nn.Dropout(dropout_rate)
        self.dec_dropout = nn.Dropout(dropout_rate)
        self.syb_emb = nn.Embedding.from_pretrained(new_glove_voc, freeze=False)
        self.syb_mlp_sequence = nn.Sequential(nn.Linear(300, 2048),
                                     nn.ReLU(inplace=True),
                                     # nn.Linear(2048, hidden_size)
                                              )

        # self.syb_mlp = nn.Linear(300, 2048)
        self.syb_mlp2 = nn.Linear(2048, self.hidden_size)
        self.v_mlp = nn.Sequential(nn.Linear(2048, 2048),
                                   nn.ReLU(inplace=True))
        # self.v_mlp = nn.Sequential(nn.Linear(2048, self.hidden_size),
        #                            nn.ReLU(inplace=True),
        #                            nn.Linear(self.hidden_size, self.hidden_size))

        self.v_positional_encoding = nn.Sequential(embedding(maxlen_v, self.hidden_size, zeros_pad=False, scale=False),
                                                   nn.Dropout(self.dropout_rate))

        self.input_proj = nn.Linear(2048, self.hidden_size)

        for i in range(self.num_blocks):
            self.__setattr__('enc_self_attention_%d' % i, new_multihead_attention(num_units=self.hidden_size,
                                                                                  num_heads=self.num_heads,
                                                                                  dropout_rate=0,  # self.dropout_rate,
                                                                                  causality=False))
            self.__setattr__('enc_feed_forward_%d' % i, feedforward(self.hidden_size,
                                                                    [4 * self.hidden_size,
                                                                     self.hidden_size]))

        self.q_mlp = nn.Sequential(nn.Linear(300, self.hidden_size),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.hidden_size, self.hidden_size))

        self.q_positional_encoding = nn.Sequential(
            embedding(self.maxlen_q, self.hidden_size, zeros_pad=False, scale=False),
            nn.Dropout(self.dropout_rate))
        self.syb_positional_encoding = nn.Sequential(embedding(maxlen, hidden_size, zeros_pad=False, scale=False),
                                                     nn.Dropout(dropout_rate))
        # self.dec_emb = embedding(2, self.hidden_size, scale=True)
        # self.dec_positional_encoding = nn.Sequential(embedding(2, self.hidden_size, zeros_pad=False, scale=False),
        #                                     nn.Dropout(self.dropout_rate))
        self.dec_emb = embedding(num_classes, hidden_size, scale=True)
        self.dec_positional_encoding = embedding(maxlen, hidden_size, zeros_pad=False, scale=False)

        # for i in range(self.num_blocks):
        #     self.__setattr__('dec_vanilla_attention_%d' % i,
        #                      new_multihead_attention_with_graph_mask(num_units=self.hidden_size,
        #                                          num_heads=self.num_heads,
        #                                          dropout_rate=0,#self.dropout_rate,
        #                                          causality=False))
        #     self.__setattr__('dec_feed_forward_%d' % i, feedforward(self.hidden_size,
        #                                                             [4 * self.hidden_size,
        #                                                              self.hidden_size]))
        for i in range(self.num_blocks):
            self.__setattr__('dec_self_attention_%d' % i,
                             multihead_attention(num_units=self.hidden_size,
                                                 num_heads=self.num_heads,
                                                 dropout_rate=0,
                                                 causality=True))
            self.__setattr__('dec_vanilla_attention_%d' % i,
                             multihead_attention(num_units=self.hidden_size,
                                                 num_heads=self.num_heads,
                                                 dropout_rate=0,
                                                 causality=False))
            self.__setattr__('dec_feed_forward_%d' % i, feedforward(self.hidden_size,
                                                                    [4 * self.hidden_size,
                                                                     self.hidden_size]))

    def forward(self, vis_fea, vis_mask, vis_graph, q_fea, q_graph, q_mask):
        # vis_fea: bs x gridx x gridy x fea_size
        batch_size = vis_fea.size(0)
        if len(vis_fea.size()) == 4:
            vis_fea = vis_fea.reshape(-1, vis_fea.size(1) * vis_fea.size(2), vis_fea.size(3))

        vis_fea = self.v_mlp(vis_fea) #New
        # vis_enc_ipt = torch.unsqueeze(torch.arange(0, vis_fea.shape[1]), 0).repeat(vis_fea.shape[0], 1).long().cuda()
        # # vis_fea += self.v_positional_encoding(vis_enc_ipt)
        # vis_fea += self.syb_positional_encoding(vis_enc_ipt)

        # q_fea = self.q_mlp(q_fea)


        q_fea = self.syb_emb(q_fea)
        # q_fea = self.syb_mlp(q_fea)
        q_fea = self.syb_mlp_sequence(q_fea) #New


        # # q_fea += self.q_positional_encoding(
        # #         Variable(torch.unsqueeze(torch.arange(0, q_fea.shape[1]), 0).repeat(q_fea.shape[0], 1).long().cuda()))
        # q_fea += self.syb_positional_encoding(
        #         Variable(torch.unsqueeze(torch.arange(vis_fea.shape[1], vis_fea.shape[1]+q_fea.shape[1]), 0).repeat(q_fea.shape[0], 1).long().cuda()))

        fea_ipt = torch.cat([vis_fea, q_fea], dim=1)
        fea_ipt = self.syb_mlp2(fea_ipt)
        fea_ipt += self.syb_positional_encoding(
            Variable(torch.unsqueeze(torch.arange(0, fea_ipt.shape[1]), 0).repeat(fea_ipt.shape[0], 1).long().cuda()))
        fea_ipt = self.enc_dropout(fea_ipt)
        size_graph = vis_fea.shape[1] + q_graph.shape[1]
        mask = torch.zeros((vis_fea.shape[0], size_graph, size_graph)).cuda()
        graph_diag = torch.zeros((vis_fea.shape[0], size_graph, size_graph)).cuda()

        for id in range(mask.size(0)):
            mask[id] = torch.block_diag(vis_mask[id], q_mask[id])
            graph_diag[id, -q_mask[id].size(1):, -q_mask[id].size(1):] = q_mask[id]
        graph_cross = 1 - mask

        graph = graph_cross
        graph[:, :vis_fea.shape[1], :vis_fea.shape[1]] = vis_graph
        graph[:, vis_fea.shape[1]:, vis_fea.shape[1]:] = q_graph

        fea_att = fea_ipt

        # only consider each individual one.
        for i in range(2):
            fea_att = self.__getattr__('enc_self_attention_%d' % i)(fea_att, fea_att, fea_att, graph_diag)
            fea_att = self.__getattr__('enc_feed_forward_%d' % i)(fea_att)

        for i in range(2, 4):
            fea_att = self.__getattr__('enc_self_attention_%d' % i)(fea_att, fea_att, fea_att, graph_cross)
            fea_att = self.__getattr__('enc_feed_forward_%d' % i)(fea_att)

        fea_graph = fea_att
        for i in range(4, self.num_blocks):
            fea_graph = self.__getattr__('enc_self_attention_%d' % i)(fea_graph, fea_graph, fea_graph, graph)
            fea_graph = self.__getattr__('enc_feed_forward_%d' % i)(fea_graph)

        # fea_graph = fea_graph * mask

        decoder_inputs = Variable(torch.ones(batch_size, 1).cuda() * 2).long()
        # Decoder
        dec = self.dec_emb(decoder_inputs)
        dec += self.dec_positional_encoding(Variable(
            torch.unsqueeze(torch.arange(0, decoder_inputs.size()[1]), 0).repeat(decoder_inputs.size(0),
                                                                                 1).long().cuda()))
        dec = self.dec_dropout(dec)
        for i in range(self.num_blocks):
            # self-attention
            dec = self.__getattr__('dec_self_attention_%d' % i)(dec, dec, dec)
            # vanilla attention
            dec = self.__getattr__('dec_vanilla_attention_%d' % i)(dec, fea_graph, fea_graph)
            # feed forward
            dec = self.__getattr__('dec_feed_forward_%d' % i)(dec)

        return dec


class AttModel_syb(nn.Module):
    def __init__(self, glove, hidden_size, maxlen, maxlen_q, num_blocks, num_heads, dropout_rate, num_classes):
        super(AttModel_syb, self).__init__()

        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        new_glove_voc = Parameter(torch.rand(407000, 300))
        nn.init.xavier_normal(new_glove_voc)
        new_glove_voc[:glove.vectors.shape[0], :] = glove.vectors

        self.syb_emb = nn.Embedding.from_pretrained(new_glove_voc, freeze=False)
        self.syb_mlp = nn.Sequential(nn.Linear(300, 2048),
                                     # nn.ReLU(inplace = True),
                                     nn.Linear(2048, hidden_size))
        self.syb_mlp_sequence = nn.Sequential(nn.Linear(300, 2048),
                                     nn.ReLU(inplace = True),
                                     nn.Linear(2048, hidden_size))

        # self.syb_positional_encoding = nn.Sequential(embedding(maxlen, hidden_size, zeros_pad=False, scale=False),
        #                             nn.Dropout(dropout_rate))

        self.enc_dropout = nn.Dropout(dropout_rate)
        self.syb_positional_encoding = embedding(maxlen, hidden_size, zeros_pad=False, scale=False)

        self.q_mlp = nn.Sequential(nn.Linear(300, hidden_size),
                                   # nn.ReLU(inplace = True),
                                   nn.Linear(hidden_size, hidden_size))

        self.q_positional_encoding = nn.Sequential(embedding(maxlen_q, hidden_size, zeros_pad=False, scale=False),
                                                   nn.Dropout(dropout_rate))

        # self.dec_emb = embedding(2, self.hidden_size, scale=True)
        # self.dec_positional_encoding = nn.Sequential(embedding(2, self.hidden_size, zeros_pad=False, scale=False),
        #                                     nn.Dropout(dropout_rate))
        # self.dec_positional_encoding = embedding(2, self.hidden_size, zeros_pad=False, scale=False)
        self.dec_emb = embedding(num_classes, hidden_size, scale=True)
        self.dec_positional_encoding = embedding(maxlen, hidden_size, zeros_pad=False, scale=False)
        self.dec_dropout = nn.Dropout(dropout_rate)

        for i in range(self.num_blocks):
            self.__setattr__('dec_self_attention_%d' % i,
                             multihead_attention(num_units=self.hidden_size,
                                                 num_heads=self.num_heads,
                                                 dropout_rate=0,
                                                 causality=True))
            self.__setattr__('dec_vanilla_attention_%d' % i,
                             multihead_attention(num_units=self.hidden_size,
                                                 num_heads=self.num_heads,
                                                 dropout_rate=0,
                                                 causality=False))
            self.__setattr__('dec_feed_forward_%d' % i, feedforward(self.hidden_size,
                                                                    [4 * self.hidden_size,
                                                                     self.hidden_size]))

        for i in range(self.num_blocks):
            self.__setattr__('enc_self_attention_%d' % i, new_multihead_attention(num_units=hidden_size,
                                                                                  num_heads=self.num_heads,
                                                                                  dropout_rate=0,  # dropout_rate,
                                                                                  causality=False))
            self.__setattr__('enc_feed_forward_%d' % i, feedforward(hidden_size,
                                                                    [4 * hidden_size,
                                                                     hidden_size]))

    def forward(self, syb_ipt, syb_mask, syb_graph, q_fea, q_graph, q_mask):
        batch_size = syb_ipt.shape[0]

        # syb_fea = self.syb_emb(syb_ipt)
        # syb_fea = self.syb_mlp(syb_fea)
        # try:
        #     pos_ipt = torch.unsqueeze(torch.arange(0, syb_ipt.size(1)), 0).repeat(syb_ipt.size(0), 1).long().cuda()
        # except:
        #     pdb.set_trace()
        #     pass
        # pos_fea = self.syb_positional_encoding(pos_ipt)
        #
        # syb_fea += pos_fea
        # # Now the qeustion.
        # # q_fea = self.q_mlp(q_fea) #already Embedded
        # q_fea = self.syb_emb(q_fea)
        # q_fea = self.syb_mlp(q_fea)
        # # q_fea += self.q_positional_encoding(
        # #         Variable(torch.unsqueeze(torch.arange(0, q_fea.shape[1]), 0).repeat(q_fea.shape[0], 1).long().cuda()))
        # q_fea += self.syb_positional_encoding(
        #         Variable(torch.unsqueeze(torch.arange(syb_ipt.size(1), syb_ipt.size(1)+q_fea.shape[1]), 0).repeat(q_fea.shape[0], 1).long().cuda()))
        #
        # fea = torch.cat([syb_fea, q_fea], dim = 1)

        fea = torch.cat([syb_ipt, q_fea], dim=1)
        fea = self.syb_emb(fea)
        # fea = self.syb_mlp(fea)
        fea = self.syb_mlp_sequence(fea)
        try:
            pos_ipt = torch.unsqueeze(torch.arange(0, fea.size(1)), 0).repeat(fea.size(0), 1).long().cuda()
        except:
            pdb.set_trace()
            pass
        pos_fea = self.syb_positional_encoding(pos_ipt)
        fea += pos_fea
        fea = self.enc_dropout(fea)

        size_graph = syb_ipt.shape[1] + q_graph.shape[1]
        mask = torch.zeros((syb_ipt.shape[0], size_graph, size_graph)).cuda()
        graph_diag = torch.zeros((syb_ipt.shape[0], size_graph, size_graph)).cuda()

        for id in range(mask.size(0)):
            mask[id] = torch.block_diag(syb_mask[id], q_mask[id])
            graph_diag[id, -q_mask[id].size(1):, -q_mask[id].size(1):] = q_mask[id]
        graph_cross = 1 - mask

        graph = graph_cross
        graph[:, :syb_ipt.shape[1], :syb_ipt.shape[1]] = syb_graph
        graph[:, syb_ipt.shape[1]:, syb_ipt.shape[1]:] = q_graph
        fea_att = fea

        for i in range(2):
            fea_att = self.__getattr__('enc_self_attention_%d' % i)(fea_att, fea_att, fea_att, graph_diag)
            fea_att = self.__getattr__('enc_feed_forward_%d' % i)(fea_att)

        for i in range(2, 4):
            fea_att = self.__getattr__('enc_self_attention_%d' % i)(fea_att, fea_att, fea_att, graph_cross)
            fea_att = self.__getattr__('enc_feed_forward_%d' % i)(fea_att)

        fea_graph = fea_att

        for i in range(4, self.num_blocks):
            fea_graph = self.__getattr__('enc_self_attention_%d' % i)(fea_graph, fea_graph, fea_graph, graph)
            fea_graph = self.__getattr__('enc_feed_forward_%d' % i)(fea_graph)

        # fea = fea_graph * mask.unsqueeze(2)

        decoder_inputs = Variable(torch.ones(batch_size, 1).cuda() * 2).long()
        # Decoder
        dec = self.dec_emb(decoder_inputs)
        dec += self.dec_positional_encoding(Variable(
            torch.unsqueeze(torch.arange(0, decoder_inputs.size()[1]), 0).repeat(decoder_inputs.size(0),
                                                                                 1).long().cuda()))

        dec = self.dec_dropout(dec)
        for i in range(self.num_blocks):
            # self-attention
            dec = self.__getattr__('dec_self_attention_%d' % i)(dec, dec, dec)
            # vanilla attention
            dec = self.__getattr__('dec_vanilla_attention_%d' % i)(dec, fea_graph, fea_graph)
            # feed forward
            dec = self.__getattr__('dec_feed_forward_%d' % i)(dec)
        return dec


class CompactBilinearPooling(nn.Module):
    def __init__(self, input_dims, output_dim):
        super(CompactBilinearPooling, self).__init__()

        input_dim1 = input_dims
        input_dim2 = input_dims
        self.output_dim = output_dim
        generate_sketch_matrix = lambda rand_h, rand_s, input_dim, output_dim: torch.sparse.FloatTensor(
            torch.stack([torch.arange(input_dim, out=torch.LongTensor()), rand_h.long()]), rand_s.float(),
            [input_dim, output_dim]).to_dense()
        self.sketch1 = torch.nn.Parameter(generate_sketch_matrix(torch.randint(output_dim, size=(input_dim1,)),
                                                                 2 * torch.randint(2, size=(input_dim1,)) - 1,
                                                                 input_dim1, output_dim), requires_grad=False)
        self.sketch2 = torch.nn.Parameter(generate_sketch_matrix(torch.randint(output_dim, size=(input_dim2,)),
                                                                 2 * torch.randint(2, size=(input_dim2,)) - 1,
                                                                 input_dim2, output_dim), requires_grad=False)

    def forward(self, x1, x2):
        fft1 = torch.rfft(x1.matmul(self.sketch1), signal_ndim=1)
        fft2 = torch.rfft(x2.matmul(self.sketch2), signal_ndim=1)
        fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1],
                                   fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim=-1)
        cbp = torch.irfft(fft_product, signal_ndim=1, signal_sizes=(self.output_dim,)) * self.output_dim
        cbp_signed_sqrt = torch.sqrt(F.relu(cbp)) - torch.sqrt(F.relu(-cbp))
        cbp_norm = F.normalize(cbp_signed_sqrt, dim=0, p=2)
        return cbp_norm


class AttModel(nn.Module):
    def __init__(self, glove, hidden_size, num_classes, maxlen_q, maxlen, maxlen_v, num_blocks, num_heads, dropout_rate,
                 dropout_rate_mcb):
        super(AttModel, self).__init__()
        self.att_vis_grid = AttModel_vis_grid(glove, hidden_size, maxlen, maxlen_q, num_blocks, num_heads, dropout_rate,
                                              maxlen_v, num_classes)

        self.att_syb = AttModel_syb(glove, hidden_size, maxlen, maxlen_q, num_blocks, num_heads, dropout_rate,
                                    num_classes)
        self.cls = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size),
                                 nn.ReLU(),
                                 nn.Dropout(dropout_rate, inplace=True),
                                 nn.Linear(hidden_size, num_classes))
        # self.cls = nn.Linear(hidden_size * 2, num_classes)

        self.mcb_out = 16000
        self.mcb = CompactBilinearPooling(hidden_size, self.mcb_out)
        self.mcb_dropout = nn.Dropout(dropout_rate_mcb)
        self.cls_mcb = nn.Sequential(nn.Linear(self.mcb_out, hidden_size),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_rate, inplace=True),
                                     nn.Linear(hidden_size, num_classes))

        self.label_smoothing = label_smoothing()

        new_glove_voc = Parameter(torch.rand(407000, 300))
        nn.init.xavier_normal(new_glove_voc)
        new_glove_voc[:glove.vectors.shape[0], :] = glove.vectors

        self.q_emb = nn.Embedding.from_pretrained(new_glove_voc, freeze=False)

    def forward(self, vis_fea, vis_mask, q_ipt, q_mask, q_graph, syb_ipt, syb_mask, syb_graph, vis_graph):
        # q_fea = self.q_emb(q_ipt)
        q_fea = q_ipt

        fea_syb = self.att_syb(syb_ipt, syb_mask, syb_graph, q_fea, q_graph, q_mask)
        fea_vis_grid = self.att_vis_grid(vis_fea, vis_mask, vis_graph, q_fea, q_graph, q_mask)
        # fea = self.mcb(fea_syb, fea_vis_grid)
        # fea = self.mcb_dropout(fea)
        # logits = self.cls_mcb(fea).squeeze(1)
        # print (logits.size()) #64, 1, 914

        fea = torch.cat((fea_syb.squeeze(), fea_vis_grid.squeeze()), 1)
        logits = self.cls(fea)  # 64, 914
        # print (logits.size())
        return logits

