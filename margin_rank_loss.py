from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

#class ATTMILLoss(nn.Module):
#    def __init__(self, margin = 0.3):
#        super(ATTMILLoss, self).__init__()
#
#        self.mr_loss = nn.MarginRankingLoss(margin)
#
#    
#    def forward(self, idx_of_objs, syb_graph, att_weights, vis_len):
#        blocks = len(att_weights)
#        batch_size = syb_graph.shape[0]
#        keep_weights = []
#        pos_ws = []
#        neg_ws = []
#
#        for b in range(blocks):
#            for i in range(batch_size):
#                syb_w = att_weights[b][i][vis_len:, vis_len:] 
#                for j in range(idx_of_objs[i].size(0)):
#                    att_row = syb_w[idx_of_objs[i][j],:]
#                    idx_row = syb_graph[i][idx_of_objs[i][j],:]
#                    pos_w = att_row[idx_row > 0].sum()
#                    neg_w = att_row[idx_row == 0].sum()
#
#                    pos_ws.append(pos_w)
#                    neg_ws.append(neg_w)
#
#        pos_ws = torch.stack(pos_ws, dim = 0)
#        neg_ws = torch.stack(neg_ws, dim = 0)
#
#        idx = torch.ones((neg_ws.shape[0],)).cuda()
#        loss = self.mr_loss(pos_ws, neg_ws, idx)
#         
#        return loss

class ATTMILLoss(nn.Module):
    def __init__(self, margin = 0.6):
        super(ATTMILLoss, self).__init__()

        self.mr_loss = nn.MarginRankingLoss(margin)


    def forward(self, idx_of_objs, valid2all, syb_graph, att_weights, vis_len):
        blocks = len(att_weights) #1 blocks  [1, 256, 73, 199]
        batch_size = syb_graph.shape[0]
        keep_weights = []
        # pos_ws = []
        # neg_ws = []
        id = 0
        pos_ws = torch.zeros(blocks*batch_size*syb_graph[0].size(0), ).cuda()
        neg_ws = torch.zeros(blocks * batch_size * syb_graph[0].size(0), ).cuda()
        for b in range(blocks):
            for i in range(batch_size):
                vis_w = att_weights[b][i] #[73, 199]
                for j in range(vis_w.size(0)): #vis
                    if j in valid2all[i]:
                        j_ = (valid2all[i] == j).nonzero()[0][0]
                        att_row = vis_w[j_,:] #input length and feature dim
                        idx_row = syb_graph[i][idx_of_objs[i][j_],:]
                        pos_w = att_row[idx_row > 0].sum()
                        neg_w = att_row[idx_row == 0].sum()

                        pos_ws[id] = pos_w
                        neg_ws[id] = neg_w
                        id += 1
                    # pos_ws.append(pos_w)
                    # neg_ws.append(neg_w)

        # pos_ws = torch.stack(pos_ws, dim = 0)
        # neg_ws = torch.stack(neg_ws, dim = 0)

        idx = torch.ones((neg_ws.shape[0],)).cuda()
        loss = self.mr_loss(pos_ws, neg_ws, idx)

        return loss

# class ATTMILLoss(nn.Module):
#     def __init__(self, margin=0.6):
#         super(ATTMILLoss, self).__init__()
#
#         self.mr_loss = nn.MarginRankingLoss(margin)
#
#     def forward(self, vis_graph, att_weights, vis_len):
#         blocks = len(att_weights)  # 6 blocks [1, 256, 73, 199]
#         batch_size = vis_graph.shape[0]
#         keep_weights = []
#         # pos_ws = []
#         # neg_ws = []
#
#         pos_ws = torch.zeros(blocks*batch_size*vis_graph[0].size(0), ).cuda()
#         neg_ws = torch.zeros(blocks * batch_size * vis_graph[0].size(0), ).cuda()
#         id = 0
#         for b in range(blocks):
#             for i in range(batch_size):
#                 vis_w = att_weights[b][i] #[73, 199]
#                 print ('1', vis_w.size())
#                 print ('2', vis_graph[i].size())
#                 vis_w = vis_w[:vis_graph[i].size(0),:vis_graph[i].size(0)]
#                 for j in range(vis_w.size(0)):  # vis
#                     att_row = vis_w[j, :]  # input length and feature dim
#                     idx_row = vis_graph[i][j, :]
#                     pos_w = att_row[idx_row > 0].sum()
#                     neg_w = att_row[idx_row == 0].sum()
#                     pos_ws[id] = pos_w
#                     neg_ws[id] = neg_w
#                     id += 1
#                     # pos_ws.append(pos_w)
#                     # neg_ws.append(neg_w)
#
#         # pos_ws = torch.stack(pos_ws, dim=0)
#         # neg_ws = torch.stack(neg_ws, dim=0)
#
#         idx = torch.ones((neg_ws.shape[0],)).cuda()
#         loss = self.mr_loss(pos_ws, neg_ws, idx)
#
#         return loss