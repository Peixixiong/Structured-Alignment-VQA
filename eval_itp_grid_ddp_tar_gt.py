# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse

from torch.autograd import Variable
import os
from AttModel_x import AttModel
from AttModel_x2 import AttModel as AttModel_v2
from AttModel_x2_dec import AttModel as AttModel_v2_dec
from AttModel_x3 import AttModel as AttModel_v3
import torch
import codecs

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from data_loader_itp_gt import GQADataset, collate_fn, load_answer_vocab
import time

import pickle
import numpy as np
from torchtext import vocab

import logging
import datetime
from misc import AverageMeter

import tarfile

import pdb
from modules import label_smoothing

def eval(model, data_loader_val, log_step, rank, with_smooth_labeling):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    lbl_sm = label_smoothing()
    with torch.no_grad():
        cnt = 0
        cnt_correct = 0
        loss_meter = AverageMeter()
        for iidx, data in enumerate(data_loader_val):

            batch_size = data['vis_fea'].shape[0]
            vis_fea = data['vis_fea'].cuda()
            vis_syb_ipt = data['vis_syb_ipt'].cuda()
            vis_syb_mask = data['vis_syb_mask'].cuda()
            vis_syb_graph = data['vis_syb_graph'].cuda()

            q_ipt = data['q_ipt'].cuda()
            q_ipt_mask = data['q_ipt_mask'].cuda()
            q_ipt_graph = data['q_ipt_graph'].cuda()

            answer = data['answer'].cuda()


            logits = model(vis_fea, q_ipt, q_ipt_mask, q_ipt_graph, vis_syb_ipt,  \
                   vis_syb_mask, vis_syb_graph)
            
            _, pred_lbls = torch.max(logits, dim = 1) 
            cnt_correct += (pred_lbls == answer).long().sum().item()
            cnt += batch_size
            if with_smooth_labeling:
                log_softmax = F.log_softmax(logits, -1)
                one_hot_answer = torch.zeros((logits.size(0), logits.size(1))).cuda()
                one_hot_answer.scatter_(1, answer.view(-1, 1), 1)
                one_hot_answer = lbl_sm(one_hot_answer)
                loss = -(one_hot_answer * log_softmax).sum(-1)
                loss = loss.mean()
            else:
                loss = criterion(logits, answer)
            loss_meter.update(loss.cpu().item(), batch_size)
            if rank == 0 and (iidx + 1 ) % log_step == 0:
                print('Time {}, Step [{}/{}], Loss: {}, Avg Loss: {}'.format(datetime.datetime.now(), iidx, len(data_loader_val), loss.item(), loss_meter.avg))
                logging.info('Time {}, Step [{}/{}], Loss: {}, Avg Loss: {}'.format(datetime.datetime.now(), iidx, len(data_loader_val), loss.item(), loss_meter.avg))
        return loss_meter.avg, cnt_correct, cnt

def main(gpu_rank, args):

    rank = gpu_rank
    torch.cuda.set_device(gpu_rank)
    
    cache_dir = os.path.join(args.data_dir_azure, 'vector_cache')
    if not os.path.isdir(cache_dir):
        cache_dir = '.vector_cache'

    w2idx, idx2w = load_answer_vocab( os.path.join(args.data_dir_azure, args.ans_vocab_fn), args.min_cnt)
    args.num_classes = len(w2idx) + 1 # all other class

    glove = vocab.GloVe(name='6B', dim=300, cache = cache_dir)

    print('Usage model version:{}'.format(args.model_v))
    if args.model_v == 1:
        model = AttModel(glove, args.hidden_size, args.num_classes, args.maxlen_q, args.maxlen, args.num_blocks, args.num_heads, args.dropout_rate)
    elif args.model_v == 2:
        print('args.model_v ==', args.model_v)
        if args.with_dec:
            print('Using with decoder.------------')
            model = AttModel_v2_dec(glove, args.hidden_size, args.num_classes, args.maxlen_q, args.maxlen, args.num_blocks, args.num_heads, args.dropout_rate)
        else:
            model = AttModel_v2(glove, args.hidden_size, args.num_classes, args.maxlen_q, args.maxlen, args.num_blocks, args.num_heads, args.dropout_rate)
    elif args.model_v == 3:
        model = AttModel_v3(glove, args.hidden_size, args.num_classes, args.maxlen_q, args.maxlen, args.num_blocks, args.num_heads, args.dropout_rate, args.pool)
    dict_weights = {}
    old_dict = torch.load(args.weight_fn)
    for key in old_dict:
        if key.startswith('module'):
            nkey = key[7:]
        else:
            nkey = key
        dict_weights[nkey] =  old_dict[key]

    model.load_state_dict(dict_weights)
    model.cuda()

    val_ds = GQADataset(args, args.fea_tar_fn_val, args.q_tar_fn_val, args.gt_graph_fn_val, args.with_loc) 

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        collate_fn = collate_fn
    )
    criterion = nn.CrossEntropyLoss()

    lbl_sm = label_smoothing()

    loss_val, cnt_correct, cnt= eval(model, val_loader, args.log_steps_val, rank, args.with_smooth_labeling)

    print('Time {}, Val Loss: {}, accuracy: {}/{} = {}'.format(datetime.datetime.now(), loss_val, cnt_correct, cnt, cnt_correct/cnt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data related setting for azure and the input
    parser.add_argument('--weight_fn', type = str)
    parser.add_argument('--data_dir_azure', type = str, default= os.environ.get('PT_DATA_DIR','./tmp'))

    parser.add_argument('--fea_tar_fn_val', default='spatial_npz_fea.tar')
    parser.add_argument('--q_tar_fn_val', default='val.tar')
    parser.add_argument('--gt_graph_fn_val', default='val_sceneGraphs.json')

    parser.add_argument('--gt_relation_fn', default='GT_relations_dict.json')
    parser.add_argument('--obj_vocab_fn', type = str, default='objects_vocab.txt')
    parser.add_argument('--attr_vocab_fn', type = str, default='attributes_vocab.txt')
    parser.add_argument('--bbox_bin_num', type = int, default = 64)
    parser.add_argument('--enc_vocab_fn', type = str, default = 'preprocessed/de.vocab.tsv')
    parser.add_argument('--ans_vocab_fn', type = str, default = 'preprocessed/en.vocab.tsv')

    parser.add_argument('--batch_size', type = int,  default = 256)
    parser.add_argument('--lr', type = float,  default = 0.0001)

    parser.add_argument('--output_dir', type = str,  default = os.environ.get('PT_OUTPUT_DIR', './tmp'))

    parser.add_argument('--maxlen', type = int,  default = 300)
    parser.add_argument('--maxlen_q', type = int,  default = 50)
    parser.add_argument('--hidden_size', type = int,  default = 512)
    parser.add_argument('--num_blocks', type = int,  default = 6)
    parser.add_argument('--num_epochs', type = int,  default = 40)
    parser.add_argument('--num_heads', type = int,  default = 8)
    parser.add_argument('--min_cnt', type = int,  default = 10)
    parser.add_argument('--dropout_rate', type = int,  default = 0.5)
    parser.add_argument('--sinusoid', action = 'store_true')
    parser.add_argument('--with_dec', action = 'store_true')
    parser.add_argument('--with_loc', action = 'store_true')
    parser.add_argument('--with_smooth_labeling', action = 'store_true')

    parser.add_argument('--log_steps', type=int, default = 100)
    parser.add_argument('--log_steps_val', type=int, default = 100)

    parser.add_argument('--model_v', type=int, default = 1)
    parser.add_argument('--pool', type=str, default = 'mean')

    #distributed
    parser.add_argument('--ngpus', help='Number of epochs', type=int, default=-1)
    parser.add_argument('--nr', help='rank of node', type=int, default=0)
    parser.add_argument('--num_nodes', help='rank of node', type=int, default=1)
    parser.add_argument('--num_workers', type = int,  default = 4)

    args = parser.parse_args()
    if args.ngpus == -1:
        args.ngpus = torch.cuda.device_count()

    args.world_size = args.ngpus * args.num_nodes
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.world_size = 1
    args.ngpus = 1
    main(0, args) 
