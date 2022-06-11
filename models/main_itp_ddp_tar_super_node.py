# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
from azureml.core import Run
from torch.autograd import Variable
import os
from AttModel_x3 import AttModel as AttModel_v3
import torch
import codecs

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import warnings

from data_loader_itp_bbox_super_node import GQADataset_super_node as GQADataset_bbox_super_node
from data_loader_itp_bbox_super_node import collate_fn as collate_fn_bbox_super_node
from data_loader_itp_bbox_super_node import *
from data_loader_itp_bbox_super_node_onlyobj import GQADataset_super_node as GQADataset_bbox_super_node_onlyobj
from data_loader_itp_bbox_super_node_onlyobj import collate_fn as collate_fn_bbox_super_node_onlyobj
# from data_loader_itp_bbox_super_node_onlyobj import *

import time

import pickle
import numpy as np
from torchtext import vocab

import logging
import datetime
from misc import AverageMeter

from margin_rank_loss import ATTMILLoss

import tarfile

import pdb
from modules import label_smoothing
def eval(model, data_loader_val, log_step, rank, with_smooth_labeling, args):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    criterion_mil = ATTMILLoss()
    lbl_sm = label_smoothing()
    with torch.no_grad():
        cnt = 0
        cnt_correct = 0
        loss_meter = AverageMeter()
        loss_rank_meter = AverageMeter()

        for iidx, data in enumerate(data_loader_val):
            batch_size = data['vis_fea'].shape[0]
            vis_fea = data['vis_fea'].cuda()
            if 'vis_fea_mask' in data:
                vis_fea_mask = data['vis_fea_mask'].cuda()
            else:
                vis_fea_mask = torch.empty((args.batch_size, 0))

            macro_node_ipt = data['macro_node_ipt'].cuda()
            macro_node_mask = data['macro_node_mask'].cuda()
            macro_graph_ipt = data['macro_graph_ipt'].cuda()

            macro_obj_loc_ipt = data['macro_obj_loc_ipt'].cuda()
            # macro_obj_loc_mask = data['macro_obj_loc_mask'].cuda()
            # macro_rel_loc_ipt = data['macro_rel_loc_ipt'].cuda()
            # macro_rel_loc_mask = data['macro_rel_loc_mask'].cuda()

            micro_positive_obj_ipt = data['micro_positive_obj_ipt'].cuda()
            micro_negative_obj_ipt = data['micro_negative_obj_ipt'].cuda()
            micro_obj_mask = data['micro_obj_mask'].cuda()

            if 'micro_positive_rel_ipt' in data:
                micro_positive_rel_ipt = data['micro_positive_rel_ipt'].cuda()
            else:
                micro_positive_rel_ipt = torch.empty((args.batch_size, 0))

            if 'micro_negative_rel_ipt' in data:
                micro_negative_rel_ipt = data['micro_negative_rel_ipt'].cuda()
            else:
                micro_negative_rel_ipt = torch.empty((args.batch_size, 0))

            if 'micro_positive_rel_loc' in data:
                micro_positive_rel_loc = data['micro_positive_rel_loc'].cuda()
            else:
                micro_positive_rel_loc = torch.empty((args.batch_size, 0))

            if 'micro_negative_rel_loc' in data:
                micro_negative_rel_loc = data['micro_negative_rel_loc'].cuda()
            else:
                micro_negative_rel_loc = torch.empty((args.batch_size, 0))
            # micro_rel_mask = data['micro_rel_mask'].cuda()

            q_ipt = data['q_ipt'].cuda()
            q_ipt_mask = data['q_ipt_mask'].cuda()
            q_ipt_graph = data['q_ipt_graph'].cuda()

            answer = data['answer'].cuda()

            if args.model_v == 3:
                # idx_of_obj = [ idx.cuda() for idx in data['idx_of_obj'] ]
                logits_concat, logits_vis, logits_syb, mil_nce_obj, mil_nce_rel = model(vis_fea, vis_fea_mask, q_ipt,q_ipt_mask, q_ipt_graph, \
                                                                                        macro_node_ipt, macro_node_mask,macro_graph_ipt,macro_obj_loc_ipt, \
                                                                                        micro_positive_obj_ipt,micro_negative_obj_ipt,micro_obj_mask,micro_positive_rel_ipt, \
                                                                                        micro_negative_rel_ipt, micro_positive_rel_loc, micro_negative_rel_loc,\
                                                                                        decMask=args.decMask, mcb=args.mcb)
                if args.only_obj == False:
                    mil_nce_loss = -mil_nce_obj - mil_nce_rel
                else:
                    mil_nce_loss = -mil_nce_obj
            if args.model_v == 3:
                log_softmax_vis = F.log_softmax(logits_vis, -1)
                log_softmax_syb = F.log_softmax(logits_syb, -1)
                log_softmax_concat = F.log_softmax(logits_concat, -1)

                log_softmax = (log_softmax_vis + log_softmax_syb + log_softmax_concat) / 3
                one_hot_answer = torch.zeros((logits_concat.size(0), logits_concat.size(1))).cuda()
                one_hot_answer.scatter_(1, answer.view(-1, 1), 1)
                one_hot_answer = lbl_sm(one_hot_answer)

                loss = -(one_hot_answer * log_softmax).sum(-1)
                loss = loss.mean()

                _, pred_lbls = torch.max(log_softmax, dim=1)
                cnt_correct += (pred_lbls[torch.nonzero(answer)] == answer[torch.nonzero(answer)]).long().sum().item()
                cnt += batch_size

            if args.model_v == 3:
                if args.with_MILNCE_loss:
                    loss += mil_nce_loss
                loss_rank_meter.update(mil_nce_loss.cpu().item(), batch_size)
            loss_meter.update(loss.cpu().item(), batch_size)

            if rank == 0 and (iidx + 1 ) % log_step == 0:
                if args.model_v == 3:
                    print('Time {}, Step [{}/{}], Loss: {}, MIL NCE Loss:{}, Avg Loss: {}, Avg MILNCE_loss: {}'.format(datetime.datetime.now(), iidx, len(data_loader_val), loss.item(), mil_nce_loss.item(), loss_meter.avg, loss_rank_meter.avg))
                    logging.info('Time {}, Step [{}/{}], Loss: {}, MIL NCE Loss:{}, Avg Loss: {}, Avg MILNCE_loss: {}'.format(datetime.datetime.now(), iidx, len(data_loader_val), loss.item(), mil_nce_loss.item(), loss_meter.avg, loss_rank_meter.avg))
                else:
                    print('Time {}, Step [{}/{}], Loss: {}, Avg Loss: {}'.format(datetime.datetime.now(), iidx, len(data_loader_val), loss.item(), loss_meter.avg))
                    logging.info('Time {}, Step [{}/{}], Loss: {}, Avg Loss: {}'.format(datetime.datetime.now(), iidx, len(data_loader_val), loss.item(), loss_meter.avg))
        return loss_meter.avg, cnt_correct, cnt

def main(gpu_rank, args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,5,6,7"
    torch.autograd.set_detect_anomaly(True)
    run = Run.get_context()
    print('ngpus:', args.ngpus)

    rank = gpu_rank
    torch.cuda.set_device(gpu_rank)
    
    if args.world_size > 1:
        dist.init_process_group(
            backend = 'nccl',
            init_method = 'tcp://{}:{}'.format(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']),
            world_size = args.world_size,
            rank = rank)

    if rank == 0:
        logging.basicConfig(level=logging.INFO,
                                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                                                        datefmt='%m-%d %H:%M',
                                                        filename='{0}/training.log'.format(os.path.join(args.data_dir_azure, args.output_dir)),
                                                        filemode='w')
        
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter( "%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        print(args)

    current_batches = 0
    

    cache_dir = os.path.join(args.data_dir_azure, 'vector_cache')
    if not os.path.isdir(cache_dir):
        cache_dir = '.vector_cache'

    w2idx, idx2w = load_answer_vocab(os.path.join(args.data_dir_azure, args.ans_vocab_fn), args.min_cnt)
    args.num_classes = len(w2idx) + 1 # all other class
    with open(os.path.join(args.data_dir_azure, args.obj_vocab_fn)) as fid:
        args.bg_class = len(fid.readlines()) + 1
    print('bg_class:', args.bg_class)

    glove = vocab.GloVe(name='6B', dim=300, cache = cache_dir)

    print('Usage model version:{}'.format(args.model_v))
    if args.model_v == 3:
        gt_relation_fn = os.path.join(args.data_dir_azure, args.gt_relation_fn)
        gt_relations = json.load(open(gt_relation_fn, 'r'))
        gt_relation_clean = list(set(gt_relations.values()))  # No redundant rel
        num_relations = len(gt_relation_clean) + 1 #Include 'no relation'
        print ('Total relation number (no redundant, including __no_relation__) is ', num_relations)
        model = AttModel_v3(glove, args.hidden_size, args.hidden_size_mil, args.num_classes, args.maxlen_q, args.maxlen, args.maxlen_v,
                                args.num_blocks, args.num_heads, args.dropout_rate, args.dropout_rate_mcb, num_relations, args.only_obj)
 
    model.train()
    model.cuda()
    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [gpu_rank], find_unused_parameters=True)
    

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    if args.only_obj:
        GQADataset = GQADataset_bbox_super_node_onlyobj
        collate_fn = collate_fn_bbox_super_node_onlyobj
    else:
        GQADataset = GQADataset_bbox_super_node
        collate_fn = collate_fn_bbox_super_node


    # else:
    #     GQADataset = GQADataset_grid
    #     collate_fn = collate_fn_grid

    #setup dataset
    train_ds = GQADataset('train', args, args.fea_tar_fn_train, args.q_tar_fn_train, args.g_tar_fn_train, args.topN, args.with_loc)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        num_replicas = args.world_size,
        rank = rank)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        drop_last = True,
        collate_fn = collate_fn,
        sampler = train_sampler
    )
    # val dataset
    val_ds = GQADataset('val', args, args.fea_tar_fn_val, args.q_tar_fn_val, args.g_tar_fn_val, args.topN, args.with_loc)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_ds,
        num_replicas = args.world_size,
        rank = rank) 

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        drop_last = True,
        collate_fn = collate_fn,
        sampler = val_sampler
    )
    if rank == 0:
        print('Train loader {}, val_loader {}'.format(len(train_loader), len(val_loader)))

    criterion = nn.CrossEntropyLoss()
    criterion_mil = ATTMILLoss()

    best_loss = 1e10

    lbl_sm = label_smoothing()

    for epoch in range(args.num_epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        loss_meter = AverageMeter()

        loss_rank_meter = AverageMeter()

        for i, data in enumerate(train_loader):
            batch_size = data['vis_fea'].shape[0]
            if batch_size != 0:
                vis_fea = data['vis_fea'].cuda()
                if 'vis_fea_mask' in data:
                    vis_fea_mask = data['vis_fea_mask'].cuda()
                else:
                    vis_fea_mask = torch.empty((args.batch_size, 0))

                macro_node_ipt = data['macro_node_ipt'].cuda()
                macro_node_mask = data['macro_node_mask'].cuda()
                macro_graph_ipt = data['macro_graph_ipt'].cuda()

                macro_obj_loc_ipt = data['macro_obj_loc_ipt'].cuda()
                # macro_obj_loc_mask = data['macro_obj_loc_mask'].cuda()
                # macro_rel_loc_ipt = data['macro_rel_loc_ipt'].cuda()
                # macro_rel_loc_mask = data['macro_rel_loc_mask'].cuda()

                micro_positive_obj_ipt = data['micro_positive_obj_ipt'].cuda()
                micro_negative_obj_ipt = data['micro_negative_obj_ipt'].cuda()
                micro_obj_mask = data['micro_obj_mask'].cuda()

                if 'micro_positive_rel_ipt' in data:
                    micro_positive_rel_ipt = data['micro_positive_rel_ipt'].cuda()
                else:
                    micro_positive_rel_ipt = torch.empty((args.batch_size, 0))

                if 'micro_negative_rel_ipt' in data:
                    micro_negative_rel_ipt = data['micro_negative_rel_ipt'].cuda()
                else:
                    micro_negative_rel_ipt = torch.empty((args.batch_size, 0))

                if 'micro_positive_rel_loc' in data:
                    micro_positive_rel_loc = data['micro_positive_rel_loc'].cuda()
                else:
                    micro_positive_rel_loc = torch.empty((args.batch_size, 0))

                if 'micro_negative_rel_loc' in data:
                    micro_negative_rel_loc = data['micro_negative_rel_loc'].cuda()
                else:
                    micro_negative_rel_loc = torch.empty((args.batch_size, 0))

                # micro_rel_mask = data['micro_rel_mask'].cuda()

                q_ipt = data['q_ipt'].cuda()
                q_ipt_mask = data['q_ipt_mask'].cuda()
                q_ipt_graph = data['q_ipt_graph'].cuda()

                answer = data['answer'].cuda()

                if args.model_v == 3:
                    # idx_of_obj = [ idx.cuda() for idx in data['idx_of_obj'] ]
                    # os.system('nvidia-smi')
                    logits_concat, logits_vis, logits_syb, mil_nce_obj, mil_nce_rel = model(vis_fea, vis_fea_mask, q_ipt, q_ipt_mask, q_ipt_graph, \
                                                                  macro_node_ipt, macro_node_mask, macro_graph_ipt, macro_obj_loc_ipt, \
                                                                  micro_positive_obj_ipt, micro_negative_obj_ipt, micro_obj_mask, micro_positive_rel_ipt, \
                                                                  micro_negative_rel_ipt, micro_positive_rel_loc, micro_negative_rel_loc,\
                                                                                            decMask=args.decMask,mcb=args.mcb)
                    if args.only_obj == False:
                        mil_nce_loss = -mil_nce_obj - mil_nce_rel
                    else:
                        mil_nce_loss = -mil_nce_obj


                optimizer.zero_grad()

                if args.model_v == 3:
                    log_softmax_vis = F.log_softmax(logits_vis, -1)
                    log_softmax_syb = F.log_softmax(logits_syb, -1)
                    log_softmax_concat = F.log_softmax(logits_concat, -1)

                    log_softmax = (log_softmax_vis + log_softmax_syb + log_softmax_concat) / 3
                    one_hot_answer = torch.zeros((logits_concat.size(0), logits_concat.size(1))).cuda()
                    one_hot_answer.scatter_(1, answer.view(-1, 1), 1)
                    one_hot_answer = lbl_sm(one_hot_answer)

                    loss = -(one_hot_answer * log_softmax).sum(-1)
                    loss = loss.mean()
                # else:
                #     if args.with_smooth_labeling:
                #         log_softmax = F.log_softmax(logits, -1)
                #         one_hot_answer = torch.zeros((logits.size(0), logits.size(1))).cuda()
                #         one_hot_answer.scatter_(1, answer.view(-1, 1), 1)
                #         one_hot_answer = lbl_sm(one_hot_answer)
                #         loss = -(one_hot_answer * log_softmax).sum(-1)
                #         loss = loss.mean()
                #     else:
                #         loss = criterion(logits, answer)


                if args.model_v == 3:
                    if args.with_MILNCE_loss:
                        loss += mil_nce_loss
                    loss_rank_meter.update(mil_nce_loss.cpu().item(), batch_size)

                loss.backward()
                # if i == 0 and rank == 0:
                #     os.system('nvidia-smi')
                optimizer.step()

                loss_meter.update(loss.cpu().item(), batch_size)
                if rank == 0 and (i + 1) % args.log_steps == 0:
                    if args.model_v == 3:
                        print('Time {}, Epoch [{}/{}], Step [{}/{}], Loss: {}, MIL NCE Loss: {}, Avg Loss: {}, Avg MILNCE_loss: {}'.format(datetime.datetime.now(), epoch + 1, args.num_epochs + 1, i+1, len(train_loader), loss.item(), mil_nce_loss.item(), loss_meter.avg, loss_rank_meter.avg))
                        run.log('Avg Loss', np.float(loss_meter.avg))
                        run.log('Avg MILNCE_loss', np.float(loss_rank_meter.avg))
                        logging.info('Time {}, Epoch [{}/{}], Step [{}/{}], Loss: {}, MIL NCE Loss: {}, Avg Loss: {}, Avg MILNCE_loss: {}'.format(datetime.datetime.now(), epoch + 1, args.num_epochs + 1, i+1, len(train_loader), loss.item(), mil_nce_loss.item(), loss_meter.avg, loss_rank_meter.avg))
                    else:
                        print('Time {}, Epoch [{}/{}], Step [{}/{}], Loss: {}, Avg Loss: {}'.format(datetime.datetime.now(), epoch + 1, args.num_epochs + 1, i+1, len(train_loader), loss.item(), loss_meter.avg))
                        run.log('Avg Loss', np.float(loss_meter.avg))
                        logging.info('Time {}, Epoch [{}/{}], Step [{}/{}], Loss: {}, Avg Loss: {}'.format(datetime.datetime.now(), epoch + 1, args.num_epochs + 1, i+1, len(train_loader), loss.item(), loss_meter.avg))

        loss_val_scalar, cnt_correct_scalar, cnt_scalar = eval(model, val_loader, args.log_steps_val, rank, args.with_smooth_labeling, args)
        loss_train_scalar, cnt_correct_scalar_train, cnt_scalar_train = eval(model, train_loader, args.log_steps, rank,
                                                               args.with_smooth_labeling, args)
        tensor_vals = torch.zeros(3).float().cuda()
        tensor_vals[0] = loss_val_scalar
        tensor_vals[1] = cnt_correct_scalar
        tensor_vals[2] = cnt_scalar
        gather_vals = [ torch.zeros(3).float().cuda() for i in range(dist.get_world_size())]
        dist.all_gather(gather_vals, tensor_vals)
        avg = torch.stack(gather_vals)
        loss_val = avg[:,0].mean().item()
        cnt_correct = avg[:,1].sum().item()
        cnt = avg[:,2].sum().item()


        tensor_trains = torch.zeros(3).float().cuda()
        tensor_trains[0] = loss_train_scalar
        tensor_trains[1] = cnt_correct_scalar_train
        tensor_trains[2] = cnt_scalar_train
        gather_trains = [ torch.zeros(3).float().cuda() for i in range(dist.get_world_size())]
        dist.all_gather(gather_trains, tensor_trains)
        avg_train = torch.stack(gather_trains)
        loss_train = avg_train[:,0].mean().item()
        cnt_correct_train = avg_train[:,1].sum().item()
        cnt_train = avg_train[:,2].sum().item()

        if rank == 0: 
            print('Time {}, Epoch [{}/{}], Val Loss: {}, accuracy: {}/{} = {}'.format(datetime.datetime.now(), epoch + 1, args.num_epochs + 1, loss_val, cnt_correct, cnt, cnt_correct/cnt))
            logging.info('Time {}, Epoch [{}/{}], Val Loss: {}, accuracy: {}/{} = {}'.format(datetime.datetime.now(), epoch + 1, args.num_epochs + 1, loss_val, cnt_correct, cnt, cnt_correct/cnt))

            print(
                'Time {}, Epoch [{}/{}], Train Loss: {}, accuracy: {}/{} = {}'.format(datetime.datetime.now(), epoch + 1,
                                                                                    args.num_epochs + 1, loss_train,
                                                                                    cnt_correct_train, cnt_train,
                                                                                    cnt_correct_train / cnt_train))
            logging.info(
                'Time {}, Epoch [{}/{}], Train Loss: {}, accuracy: {}/{} = {}'.format(datetime.datetime.now(), epoch + 1,
                                                                                    args.num_epochs + 1, loss_train,
                                                                                    cnt_correct_train, cnt_train,
                                                                                    cnt_correct_train / cnt_train))
            train_acc = cnt_correct_train / cnt_train
            test_acc = cnt_correct / cnt
            run.log('Test Acc', np.float(test_acc))
            run.log('Train Acc', np.float(train_acc))
            # if best_loss > loss_val:
            if 1:
                best_loss = loss_val
                checkpoint_path = os.path.join(os.path.join(args.data_dir_azure, args.output_dir), 'model_' +  str(epoch + 1) + '.pth')
                torch.save(model.state_dict(), checkpoint_path)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    # data related setting for azure and the input
    parser.add_argument('--data_dir_azure', type = str, default= os.environ.get('PT_DATA_DIR','./tmp'))

    parser.add_argument('--fea_tar_fn_train', default='gt_bua_npz.tar')
    parser.add_argument('--q_tar_fn_train', default='train.tar')
    parser.add_argument('--g_tar_fn_train', default='gt_bua_npz.tar')  #'bua_npz_topk.tar'

    parser.add_argument('--fea_tar_fn_val', default='gt_bua_npz.tar')
    parser.add_argument('--q_tar_fn_val', default='val.tar') #'val.tar' 'test2.tar
    parser.add_argument('--g_tar_fn_val', default='gt_bua_npz.tar')  #'bua_npz_topk.tar'

    parser.add_argument('--gt_relation_fn', default='GT_relations_dict_compsite.json')
    parser.add_argument('--obj_vocab_fn', type = str, default='objects_vocab.txt')
    parser.add_argument('--attr_vocab_fn', type = str, default='attributes_vocab.txt')
    parser.add_argument('--bbox_bin_num', type = int, default = 64)
    parser.add_argument('--enc_vocab_fn', type = str, default = 'preprocessed/de.vocab.composite2.tsv')
    parser.add_argument('--ans_vocab_fn', type = str, default = 'preprocessed/en.vocab.tsv')

    parser.add_argument('--batch_size', type = int,  default = 256)
    parser.add_argument('--lr', type = float,  default = 0.0001)

    parser.add_argument('--output_dir', type = str,  default = os.environ.get('PT_OUTPUT_DIR', './tmp'))

    parser.add_argument('--maxlen', type = int,  default = 300)
    parser.add_argument('--maxlen_q', type = int,  default = 50)
    parser.add_argument('--maxlen_v', type = int,  default = 49)
    parser.add_argument('--hidden_size', type = int,  default = 512) #512
    parser.add_argument('--hidden_size_mil', type=int, default= 64)
    parser.add_argument('--num_blocks', type = int,  default = 6)
    parser.add_argument('--num_epochs', type = int,  default = 40)
    parser.add_argument('--num_heads', type = int,  default = 8)
    parser.add_argument('--min_cnt', type = int,  default = 10)
    parser.add_argument('--dropout_rate', type = float,  default = 0.5)
    parser.add_argument('--dropout_rate_mcb', type = float, default = 0.1)
    parser.add_argument('--aug_rate', type = float, default = 0.5)
    parser.add_argument('--topN', type = int,  default = 1)

    parser.add_argument('--sinusoid', action = 'store_true')
    parser.add_argument('--with_dec', action = 'store_true')
    parser.add_argument('--with_loc', action = 'store_true')
    parser.add_argument('--with_smooth_labeling', action = 'store_true')
    parser.add_argument('--with_bbox', action = 'store_true')
    parser.add_argument('--with_rank_loss', action = 'store_true')
    parser.add_argument('--with_MILNCE_loss', action='store_true')
    parser.add_argument('--with_gt_relation', action = 'store_true')
    parser.add_argument('--local_debug', action = 'store_true')
    parser.add_argument('--decMask',action = 'store_true')
    parser.add_argument('--visGraph', action='store_true')
    parser.add_argument('--mcb', action='store_true')
    parser.add_argument('--dataAug', action='store_true')
    parser.add_argument('--gtNode', action='store_true')
    parser.add_argument('--gtWpred', action='store_true')
    parser.add_argument('--GTRelPredNode', action='store_true')
    parser.add_argument('--only_obj', action='store_true')
    parser.add_argument('--pred_rel', action='store_true')

    parser.add_argument('--log_steps', type=int, default = 100) #100
    parser.add_argument('--log_steps_val', type=int, default = 100)

    parser.add_argument('--model_v', type=int, default = 1)
    parser.add_argument('--pool', type=str, default = 'mean')

    #distributed
    parser.add_argument('--ngpus', help='Number of gpus', type=int, default=-1)
    parser.add_argument('--nr', help='rank of node', type=int, default=0)
    parser.add_argument('--num_nodes', help='rank of node', type=int, default=1)
    parser.add_argument('--num_workers', type = int,  default = 4)

    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '7787'
    if args.ngpus == -1:
        args.ngpus = torch.cuda.device_count()

    args.world_size = args.ngpus * args.num_nodes
    if not os.path.exists(os.path.join(args.data_dir_azure, args.output_dir)):
        os.makedirs(os.path.join(args.data_dir_azure, args.output_dir))

    if args.local_debug:
        args.world_size = 1
        args.ngpus = 1
        main(0, args) 
    else:
        mp.spawn(main, nprocs = args.ngpus, args = (args,))
