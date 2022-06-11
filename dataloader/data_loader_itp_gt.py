from __future__ import print_function

import numpy as numpy
import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
import math
import random
from PIL import Image
from io import BytesIO

import tarfile
import argparse
import pdb
import codecs

def load_graph_vocab(de_vocab_fn):
    vocab = [line.split()[0] for line in codecs.open(de_vocab_fn, 'r', 'utf-8').read().splitlines() ]
    index = [int(line.split()[1]) for line in codecs.open(de_vocab_fn, 'r', 'utf-8').read().splitlines() ]

    word2idx = {word: index[idx] for idx, word in enumerate(vocab)}
    idx2word = {index[idx]: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_answer_vocab(en_vocab_fn, min_cnt):
    vocab = [' '.join(line.split()[:-1]) for line in codecs.open(en_vocab_fn, 'r', 'utf-8').read().splitlines() if int(line.split()[-1])>=min_cnt]
    word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}
    idx2word = {idx + 1: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

PAD = 400000
UNK = 400001
END = 400003


class GQADataset(data.Dataset):
    def __init__(self, opt, fea_tar_fn, q_tar_fn, gt_graph_fn, with_loc = False):
        super(GQADataset, self).__init__()
        self.opt = opt

        self.with_loc = with_loc
        self.pos_grid_num = 10

        self.fea_tar_fn = os.path.join(opt.data_dir_azure, fea_tar_fn)
        self.q_tar_fn = os.path.join(opt.data_dir_azure, q_tar_fn)
        self.gt_graph_fn = os.path.join(opt.data_dir_azure, gt_graph_fn)

        self.gt_relation_fn = os.path.join(opt.data_dir_azure, opt.gt_relation_fn)
        self.enc_vocab_fn = os.path.join(opt.data_dir_azure, opt.enc_vocab_fn)
        self.ans_vocab_fn = os.path.join(opt.data_dir_azure, opt.ans_vocab_fn)

        self.enc_w2id, _ = load_graph_vocab(self.enc_vocab_fn)
        self.ans_w2id, _ = load_answer_vocab(self.ans_vocab_fn, opt.min_cnt)

        self.fea_dict = self.load_tar_infos(self.fea_tar_fn)
        self.q_list = self.load_tar_infos_list(self.q_tar_fn)
        with open(self.gt_graph_fn) as fid:
            self.gt_graph = json.load(fid)

        print(len(self.q_list))

        self.gt_relations = json.load(open(self.gt_relation_fn, 'r'))


        vg_classes = []
        with open(os.path.join(opt.data_dir_azure, opt.obj_vocab_fn)) as f:
            for object in f.readlines():
                vg_classes.append(object.split(',')[0].lower().strip())

        self.vg_classes = vg_classes 

        vg_attrs = []
        with open(os.path.join(opt.data_dir_azure, opt.attr_vocab_fn)) as f:
            for object in f.readlines():
                vg_attrs.append(object.split(',')[0].lower().strip())
        self.vg_attrs= vg_attrs

    def load_tar_infos(self, tar_fn):
        dict_fn2tar = {}
        tar_fid = tarfile.open(tar_fn)
        tar_members = tar_fid.getmembers()
        for member in tar_members:
            key = os.path.basename(member.name)
            key = os.path.splitext(key)[0]
            dict_fn2tar[key] = member
        tar_fid.close()
        return dict_fn2tar

    def load_tar_infos_list(self, tar_fn, ext = '.json'):
        lst_fn2tar = []
        tar_fid = tarfile.open(tar_fn)
        tar_members = tar_fid.getmembers()
        tar_fid.close()

        tar_ret = []
        for tar_mem in tar_members:
            if tar_mem.name.endswith(ext):
                tar_ret.append(tar_mem)
        return tar_ret

    def convert_graph(self, gt_graph):
        # do not use the gt relation,
        # instead use the statistically frequent relation.

        idx_obj = []
        nodes_obj = []
        relation = []
        nodes = []
        for obj in gt_graph['objects']:
            data = gt_graph['objects'][obj]
            x,y,w,h = data['x'], data['y'], data['w'], data['h']

            nodes.append(data['name'].strip().replace(' ', ''))
            pos_obj = len(nodes)
            if len(data['attributes']) > 0:
                nodes.append(data['attributes'][0])
                pos_attr = len(nodes)
                relation.append([pos_obj, pos_attr])
                relation.append([pos_attr, pos_obj])
            idx_obj.append(pos_obj)
            nodes_obj.append(data['name'])
            if self.with_loc:
                # add position
                cx = x + w / 2
                cy = y + h / 2
                pos_node_name = 'x' + str(math.floor(cx / gt_graph['width'] * self.pos_grid_num)) + 'y' + str(math.floor(cy / gt_graph['height']) * self.pos_grid_num)
                nodes.append(pos_node_name)
                pos_pos = len(nodes)
                relation.append([pos_obj, pos_pos])
                relation.append([pos_pos, pos_obj])


        num_obj = len(idx_obj)
        dict_rel2pos = {}
        for i in range(num_obj):
            for j in range(num_obj):
                if j == i:
                    continue

                key = nodes_obj[i] + ',' + nodes_obj[j]
                if key in self.gt_relations:
                    r_name = self.gt_relations[key]
                    pos_rel = len(nodes)

                    if r_name in dict_rel2pos:
                        pos_rel = dict_rel2pos[r_name]
                    else:
                        dict_rel2pos[r_name] = pos_rel
                        r_name = ''.join(r_name.split())
                        nodes.append(r_name)
                    relation.append([idx_obj[i], pos_rel])
                    relation.append([pos_rel, idx_obj[j]])

        return nodes, relation

    def __getitem__(self, index):
        q_mem = self.q_list[index]
        with tarfile.open(self.q_tar_fn) as tar_fid:
            qinfo = json.load(tar_fid.extractfile(q_mem))
        qnode = qinfo['node_list']
        qedge = qinfo['edge_pair']
        answer = qinfo['answer']
        answer = self.ans_w2id.get(answer, 0) # default all 0 classes
        answer = np.asarray(answer).astype('int32')

        image_id = qinfo['image_id']
        # load grid image_fea
        with tarfile.open(self.fea_tar_fn) as tar_fid:
            fea_mem = self.fea_dict[image_id]
            array_file = BytesIO()
            array_file.write(tar_fid.extractfile(fea_mem).read())
            array_file.seek(0)
            vis_fea = np.load(array_file)
            vis_fea = vis_fea['fea']
        # gt graph. 
        gt_graph = self.gt_graph[image_id]
        nodes, edges = self.convert_graph(gt_graph)
        nodes_idx = []
        for node in nodes:
            nodes_idx.append(self.enc_w2id.get(node, UNK))

        qnode_idx = []
        for qn in qnode:
            qnode_idx.append(self.enc_w2id.get(qn, UNK))
        
        return vis_fea, np.asarray(nodes_idx).astype('int64'), edges, np.asarray(qnode_idx).astype('int64'), qedge, answer
    
    def __len__(self):
        return len(self.q_list)

def collate_fn(data):
    vis_fea, nodes_idx, edges, qnode_idx, qedge, answer = zip(*data)
    answer = np.stack(answer, axis = 0)
    max_node_len = 0
    batch_size = len(nodes_idx)
    for node in nodes_idx:
        max_node_len = max(max_node_len, node.shape[0])

    # Process input 
    node_ipt = np.zeros((batch_size, max_node_len), dtype = 'int64')  
    node_ipt_mask = np.zeros((batch_size, max_node_len), dtype = 'int32')  
    node_ipt_graph = np.zeros((batch_size, max_node_len, max_node_len), dtype = 'int32')  
    node_ipt[:] = PAD

    for i in range(batch_size):
        node_ipt[i,0:nodes_idx[i].shape[0]] = nodes_idx[i]
        node_ipt_mask[i, 0:nodes_idx[i].shape[0]] = 1
    # generate graph.
    for row, edge in enumerate(edges):
        edge = np.asarray(edge).astype('int32')
        if edge.size == 0:
            continue
        if len(edge.shape) == 1:
            edge = edge[np.newaxis,:]
        node_ipt_graph[row, edge[:,0], edge[:,1]] = 1
    
    # process question node
    max_q_len = 0
    for node in qnode_idx:
        max_q_len = max(max_q_len, node.shape[0])

    node_q_ipt = np.zeros((batch_size, max_q_len), dtype = 'int64')  
    node_q_ipt_mask = np.zeros((batch_size, max_q_len), dtype = 'int32')  
    node_q_ipt_graph = np.zeros((batch_size, max_q_len, max_q_len), dtype = 'int32')  
    node_q_ipt[:] = PAD

    for i in range(batch_size):
        node_q_ipt[i,0:qnode_idx[i].shape[0]] = qnode_idx[i]
        node_q_ipt_mask[i, 0:qnode_idx[i].shape[0]] = 1
    # generate graph.
    for row, edge in enumerate(qedge):
        edge = np.asarray(edge).astype('int32')
        node_q_ipt_graph[row, edge[:,0], edge[:,1]] = 1

    data_ipt = {}
    vis_fea = np.stack(vis_fea, axis = 0)
    data_ipt['vis_fea'] = torch.from_numpy(vis_fea)
    data_ipt['vis_syb_ipt'] = torch.from_numpy(node_ipt)
    data_ipt['vis_syb_graph'] = torch.from_numpy(node_ipt_graph)
    data_ipt['vis_syb_mask'] = torch.from_numpy(node_ipt_mask)
    data_ipt['q_ipt'] = torch.from_numpy(node_q_ipt)
    data_ipt['q_ipt_mask'] = torch.from_numpy(node_q_ipt_mask)
    data_ipt['q_ipt_graph'] = torch.from_numpy(node_q_ipt_graph)
    data_ipt['answer'] = torch.from_numpy(answer).long()


    return data_ipt

if __name__ == '__main__':
    #opt = {'keep_res':True, 'pad':False}
    parser = argparse.ArgumentParser()
    # basic experiment setting
    parser.add_argument('--data_dir_azure', default='./tmp')
    parser.add_argument('--fea_tar_fn', default='spatial_npz_fea.tar')
    parser.add_argument('--q_tar_fn', default='train.tar')
    parser.add_argument('--gt_graph_fn', default='train_sceneGraphs.json')
    parser.add_argument('--gt_relation_fn', default='GT_relations_dict.json')
    parser.add_argument('--obj_vocab_fn', type = str, default='objects_vocab.txt')
    parser.add_argument('--attr_vocab_fn', type = str, default='attributes_vocab.txt')
    parser.add_argument('--bbox_bin_num', type = int, default = 64)
    parser.add_argument('--min_cnt', type = int, default = 4)
    parser.add_argument('--enc_vocab_fn', type = str, default = 'preprocessed/de.vocab.tsv.composite')
    parser.add_argument('--ans_vocab_fn', type = str, default = 'preprocessed/answer.txt')
    opt = parser.parse_args()

    data = GQADataset(opt, opt.fea_tar_fn, opt.q_tar_fn, opt.gt_graph_fn, True) 
    loader = torch.utils.data.DataLoader(
        data,
        batch_size = 4,
        num_workers = 0,
        pin_memory = True,
        drop_last = True,
        collate_fn = collate_fn
    ) 


    #for i in range(len(data)):
    #    if i % 1000 == 0:
    #        print(i, len(data))
    #    data[i]
    max_len = 0
    for i, data in enumerate(loader):
        if len(data['vis_syb_ipt'].shape) <= 1:
            print('error shape', data['vis_syb_ipt'].shape)
        if data['vis_syb_ipt'].shape[1] > max_len:
            max_len = data['vis_syb_ipt'].shape[1]
            print(max_len)
        if i % 1000 == 0:
            print(' i = {}'.format(i))
