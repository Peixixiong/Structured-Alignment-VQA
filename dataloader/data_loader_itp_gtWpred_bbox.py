from __future__ import print_function

import numpy as numpy
import torch.utils.data as data
import numpy as np
import torch
import json
import os
import math
import random
from random import choices
from io import BytesIO
from synonym_word_converter import *

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
VIS_PAD = -1


class GQADataset_topN(data.Dataset):
    def __init__(self, split, opt, fea_tar_fn, q_tar_fn, g_tar_fn, topN, with_loc = True, with_gt_relation = False):
        super(GQADataset_topN, self).__init__()
        self.split = split
        self.opt = opt
        self.with_loc = with_loc
        self.pos_grid_num = 10
        self.topN = topN

        self.fea_tar_fn = os.path.join(opt.data_dir_azure, fea_tar_fn)
        self.q_tar_fn = os.path.join(opt.data_dir_azure, q_tar_fn)
        self.g_tar_fn = os.path.join(opt.data_dir_azure, g_tar_fn)
        if split == 'train' :
            self.gt_graph_fn = os.path.join(opt.data_dir_azure, 'train_sceneGraphs.json')
        else:
            self.gt_graph_fn = os.path.join(opt.data_dir_azure, 'val_sceneGraphs.json')

        self.gt_relation_fn = os.path.join(opt.data_dir_azure, opt.gt_relation_fn)
        self.enc_vocab_fn = os.path.join(opt.data_dir_azure, opt.enc_vocab_fn)
        self.ans_vocab_fn = os.path.join(opt.data_dir_azure, opt.ans_vocab_fn)

        self.enc_w2id, _ = load_graph_vocab(self.enc_vocab_fn)
        self.ans_w2id, _ = load_answer_vocab(self.ans_vocab_fn, opt.min_cnt)

        self.fea_dict = self.load_tar_infos(self.fea_tar_fn)
        self.g_dict = self.load_tar_infos(self.g_tar_fn)
        self.q_list = self.load_tar_infos_list(self.q_tar_fn)

        self.bg_class = opt.bg_class # this equals to the bg cls idx

        # load GT relation
        self.gt_relations = json.load(open(self.gt_relation_fn, 'r'))

        with open(self.gt_graph_fn) as fid:
            self.gt_graph = json.load(fid)

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

        self.word_converter = {}
        for key in syn_dict_composit_multiwrds:
            new_key = key.replace(' ', '')
            if new_key != syn_dict_composit_multiwrds[key]:
                self.word_converter[new_key] = syn_dict_composit_multiwrds[key]

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




    def convert_graph_visrel(self, data_info, bg_class, bbox):
        vis_nodes_obj = []
        vis_nodes_attr = []
        vis_dict_attr2idx = {}
        vis_dict_rel2pos = {}
        vis_dict_pos2idx = {}

        nodes_obj = []
        nodes_attr = []
        dict_obj_idx = {}
        dict_attr2idx = {}
        dict_rel2pos = {}
        dict_pos2idx = {}
        valid2all = []

        keep_idx = np.zeros(data_info['objects_id'].shape, dtype = 'int32')
        for row_idx, (obj_idx, attr_idx) in enumerate(zip(data_info['objects_id'], data_info['attrs_id'])):
            if obj_idx >= len(self.vg_classes):
                vis_nodes_obj.append('__background__')
                vis_nodes_attr.append('__background_attr__')
                continue
            keep_idx[row_idx] = 1
            # valid2all[len(nodes_obj)] = row_idx
            valid2all.append(row_idx)
            nodes_obj.append(self.vg_classes[obj_idx].replace(' ',''))
            vis_nodes_obj.append(self.vg_classes[obj_idx].replace(' ',''))
            nodes_attr.append(self.vg_attrs[attr_idx].replace(' ',''))
            vis_nodes_attr.append(self.vg_attrs[attr_idx].replace(' ', ''))

        # Now, build VIS nodes include obj and attr first and relations.
        syb2vis = []
        vis_num_obj = len(vis_nodes_obj)
        vis_idx_obj = []
        vis_relation = []
        vis_nodes = []
        for i in range(vis_num_obj):
            pos_obj = len(vis_nodes) #Multi-Words
            syb2vis.append([pos_obj, i])
            vis_nodes.append(vis_nodes_obj[i])
            if vis_nodes_attr[i] != '__background_attr__':
                if vis_nodes_attr[i] in vis_dict_attr2idx:
                    pos_attr = vis_dict_attr2idx[vis_nodes_attr[i]]
                else:
                    pos_attr = len(vis_nodes)
                    vis_dict_attr2idx[vis_nodes_attr[i]] = pos_attr
                    vis_nodes.append(vis_nodes_attr[i])
                # undirected graph
                vis_relation.append([pos_obj, pos_attr])
                vis_relation.append([pos_attr, pos_obj])

            vis_idx_obj.append(pos_obj)

            if self.with_loc:
                pos_node_name = 'x' + str(bbox[i][0].item()) + 'y' + str(bbox[i][1])
                if pos_node_name in vis_dict_pos2idx:
                    pos_pos = vis_dict_pos2idx[pos_node_name]
                else:
                    pos_pos = len(vis_nodes)
                    vis_dict_pos2idx[pos_node_name] = pos_pos
                    vis_nodes.append(pos_node_name)
                vis_relation.append([pos_obj, pos_pos])
                vis_relation.append([pos_pos, pos_obj])

                pos_node_name = 'x' + str(bbox[i][2].item()) + 'y' + str(bbox[i][3])
                if pos_node_name in vis_dict_pos2idx:
                    pos_pos = vis_dict_pos2idx[pos_node_name]
                else:
                    pos_pos = len(vis_nodes)
                    vis_dict_pos2idx[pos_node_name] = pos_pos
                    vis_nodes.append(pos_node_name)
                vis_relation.append([pos_obj, pos_pos])
                vis_relation.append([pos_pos, pos_obj])

        for i in range(vis_num_obj):
            for j in range(vis_num_obj):
                key = vis_nodes_obj[i] + ',' + vis_nodes_obj[j] #MultiWords inside
                if key in self.gt_relations:
                    r_name = self.gt_relations[key].replace(' ','')
                    pos_rel = len(vis_nodes)

                    if r_name in vis_dict_rel2pos:
                        pos_rel = vis_dict_rel2pos[r_name]
                    else:
                        vis_dict_rel2pos[r_name] = pos_rel
                        r_name = ''.join(r_name.split())
                        if 'left' in r_name and (bbox[i][0].item()+bbox[i][2].item()) > (bbox[j][0].item()+bbox[j][2].item()):
                            r_name = 'right'
                        if 'right' in r_name and (bbox[i][0].item() + bbox[i][2].item()) < (
                                bbox[j][0].item() + bbox[j][2].item()):
                            r_name = 'left'
                        if 'bottom' in r_name and (bbox[i][1].item()+bbox[i][3].item()) < (bbox[j][1].item()+bbox[j][3].item()):
                            r_name = 'top'
                        if 'top' in r_name and (bbox[i][1].item() + bbox[i][3].item()) > (
                                bbox[j][1].item() + bbox[j][3].item()):
                            r_name = 'bottom'
                        vis_nodes.append(r_name)
                    vis_relation.append([vis_idx_obj[i], pos_rel])
                    vis_relation.append([pos_rel, vis_idx_obj[j]])

        #Make Background Visual Feature Connect to every other Visual Feature.
        for id, item in enumerate(vis_nodes_obj):
            if item == '__background__':
                for i in range(vis_num_obj):
                    vis_relation.append([id, i])
                    vis_relation.append([i, id])

        # Now, build nodes include obj and attr first and relations.
        num_obj = len(nodes_obj)
        idx_obj = []
        relation = []
        nodes = []
        for i in range(num_obj):
            pos_obj = len(nodes) #Multi-Words
            nodes.append(nodes_obj[i])
            if nodes_attr[i] in dict_attr2idx:
                pos_attr = dict_attr2idx[nodes_attr[i]]
            else:
                pos_attr = len(nodes)
                dict_attr2idx[nodes_attr[i]] = pos_attr
                nodes.append(nodes_attr[i])
            # undirected graph
            relation.append([pos_obj, pos_attr])
            relation.append([pos_attr, pos_obj])

            idx_obj.append(pos_obj)

            if self.with_loc:
                bbox_i = valid2all[i]
                pos_node_name = 'x' + str(bbox[bbox_i][0].item()) + 'y' + str(bbox[bbox_i][1])
                if pos_node_name in dict_pos2idx:
                    pos_pos = dict_pos2idx[pos_node_name]
                else:
                    pos_pos = len(nodes)
                    dict_pos2idx[pos_node_name] = pos_pos
                    nodes.append(pos_node_name)
                relation.append([pos_obj, pos_pos])
                relation.append([pos_pos, pos_obj])

                pos_node_name = 'x' + str(bbox[bbox_i][2].item()) + 'y' + str(bbox[bbox_i][3])
                if pos_node_name in dict_pos2idx:
                    pos_pos = dict_pos2idx[pos_node_name]
                else:
                    pos_pos = len(nodes)
                    dict_pos2idx[pos_node_name] = pos_pos
                    nodes.append(pos_node_name)
                relation.append([pos_obj, pos_pos])
                relation.append([pos_pos, pos_obj])

        for i in range(num_obj):
            for j in range(num_obj):
                key = nodes_obj[i] + ',' + nodes_obj[j] #MultiWords inside
                if key in self.gt_relations:
                    r_name = self.gt_relations[key].replace(' ','')
                    pos_rel = len(nodes)

                    if r_name in dict_rel2pos:
                        pos_rel = dict_rel2pos[r_name]
                    else:
                        dict_rel2pos[r_name] = pos_rel
                        r_name = ''.join(r_name.split())
                        bbox_i = valid2all[i]
                        bbox_j = valid2all[j]
                        if 'left' in r_name and (bbox[bbox_i][0].item()+bbox[bbox_i][2].item()) > (bbox[bbox_j][0].item()+bbox[bbox_j][2].item()):
                            r_name = 'right'
                        if 'right' in r_name and (bbox[bbox_i][0].item() + bbox[bbox_i][2].item()) < (
                                bbox[bbox_j][0].item() + bbox[bbox_j][2].item()):
                            r_name = 'left'
                        if 'bottom' in r_name and (bbox[bbox_i][1].item()+bbox[bbox_i][3].item()) < (bbox[bbox_j][1].item()+bbox[bbox_j][3].item()):
                            r_name = 'top'
                        if 'top' in r_name and (bbox[bbox_i][1].item() + bbox[bbox_i][3].item()) > (
                                bbox[bbox_j][1].item() + bbox[bbox_j][3].item()):
                            r_name = 'bottom'
                        nodes.append(r_name)

                    relation.append([idx_obj[i], pos_rel])
                    relation.append([pos_rel, idx_obj[j]])
        return nodes, relation, vis_relation, keep_idx, idx_obj, vis_nodes, syb2vis, valid2all


    def convert_graph(self, data_info, bg_class, bbox, gt_graph):
        nodes_objs = []
        nodes_attrs = []
        dict_attr2idx = {}
        dict_rel2pos = {}
        dict_pos2idx = {}
        valid2all = []

        keep_idx = np.ones(len(data_info['objects_id']), dtype = 'int32')
        for row_idx, (obj_idxs, obj, attr_idx) in enumerate(zip(data_info['objects_id'], gt_graph['objects'], data_info['attrs_id'])):
            nodes_obj = []
            for obj_idx in obj_idxs:
                if len(nodes_obj) < self.topN:
                    if obj_idx < len(self.vg_classes):
                        # keep_idx[row_idx] = 1
                        valid2all.append(row_idx) #from syb idx to bbox idx [0,0,0,0,0,1,1,1,1,1]
                        nodes_obj.append(self.vg_classes[obj_idx].replace(' ', ''))
                else:
                    break
            gt_obj_name = gt_graph['objects'][obj]['name'].strip().replace(' ', '')
            nodes_obj[-1] = gt_obj_name
            nodes_attrs.append([self.vg_attrs[attr_idx].replace(' ', '')])
            nodes_objs.append(nodes_obj)

        # Now, build nodes include obj and attr first and relations.
        num_obj = len(nodes_objs)
        idx_objs = []
        relation = []
        nodes = []
        syb2vis = {}
        vis_relation = []

        for i in range(num_obj):
            nodes_obj = nodes_objs[i]
            idx_obj = []
            previous_pos = len(nodes)
            for i_ in range(len(nodes_obj)):
                pos_obj = len(nodes)
                if previous_pos != pos_obj:  # obj connected with obj
                    relation.append([previous_pos, pos_obj])
                    relation.append([pos_obj, previous_pos])
                    previous_pos = pos_obj

                syb2vis[pos_obj] = i
                nodes.append(nodes_obj[i_])

                for nodes_attr in nodes_attrs[i]:
                    if nodes_attr in dict_attr2idx:
                        pos_attr = dict_attr2idx[nodes_attr]
                    else:
                        pos_attr = len(nodes)
                        dict_attr2idx[nodes_attr] = pos_attr
                        nodes.append(nodes_attr)

                    relation.append([pos_obj, pos_attr])
                    relation.append([pos_attr, pos_obj])

                idx_obj.append(pos_obj)

                if self.with_loc:
                    bbox_i = i
                    pos_node_name = 'x' + str(bbox[bbox_i][0].item()) + 'y' + str(bbox[bbox_i][1])
                    if pos_node_name in dict_pos2idx:
                        pos_pos = dict_pos2idx[pos_node_name]
                    else:
                        pos_pos = len(nodes)
                        dict_pos2idx[pos_node_name] = pos_pos
                        nodes.append(pos_node_name)
                    relation.append([pos_obj, pos_pos])
                    relation.append([pos_pos, pos_obj])

                    pos_node_name = 'x' + str(bbox[bbox_i][2].item()) + 'y' + str(bbox[bbox_i][3])
                    if pos_node_name in dict_pos2idx:
                        pos_pos = dict_pos2idx[pos_node_name]
                    else:
                        pos_pos = len(nodes)
                        dict_pos2idx[pos_node_name] = pos_pos
                        nodes.append(pos_node_name)
                    relation.append([pos_obj, pos_pos])
                    relation.append([pos_pos, pos_obj])
            idx_objs.append(idx_obj)

        for i in range(num_obj):
            for j in range(num_obj):
                nodes_obj_is = nodes_objs[i]
                nodes_obj_js = nodes_objs[j]
                for i_ in range(len(nodes_obj_is)):
                    for j_ in range(len(nodes_obj_js)):
                        key = nodes_obj_is[i_] + ',' + nodes_obj_js[j_]
                        if key in self.gt_relations:
                            r_name = self.gt_relations[key].replace(' ', '')
                            pos_rel = len(nodes)

                            if r_name in dict_rel2pos:
                                pos_rel = dict_rel2pos[r_name]
                            else:
                                dict_rel2pos[r_name] = pos_rel
                                r_name = ''.join(r_name.split())
                                bbox_i = i
                                bbox_j = j
                                if 'left' in r_name and (bbox[bbox_i][0].item() + bbox[bbox_i][2].item()) > (
                                        bbox[bbox_j][0].item() + bbox[bbox_j][2].item()):
                                    r_name = 'right'
                                if 'right' in r_name and (bbox[bbox_i][0].item() + bbox[bbox_i][2].item()) < (
                                        bbox[bbox_j][0].item() + bbox[bbox_j][2].item()):
                                    r_name = 'left'
                                if 'bottom' in r_name and (bbox[bbox_i][1].item() + bbox[bbox_i][3].item()) < (
                                        bbox[bbox_j][1].item() + bbox[bbox_j][3].item()):
                                    r_name = 'top'
                                if 'top' in r_name and (bbox[bbox_i][1].item() + bbox[bbox_i][3].item()) > (
                                        bbox[bbox_j][1].item() + bbox[bbox_j][3].item()):
                                    r_name = 'bottom'
                                nodes.append(r_name)
                            relation.append([idx_objs[i][i_], pos_rel])
                            relation.append([pos_rel, idx_objs[j][j_]])
                            vis_relation.append([valid2all[syb2vis[idx_objs[i][i_]]], valid2all[syb2vis[idx_objs[j][j_]]]])
        return nodes, relation, vis_relation, keep_idx, idx_obj, [None], [-1], valid2all

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
        try:
            gt_graph = self.gt_graph[image_id]
            # load grid image_fea
            with tarfile.open(self.fea_tar_fn) as tar_fid:
                fea_mem = self.fea_dict[image_id]
                array_file = BytesIO()
                array_file.write(tar_fid.extractfile(fea_mem).read())
                array_file.seek(0)
                vis_fea = np.load(array_file)
                vis_fea = vis_fea['x']

            with tarfile.open(self.g_tar_fn) as tar_fid:
                graph_mem = self.g_dict[image_id]
                array_file = BytesIO()
                array_file.write(tar_fid.extractfile(graph_mem).read())
                array_file.seek(0)
                data = np.load(array_file, allow_pickle = True)

                bbox = data['bbox']
                if len(bbox.shape) == 1:
                    bbox = np.reshape(bbox, (1, bbox.size))
                bbox[:,0] /= data['image_w']
                bbox[:,2] /= data['image_w']
                bbox[:,1] /= data['image_h']
                bbox[:,3] /= data['image_h']

                bbox = np.floor(bbox * self.opt.bbox_bin_num).astype('int32')

                data_info = data['info'].tolist()
                if self.opt.model_v != 23:
                    nodes, edges, vis_edges, keep_bbox, idx_of_obj, vis_nodes, syb2vis, valid2all  = self.convert_graph(data_info, self.opt.bg_class, bbox, gt_graph)
                else:
                    nodes, edges, vis_edges, keep_bbox, idx_of_obj, vis_nodes, syb2vis, valid2all = self.convert_graph_visrel(data_info, self.opt.bg_class, bbox)
                idx_of_obj = np.asarray(idx_of_obj).astype('int64')
                valid2all = np.asarray(valid2all).astype('int64')

                bbox = bbox[keep_bbox > 0,:]
            nodes_idx = []
            vis_nodes_idx = []

            for node in nodes:
                if node in self.word_converter:
                    node_ = self.word_converter[node]
                else:
                    node_ = node
                nodes_idx.append(self.enc_w2id.get(node_, UNK))
            # if len(nodes_idx) >= 600:
            #     # print ('val maxlen',len(nodes_idx))
            #     return None
            for node in vis_nodes:
                if node in self.word_converter:
                    node_ = self.word_converter[node]
                else:
                    node_ = node
                vis_nodes_idx.append(self.enc_w2id.get(node_, UNK))

            qnode_idx = []
            for qn in qnode:
                qnode_idx.append(self.enc_w2id.get(qn, UNK))

            return vis_fea, np.asarray(nodes_idx).astype('int64'), edges, vis_edges, bbox, np.asarray(qnode_idx).astype(
                'int64'), \
                   qedge, answer, idx_of_obj, np.asarray(vis_nodes_idx).astype('int64'), np.asarray(syb2vis).astype(
                'int64'), valid2all
        except:
            return None

    def __len__(self):
        return len(self.q_list)



def collate_fn(data):
    data = list(filter(lambda x: x is not None, data))
    vis_fea, nodes_idx, edges, vis_edges, bbox, qnode_idx, qedge, answer, idx_of_obj, vis_nodes_idx, syb2vis, valid2all = zip(*data)
    idx_of_obj = [ torch.from_numpy(idx) for idx in idx_of_obj ]
    valid2all = [torch.from_numpy(idx) for idx in valid2all]

    answer = np.stack(answer, axis = 0)
    max_node_len = 0
    batch_size = len(nodes_idx)
    # process the vis feature.
    max_fea_len = 0
    for fea in vis_fea:
        max_fea_len = max(max_fea_len, fea.shape[0])
    fea_ipt = np.zeros((batch_size, max_fea_len, vis_fea[0].shape[1]), dtype = 'float32')
    fea_ipt_mask = np.zeros((batch_size, max_fea_len, max_fea_len), dtype = 'int32')
    for i in range(batch_size):
        fea_ipt[i, 0:vis_fea[i].shape[0], :] = vis_fea[i]
        fea_ipt_mask[i,0:vis_fea[i].shape[0], 0:vis_fea[i].shape[0]] = 1


    # Process syb input
    for node in nodes_idx:
        max_node_len = max(max_node_len, node.shape[0])
    # print (max_node_len)
    # if max_node_len >= 600:
    #     print ('max_node_len, val', max_node_len)

    node_ipt = np.zeros((batch_size, max_node_len), dtype = 'int64')
    node_ipt_mask = np.zeros((batch_size, max_node_len, max_node_len), dtype = 'int32')
    node_ipt_graph = np.zeros((batch_size, max_node_len, max_node_len), dtype = 'int32')
    node_ipt_vis_graph = np.zeros((batch_size, max_fea_len, max_fea_len), dtype='int32')
    node_ipt[:] = PAD

    for i in range(batch_size):
        node_ipt[i,0:nodes_idx[i].shape[0]] = nodes_idx[i]
        node_ipt_mask[i, 0:nodes_idx[i].shape[0], 0:nodes_idx[i].shape[0]] = 1

    # generate graph.
    for row, edge in enumerate(edges):
        edge = np.asarray(edge).astype('int32')
        if edge.size == 0:
            continue
        if len(edge.shape) == 1:
            edge = edge[np.newaxis,:]
        node_ipt_graph[row, edge[:,0], edge[:,1]] = 1
    # generate vis graph.
    for row, edge in enumerate(vis_edges):
        edge = np.asarray(edge).astype('int32')
        if edge.size == 0:
            continue
        if len(edge.shape) == 1:
            edge = edge[np.newaxis,:]
        node_ipt_vis_graph[row, edge[:,0], edge[:,1]] = 1

    # process question node
    max_q_len = 0
    for node in qnode_idx:
        max_q_len = max(max_q_len, node.shape[0])

    node_q_ipt = np.zeros((batch_size, max_q_len), dtype = 'int64')
    node_q_ipt_mask = np.zeros((batch_size, max_q_len, max_q_len), dtype = 'int32')
    node_q_ipt_graph = np.zeros((batch_size, max_q_len, max_q_len), dtype = 'int32')
    node_q_ipt[:] = PAD

    for i in range(batch_size):
        node_q_ipt[i,0:qnode_idx[i].shape[0]] = qnode_idx[i]
        node_q_ipt_mask[i, 0:qnode_idx[i].shape[0], 0:qnode_idx[i].shape[0]] = 1
    # generate graph.
    for row, edge in enumerate(qedge):
        edge = np.asarray(edge).astype('int32')
        node_q_ipt_graph[row, edge[:,0], edge[:,1]] = 1

    # process bbox
    max_obj = 0
    for bb in bbox:
        max_obj = max(max_obj, bb.shape[0])

    bbox_ipt = np.zeros((batch_size, max_obj, 4), dtype = 'int32')
    bbox_ipt_mask = np.zeros((batch_size, max_obj), dtype = 'int32')
    for i in range(batch_size):
        bbox_ipt[i,0:bbox[i].shape[0],:] = bbox[i]
        bbox_ipt_mask[i,0:bbox[i].shape[0]] = 1
    data_ipt = {}
    data_ipt['vis_fea'] = torch.from_numpy(fea_ipt)
    data_ipt['vis_fea_mask'] = torch.from_numpy(fea_ipt_mask) #vis_mask
    data_ipt['vis_syb_ipt'] = torch.from_numpy(node_ipt)
    data_ipt['vis_syb_graph'] = torch.from_numpy(node_ipt_graph) #syb graph
    data_ipt['vis_vis_graph'] = torch.from_numpy(node_ipt_vis_graph) #vis graph
    data_ipt['vis_syb_mask'] = torch.from_numpy(node_ipt_mask) #syb_mask
    data_ipt['q_ipt'] = torch.from_numpy(node_q_ipt)
    data_ipt['q_ipt_mask'] = torch.from_numpy(node_q_ipt_mask)
    data_ipt['q_ipt_graph'] = torch.from_numpy(node_q_ipt_graph)
    data_ipt['bbox_ipt'] = torch.from_numpy(bbox_ipt)
    data_ipt['bbox_ipt_mask'] = torch.from_numpy(bbox_ipt_mask)
    data_ipt['answer'] = torch.from_numpy(answer).long()
    data_ipt['idx_of_obj'] = idx_of_obj
    data_ipt['valid2all'] = valid2all

    return data_ipt

if __name__ == '__main__':
    #opt = {'keep_res':True, 'pad':False}
    parser = argparse.ArgumentParser()
    # basic experiment setting
    parser.add_argument('--data_dir_azure', default='./tmp2')
    parser.add_argument('--fea_tar_fn', default='spatial_npz_fea.tar')
    parser.add_argument('--q_tar_fn', default='train.tar')
    parser.add_argument('--g_tar_fn', default='bua_npz.tar')
    parser.add_argument('--gt_relation_fn', default='GT_relations_dict_compsite.json')
    parser.add_argument('--bg_class', type = int, default=1704)
    parser.add_argument('--obj_vocab_fn', type = str, default='objects_vocab.txt')
    parser.add_argument('--attr_vocab_fn', type = str, default='attributes_vocab.txt')
    parser.add_argument('--bbox_bin_num', type = int, default = 64)
    parser.add_argument('--min_cnt', type = int, default = 4)
    parser.add_argument('--enc_vocab_fn', type = str, default = 'preprocessed/de.vocab.composite2.tsv')
    parser.add_argument('--ans_vocab_fn', type = str, default = 'preprocessed/en.vocab.tsv')
    opt = parser.parse_args()

    data = GQADataset(opt, opt.fea_tar_fn, opt.q_tar_fn, opt.g_tar_fn) 
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

    for i, data in enumerate(loader):
        if i % 1000 == 0:
            print(i)
