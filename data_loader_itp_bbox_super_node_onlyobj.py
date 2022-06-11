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
INVALID = 400003
VIS_PAD = -1
LOC_PAD = -1


class GQADataset_super_node(data.Dataset):
    def __init__(self, split, opt, fea_tar_fn, q_tar_fn, g_tar_fn, topN, with_loc = True):
        super(GQADataset_super_node, self).__init__()
        self.split = split
        self.opt = opt
        self.with_loc = with_loc
        self.pos_grid_num = 10
        self.topN = topN
        self.len_threshold = opt.maxlen

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
        self.gt_relation_clean = list(set(self.gt_relations.values())) #No redundant rel
        self.num_relations = len(self.gt_relation_clean)

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

    def convert_graph(self, data_info, bg_class, bbox, gt_graph):
        micro_positive_node = []
        micro_negative_node = []
        macro_empty_node = []

        nodes_attr = []
        correct_nodes =[] #for opt.pred_rel
        dict_attr2idx = {}
        dict_pos2idx = {}
        for row_idx, (obj_idxs, obj, attr_idx) in enumerate(zip(data_info['objects_id'], gt_graph['objects'], data_info['attrs_id'])):
            nodes_obj = []
            gt_obj_name = gt_graph['objects'][obj]['name'].strip().replace(' ', '')
            nodes_obj.append(gt_obj_name)
            corr_flag = 0
            for obj_idx in obj_idxs:
                if len(nodes_obj) < self.topN:
                    if obj_idx < len(self.vg_classes) and self.vg_classes[obj_idx].replace(' ', '') != gt_obj_name:
                        nodes_obj.append(self.vg_classes[obj_idx].replace(' ', ''))
                    if obj_idx < len(self.vg_classes) and self.vg_classes[obj_idx].replace(' ', '') == gt_obj_name:
                        corr_flag = 1 #correctly detected
                else:
                    break
            correct_nodes.append(corr_flag)
            macro_empty_node.append(PAD)
            nodes_attr.append(self.vg_attrs[attr_idx].replace(' ', ''))
            micro_positive_node.append(nodes_obj)
            
            #random sample the rest list for the negative syb node
            vg_classes_neg = [j.replace(' ', '') for j in self.vg_classes if j.replace(' ', '') not in nodes_obj]
            micro_negative_node.append(random.sample(vg_classes_neg, self.topN))

        # Build Macro Empty Node
        num_obj = len(macro_empty_node)
        idx_obj = []
        macro_relation = []
        macro_node = []
        macro_obj_loc = [] #Record obj node location

        for i in range(num_obj):
            pos_obj = len(macro_node)
            macro_node.append(macro_empty_node[i])
            macro_obj_loc.append(pos_obj)
            
            if nodes_attr[i] in dict_attr2idx:
                pos_attr = dict_attr2idx[nodes_attr[i]]
            else:
                pos_attr = len(macro_node)
                dict_attr2idx[nodes_attr[i]] = pos_attr
                macro_node.append(nodes_attr[i])
            # undirected graph
            macro_relation.append([pos_obj, pos_attr])
            macro_relation.append([pos_attr, pos_obj])
            idx_obj.append(pos_obj)

            if self.with_loc:
                pos_node_name = 'x' + str(bbox[i][0].item()) + 'y' + str(bbox[i][1])
                if pos_node_name in dict_pos2idx:
                    pos_pos = dict_pos2idx[pos_node_name]
                else:
                    pos_pos = len(macro_node)
                    dict_pos2idx[pos_node_name] = pos_pos
                    macro_node.append(pos_node_name)
                macro_relation.append([pos_obj, pos_pos])
                macro_relation.append([pos_pos, pos_obj])

                pos_node_name = 'x' + str(bbox[i][2].item()) + 'y' + str(bbox[i][3])
                if pos_node_name in dict_pos2idx:
                    pos_pos = dict_pos2idx[pos_node_name]
                else:
                    pos_pos = len(macro_node)
                    dict_pos2idx[pos_node_name] = pos_pos
                    macro_node.append(pos_node_name)
                macro_relation.append([pos_obj, pos_pos])
                macro_relation.append([pos_pos, pos_obj])

        # Build Top1 Macro Rel Node
        dict_rel2pos = {}
        for i in range(num_obj):
            for j in range(num_obj):
                if self.opt.pred_rel:
                    if correct_nodes[i] == 1:
                        nodes_obj_i = micro_positive_node[i][0]
                    else:
                        nodes_obj_i = micro_positive_node[i][1]
                    if correct_nodes[j] == 1:
                        nodes_obj_j = micro_positive_node[j][0]
                    else:
                        nodes_obj_j = micro_positive_node[j][1]
                else:
                    nodes_obj_i = micro_positive_node[i][0]
                    nodes_obj_j = micro_positive_node[j][0]
                key = nodes_obj_i + ',' + nodes_obj_j
                if key in self.gt_relations:
                    r_name = self.gt_relations[key].replace(' ', '')
                    pos_rel = len(macro_node)

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
                        macro_node.append(r_name)

                    macro_relation.append([idx_obj[i], pos_rel])
                    macro_relation.append([pos_rel, idx_obj[j]])

        return macro_node, macro_relation, macro_obj_loc, micro_positive_node, micro_negative_node
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
                macro_nodes, macro_edges, macro_obj_locs, micro_positive_nodes, micro_negative_nodes \
                    = self.convert_graph(data_info, self.opt.bg_class, bbox, gt_graph)

            macro_nodes_idx = []
            micro_positive_nodes_wrd = []
            micro_negative_nodes_wrd = []
            #Macro
            for node in macro_nodes:
                if node == PAD:
                    macro_nodes_idx.append(PAD)
                else:
                    if node in self.word_converter:
                        node_ = self.word_converter[node]
                    else:
                        node_ = node
                    macro_nodes_idx.append(self.enc_w2id.get(node_, UNK))

            qnode_idx = []
            for qn in qnode:
                qnode_idx.append(self.enc_w2id.get(qn, UNK))

            if len(macro_nodes_idx) + len(qnode_idx) >= self.len_threshold:
                if self.split == 'val' or self.split == 'test':
                    print ('len', len(macro_nodes_idx) + len(qnode_idx))
                return None

            else:
                #Micro
                for nodes_pos in micro_positive_nodes:
                    node_idx_pos = []
                    for node_pos in nodes_pos:
                        if node_pos in self.word_converter:
                            node_pos = self.word_converter[node_pos]
                        node_idx_pos.append(self.enc_w2id.get(node_pos, UNK))
                    micro_positive_nodes_wrd.append(node_idx_pos)

                for nodes_neg in micro_negative_nodes:
                    node_idx_neg = []
                    for node_neg in nodes_neg:
                        if node_neg in self.word_converter:
                            node_neg = self.word_converter[node_neg]
                        node_idx_neg.append(self.enc_w2id.get(node_neg, UNK))
                    micro_negative_nodes_wrd.append(node_idx_neg)



                return vis_fea, np.asarray(macro_nodes_idx).astype('int64'), np.asarray(macro_obj_locs).astype('int64'), \
                       macro_edges, np.asarray(micro_positive_nodes_wrd).astype('int64'), np.asarray(micro_negative_nodes_wrd).astype('int64'), \
                       np.asarray(qnode_idx).astype('int64'), qedge, answer, self.topN
        except:
            return None

    def __len__(self):
        return len(self.q_list)



def collate_fn(data):
    data = list(filter(lambda x: x is not None, data))
    vis_fea, macro_nodes_idx, macro_obj_locs, macro_edges, micro_positive_nodes_wrd, micro_negative_nodes_wrd, \
    qnode_idx, qedge, answer, topN = zip(*data)
    topN = topN[0]

    answer = np.stack(answer, axis = 0)
    batch_size = len(macro_nodes_idx)

    # Process the vis feature.
    max_fea_len = 0
    for fea in vis_fea:
        max_fea_len = max(max_fea_len, fea.shape[0])
    fea_ipt = np.zeros((batch_size, max_fea_len, vis_fea[0].shape[1]), dtype='float32')
    fea_ipt_mask = np.zeros((batch_size, max_fea_len, max_fea_len), dtype='int32')
    for i in range(batch_size):
        fea_ipt[i, 0:vis_fea[i].shape[0], :] = vis_fea[i]
        fea_ipt_mask[i, 0:vis_fea[i].shape[0], 0:vis_fea[i].shape[0]] = 1

    #Process the syb feature
    #max node len
    max_node_len = 0
    for node in macro_nodes_idx:
        max_node_len = max(max_node_len, node.shape[0])
    # print ('max_node_len', max_node_len)

    macro_node_ipt = np.zeros((batch_size, max_node_len), dtype='int64')
    macro_node_ipt_mask = np.zeros((batch_size, max_node_len, max_node_len), dtype = 'int32')
    macro_node_ipt_graph = np.zeros((batch_size, max_node_len, max_node_len), dtype = 'int32')
    macro_node_ipt[:] = PAD

    for i in range(batch_size):
        macro_node_ipt[i, 0:macro_nodes_idx[i].shape[0]] = macro_nodes_idx[i]
        macro_node_ipt_mask[i, 0:macro_nodes_idx[i].shape[0], 0:macro_nodes_idx[i].shape[0]] = 1

    # generate graph.
    for row, edge in enumerate(macro_edges):
        edge = np.asarray(edge).astype('int32')
        if edge.size == 0:
            continue
        if len(edge.shape) == 1:
            edge = edge[np.newaxis, :]
        macro_node_ipt_graph[row, edge[:, 0], edge[:, 1]] = 1

    #max obj len
    max_obj_len = max_fea_len
    macro_obj_loc_ipt = np.zeros((batch_size, max_obj_len), dtype='int64')
    macro_obj_loc_ipt[:] = LOC_PAD

    micro_positive_obj_ipt = np.zeros((batch_size, max_obj_len, topN), dtype='int64')
    micro_negative_obj_ipt = np.zeros((batch_size, max_obj_len, topN), dtype='int64')
    micro_positive_obj_ipt[:] = PAD
    micro_negative_obj_ipt[:] = PAD

    micro_obj_ipt_mask = np.zeros((batch_size, max_obj_len, topN), dtype='int32')

    for i in range(batch_size):
        macro_obj_loc_ipt[i, 0:macro_obj_locs[i].shape[0]] = macro_obj_locs[i]
        micro_positive_obj_ipt[i, 0:micro_positive_nodes_wrd[i].shape[0], :] = micro_positive_nodes_wrd[i]
        micro_negative_obj_ipt[i, 0:micro_negative_nodes_wrd[i].shape[0], :] = micro_negative_nodes_wrd[i]

        micro_obj_ipt_mask[i, 0:macro_obj_locs[i].shape[0], :] = 1

    # Process question node
    max_q_len = 0
    for node in qnode_idx:
        max_q_len = max(max_q_len, node.shape[0])

    node_q_ipt = np.zeros((batch_size, max_q_len), dtype='int64')
    node_q_ipt_mask = np.zeros((batch_size, max_q_len, max_q_len), dtype='int32')
    node_q_ipt_graph = np.zeros((batch_size, max_q_len, max_q_len), dtype='int32')
    node_q_ipt[:] = PAD

    for i in range(batch_size):
        node_q_ipt[i, 0:qnode_idx[i].shape[0]] = qnode_idx[i]
        node_q_ipt_mask[i, 0:qnode_idx[i].shape[0], 0:qnode_idx[i].shape[0]] = 1
    # generate graph.
    for row, edge in enumerate(qedge):
        edge = np.asarray(edge).astype('int32')
        node_q_ipt_graph[row, edge[:, 0], edge[:, 1]] = 1

    data_ipt = {}
    data_ipt['vis_fea'] = torch.from_numpy(fea_ipt)
    data_ipt['vis_fea_mask'] = torch.from_numpy(fea_ipt_mask) #vis_mask


    data_ipt['macro_node_ipt'] = torch.from_numpy(macro_node_ipt)
    data_ipt['macro_graph_ipt'] = torch.from_numpy(macro_node_ipt_graph) #syb graph
    data_ipt['macro_node_mask'] = torch.from_numpy(macro_node_ipt_mask)  # syb_mask

    data_ipt['macro_obj_loc_ipt'] = torch.from_numpy(macro_obj_loc_ipt)
    # data_ipt['macro_obj_loc_mask'] = torch.from_numpy(macro_obj_loc_ipt_mask)
    # data_ipt['macro_rel_loc_ipt'] = torch.from_numpy(macro_rel_loc_ipt)
    # data_ipt['macro_rel_loc_mask'] = torch.from_numpy(macro_rel_loc_ipt_mask)

    data_ipt['micro_positive_obj_ipt'] = torch.from_numpy(micro_positive_obj_ipt)
    data_ipt['micro_negative_obj_ipt'] = torch.from_numpy(micro_negative_obj_ipt)
    data_ipt['micro_obj_mask'] = torch.from_numpy(micro_obj_ipt_mask)
    # data_ipt['micro_rel_mask'] = torch.from_numpy(micro_rel_ipt_mask)

    data_ipt['q_ipt'] = torch.from_numpy(node_q_ipt)
    data_ipt['q_ipt_mask'] = torch.from_numpy(node_q_ipt_mask)
    data_ipt['q_ipt_graph'] = torch.from_numpy(node_q_ipt_graph)
    data_ipt['answer'] = torch.from_numpy(answer).long()
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
