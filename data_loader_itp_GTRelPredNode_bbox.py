from __future__ import print_function

import numpy as numpy
import torch.utils.data as data
import numpy as np
import torch
import json
import os
import math
import random
from io import BytesIO
from synonym_word_converter import *

import tarfile
import argparse
import pdb
import codecs


def load_graph_vocab(de_vocab_fn):
    vocab = [line.split()[0] for line in codecs.open(de_vocab_fn, 'r', 'utf-8').read().splitlines()]
    index = [int(line.split()[1]) for line in codecs.open(de_vocab_fn, 'r', 'utf-8').read().splitlines()]

    word2idx = {word: index[idx] for idx, word in enumerate(vocab)}
    idx2word = {index[idx]: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_answer_vocab(en_vocab_fn, min_cnt):
    vocab = [' '.join(line.split()[:-1]) for line in codecs.open(en_vocab_fn, 'r', 'utf-8').read().splitlines() if
             int(line.split()[-1]) >= min_cnt]
    word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}
    idx2word = {idx + 1: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


PAD = 400000
UNK = 400001
END = 400003


class GQADataset_topN(data.Dataset):
    def __init__(self, split, opt, fea_tar_fn, q_tar_fn, g_tar_fn, topN, with_loc = True, with_gt_relation = True):
        super(GQADataset_topN, self).__init__()
        self.split = split
        self.opt = opt
        self.with_loc = with_loc
        self.pos_grid_num = 10
        self.topN = topN
        self.with_gt_relation = with_gt_relation
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

    def load_tar_infos_list(self, tar_fn, ext='.json'):
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
        # do not use the gt relation,
        # instead use the statistically frequent relation.

        idx_obj = []
        nodes_obj = []
        posi_obj = []
        relation = []
        nodes = []
        dict_attr2idx = {}
        dict_obj2idx = {}
        dict_rel2pos = {}
        dict_pos2idx = {}
        for row_idx, (obj_idxs, obj) in enumerate(zip(data_info['objects_id'], gt_graph['objects'])):
            dict_obj2idx[obj] = len(dict_obj2idx)
            for obj_idx in obj_idxs:
                if obj_idx < len(self.vg_classes):
                    pred_node = self.vg_classes[obj_idx].replace(' ', '')
                    break
            nodes.append(pred_node)
            # nodes.append(gt_graph['objects'][obj]['name'].strip().replace(' ', ''))

        for obj in gt_graph['objects']:
            data = gt_graph['objects'][obj]
            x, y, w, h = data['x'], data['y'], data['w'], data['h']
            pos_obj = dict_obj2idx[obj]
            # pos_obj = len(nodes)
            if len(data['attributes']) > 0:
                attr_name = data['attributes'][0]
                if attr_name in dict_attr2idx:
                    pos_attr = dict_attr2idx[attr_name]
                else:
                    pos_attr = len(nodes)
                    nodes.append(data['attributes'][0].replace(' ', ''))
                    dict_attr2idx[attr_name] = pos_attr

                relation.append([pos_obj, pos_attr])
                relation.append([pos_attr, pos_obj])

            if self.with_gt_relation:
                for rel in data['relations']:
                    # {'object': '4201979', 'name': 'to the left of'}
                    tgt_obj_idx = dict_obj2idx[rel['object']]
                    r_name = rel['name'].replace(' ', '')
                    pos_rel = len(nodes)

                    if r_name in dict_rel2pos:
                        pos_rel = dict_rel2pos[r_name]
                    else:
                        dict_rel2pos[r_name] = pos_rel
                        r_name = ''.join(r_name.split())
                        nodes.append(r_name)
                    relation.append([pos_obj, pos_rel])
                    relation.append([pos_rel, tgt_obj_idx])

            idx_obj.append(pos_obj)
            nodes_obj.append(data['name'])
            posi_obj.append([x + w / 2, y + h / 2])
            if self.with_loc:
                # add position
                for cx, cy in zip([x, x + w], [y, y + h]):
                    pos_node_name = 'x' + str(math.floor(cx / gt_graph['width'] * self.pos_grid_num)) + 'y' + str(
                        math.floor(cy / gt_graph['height']) * self.pos_grid_num)
                    if pos_node_name in dict_pos2idx:
                        pos_pos = dict_pos2idx[pos_node_name]
                    else:
                        pos_pos = len(nodes)
                        dict_pos2idx[pos_node_name] = pos_pos
                        nodes.append(pos_node_name)

                    relation.append([pos_obj, pos_pos])
                    relation.append([pos_pos, pos_obj])

        if not self.with_gt_relation:
            num_obj = len(idx_obj)
            for i in range(num_obj):
                for j in range(num_obj):
                    if j == i:
                        continue
                    key = nodes_obj[i] + ',' + nodes_obj[j]
                    if key in self.gt_relations:
                        r_name = self.gt_relations[key].replace(' ', '')
                        pos_rel = len(nodes)

                        if r_name in dict_rel2pos:
                            pos_rel = dict_rel2pos[r_name]
                        else:
                            dict_rel2pos[r_name] = pos_rel
                            r_name_t = ''.join(r_name.split())
                            nodes.append(r_name_t)
                        if not ('left' in r_name and posi_obj[i][0] < posi_obj[j][0] or
                                'right' in r_name and posi_obj[i][0] > posi_obj[j][0] or
                                'top' in r_name and posi_obj[i][1] < posi_obj[j][1] or
                                'under' in r_name and posi_obj[i][1] > posi_obj[j][1]):
                            continue
                        relation.append([idx_obj[i], pos_rel])
                        relation.append([pos_rel, idx_obj[j]])

        return nodes, relation, idx_obj

    def __getitem__(self, index):
        q_mem = self.q_list[index]
        with tarfile.open(self.q_tar_fn) as tar_fid:
            qinfo = json.load(tar_fid.extractfile(q_mem))
        qnode = qinfo['node_list']
        qedge = qinfo['edge_pair']
        answer = qinfo['answer']
        answer = self.ans_w2id.get(answer, 0)  # default all 0 classes
        answer = np.asarray(answer).astype('int32')

        image_id = qinfo['image_id']

        try:
            # load grid image_fea
            with tarfile.open(self.fea_tar_fn) as tar_fid:
                fea_mem = self.fea_dict[image_id]
                array_file = BytesIO()
                array_file.write(tar_fid.extractfile(fea_mem).read())
                array_file.seek(0)
                vis_fea = np.load(array_file)
                vis_fea = vis_fea['x']
            # gt graph.
            gt_graph = self.gt_graph[image_id]

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

                nodes, edges, idx_of_obj = self.convert_graph(data_info, self.opt.bg_class, bbox, gt_graph)
                idx_of_obj = np.asarray(idx_of_obj).astype('int64')

            nodes_idx = []
            for node in nodes:
                if node in self.word_converter:
                    node_ = self.word_converter[node]
                else:
                    node_ = node
                nodes_idx.append(self.enc_w2id.get(node_, UNK))

            qnode_idx = []
            for qn in qnode:
                qnode_idx.append(self.enc_w2id.get(qn, UNK))
            return vis_fea, np.asarray(nodes_idx).astype('int64'), edges, np.asarray(qnode_idx).astype(
                'int64'), qedge, answer, idx_of_obj
        except:
            return None

    def __len__(self):
        return len(self.q_list)


def collate_fn(data):
    data = list(filter(lambda x: x is not None, data))
    vis_fea, nodes_idx, edges, qnode_idx, qedge, answer, idx_of_obj = zip(*data)
    idx_of_obj = [torch.from_numpy(idx) for idx in idx_of_obj]

    answer = np.stack(answer, axis=0)
    max_node_len = 0
    batch_size = len(nodes_idx)
    # process the vis feature.
    max_fea_len = 0
    for fea in vis_fea:
        max_fea_len = max(max_fea_len, fea.shape[0])
    fea_ipt = np.zeros((batch_size, max_fea_len, vis_fea[0].shape[1]), dtype='float32')
    fea_ipt_mask = np.zeros((batch_size, max_fea_len, max_fea_len), dtype='int32')
    for i in range(batch_size):
        fea_ipt[i, 0:vis_fea[i].shape[0], :] = vis_fea[i]
        fea_ipt_mask[i, 0:vis_fea[i].shape[0], 0:vis_fea[i].shape[0]] = 1

    # Process syb input
    for node in nodes_idx:
        max_node_len = max(max_node_len, node.shape[0])
    if max_node_len >= 245:
        print('max_node_len, val', max_node_len)

    node_ipt = np.zeros((batch_size, max_node_len), dtype='int64')
    node_ipt_mask = np.zeros((batch_size, max_node_len, max_node_len), dtype='int32')
    node_ipt_graph = np.zeros((batch_size, max_node_len, max_node_len), dtype='int32')
    node_ipt_vis_graph = np.zeros((batch_size, max_fea_len, max_fea_len), dtype='int32')
    node_ipt[:] = PAD

    for i in range(batch_size):
        node_ipt[i, 0:nodes_idx[i].shape[0]] = nodes_idx[i]
        node_ipt_mask[i, 0:nodes_idx[i].shape[0], 0:nodes_idx[i].shape[0]] = 1

    # generate graph.
    for row, edge in enumerate(edges):
        edge = np.asarray(edge).astype('int32')
        if edge.size == 0:
            continue
        if len(edge.shape) == 1:
            edge = edge[np.newaxis, :]
        node_ipt_graph[row, edge[:, 0], edge[:, 1]] = 1

    # process question node
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
    data_ipt['vis_fea_mask'] = torch.from_numpy(fea_ipt_mask)  # vis_mask
    data_ipt['vis_syb_ipt'] = torch.from_numpy(node_ipt)
    data_ipt['vis_syb_graph'] = torch.from_numpy(node_ipt_graph)  # syb graph
    data_ipt['vis_vis_graph'] = torch.from_numpy(node_ipt_vis_graph)  # vis graph
    data_ipt['vis_syb_mask'] = torch.from_numpy(node_ipt_mask)  # syb_mask
    data_ipt['q_ipt'] = torch.from_numpy(node_q_ipt)
    data_ipt['q_ipt_mask'] = torch.from_numpy(node_q_ipt_mask)
    data_ipt['q_ipt_graph'] = torch.from_numpy(node_q_ipt_graph)
    data_ipt['answer'] = torch.from_numpy(answer).long()
    data_ipt['idx_of_obj'] = idx_of_obj
    return data_ipt


def collate_fn_gt(data):
    vis_fea, nodes_idx, edges, qnode_idx, qedge, answer, idx_of_obj = zip(*data)

    idx_of_obj = [torch.from_numpy(idx) for idx in idx_of_obj]

    batch_size = len(nodes_idx)
    # process the vis feature.
    max_fea_len = 0
    for fea in vis_fea:
        max_fea_len = max(max_fea_len, fea.shape[0])

    fea_ipt = np.zeros((batch_size, max_fea_len, vis_fea[0].shape[1]), dtype='float32')
    fea_ipt_mask = np.zeros((batch_size, max_fea_len), dtype='int32')
    for i in range(batch_size):
        fea_ipt[i, 0:vis_fea[i].shape[0], :] = vis_fea[i]
        fea_ipt_mask[i, 0:vis_fea[i].shape[0]] = 1

    answer = np.stack(answer, axis=0)
    max_node_len = 0
    for node in nodes_idx:
        max_node_len = max(max_node_len, node.shape[0])

    # Process input
    node_ipt = np.zeros((batch_size, max_node_len), dtype='int64')
    node_ipt_mask = np.zeros((batch_size, max_node_len), dtype='int32')
    node_ipt_graph = np.zeros((batch_size, max_node_len, max_node_len), dtype='int32')
    node_ipt[:] = PAD

    for i in range(batch_size):
        node_ipt[i, 0:nodes_idx[i].shape[0]] = nodes_idx[i]
        node_ipt_mask[i, 0:nodes_idx[i].shape[0]] = 1
    # generate graph.
    for row, edge in enumerate(edges):
        edge = np.asarray(edge).astype('int32')
        if edge.size == 0:
            continue
        if len(edge.shape) == 1:
            edge = edge[np.newaxis, :]
        node_ipt_graph[row, edge[:, 0], edge[:, 1]] = 1

    # process question node
    max_q_len = 0
    for node in qnode_idx:
        max_q_len = max(max_q_len, node.shape[0])

    node_q_ipt = np.zeros((batch_size, max_q_len), dtype='int64')
    node_q_ipt_mask = np.zeros((batch_size, max_q_len), dtype='int32')
    node_q_ipt_graph = np.zeros((batch_size, max_q_len, max_q_len), dtype='int32')
    node_q_ipt[:] = PAD

    for i in range(batch_size):
        node_q_ipt[i, 0:qnode_idx[i].shape[0]] = qnode_idx[i]
        node_q_ipt_mask[i, 0:qnode_idx[i].shape[0]] = 1
    # generate graph.
    for row, edge in enumerate(qedge):
        edge = np.asarray(edge).astype('int32')
        node_q_ipt_graph[row, edge[:, 0], edge[:, 1]] = 1

    data_ipt = {}
    data_ipt['vis_fea'] = torch.from_numpy(fea_ipt)
    data_ipt['vis_fea_mask'] = torch.from_numpy(fea_ipt_mask)
    data_ipt['vis_syb_ipt'] = torch.from_numpy(node_ipt)
    data_ipt['vis_syb_graph'] = torch.from_numpy(node_ipt_graph)
    data_ipt['vis_syb_mask'] = torch.from_numpy(node_ipt_mask)
    data_ipt['q_ipt'] = torch.from_numpy(node_q_ipt)
    data_ipt['q_ipt_mask'] = torch.from_numpy(node_q_ipt_mask)
    data_ipt['q_ipt_graph'] = torch.from_numpy(node_q_ipt_graph)
    data_ipt['answer'] = torch.from_numpy(answer).long()
    data_ipt['idx_of_obj'] = idx_of_obj

    return data_ipt


if __name__ == '__main__':
    # opt = {'keep_res':True, 'pad':False}
    parser = argparse.ArgumentParser()
    # basic experiment setting
    parser.add_argument('--data_dir_azure', default='./tmp')
    parser.add_argument('--fea_tar_fn', default='gt_bua_npz.tar')
    parser.add_argument('--q_tar_fn', default='train.tar')
    parser.add_argument('--gt_graph_fn', default='train_sceneGraphs.json')
    parser.add_argument('--gt_relation_fn', default='GT_relations_dict.json')
    parser.add_argument('--obj_vocab_fn', type=str, default='objects_vocab.txt')
    parser.add_argument('--attr_vocab_fn', type=str, default='attributes_vocab.txt')
    parser.add_argument('--bbox_bin_num', type=int, default=64)
    parser.add_argument('--min_cnt', type=int, default=4)
    parser.add_argument('--enc_vocab_fn', type=str, default='preprocessed/de.vocab.tsv.composite')
    parser.add_argument('--ans_vocab_fn', type=str, default='preprocessed/answer.txt')
    opt = parser.parse_args()

    data = GQADataset(opt, opt.fea_tar_fn, opt.q_tar_fn, opt.gt_graph_fn, True, True)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=4,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    # for i in range(len(data)):
    #    if i % 1000 == 0:
    #        print(i, len(data))
    #    data[i]
    max_len = 0
    for i, data in enumerate(loader):
        print('i = ', i)
        if len(data['vis_syb_ipt'].shape) <= 1:
            print('error shape', data['vis_syb_ipt'].shape)
        if data['vis_syb_ipt'].shape[1] > max_len:
            max_len = data['vis_syb_ipt'].shape[1]
            print(max_len)
        if i % 1000 == 0:
            print(' i = {}'.format(i))
