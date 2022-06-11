# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

import numpy as np
import codecs
import random
import torch
import h5py
import tqdm
from hyperparams import Hyperparams as hp
import os
import json

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def load_de_vocab(de_vocab_fn):
    print(de_vocab_fn)
    vocab = [line.split()[0] for line in codecs.open(de_vocab_fn, 'r', 'utf-8').read().splitlines() ]
    index = [line.split()[1] for line in codecs.open(de_vocab_fn, 'r', 'utf-8').read().splitlines() ]

    word2idx = {word: index[idx] for idx, word in enumerate(vocab)}
    idx2word = {index[idx]: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_en_vocab(en_vocab_fn, min_cnt):
    vocab = [' '.join(line.split()[:-1]) for line in codecs.open(en_vocab_fn, 'r', 'utf-8').read().splitlines() if int(line.split()[-1])>=min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def create_data(source_sents_syb, source_sents_vis, target_sents, graph_edges_syb, graph_edges_vis, source_sents0, q_id_list, split, root_dir):  
    de2idx, idx2de = load_de_vocab(root_dir + '/preprocessed/de.vocab.tsv')
    en2idx, idx2en = load_en_vocab(root_dir + '/preprocessed/en.vocab.tsv', 10)
    
    dict_id2img_id = {}
    id2img_id_fn = ''
    if split == 'train' and os.path.isfile(root_dir + hp.train_id2img_id_fn):
        id2img_id_fn = root_dir + hp.train_id2img_id_fn
        print('Loading training idmap from {}'.format(id2img_id_fn))

    elif split == 'test' and os.path.isfile(root_dir + hp.test_id2img_id_fn):
        id2img_id_fn = root_dir + hp.test_id2img_id_fn
        print('Loading testing idmap from {}'.format(id2img_id_fn))
    
    if os.path.isfile(id2img_id_fn):
        dict_id2img_id = json.load(open(id2img_id_fn))
    
    x_list_syb, x_list_vis, y_list, g_list_syb, g_list_vis, x_list0, q_list = [], [], [], [], [], [], []
    idx = 0
    for source_sent_syb, source_sent_vis, target_sent, graph_link_syb, graph_link_vis, q_id in tqdm.tqdm(zip(source_sents_syb, source_sents_vis, target_sents, graph_edges_syb, graph_edges_vis, q_id_list), mininterval = 120, miniters = 50):        
        x_syb = [de2idx.get(word, '400001') for word in (source_sent_syb + u" </s>").split()] # 1: OOV, </S>: End of Text
        x_vis = [de2idx.get(word, '400001') for word in (source_sent_vis + u" </s>").split()] # 1: OOV, </S>: End of Text
        #y = [en2idx.get(word, 1) for word in [target_sent, u"</s>"]]  
        y = [en2idx.get(word, 1) for word in [target_sent]] 
        #img_idx = str(idx)
        if len(dict_id2img_id) > 0:
            img_idx = dict_id2img_id[str(idx)]
        if len(x_syb) <= hp.maxlen and source_sents0[img_idx].shape[0]!=0:            
            x_list_syb.append(np.array(x_syb))
            x_list_vis.append(np.array(x_vis))
            y_list.append(np.array(y))
            g_list_syb.append(np.array(graph_link_syb))
            g_list_vis.append(np.array(graph_link_vis))
            x_list0.append(source_sents0[img_idx])
            q_list.append(q_id+1)
        idx += 1

    X_syb = np.zeros([len(x_list_syb), hp.maxlen], np.int32)
    X_vis = np.zeros([len(x_list_syb), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), 1], np.int32)
    G_syb1 = np.zeros([len(g_list_syb), hp.maxlen, hp.maxlen], np.int8)
    G_vis1 = np.zeros([len(g_list_vis), hp.maxlen, hp.maxlen], np.int8)

    G_syb2 = np.zeros([len(g_list_syb), hp.maxlen, hp.maxlen], np.int8)
    G_vis2 = np.zeros([len(g_list_vis), hp.maxlen, hp.maxlen], np.int8)

    G_syb3 = np.zeros([len(g_list_syb), hp.maxlen, hp.maxlen], np.int8)
    G_vis3 = np.zeros([len(g_list_vis), hp.maxlen, hp.maxlen], np.int8)
    for i, (x_syb, x_vis, y, g_syb, g_vis, x0, q_length) in tqdm.tqdm(enumerate(zip(x_list_syb, x_list_vis, y_list, g_list_syb, g_list_vis, x_list0, q_list)), mininterval = 120, miniters = 50):
        X_syb[i] = np.lib.pad(x_syb, [0, hp.maxlen-len(x_syb)], 'constant', constant_values=(400000, 400000))
        X_vis[i] = np.lib.pad(x_vis, [x0.shape[0], hp.maxlen-len(x_vis)-x0.shape[0]], 'constant', constant_values=(400000, 400000))
        Y[i] = y
        
        G_syb1[i, :len(x_syb) - q_length, len(x_syb) - q_length:len(x_syb)] = 1
        G_syb1[i, len(x_syb) - q_length:len(x_syb), :len(x_syb) - q_length] = 1
        G_syb2[i, :len(x_syb) - q_length, len(x_syb) - q_length:len(x_syb)] = 1
        G_syb2[i, len(x_syb) - q_length:len(x_syb), :len(x_syb) - q_length] = 1
        G_syb3[i, len(x_syb) - q_length:len(x_syb), len(x_syb) - q_length:len(x_syb)] = 1


        G_vis1[i, :x0.shape[0], x0.shape[0]:x0.shape[0]+len(x_vis)] = 1
        G_vis1[i, x0.shape[0]:x0.shape[0]+len(x_vis), :x0.shape[0]] = 1

        G_vis2[i, :x0.shape[0], x0.shape[0]:x0.shape[0]+len(x_vis)] = 1
        G_vis2[i, x0.shape[0]:x0.shape[0]+len(x_vis), :x0.shape[0]] = 1

        G_vis3[i, x0.shape[0]:x0.shape[0]+len(x_vis), x0.shape[0]:x0.shape[0]+len(x_vis)] = 1

        for g_syb_ in g_syb:
            G_syb1[i][g_syb_[0],g_syb_[1]] = 1

        for g_vis_ in g_vis:
            G_vis1[i][g_vis_[0],g_vis_[1]] = 1   
     
    return X_syb, X_vis, Y, G_syb1, G_vis1, G_syb2, G_vis2, G_syb3, G_vis3, x_list0


    
def load_train_data(root_dir, split = 'train'):
    print ('begin')
    input_sents0 = h5py.File(root_dir + hp.source_train0, 'r')
    print ('finished input_sents0')
    input_sents_syb = np.load(root_dir + hp.source_train_syb)
    print ('finished input_sents_syb')
    input_sents_vis = np.load(root_dir + hp.source_train_vis)
    print ('finished input_sents_vis')
    output_sents = np.load(root_dir + hp.target_train)
    print ('finished output_sents')
    graph_edge_syb = np.load(root_dir + hp.train_graph_syb, allow_pickle=True).tolist()
    print ('finished graph_edge_syb')
    graph_edge_vis = np.load(root_dir + hp.train_graph_vis, allow_pickle=True).tolist()
    print ('finished graph_edge_vis')
    q_id = np.load(root_dir + hp.train_q_id)
    print ('finished q_id')
   
    X_syb, X_vis, Y, Graphs_syb1, Graphs_vis1, Graphs_syb2, Graphs_vis2, Graphs_syb3, Graphs_vis3, X0 = create_data(input_sents_syb, input_sents_vis, output_sents, graph_edge_syb, graph_edge_vis, input_sents0, q_id, split, root_dir)
    return X_syb, X_vis, Y, Graphs_syb1, Graphs_vis1, Graphs_syb2, Graphs_vis2, Graphs_syb3, Graphs_vis3, X0


def load_test_data(root_dir, split = 'test'):
    input_sents_syb = np.load(root_dir + hp.source_test_syb)
    print ('finished input_sents_syb')
    input_sents_vis = np.load(root_dir + hp.source_test_vis)
    print ('finished input_sents_vis')
    input_sents0 = h5py.File(root_dir + hp.source_test0, 'r')
    print ('finished input_sents0')
    output_sents = np.load(root_dir + hp.target_test)
    print ('finished output_sents')
    graph_edge_syb = np.load(root_dir + hp.test_graph_syb, allow_pickle=True).tolist()
    print ('finished graph_edge_syb')
    graph_edge_vis = np.load(root_dir + hp.test_graph_vis, allow_pickle=True).tolist()
    print ('finished graph_edge_vis')
    q_id = np.load(root_dir + hp.test_q_id)
    print ('finished q_id')

    X_syb, X_vis, Y, Graphs_syb1, Graphs_vis1, Graphs_syb2, Graphs_vis2, Graphs_syb3, Graphs_vis3, X0 = create_data(input_sents_syb, input_sents_vis, output_sents, graph_edge_syb, graph_edge_vis, input_sents0, q_id, split, root_dir)
    return X_syb, X_vis, Y, Graphs_syb1, Graphs_vis1, Graphs_syb2, Graphs_vis2, Graphs_syb3, Graphs_vis3, X0

def get_batch_indices(total_length, batch_size):
    current_index = 0
    indexs = [i for i in xrange(total_length)]
    random.shuffle(indexs)
    while 1:
        if current_index + batch_size >= total_length:
            break
        current_index += batch_size
        yield indexs[current_index: current_index + batch_size], current_index

