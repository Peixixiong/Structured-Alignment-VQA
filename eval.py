# -*- coding: utf-8 -*-

from __future__ import print_function
import codecs
import os
import argparse
import numpy as np

from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_de_vocab, load_en_vocab, load_test_data2
#from nltk.translate.bleu_score import corpus_bleu
from AttModel import AttModel
from torch.autograd import Variable
import torch
from torchtext import vocab
import collections

def dictCoulnt(type_, old_dict, correct):
    if type_ not in old_dict:
        if correct == 1:
            old_dict[type_] = [1,1]
        else:
            old_dict[type_] = [0,1]            
    else:
        if correct == 1:
            old_dict[type_] = [old_dict[type_][0]+1, old_dict[type_][1]+1]
        else:
            old_dict[type_] = [old_dict[type_][0], old_dict[type_][1]+1]
    return old_dict

def eval(root_dir):
    # Load data
    X_syb, X_vis, Y, Graphs_syb, Graphs_vis, X0, type1_list, type2_list, type3_list = load_test_data2(root_dir)
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    enc_voc = len(de2idx)
    dec_voc = len(en2idx)
    # load model
    glove = vocab.GloVe(name='6B', dim=300)
    model = AttModel(glove, hp, enc_voc, dec_voc)

    model_dic = torch.load(root_dir + '/outputs/catpred_model_epoch_%02d' % hp.eval_epoch + '.pth')
    new_state_dict = collections.OrderedDict()
    for k, v in model_dic.items():
        #name = k.replace('module.', '')# remove `module.`
        name = k
        new_state_dict[name] = v
    if torch.cuda.device_count() >= 1:
        model = torch.nn.DataParallel(model).cuda() 
    model.load_state_dict(new_state_dict)

    print('Model Loaded.')
    count,total = 0,0
    type1_dict = {}
    type2_dict = {}
    type3_dict = {}  
    model.eval()
    for i in range(len(X_syb) // hp.test_batch_size):
        print (i,len(X_syb) // hp.test_batch_size)
        # Get mini-batches
        type1 = type1_list[i * hp.test_batch_size: (i + 1) * hp.test_batch_size]
        type2 = type2_list[i * hp.test_batch_size: (i + 1) * hp.test_batch_size]
        type3 = type3_list[i * hp.test_batch_size: (i + 1) * hp.test_batch_size]        
        x_syb = X_syb[i * hp.test_batch_size: (i + 1) * hp.test_batch_size]
        x_vis = X_vis[i * hp.test_batch_size: (i + 1) * hp.test_batch_size]
            
        g_syb = Graphs_syb[i * hp.test_batch_size: (i + 1) * hp.test_batch_size]
        g_vis = Graphs_vis[i * hp.test_batch_size: (i + 1) * hp.test_batch_size]

        y = Y[i * hp.test_batch_size: (i + 1) * hp.test_batch_size]
        # Autoregressive inference
        x_syb_ = Variable(torch.LongTensor(x_syb).cuda())
        x_vis_ = Variable(torch.LongTensor(x_vis).cuda())
            
        g_syb_ = Variable(torch.LongTensor(g_syb).cuda())
        g_vis_ = Variable(torch.LongTensor(g_vis).cuda())
        y_ = Variable(torch.LongTensor(y).cuda())
                
        X0_batch = np.zeros([x_syb_.shape[0], hp.maxlen, 2048], np.float32)
        id2 = 0
        for idx in range(i * hp.test_batch_size, (i + 1) * hp.test_batch_size):       
            X0_batch[id2] = np.lib.pad(np.array(X0[idx]), [0, hp.maxlen-np.array(X0[idx]).shape[0]], 'constant', constant_values=(-1, -1))[:,:2048]
            id2 += 1  
                
        x0_ = Variable(torch.Tensor(X0_batch).cuda())                        
        preds_t = torch.LongTensor(np.zeros((hp.test_batch_size, 2), np.int32)).cuda()
        preds = Variable(preds_t)
        for j in range(2): #range(hp.maxlen):
            _, _preds, _ = model(x_syb_, x_vis_, preds, g_syb_, g_vis_, x0_)
            preds_t[:, j] = _preds.data[:, j]
            preds = Variable(preds_t.long())                   
        #preds = preds.data.cpu().numpy()
        y_1 = y_.eq(3.).float()
        y_2 = y_.eq(0.).float()
        istarget = (1. - (y_1 + y_2)).view(-1)        
        for target, pred, type1_, type2_, type3_ in zip(y_, preds, type1, type2, type3):
            type1_ = str(type1_)
            type2_ = str(type2_)
            type3_ = str(type3_)
            if target[0] == pred[0]:
                type1_dict = dictCoulnt(type1_, type1_dict, 1)
                type2_dict = dictCoulnt(type2_, type2_dict, 1)
                type3_dict = dictCoulnt(type3_, type3_dict, 1)
            else:
                type1_dict = dictCoulnt(type1_, type1_dict, 0)
                type2_dict = dictCoulnt(type2_, type2_dict, 0)
                type3_dict = dictCoulnt(type3_, type3_dict, 0)            
        count += torch.sum(preds.eq(y_).float().view(-1) * istarget) 
        total += torch.sum(istarget)
        print (count,total)
        print (count/total)
    print (type1_dict)
    print (type2_dict)
    print (type3_dict)  
      
def eval_fast(root_dir):
    # Load data
    X_syb, X_vis, Y, Graphs_syb, Graphs_vis, X0= load_test_data(root_dir)
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    enc_voc = len(de2idx)
    dec_voc = len(en2idx)
    # load model
    glove = vocab.GloVe(name='6B', dim=300)
    model = AttModel(glove, hp, enc_voc, dec_voc)

    model_dic = torch.load(root_dir + '/outputs/catpred_model_epoch_%02d' % hp.eval_epoch + '.pth')
    new_state_dict = collections.OrderedDict()
    for k, v in model_dic.items():
        #name = k.replace('module.', '')# remove `module.`
        name = k
        new_state_dict[name] = v
    if torch.cuda.device_count() >= 1:
        model = torch.nn.DataParallel(model).cuda() 
    model.load_state_dict(new_state_dict)

    print('Model Loaded.')
    count,total = 0,0
    type1_dict = {}
    type2_dict = {}
    type3_dict = {}  
    model.eval()
    for i in range(len(X_syb) // hp.test_batch_size):
        print (i,len(X_syb) // hp.test_batch_size)
        # Get mini-batches     
        x_syb = X_syb[i * hp.test_batch_size: (i + 1) * hp.test_batch_size]
        x_vis = X_vis[i * hp.test_batch_size: (i + 1) * hp.test_batch_size]
            
        g_syb = Graphs_syb[i * hp.test_batch_size: (i + 1) * hp.test_batch_size]
        g_vis = Graphs_vis[i * hp.test_batch_size: (i + 1) * hp.test_batch_size]

        y = Y[i * hp.test_batch_size: (i + 1) * hp.test_batch_size]
        # Autoregressive inference
        x_syb_ = Variable(torch.LongTensor(x_syb).cuda())
        x_vis_ = Variable(torch.LongTensor(x_vis).cuda())
            
        g_syb_ = Variable(torch.LongTensor(g_syb).cuda())
        g_vis_ = Variable(torch.LongTensor(g_vis).cuda())
        y_ = Variable(torch.LongTensor(y).cuda())
                
        X0_batch = np.zeros([x_syb_.shape[0], hp.maxlen, 2048], np.float32)
        id2 = 0
        for idx in range(i * hp.test_batch_size, (i + 1) * hp.test_batch_size):       
            X0_batch[id2] = np.lib.pad(np.array(X0[idx]), [0, hp.maxlen-np.array(X0[idx]).shape[0]], 'constant', constant_values=(-1, -1))[:,:2048]
            id2 += 1  
                
        x0_ = Variable(torch.Tensor(X0_batch).cuda())                        
        preds_t = torch.LongTensor(np.zeros((hp.test_batch_size, 2), np.int32)).cuda()
        preds = Variable(preds_t)
        for j in range(2): #range(hp.maxlen):
            _, _preds, _ = model(x_syb_, x_vis_, preds, g_syb_, g_vis_, x0_)
            preds_t[:, j] = _preds.data[:, j]
            preds = Variable(preds_t.long())                   
        #preds = preds.data.cpu().numpy()
        y_1 = y_.eq(3.).float()
        y_2 = y_.eq(0.).float()
        istarget = (1. - (y_1 + y_2)).view(-1)        
           
        count += torch.sum(preds.eq(y_).float().view(-1) * istarget) 
        total += torch.sum(istarget)
        print (count,total)
        print (count/total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
    args = parser.parse_args()

    data_folder = args.data_folder
    eval_fast(data_folder)
    print('Done')



