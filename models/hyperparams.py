# -*- coding: utf-8 -*-
#/usr/bin/python2


class Hyperparams:
    '''Hyperparameters'''
    # data
    root = '/'

    train_graph_syb = root + 'train_graph_syb.npy'
    train_graph_vis = root + 'train_graph_vis.npy'    
    source_train_syb = root + 'source_train_syb.npy'
    source_train_vis = root + 'source_train_vis.npy'
    source_train0 = root + 'source_train_vis.h5'
    target_train = root + 'target_train_syb.npy'
    train_q_id = root + 'train_q_id.npy'
    '''
    test_graph_syb = root + 'train_graph_syb.npy'
    test_graph_vis = root + 'train_graph_vis.npy'    
    source_test_syb = root + 'source_train_syb.npy'
    source_test_vis = root + 'source_train_vis.npy'
    source_test0 = root + 'source_train_vis.h5'
    target_test = root + 'target_train_syb.npy'
    test_q_id = root + 'train_q_id.npy'
    
    ''' 
    #test_graph_syb = root + 'test2_graph_syb.npy'
    #test_graph_vis = root + 'test2_graph_vis.npy'    
    #source_test_syb = root + 'source2_test_syb.npy'
    #source_test_vis = root + 'source2_test_vis.npy'
    #source_test0 = root + 'source2_test_vis.h5'
    #target_test = root + 'target2_test_syb.npy'
    #test_q_id = root + 'test2_q_id.npy'
    #type1_test = root + 'test2_type1.npy'
    #type2_test = root + 'test2_type2.npy'
    #type3_test = root + 'test2_type3.npy'
    test_graph_syb = root + 'val_graph_syb.npy'
    test_graph_vis = root + 'val_graph_vis.npy'    
    source_test_syb = root + 'source_val_syb.npy'
    source_test_vis = root + 'source_val_vis.npy'
    source_test0 = root + 'source_val_vis.h5'
    target_test = root + 'target_val_syb.npy'
    test_q_id = root + 'val_q_id.npy'

    

#     source_train = 'corpora/train.tags.de-en.de'
#     target_train = 'corpora/train.tags.de-en.en'
#     source_test = 'corpora/IWSLT16.TED.tst2014.de-en.de.xml'
#     target_test = 'corpora/IWSLT16.TED.tst2014.de-en.en.xml'
    
    # training
    batch_size = 6##32 # alias = N
    test_batch_size = 256 #256 32
    lr = 0.0001 # lr = 0.0001 learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir' # log directory

    model_dir = './models/'  # saving directory
    output_dir = 'outputs/'

    # model
    maxlen = 100 #222 # 170 Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
                # GQA_sgg_official/ 182
    min_cnt = 10 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # 512 alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 40
    num_heads = 8
    dropout_rate = 0.5
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    eval_epoch = 5  # epoch of model for eval
    preload = None  # epcho of preloaded model for resuming training
    
    
    
    
