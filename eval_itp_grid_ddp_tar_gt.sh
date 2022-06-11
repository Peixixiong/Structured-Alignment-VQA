
weight_fn=~/src/GraphVQA/aml_weights/pt-EXP_GQA_V2_GT_LOC-3ab34615_1602720796_8bfc93dc_with_dec/model_6.pth
#CUDA_VISIBLE_DEVICES=0 python eval_itp_grid_ddp_tar_gt.py --batch_size 32  --model_v 2 --with_dec --enc_vocab_fn preprocessed/de.vocab.tsv.composite --ans_vocab_fn preprocessed/answer.txt --with_loc --with_dec --maxlen 500 --maxlen_q 100 --weight_fn $weight_fn
CUDA_VISIBLE_DEVICES=0 python eval_itp_grid_ddp_tar_gt.py --batch_size 32  --model_v 2 --with_dec --enc_vocab_fn preprocessed/de.vocab.tsv.composite --ans_vocab_fn preprocessed/answer.txt --with_loc --with_dec --maxlen 500 --maxlen_q 100 --weight_fn $weight_fn --q_tar_fn_val train.tar --gt_graph_fn_val train_sceneGraphs.json
