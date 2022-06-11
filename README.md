# SA-VQA: Structured Alignment of Visual and Semantic Representations for Visual Question Answering

This repository is the dev version of implementing the work [SA-VQA: Structured Alignment of Visual and Semantic Representations for Visual Question Answering](https://arxiv.org/pdf/2201.10654.pdf). 

More details will be updated.

# Reference
If you use our code or features, please cite our paper:
```
@article{xiong2022sa,
  title={SA-VQA: Structured Alignment of Visual and Semantic Representations for Visual Question Answering},
  author={Xiong, Peixi and You, Quanzeng and Yu, Pei and Liu, Zicheng and Wu, Ying},
  journal={arXiv preprint arXiv:2201.10654},
  year={2022}
}
```

# License
The SA-VQA is released under the MIT License (refer to the LICENSE file for details).

# Requirements
This work is implemented on [Microsoft Azure Cloud](https://azure.microsoft.com/en-us/). Please modify the corresponding settings in the ```submit.py``` file.

The [Azureml Core](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core?view=azure-ml-py) package is required to install.

# Training
```submit.py``` is the script to run the [Azure Machine Learning SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py) for Python.
```main_itp_ddp_tar_super_node.py``` is used for training, and it is the script that will be run in parallel on multiple nodes. The settings are defined in the argument section in the *.py file.

```
$ python3 main_itp_ddp_tar_super_node.py --model_v 3 \
--enc_vocab_fn preprocessed/de.vocab.composite2.tsv \
--ans_vocab_fn preprocessed/answer.txt
```

## Files
All related scripts are in the ```models``` folder, while the rest of the folders are for ablation study usage only.

```submit.py``` is used to train on Azure Cloud.

```main_itp_ddp_tar_super_node.py``` is the training file.

```data_loader_itp_bbox_super_node_onlyobj.py``` is the corresponding data loader.

```Att_Model_x3.py``` is the file about our semantic transformer and visual transformer.

For each transformer, the graph-guided multi-head attention and some other embedding functions are in ```modules.py```.

# Testing
```eval_itp_grid_ddp_tar_gt.py``` is used for evaluation. The settings are defined in the argument section in the *.py file.


```
$ python3 eval_itp_grid_ddp_tar_gt.py --model_v 3 \
--enc_vocab_fn preprocessed/de.vocab.composite2.tsv \
--ans_vocab_fn preprocessed/answer.txt
```
