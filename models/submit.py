#!/usr/bin/env python
import azureml.core

print("SDK version:", azureml.core.VERSION)

from azureml.telemetry import set_diagnostics_collection

set_diagnostics_collection(send_diagnostics=True)


from azureml.core.workspace import Workspace

ws = Workspace.from_config()
print('Workspace name: ' + ws.name,
      'Azure region: ' + ws.location,
      'Subscription id: ' + ws.subscription_id,
      'Resource group: ' + ws.resource_group, sep='\n')

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# choose a name for your cluster
cluster_name = "GPUCompute"

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target.')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_ND40RS_V2',max_nodes=4, min_nodes=4)

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True)

# use get_status() to get a detailed status for the current AmlCompute.
print(compute_target.get_status().serialize())


from azureml.core import Datastore
datastore = Datastore.register_azure_blob_container(workspace=ws,
                                                      datastore_name='ds_oi',
                                                      container_name='intern',
                                                      account_name='visioneve',
                                                      account_key='FGvn3EG//vgPAMfcPEVK1aI5SvebwYfDvVmRW8cybrzxbd/4uVBNk5lo0KpiTd3tylhPoPGLtnOVJdn6dlqLxg==',
                                                      create_if_not_exists=True)


from azureml.core import Experiment
from azureml.core.container_registry import ContainerRegistry

experiment_name = 'px_ddp_experiment' #px_ddp px_ddp_experiment
experiment = Experiment(ws, name=experiment_name)
from azureml.train.estimator import Estimator

from azureml.train.dnn import PyTorch, Mpi

project_folder = "."

custom_docker_image=""
details = ContainerRegistry()
details.address = ""
details.username = ""
details.password = ""




from azureml.train.estimator import Estimator

script_params = {
    # to mount files referenced by mnist dataset
    #'--data-folder': ds.as_named_input('mnist').as_mount(),
    #'--data-folder': datastore.path('GQA_sgg_official/tar_data/').as_mount(), #BUA_pred2/ corpora/corpora/corpora/
    '--batch_size': 32, #####4 for rel and obj, 32 for obj only
    '--model_v': 3,
    '--data_dir_azure': datastore.path('GQA/tar_data/').as_mount(),
    '--output_dir':  'output_preproc_supernode_onlyobj_predrel', ##2 output_test_fusion_novis_preproc_gttopk
    '--q_tar_fn_train': 'train.tar',
    '--q_tar_fn_val': 'val.tar',
    '--with_loc': '',
    '--with_dec':'',
    '--enc_vocab_fn': 'preprocessed/de.vocab.composite2.tsv',
    '--ans_vocab_fn': 'preprocessed/answer.txt',
    '--with_bbox': '',
    '--maxlen': 450, ####1600 for rel and obj, 450 for obj only
    '--maxlen_q' : 40,
    # '--maxlen_v': 40,
    '--num_blocks': 6,
    '--fea_tar_fn_train': 'gt_bua_npz_topk.tar', #'bua_npz_topk.tar', gt_bua_npz gt_bua_npz_topk
    '--fea_tar_fn_val': 'gt_bua_npz_topk.tar', #'bua_npz_topk.tar',
    '--g_tar_fn_train': 'gt_bua_npz_topk.tar',
    '--g_tar_fn_val': 'gt_bua_npz_topk.tar',
    '--with_smooth_labeling': '',
    '--min_cnt': 50,
    '--decMask': '',
    '--dropout_rate': 0.5,
    '--with_MILNCE_loss': '',
    '--topN': 5,
    '--hidden_size_mil': 1024, #####64 for for rel and obj, 1024 for obj only
    '--log_steps': 100,
    '--only_obj': '',
    '--pred_rel':''


    # '--GTRelPredNode': '',
    # '--gtWpred': '',
    # '--ngpus': 7
    # '--gtNode': '',
    # '--with_gt_relation': ''
    # '--with_rank_loss': '',
    # '--visGraph': '', ##3 #visGraph: if use object connect info as attention mask in visual branch
    # '--mcb': '',
    # '--dataAug': '',
    # '--aug_rate': 1.0, ##4 #The probability that change the object and attributes during training phrase

}

sk_est = Estimator(source_directory=project_folder,
                   script_params=script_params,
                   compute_target=compute_target,
                   node_count=1,
                   distributed_training=Mpi(),
                   custom_docker_image=custom_docker_image,
                   user_managed = False,
                   shm_size = '512g',
                   #user_managed = True,
                   image_registry_details=details,
                   pip_packages= ['torchtext', 'h5py','tqdm'],
                   entry_script='main_itp_ddp_tar_super_node.py',
                   use_gpu=True)

run = experiment.submit(sk_est)
print(run.get_portal_url())
