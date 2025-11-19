import yaml
import argparse
from easydict import EasyDict as edict
from utils import convert_arg_line_to_args


def load_config(config_path, return_edict=False):

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    return edict(cfg) if return_edict else cfg


def update_config(cfg, info):


    pretrained_name = cfg.encoder
    cfg.pretrain = info["PRETRAINED_MODELS"][pretrained_name]
    
    cfg.data_path = info[cfg.dataset]["dataset_dir"]
    cfg.gt_path = cfg.data_path
    cfg.filenames_file  = info[cfg.dataset]["train_img_txt"]
    if cfg.dataset_eval != "openworld":
        cfg.data_path_eval = info[cfg.dataset_eval]["val_dataset_dir"]
    cfg.filenames_file_eval = info[cfg.dataset_eval]["val_img_txt"]
    cfg.gt_path_eval = cfg.data_path_eval
    

    if cfg.dataset in ["hypersim", "nyudmt", "nyudmt_reverse"]:
        cfg.boundary_lambda = 1.1

    return cfg


def parse_args(arguments=None):

    parser = argparse.ArgumentParser(description='MoDOT PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    # basic
    parser.add_argument('--config_path',               type=str, default="./configs/config.yml", help='datasets info and backbone path')
    parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
    parser.add_argument('--model_name',                type=str,   help='model name', default='MoDOT')
    parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07', default='large07')
    parser.add_argument('--pretrain',                  type=str,   help='path of pretrained encoder', default=None)
    parser.add_argument('--wandb_tags',                type=str,   help='tags', default='train,newcrfs,')

    # loss
    parser.add_argument('--ob_loss_weight',             type=float,   help='ob loss weight', default=1.0)
    parser.add_argument('--ob_loss_type',               type=str,   help='ob loss function type', choices=['sem_seg', 'al', 'cce', 'fl','none','normal'], default='cce')
    parser.add_argument('--boundary_weights',           type=str, default='0.5,0.5,0.5,0.5,0.5,1.1', help='')
    parser.add_argument('--boundary_lambda',            type=float, default=1.7, help='')
    parser.add_argument('--loss_type',                  type=str,   help='depth loss functions', default='silog')
    parser.add_argument('--depth_loss_weight',          type=float,   help='ob loss weight', default=1.1)
    parser.add_argument('--init_depth_loss_weight',     type=float,   help='init depth loss weight', default=0.9)
    parser.add_argument('--use_geo_consensus',          help='OBDCL loss', action='store_true')
    # parser.add_argument('--use_geo_consensus_gt',       help='depth bound loss use ob gt', action='store_true')
    parser.add_argument('--ts_checkpoint_path',         type=str, help='checkpoint for SSR training', default="")
    parser.add_argument('--use_step_OBDCL',             help='step OBDCL loss', action='store_true')

    # Dataset
    parser.add_argument('--dataset',                   type=str,   help='dataset to train on', default='synocc') # synocc is OB-FUTURE
    parser.add_argument('--data_path',                 type=str,   help='path to the data') # , required=True)
    parser.add_argument('--gt_path',                   type=str,   help='path to the ground truth data') # , required=True)
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file') # , required=True)
    parser.add_argument('--input_height',              type=int,   help='input height', default=320)
    parser.add_argument('--input_width',               type=int,   help='input width',  default=320)
    parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
    # val datasets
    parser.add_argument('--dataset_eval',              type=str,   help='dataset for eval', default="synocc") # synocc is OB-FUTURE

    # Log and save
    parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
    parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=200)
    parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=2000)
    parser.add_argument('--save_by_iter',              action='store_true', help='save checkpoint by iterations')

    # Training
    parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
    parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
    parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
    parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
    parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)
    parser.add_argument('--amp',                       action='store_true', help='amp in training')
    # scheduler, not used
    parser.add_argument('--scheduler_name', type=str, default='MultiStepLR',help='learning rate scheduler (default: MultiStepLR)')
    parser.add_argument('--scheduler_param', type=str, default="{'milestones':[6, 9, 11]}", help='')
    parser.add_argument('--scheduler_mode', type=str, default='epoch', help='epoch or iter')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='warmup_epochs')
    parser.add_argument('--is_two_stage', action='store_true', help='is two satge?')

    # Multi-gpu training from newcrfs
    parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
    parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
    parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
    parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
    parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
    parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
    parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                        'N processes per node, which has N GPUs. This is the '
                                                                        'fastest way to use PyTorch for either single node or '
                                                                        'multi node data parallel training', action='store_true',)
    # Online eval
    parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
    parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
    parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
    parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
    parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
    parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
    # parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
    parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                        'if empty outputs to checkpoint folder', default='')
    parser.add_argument('--eval_batch_size',           type=int,   help='evaluation batch size', default=1)    
    args = parser.parse_args(arguments)

    return args


if __name__ == '__main__':
    args = parse_args()
    print(dir(args))
