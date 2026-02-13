import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import nullcontext
from torch.cuda.amp import GradScaler, autocast

import pdb
import time
import os, sys, time
from telnetlib import IP
import argparse
import numpy as np
from tqdm import tqdm
from addict import Dict as adict
import matplotlib.pyplot as plt

import wandb

from utils import post_process_depth, flip_lr, compute_errors, eval_metrics, \
                       block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args


from parse_args import *
from initial import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    print("Config data is ", arg_filename_with_prefix)
    args = parse_args([arg_filename_with_prefix])
else:
    args = parse_args()


print(f"pytorch vision: {torch.__version__}")
print('*' * 80)
print('the args are the below')
print('*' * 80)
args = adict(vars(args))
info = load_config(args.config_path)
args = update_config(args, info)
for key, value in args.items():
    print('{:<20}:{}'.format(key, str(value)))
print('*' * 80 + '\n')

# torch.autograd.set_detect_anomaly(True)


def init_wandb_log(args):
    # start a new wandb run to track this script
    tags_w = args.wandb_tags.split(",")
    wandb.init(
        # set the wandb project where this run will be logged
        project="MoDOT",

        # track hyperparameters and run metadata
        config=args,
        tags=tags_w
    )


def get_model_var_sum(model):
    var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)
    return var_sum, var_cnt



def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu
    start_epoch, start_iter, end_iter = init_saving(args)
    if args.gpu is not None:
        print("== Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)


    model = init_model(args)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        print("== Model Initialized on GPU: {}".format(args.gpu))
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()
        print("== Model Initialized")



    # Training initial
    global_step = 0
    optimizer = init_optimizer(args, model)
    # scheduler = init_scheduler(args, optimizer)
    # torch.backends.cudnn.enabled = False
    model, optimizer, model_just_loaded = load_checkpoint(args, model, optimizer)
    cudnn.benchmark = True
    train_transforms, valid_transforms = init_transformations(args)
    dataloader, dataloader_eval = init_dataloader(args, is_train=True, train_transforms=train_transforms, valid_transforms=valid_transforms)
    criterion = init_criterion(args)    
    # amp
    amp_context = nullcontext
    if args.amp:
        amp_context = autocast
        scaler = GradScaler()
    

    start_time = time.time()
    duration = 0
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    var_sum, var_cnt = get_model_var_sum(model)
    print("== Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch
    running_ob_loss = 0
    running_depth_loss = 0
    running_OBDCL = 0
    # for local plot
    iterations_ob_loss = list()
    iterations_depth_loss = list()

    init_wandb_log(args)
    wandb.watch(model)
    stop_training = False 

    while epoch < args.num_epochs:
        print()
        print("Training epoch ", epoch)
        print("-"*66)
        print()
        start_e = time.time()
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)
        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            before_op_time = time.time()

            with amp_context():

                image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
                depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))
                ob_gt = torch.autograd.Variable(sample_batched['edge'].cuda(args.gpu, non_blocking=True))

                out = model(image)

                depth_loss, ob_loss, geo_loss = criterion(out, depth_gt, ob_gt, global_step)
                loss = depth_loss + ob_loss + geo_loss 


                # Check nan / inf in training loop
                if not torch.isfinite(depth_loss):
                    # Happen in OB-FUTURE
                    print(f"[Warning] depth loss is NaN or INF at step {global_step}, skipping batch.")
                    continue   # safely skip the batch
                
                running_depth_loss += depth_loss.cpu().item()
                running_ob_loss += ob_loss.cpu().item()

                if args.use_geo_consensus:
                    running_OBDCL += geo_loss.cpu().item()

                if args.amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
                    for param_group in optimizer.param_groups:
                        current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                        param_group['lr'] = current_lr
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    for param_group in optimizer.param_groups:
                        current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                        param_group['lr'] = current_lr
                    optimizer.step()

            # if args.scheduler_mode == 'iter':
            #     scheduler.step()

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.9f}, depth loss: {:.6f}, ob loss: {:.6f}, geo loss: {:.6f}'  \
                        .format(epoch, step, steps_per_epoch, global_step, current_lr, depth_loss.cpu().item(), ob_loss, geo_loss))
                # if np.isnan(loss.cpu().item()):
                #     print('NaN in loss occurred. Aborting training.')
                #     return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                var_sum, var_cnt = get_model_var_sum(model)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))



            model_just_loaded = False
            global_step += 1

            if global_step % ( args.log_freq ) == 0:
                running_ob_loss = running_ob_loss / args.log_freq
                running_depth_loss = running_depth_loss / args.log_freq
                running_geo_loss = running_OBDCL / args.log_freq
                
                iterations_depth_loss.append(running_depth_loss)
                iterations_ob_loss.append(running_ob_loss)
                # log metrics to wandb
                wandb.log({"depth_loss": running_depth_loss, "ob_loss": running_ob_loss, "OBDCL":running_OBDCL, "learning rate": current_lr, "var average": var_sum.item()/var_cnt })
                
                running_ob_loss = 0
                running_depth_loss = 0
                running_OBDCL = 0

            if args.save_by_iter and global_step >= start_iter and global_step % args.save_freq == 0:
                checkpoint = {'global_step': global_step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }
                cpks_save_path = args.log_directory + '/' + args.model_name + '/ckps/'
                if not os.path.exists(cpks_save_path):
                    os.makedirs(cpks_save_path)
                model_save_name = 'iteration_{}.pth'.format(global_step)
                torch.save(checkpoint, cpks_save_path + model_save_name)
            
            if args.save_by_iter and global_step == end_iter:
                # stop training
                epoch = 999
                stop_training = True
            else:
                stop_training = False 
        
        if stop_training:
            break

        # if args.scheduler_mode == 'epoch':
        #     scheduler.step()

        if not args.save_by_iter and epoch >= start_epoch:
            checkpoint = {'global_step': global_step,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }
            cpks_save_path = args.log_directory + '/' + args.model_name + '/ckps/'
            if not os.path.exists(cpks_save_path):
                os.makedirs(cpks_save_path)
            model_save_name = 'epoch_{}.pth'.format(epoch)
            torch.save(checkpoint, cpks_save_path + model_save_name)

        epoch += 1
        print("Train this epoch using ", (time.time() - start_e) / 60 )
        print("-"*66)
        print()
    
    wandb.finish()

    # plot by matplotlib
    iterations = list(range(0, (len(iterations_depth_loss)) * args.log_freq, args.log_freq))  

    # Plot the losses
    skipped_first_iter = 10
    plt.figure(figsize=(10, 6))
    plt.plot(iterations[skipped_first_iter:], iterations_ob_loss[skipped_first_iter:], label="OB loss", color="limegreen", linestyle="-", marker="*", markersize=4)
    plt.plot(iterations[skipped_first_iter:], iterations_depth_loss[skipped_first_iter:], label="Depth loss", color="darkorange", linestyle="-", marker="v", markersize=4)

    # Add labels, legend, and title
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.title("Loss Variation During Training", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot
    loss_png_out_path = os.path.join(args.log_directory, args.model_name) + "/training_loss.png"
    plt.savefig(loss_png_out_path, dpi=300)  # Save as an image




def main():
    if args.mode != 'train':
        print('train.py is only for training.')
        return -1

    command = 'mkdir ' + os.path.join(args.log_directory, args.model_name)
    os.system(command)

    command = 'mkdir ' + os.path.join(args.log_directory, args.model_name+"/ckps/")
    os.system(command)

    args_out_path = os.path.join(args.log_directory, args.model_name)
    command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    os.system(command)

    save_files = True
    if save_files:
        aux_out_path = os.path.join(args.log_directory, args.model_name)
        networks_savepath = os.path.join(aux_out_path, 'networks')
        dataloaders_savepath = os.path.join(aux_out_path, 'dataloaders')
        # models_savepath = os.path.join(aux_out_path, 'models')
        command = 'cp newcrfs/wandb_train.py ' + aux_out_path
        os.system(command)
        command = 'cp modot/initial.py ' + aux_out_path
        os.system(command)
        command = 'mkdir -p ' + networks_savepath + ' && cp modot/networks/*.py ' + networks_savepath
        os.system(command)
        # command = 'mkdir -p ' + models_savepath + ' && cp modot/models/*.py ' + models_savepath
        # os.system(command)
        command = 'mkdir -p ' + dataloaders_savepath + ' && cp modot/dataloaders/*.py ' + dataloaders_savepath
        os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
