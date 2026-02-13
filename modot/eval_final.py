import torch
import torch.backends.cudnn as cudnn

import cv2
import pdb
import time 
import os, sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from addict import Dict as adict

from parse_args import *
from initial import *
from ob_evals import *

from utils import post_process_depth, flip_lr, compute_errors, post_process_ob
# import warnings
# warnings.filterwarnings("ignore")


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
    
    
def eval(model, dataloader_eval, eval_save_path, post_process=False,save_interval=10,vis_res=False):

    vis_res = True
    save_interval=2

    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)

    eval_measures = torch.zeros(16).cuda()

    for _, eval_sample_batched in enumerate(dataloader_eval.data):

        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            gt_depth = eval_sample_batched['depth']
            ob_gt = eval_sample_batched['edge']

            output = model(image)

            pred_depth = output[0]
            if isinstance(output[1], list):
                pred_ob = torch.sigmoid(output[1][-1])
            else:
                pred_ob = torch.sigmoid(output[1])

            if post_process:
                image_flipped = flip_lr(image)
                output_flipped = model(image_flipped)
                pred_depth_flipped = output_flipped[0]
                if isinstance(output_flipped[1], list):
                    pred_ob_flipped = torch.sigmoid(output_flipped[1][-1])
                else:
                    pred_ob_flipped = torch.sigmoid(output_flipped[1])                
                # pred_ob_flipped = output_flipped[1][-1] 
                # pred_ob_flipped = torch.sigmoid(pred_ob_flipped)
                pred_depth = post_process_depth(pred_depth, pred_depth_flipped)
                pred_ob = post_process_ob(pred_ob, pred_ob_flipped)


            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

            # pred_ob = torch.sigmoid(pred_ob)
            # print("another sigmoid")
            # Using another sigmoid function here can improve OB-Fscore, yielding better performance than the results reported in the paper
            pred_ob = pred_ob.cpu().numpy().squeeze()
            # print(np.max(pred_ob), np.min(pred_ob))
            gt_ob = ob_gt.cpu().numpy().squeeze()

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        # synocc is OB-FUTURE
        if args.dataset_eval == 'synocc':
            valid_mask = (gt_depth < 10.0) & (gt_depth > 0.0)
            invalid_mask = gt_depth >= 10.0
        elif args.dataset_eval == 'hypersim':
            valid_mask = (~np.isnan(gt_depth)) & (gt_depth > 0.0)
            invalid_mask = None
        elif args.dataset_eval in ['nyud', 'nyudmt', 'nyudmt_reverse']:
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
            invalid_mask = None
            if args.dataset_eval in ['nyud', 'nyudmt_reverse']:
                crop_mask = np.zeros_like(gt_depth, dtype=np.bool)
                crop_mask[45:471, 41:601] = True  

                # Combine both: valid depth values inside crop
                valid_mask = valid_mask & crop_mask
        
        # gt_depth[gt_depth < args.min_depth_eval] = args.min_depth_eval
        # gt_depth[gt_depth > args.max_depth_eval] = args.max_depth_eval
        # gt_depth[np.isinf(gt_depth)] = args.max_depth_eval
        # gt_depth[np.isnan(gt_depth)] = args.min_depth_eval

        # vis eval depth exaples
        if  vis_res  and  _ % save_interval == 0:
            if args.dataset_eval in ["nyud", "nyudmt", "nyudmt_reverse"]:
                depth_save = pred_depth * 10 
            elif args.dataset_eval == "hypersim":
                depth_save = pred_depth / 100
            elif args.dataset_eval == "synocc":
                depth_save = pred_depth / 10

            depth_vis = (depth_save.min() / depth_save) * 255.0  # save inverse depth for better vis
  
            if args.dataset_eval in ["nyud"]:
                name = eval_sample_batched['name'][0].strip().split("/")[-1] + ".png"
            elif args.dataset_eval in ["hypersim","nyudmt","nyudmt_reverse"]:
                name = eval_sample_batched['meta']['name'][0] + ".png"
            elif args.dataset_eval == "synocc":
                name = eval_sample_batched['meta']['img_name'][0] + ".png"
                depth_vis[invalid_mask] = 0
            else:
                name = eval_sample_batched['name'][0] + ".png"
            plt.imsave(eval_save_path + name, depth_vis, vmin=depth_vis.min(), vmax=depth_vis.max())


        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        # # error acc/comp, used in p2orm, shapernet, displacement_filed
        # acc, comp = compute_depth_boundary_error(ob_gt.cpu().numpy().squeeze(), pred_depth / 10, valid_mask)
        measures.extend([0.0, 0.0])


        # non-NMS edge metrics
        # print(Counter(gt_ob.reshape(-1)))
        if args.dataset == "nyudmt" and args.dataset_eval == "nyudmt":
            ob_mask = (gt_ob != 255)
            edge_metrics = compute_edge_metrics(pred_ob[ob_mask], gt_ob[ob_mask], threshold=0.7, edge_nms=False)
        else:
            edge_metrics = compute_edge_metrics(pred_ob, gt_ob, threshold=0.7, edge_nms=False)
        measures.extend(  [edge_metrics["Precision"], edge_metrics["Recall"], edge_metrics["Accuracy"], edge_metrics["F1-Score"]]  )


        eval_measures[:15] += torch.tensor(measures).cuda()
        eval_measures[-1] += 1


    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[-1].item()
    eval_measures_cpu /= cnt

    print()
    str_output = "{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7} , {:>7}, {:>7}, {:>7}, {:>7} ".format('abs_rel', 'sq_rel', 'rms',
                                                                                        'log_rms', 'log10', 'silog', 'd1', 'd2',
                                                                                    'd3', "ob_acc", "ob_comp", "ob_precision", "ob_recall","acc","fscore")
    print(str_output)
    end_index = 14

    for i in range(end_index):
        print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.4f}'.format(eval_measures_cpu[end_index]))
    print()

    with open(eval_save_path+"measures.txt", "w+") as m:
        m.write('Computing errors for {} eval samples, post_process: '.format(int(cnt), post_process))
        m.write("\n")
        m.write(str_output)
        m.write("\n")
        for i in range(end_index):
            m.write('{:7.4f}, '.format(eval_measures_cpu[i]))
        m.write('{:7.4f}'.format(eval_measures_cpu[end_index]))

        m.write("\n")

    return eval_measures_cpu



def plot_metrics(metrics_list, metric_names, save_dir, epochs=None, start_epoch=10):
    """
    Plot and save metrics changes over epochs.

    Args:
        metrics_list (list of list/np.array): A list where each element is a list/array of metric values over epochs.
        metric_names (list of str): A list of names corresponding to each metric.
        save_dir (str): Directory to save the plot image.
        epochs (int, optional): Total number of epochs. If None, inferred from metrics_list.

    Returns:
        None
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Determine the number of epochs
    if epochs is None:
        epochs = len(metrics_list[0])

    # Create a range for the x-axis
    if start_epoch > 1000:
        eval_step = 2000
    else:
        eval_step = 1
    epoch_range = range(start_epoch + eval_step, epochs * eval_step + eval_step + start_epoch)


    # Plot each metric
    for metric, name in zip(metrics_list, metric_names):
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_range[:len(metric)], metric, label=name)

        # Add metric values as text on the plot
        for x, y in zip(epoch_range[:len(metric)], metric):
            plt.text(x, y, f'{y:.4f}', fontsize=8, ha='right')

        # Configure plot
        plt.xlabel('Epochs/Iterations')
        plt.ylabel('Metric Value')
        plt.title(name)
        plt.legend([name])
        plt.grid(True)

        # Save the plot
        save_path = os.path.join(save_dir, f'{name}_plot.png')
        plt.savefig(save_path)
        plt.close()
        # print(f"Plot saved at {save_path}")
    
    # plot a joint metrics
    # Define the metrics to include in the joint plot
    joint_metrics = ['abs_rel', 'rms', 'ob_precision'] # , 'ob_fscore']
    colors = ['mediumblue', 'limegreen', 'darkorange', 'c']
    markers = ['o', 's', 'D', '^']   

    print()
    # print(metric_names)
    print("Plot metrics every epoch/iterations")
    print()
    # Create a joint plot for selected metrics
    plt.figure(figsize=(12, 8))
    for metric, name, color, marker in zip(metrics_list, metric_names, colors, markers):
        if name in joint_metrics:
            plt.plot(epoch_range[:len(metric)], metric, label=name, color=color, marker=marker)

            # Add metric values as text on the plot
            for x, y in zip(epoch_range[:len(metric)], metric):
                plt.text(x, y, f'{y:.4f}', fontsize=8, ha='right')

    # Configure joint plot
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.title('Joint Metrics Plot')
    plt.legend()
    plt.grid(True)

    # Save the joint plot
    joint_save_path = os.path.join(save_dir, '0_joint_main_metrics.png')
    plt.savefig(joint_save_path)
    plt.close()

# for invpt
# def interpolate_pos_embed_inference(model, infer_img_size, device):
#     pos_embed = model.pos_embed
#     embedding_size = pos_embed.shape[-1]

#     patch_embed = model.patch_embed

#     num_patches = patch_embed.num_patches
#     num_extra_tokens = pos_embed.shape[-2] - num_patches
#     grid_size = patch_embed.grid_size

#     patch_size = patch_embed.patch_size
#     infer_grid_size = (infer_img_size[0] // patch_size[0], \
#         infer_img_size[1] // patch_size[1])

#     orig_size, new_size = grid_size, infer_grid_size
#     if orig_size != new_size:
#         # print("Position interpolate from %dx%d to %dx%d" % (orig_size[0], orig_size[1], 
#         #     new_size[0], new_size[1]))
#         extra_tokens = pos_embed[:, :num_extra_tokens]
#         # only the position tokens are interpolated
#         pos_tokens = pos_embed[:, num_extra_tokens:]
#         pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)
#         pos_tokens = torch.nn.functional.interpolate(
#             pos_tokens, size=new_size, mode='bicubic', align_corners=False)
#         pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
#         new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
#         new_pos_embed = torch.nn.Parameter(new_pos_embed).to(device)

#         model.pos_embed = new_pos_embed
#         model.patch_embed.grid_size = infer_grid_size


def main_worker(args):

    model = init_model(args)
    model = torch.nn.DataParallel(model)
    model.cuda()
    cudnn.benchmark = True
    post_process = True

    command = 'cp ' + sys.argv[1] + ' ' + args.log_directory
    os.system(command)

    command = 'cp modot/eval_final.py ' + args.log_directory
    os.system(command)

    start_eval_index, end_eval_index, eval_step, save_interval, vis_res = init_eval_saving(args)

    is_train = True
    if args.mode != "train":
        is_train = False
    train_transforms, valid_transforms = init_transformations(args)
    dataloader_eval = init_dataloader(args, is_train=is_train, valid_transforms=valid_transforms)


    metrics_recorder = {'abs_rel':[], 'sq_rel':[], 'rms':[], 'log_rms':[], 'log10':[], 'silog':[], 'd1':[], 'd2':[], 'd3':[], 
                    "ob_precision":[], "ob_recall":[],"ob_acc":[],"ob_fscore":[]}

    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            # torch.save(model.state_dict(), "./saved_models/")
            print("== Loaded checkpoint '{}'".format(args.checkpoint_path))
            del checkpoint

            eval_save_path = "./models/" + args.model_name + "/eval/" + args.checkpoint_path.split(".")[0] + "/"

            # ===== Evaluation ======
            model.eval()
            with torch.no_grad():
                eval_measures = eval(model, dataloader_eval, eval_save_path,    
                                    post_process=True,save_interval=save_interval,vis_res=vis_res)

        elif os.path.isdir(args.checkpoint_path):
            item_list = os.listdir(args.checkpoint_path)
            # for item in item_list:
            for i in range(start_eval_index, end_eval_index, eval_step):
                if args.save_by_iter:
                    item = f"iteration_{i}.pth"
                else:
                    item = f"epoch_{i}.pth"
                
                # print(item)

                start_time = time.time()
                cp_loc = args.checkpoint_path+item
                if os.path.isfile(cp_loc):
                    print()
                    print("-"*66)
                    print("== Loading checkpoint '{}'".format(args.checkpoint_path))
                    checkpoint = torch.load(cp_loc, map_location='cpu')

                    # # # Modify pos_embed to match shape
                    # if args.model == "invpt":
                    #     model_state_dict = model.state_dict()
                    #     if 'module.backbone.pos_embed' in checkpoint['model']:
                    #         pretrained_pos_embed = checkpoint['model']['module.backbone.pos_embed']
                    #         current_pos_embed = model_state_dict['module.backbone.pos_embed']

                    #         if pretrained_pos_embed.shape != current_pos_embed.shape:
                    #             print(f"Resizing pos_embed from {pretrained_pos_embed.shape} to {current_pos_embed.shape}")

                    #             # Resizing logic (e.g., interpolate positional embeddings)
                    #             from torch.nn.functional import interpolate
                    #             pretrained_pos_embed = interpolate(
                    #                 pretrained_pos_embed.permute(0, 2, 1),  # (1, 1024, 401)
                    #                 size=current_pos_embed.shape[1],  # Target sequence length
                    #                 mode='linear'
                    #             ).permute(0, 2, 1)  # Back to (1, 4490, 1024)

                    #             checkpoint['model']['module.backbone.pos_embed'] = pretrained_pos_embed

                    # Load state dict with resized pos_embed
                    model.load_state_dict(checkpoint['model'], strict=False)
                    print("== Loaded checkpoint '{}'".format(args.checkpoint_path+item))
                    del checkpoint
                    eval_save_path = args.log_directory + "/eval/" + item.split(".")[0] + "/"
                    print('Computing errors for {} eval samples'.format(len(dataloader_eval.data)), ', post_process: ', post_process)
                    print()
                    # ===== Evaluation ======
                    model.eval()
                    with torch.no_grad():
                        # eval_measures = eval(model, dataloader_eval, eval_save_path, post_process=post_process)
                        eval_measures = eval(model, dataloader_eval, eval_save_path,    
                                            post_process=True,save_interval=save_interval,vis_res=vis_res)
                    print("Evaluate this epoch using ", (time.time() - start_time) / 60 )
                    print("-"*66)
                    print()

                    metrics_recorder["abs_rel"].append(eval_measures[0])
                    metrics_recorder["sq_rel"].append(eval_measures[1])
                    metrics_recorder["rms"].append(eval_measures[2])
                    metrics_recorder["log_rms"].append(eval_measures[3])
                    metrics_recorder["log10"].append(eval_measures[4])
                    metrics_recorder["silog"].append(eval_measures[5])
                    metrics_recorder["d1"].append(eval_measures[6])
                    metrics_recorder["d2"].append(eval_measures[7])
                    metrics_recorder["d3"].append(eval_measures[8])
                    metrics_recorder["ob_precision"].append(eval_measures[11])#
                    metrics_recorder["ob_recall"].append(eval_measures[12])
                    metrics_recorder["ob_acc"].append(eval_measures[13])
                    metrics_recorder["ob_fscore"].append(eval_measures[14])
                else:
                    metrics_recorder["abs_rel"].append(0)
                    metrics_recorder["sq_rel"].append(0)
                    metrics_recorder["rms"].append(0)
                    metrics_recorder["log_rms"].append(0)
                    metrics_recorder["log10"].append(0)
                    metrics_recorder["silog"].append(0)
                    metrics_recorder["d1"].append(0)
                    metrics_recorder["d2"].append(0)
                    metrics_recorder["d3"].append(0)
                    metrics_recorder["ob_precision"].append(0)
                    metrics_recorder["ob_recall"].append(0)
                    metrics_recorder["ob_acc"].append(0)
                    metrics_recorder["ob_fscore"].append(0)     

                # update item list
                item_list = os.listdir(args.checkpoint_path)             

                      

        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))

    plot_metrics(list(metrics_recorder.values()), list(metrics_recorder.keys()), args.log_directory + "eval_plots/", start_epoch=start_eval_index)

def main():
    torch.cuda.empty_cache()
    args.distributed = False
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1
    
    main_worker(args)


if __name__ == '__main__':
    main()
