"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import network
import optimizer
import time
import torchvision.utils as vutils
import torch.nn.functional as F
from network.mynn import freeze_weights, unfreeze_weights
import numpy as np
import random

from EIL import DeepV3Plus_ei, get_train_dataloader_envs

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepR50V3PlusD',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', nargs='*', type=str, default=['gtav'],
                    help='a list of datasets; cityscapes, mapillary, camvid, kitti, gtav, mapillary, synthia')
parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                    help='uniformly sample images across the multiple source domains')
parser.add_argument('--val_dataset', nargs='*', type=str, default=['cityscapes', 'bdd100k', 'mapillary'],
                    help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
parser.add_argument('--covstat_val_dataset', nargs='*', type=str, default=['gtav'],
                    help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')
parser.add_argument('--class_uniform_pct', type=float, default=0.5,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_iter', type=int, default=60000)
parser.add_argument('--max_cu_epoch', type=int, default=10000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.5, help='level of color augmentation')
parser.add_argument('--gblur', default=True,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=0.9,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=8,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=768,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='0101',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='r50os8_gtav_isw_hvt',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=True,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')

parser.add_argument('--wt_layer', nargs='*', type=int, default=[0, 0, 2, 2, 2, 0, 0],
                    help='0: None, 1: IW/IRW, 2: ISW, 3: IS, 4: IN (IBNNet: 0 0 4 4 4 0 0)')
parser.add_argument('--wt_reg_weight', type=float, default=0.6)
parser.add_argument('--relax_denom', type=float, default=0.0)
parser.add_argument('--clusters', type=int, default=3)
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--dynamic', action='store_true', default=False)

parser.add_argument('--image_in', action='store_true', default=False,
                    help='Input Image Instance Norm')
parser.add_argument('--cov_stat_epoch', type=int, default=5,
                    help='cov_stat_epoch')
parser.add_argument('--visualize_feature', action='store_true', default=False,
                    help='Visualize intermediate feature')
parser.add_argument('--use_wtloss', action='store_true', default=False,
                    help='Automatic setting from wt_layer')
parser.add_argument('--use_isw', action='store_true', default=False,
                    help='Automatic setting from wt_layer')

parser.add_argument('--res18', type=str, default="/home/changty/test_data/resnet18-5c106cde.pth",
                    help='the pretrained model of resnet18')

##############################Ours EIL###############################
parser.add_argument('--envs_split_type', default='env_infer', choices=['env_infer', 'random'],
                    help='The environment split types')
parser.add_argument('--envs_num', default=2,
                    help='Number of splited environments')
parser.add_argument('--interval_num', default=5,
                    help='The number of intervals (epoch) the EIL module conducts')
parser.add_argument('--EI_bs_scaling_factor', default=2,
                    help='The scaling factor of batch size in Environment Inference stage')
parser.add_argument('--eta', default=0.1,
                    help='The hyperparameter η in Environment Inference module')
parser.add_argument('--lambda_1_E', default=100,
                    help='The hyperparameter λ_1^E in Environment Inference module')
parser.add_argument('--lambda_2_E', default=1,
                    help='The hyperparameter λ_2^E in Environment Inference module')
parser.add_argument('--tau', default=0.2,
                    help='The hyperparameter τ in Environment Inference module')

##############################Ours HVT###############################
parser.add_argument('--mu_g', type=float, default=0.4,
                    help='the value of hyper-parameter μ_G')
parser.add_argument('--gamma', type=float, default=0.2,
                    help='the value of hyper-parameter γ')
parser.add_argument('--mu_p', type=float, default=0.1,
                    help='the value of hyper-parameter μ_P')
parser.add_argument('--beta_p', type=float, default=0.15,
                    help='the value of hyper-parameter β')
parser.add_argument('--lambda_1_T', type=float, default=1,
                    help='the value of hyper-parameter λ1T')
parser.add_argument('--lambda_2_T', type=float, default=1,
                    help='the value of hyper-parameter λ2T')
parser.add_argument('--lambda_3_T', type=float, default=10,
                    help='the value of hyper-parameter λ3T')
parser.add_argument('--num_patch_row', type=int, default=4,
                    help='the number of rows in local transformation N'' ')
parser.add_argument('--num_patch_column', type=int, default=4,
                    help='the number of columns in local transformation M'' ')


args = parser.parse_args()

# Enable CUDNN Benchmarking optimization
#torch.backends.cudnn.benchmark = True
random_seed = cfg.RANDOM_SEED  #304
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ:
    # args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

# torch.cuda.set_device(args.local_rank)
# print('My Rank:', args.local_rank)
# Initialize distributed communication

args.dist_url = args.dist_url + str(4634 + (int(time.time()%1000))//10)

torch.distributed.init_process_group(backend='nccl',
                                     init_method=args.dist_url,
                                     world_size=args.world_size,
                                     rank=args.local_rank)

for i in range(len(args.wt_layer)):
    if args.wt_layer[i] == 1:
        args.use_wtloss = True
    if args.wt_layer[i] == 2:
        args.use_wtloss = True
        args.use_isw = True

def main():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)

    args.envs_index_list = None
    args.train_droplast = True
    train_loader, val_loaders, train_obj, extra_val_loaders, covstat_val_loaders = datasets.setup_loaders(args)

    criterion, criterion_val = loss.get_loss(args)
    criterion_aux = loss.get_loss_aux(args)
    net = network.get_net(args, criterion, criterion_aux)

    num_sample = len(train_obj.imgs)
    if 'DeepR50V3PlusD' in args.arch:
        trunk='resnet-50'
    else:
        trunk='resnet-101'
    net_ei = DeepV3Plus_ei(trunk=trunk, args=args)
    optim_ei, scheduler = optimizer.get_optimizer(args, net_ei)
    net_ei = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_ei)
    net_ei = torch.nn.DataParallel(net_ei).cuda()

    optim, scheduler = optimizer.get_optimizer(args, net)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, args.local_rank)
    epoch = 0
    i = 0

    if args.snapshot:
        epoch, mean_iu = optimizer.load_weights(net, optim, scheduler,
                            args.snapshot, args.restore_optimizer)
        if args.restore_optimizer is True:
            iter_per_epoch = len(train_loader)
            i = iter_per_epoch * (epoch + 1)
            epoch = epoch + 1
        else:
            epoch = 0

    print("#### iteration", i)
    torch.cuda.empty_cache()
    # Main Loop

    while i < args.max_iter:
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.ITER = i
        cfg.immutable(True)

        # Environment Inference for every 5 epochs
        if (epoch % args.interval_num==0):
            train_loader_envs, net_ei, optim_ei = get_train_dataloader_envs(train_loader, net, criterion, net_ei, optim_ei, num_sample, epoch, args)
        else:
            train_loader_envs = train_loader_envs

        i = train(train_loader_envs, net, optim, epoch, i, writer, scheduler, args.max_iter)
        train_loader.sampler.set_epoch(epoch + 1)

        if (epoch+1) % 1 == 0 or i >= args.max_iter:
            for dataset, val_loader in extra_val_loaders.items():
                print("Extra validating... This won't save pth file")
                validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, save_pth=False)

        if (args.dynamic and args.use_isw and epoch % (args.cov_stat_epoch + 1) == args.cov_stat_epoch) \
           or (args.dynamic is False and args.use_isw and epoch == args.cov_stat_epoch):
            net.module.reset_mask_matrix()
            for trial in range(args.trials):
                for dataset, val_loader in covstat_val_loaders.items():  # For get the statistics of covariance
                    validate_for_cov_stat(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i,
                                          save_pth=False)
                    net.module.set_mask_matrix()

        if args.local_rank == 0 and (epoch >= 25 or (epoch+1) % 5 == 0):
            print("Saving pth file...")
            evaluate_eval(args, net, optim, scheduler, None, None, [],
                        writer, epoch, "None", None, i, save_pth=True)

        if args.class_uniform_pct:
            if epoch >= args.max_cu_epoch:
                train_obj.build_epoch(cut=True)
                train_loader.sampler.set_num_samples()
            else:
                train_obj.build_epoch()

        epoch += 1

    # Validation after epochs
    if len(val_loaders) == 1:
        # Run validation only one time - To save models
        for dataset, val_loader in val_loaders.items():
            validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i)
    else:
        if args.local_rank == 0:
            print("Saving pth file...")
            evaluate_eval(args, net, optim, scheduler, None, None, [],
                        writer, epoch, "None", None, i, save_pth=True)

    for dataset, val_loader in extra_val_loaders.items():
        print("Extra validating... This won't save pth file")
        validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, save_pth=False)


def train(train_loader, net, optim, curr_epoch, curr_iter, writer, scheduler, max_iter):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    def get_robust_loss(outputs):
        outputs_index = 0
        main_loss = outputs[outputs_index]
        outputs_index += 1
        aux_loss = outputs[outputs_index]
        outputs_index += 1
        total_loss = main_loss + (0.4 * aux_loss)

        if args.use_wtloss and (not args.use_isw or (args.use_isw and curr_epoch > args.cov_stat_epoch)):
            wt_loss = outputs[outputs_index]
            # outputs_index += 1
            # total_loss = total_loss + (args.wt_reg_weight * wt_loss)
        else:
            wt_loss = 0

        return total_loss, wt_loss, outputs_index
    
    def get_epenalty_pixel_reduction(main_out, main_out_envs, gt=None):
        main_loss_pixel = []
        main_loss_envs_pixel = []
        for i in range(0, main_out.shape[0]):
            main_out_tmp = main_out[1*i:1*i+1].permute(0,2,3,1).reshape(-1, datasets.num_classes)
            main_out_envs_tmp = main_out_envs[1*i:1*i+1].permute(0,2,3,1).reshape(-1, datasets.num_classes)
            gt_tmp = gt[1*i:1*i+1].permute(1,2,0).reshape(-1)
            main_loss_pixel_tmp = criterion_pixel(main_out_tmp, gt_tmp).unsqueeze(0)
            main_loss_envs_pixel_tmp = criterion_pixel(main_out_envs_tmp, gt_tmp).unsqueeze(0)
            main_loss_pixel.append(main_loss_pixel_tmp)
            main_loss_envs_pixel.append(main_loss_envs_pixel_tmp)

        e_penalty = (torch.mean((torch.cat(main_loss_pixel, dim=0) - torch.cat(main_loss_envs_pixel, dim=0)).abs(), dim=1)).mean()
        return e_penalty
    
    net.train()

    criterion_pixel = loss.get_loss_pixel(args)[0]

    train_total_loss = AverageMeter()
    time_meter = AverageMeter()


    for i, data in enumerate(train_loader):
        if curr_iter >= max_iter:
            break

        # inputs, gts, _, aux_gts = data
        inputs, gts, _, aux_gts, envs_id = data

        # Multi source and AGG case
        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            num_domains = D
            inputs = inputs.transpose(0, 1)
            gts = gts.transpose(0, 1).squeeze(2)
            aux_gts = aux_gts.transpose(0, 1).squeeze(2)

            inputs = [input.squeeze(0) for input in torch.chunk(inputs, num_domains, 0)]
            gts = [gt.squeeze(0) for gt in torch.chunk(gts, num_domains, 0)]
            aux_gts = [aux_gt.squeeze(0) for aux_gt in torch.chunk(aux_gts, num_domains, 0)]
        else:
            B, C, H, W = inputs.shape
            num_domains = 1
            inputs = [inputs]
            gts = [gts]
            aux_gts = [aux_gts]

        batch_pixel_size = C * H * W

        for di, ingredients in enumerate(zip(inputs, gts, aux_gts)):
            input, gt, aux_gt = ingredients

            start_ts = time.time()

            img_gt = None
            input, gt = input.cuda(), gt.cuda()

            
            # the train of Environment Specific Model
            optim.zero_grad()
            if args.use_isw:
                outputs_envs, loss_hvt_envs, output_original_envs = net(input, envs_id=envs_id, is_envs_model=True, curr_iter=curr_iter, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=args.visualize_feature,
                            apply_wtloss=False if curr_epoch<=args.cov_stat_epoch else True)
            else:
                outputs_envs, loss_hvt_envs, output_original_envs = net(input, envs_id=envs_id, is_envs_model=True, curr_iter=curr_iter, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=args.visualize_feature)

            robust_loss_envs, wt_loss_envs, outputs_index = get_robust_loss(outputs_envs)
            if args.use_wtloss and (not args.use_isw or (args.use_isw and curr_epoch > args.cov_stat_epoch)):
                total_loss_envs = robust_loss_envs + args.wt_reg_weight * wt_loss_envs + loss_hvt_envs
            else:
                total_loss_envs = robust_loss_envs + loss_hvt_envs

            total_loss_envs.backward()
            optim.step()

            if args.local_rank == 0:
                if i % 30 == 29:
                    msg = '[Environment Specific Model Trainging], [epoch {}], [iter {} / {} : {}], [loss_envs {:0.6f}], [robust_loss_envs {:0.6f}], [loss_hvt_envs {:0.6f}], [lr {:0.6f}]'.format(
                        curr_epoch, i + 1, len(train_loader), curr_iter, total_loss_envs, robust_loss_envs, loss_hvt_envs, optim.param_groups[-1]['lr'])

                    print(msg)
                    logging.info(msg)

            del total_loss_envs, robust_loss_envs, wt_loss_envs, loss_hvt_envs

            # the train of Global Model
            optim.zero_grad()
            if args.use_isw:
                outputs, loss_hvt, output_original = net(input, curr_iter=curr_iter, envs_id=envs_id, is_envs_model=False, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=args.visualize_feature,
                            apply_wtloss=False if curr_epoch<=args.cov_stat_epoch else True)
            else:
                outputs, loss_hvt, output_original = net(input, curr_iter=curr_iter, envs_id=envs_id, is_envs_model=False, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=args.visualize_feature)

            robust_loss, wt_loss, outputs_index = get_robust_loss(outputs)
            if args.use_wtloss and (not args.use_isw or (args.use_isw and curr_epoch > args.cov_stat_epoch)):
                total_loss = robust_loss + args.wt_reg_weight * wt_loss + loss_hvt
            else:
                total_loss = robust_loss + loss_hvt

            # EIL Loss
            e_penalty = get_epenalty_pixel_reduction(output_original['main_out'], output_original_envs['main_out'].detach(), gt=gt) * args.tau

            if args.visualize_feature:
                f_cor_arr = outputs[outputs_index]
                outputs_index += 1

            total_loss_sum = total_loss + e_penalty
            log_total_loss = total_loss.clone().detach_()
            torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
            log_total_loss = log_total_loss / args.world_size
            train_total_loss.update(log_total_loss.item(), batch_pixel_size)

            total_loss_sum.backward()
            optim.step()

            time_meter.update(time.time() - start_ts)

            del total_loss, log_total_loss

            if args.local_rank == 0:
                if i % 30 == 29:
                    if args.visualize_feature:
                        visualize_matrix(writer, f_cor_arr, curr_iter, '/Covariance/Feature-')

                    msg = '[Global Model Trainging], [epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [robust_loss {:0.6f}], [loss_hvt {:0.6f}], [e_penalty {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                        curr_epoch, i + 1, len(train_loader), curr_iter, train_total_loss.avg, robust_loss, loss_hvt, e_penalty,
                        optim.param_groups[-1]['lr'], time_meter.avg / args.train_batch_size)

                    print(msg)
                    logging.info(msg)
                    if args.use_wtloss:
                        print("Whitening Loss", wt_loss)

                    # Log tensorboard metrics for each iteration of the training phase
                    writer.add_scalar('loss/train_loss', (train_total_loss.avg),
                                    curr_iter)
                    train_total_loss.reset()
                    time_meter.reset()

        curr_iter += 1
        scheduler.step()

        if i > 5 and args.test_mode:
            return curr_iter

    return curr_iter

def validate(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0
    error_acc = 0
    dump_images = []

    for val_idx, data in enumerate(val_loader):
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        inputs, gt_image, img_names, _ = data

        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            gt_image = gt_image.view(-1, 1, H, W)

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

        with torch.no_grad():
            if args.use_wtloss:
                output, f_cor_arr = net(inputs, visualize=True)
            else:
                output = net(inputs)

        del inputs

        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == datasets.num_classes

        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)

        del gt_cuda

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions
        predictions = output.data.max(1)[1].cpu()

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                print("validating: %d / %d", val_idx + 1, len(val_loader))
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
                             datasets.num_classes)
        del output, val_idx, data

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()

    if args.local_rank == 0:
        evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                    writer, curr_epoch, dataset, None, curr_iter, save_pth=save_pth)

        if args.use_wtloss:
            visualize_matrix(writer, f_cor_arr, curr_iter, '/Covariance/Feature-')

    return val_loss.avg

def validate_for_cov_stat(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    # net.train()#eval()
    net.eval()

    for val_idx, data in enumerate(val_loader):
        img_or, img_photometric, img_geometric, img_name = data   # img_geometric is not used.
        img_or, img_photometric = img_or.cuda(), img_photometric.cuda()

        with torch.no_grad():
            net([img_photometric, img_or], cal_covstat=True)

        del img_or, img_photometric, img_geometric

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / 100", val_idx + 1)
        del data

        if val_idx >= 499:
            return


def visualize_matrix(writer, matrix_arr, iteration, title_str):
    stage = 'valid'

    for i in range(len(matrix_arr)):
        C = matrix_arr[i].shape[1]
        matrix = matrix_arr[i][0].unsqueeze(0)    # 1 X C X C
        matrix = torch.clamp(torch.abs(matrix), max=1)
        matrix = torch.cat((torch.ones(1, C, C).cuda(), torch.abs(matrix - 1.0),
                        torch.abs(matrix - 1.0)), 0)
        matrix = vutils.make_grid(matrix, padding=5, normalize=False, range=(0,1))
        writer.add_image(stage + title_str + str(i), matrix, iteration)


def save_feature_numpy(feature_maps, iteration):
    file_fullpath = '/home/userA/projects/visualization/feature_map/'
    file_name = str(args.date) + '_' + str(args.exp)
    B, C, H, W = feature_maps.shape
    for i in range(B):
        feature_map = feature_maps[i]
        feature_map = feature_map.data.cpu().numpy()   # H X D
        file_name_post = '_' + str(iteration * B + i)
        np.save(file_fullpath + file_name + file_name_post, feature_map)


if __name__ == '__main__':
    main()
