from __future__ import print_function

import argparse
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataloader import SecenFlowLoader as DA
from dataloader import listflowfile as lt
from dataloader import ETH3D_loader as et
from dataloader import KITTI2012loader as kt2012
from dataloader import KITTIloader as kt
from dataloader import SceneFlowLoader as SFL
from dataloader import middlebury_loader as mb
from models import *

from torch.autograd import grad
from submission import dg_test

from EIL import get_train_dataloader_envs

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default="/data/changty/SceneFlow_archive/",
                    help='datapath')
parser.add_argument('--epochs', type=int, default=15,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', default="/home/changty/github_sm_ehvt/psmnet_ehvt/save_ckpt/",
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--logfile', type=str, default='/home/changty/github_sm_ehvt/psmnet_ehvt/save_ckpt/log-psm-ehvt.txt',
                    help='the domain generalization evaluation results on four realistic datasets')
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
parser.add_argument('--mu_g', type=float, default=0.8,
                    help='the value of hyper-parameter μ_G')
parser.add_argument('--gamma', type=float, default=0.2,
                    help='the value of hyper-parameter γ')
parser.add_argument('--mu_p', type=float, default=0.1,
                    help='the value of hyper-parameter μ_P')
parser.add_argument('--beta_p', type=float, default=0.15,
                    help='the value of hyper-parameter β')
parser.add_argument('--lambda_1_T', type=float, default=0.5,
                    help='the value of hyper-parameter λ1T')
parser.add_argument('--lambda_2_T', type=float, default=0.5,
                    help='the value of hyper-parameter λ2T')
parser.add_argument('--lambda_3_T', type=float, default=1,
                    help='the value of hyper-parameter λ3T')
parser.add_argument('--num_patch_row', type=int, default=4,
                    help='the number of rows in local transformation N'' ')
parser.add_argument('--num_patch_column', type=int, default=4,
                    help='the number of columns in local transformation M'' ')

args = parser.parse_args()

# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,6'

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


###########experiment_type###########
print(args.savemodel)

b_size = 8
all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)
all_img_id_list = list(range(0, len(all_left_img)))

TrainImg = DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True)

TrainImgLoader = torch.utils.data.DataLoader(
         TrainImg, 
         batch_size= b_size, shuffle= True, num_workers= 8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
         batch_size= 1, shuffle= False, num_workers= 4, drop_last=False)

model = stackhourglass(args.maxdisp, args)
model = nn.DataParallel(model)
model.cuda()
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel, map_location=torch.device('cpu'))
    model.load_state_dict(pretrain_dict['state_dict'], strict=False)

model_ei = PSMNet_ei(args.envs_num)
model_ei = nn.DataParallel(model_ei).cuda()
param_groups_ei = [
        {'params': model_ei.module.model_ei_feat.parameters(), 'lr': 0.0001},
        {'params': model_ei.module.model_ei_classify.parameters(), 'lr': 0.0001}]
optimizer_ei = optim.Adam(param_groups_ei, lr=0.001, betas=(0.9, 0.999))

def disp2distribute(disp_gt, max_disp, b=2):
    disp_gt = disp_gt.unsqueeze(1)
    disp_range = torch.arange(0, max_disp).view(1, -1, 1, 1).float().cuda()
    gt_distribute = torch.exp(-torch.abs(disp_range - disp_gt) / b)
    gt_distribute = gt_distribute / (torch.sum(gt_distribute, dim=1, keepdim=True) + 1e-8)
    return gt_distribute

def CEloss(disp_gt, max_disp, gt_distribute, pred_distribute):
    mask = (disp_gt > 0) & (disp_gt < max_disp)
    pred_distribute = torch.log(pred_distribute + 1e-8)
    ce_loss = torch.sum(-gt_distribute * pred_distribute, dim=1)
    ce_loss = torch.mean(ce_loss[mask])
    return ce_loss

def irmv1_penalty(loss):
    scale = torch.tensor(1.).to(loss.device).requires_grad_()
    erm_loss = (loss * scale).mean()
    grad_single = grad(erm_loss, [scale], create_graph=True)[0]
    penalty = grad_single.pow(2).sum()
    return penalty

def adjust_learning_rate(epoch):
    lr = 0.001
    if epoch < 10:
        lr = lr 
    if epoch >= 10 and epoch < 20:
        lr = lr * 0.5
    if epoch >= 20 and epoch < 30:
        lr = lr * 0.5 * 0.5
    if epoch >= 30 and epoch < 40:
        lr = lr * 0.5 * 0.5 * 0.5
    if epoch>=40:
        lr = lr * 0.5 * 0.5 * 0.5 * 0.5
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if epoch in [10, 20, 30, 40]:
        for param_group in optimizer_ei.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5

def generate_envs_loss_disp_hvt(imgL, imgR, imgL_org, imgR_org, disp_true, mask, batch_idx, epoch, env_id):
    outputs_list, loss_hvt = model(imgL, imgR, imgL_org, imgR_org, batch_idx, epoch, training=True, is_envs_model=True, env_id=env_id, envs_num=args.envs_num)

    b = mask.shape[0]
    loss_disp_list = []
    for i in range(b):
        if mask[i].sum()==0:
            loss_disp_list.append(torch.tensor([5.], device=imgL.device))
        else:
            loss_disp = 0
            for output in outputs_list:
                output1 = torch.squeeze(output[0],1)
                output2 = torch.squeeze(output[1],1)
                output3 = torch.squeeze(output[2],1)
                loss_disp_tmp = (0.5*F.smooth_l1_loss(output1[i][mask[i]], disp_true[i][mask[i]], size_average=True) + 0.7*F.smooth_l1_loss(output2[i][mask[i]], disp_true[i][mask[i]], size_average=True) + F.smooth_l1_loss(output3[i][mask[i]], disp_true[i][mask[i]], size_average=True)) / (0.5+0.7+1) 
                loss_disp += loss_disp_tmp
            loss_disp_list.append(loss_disp.unsqueeze(0))

    loss_disp_sum_list = torch.cat(loss_disp_list)

    return loss_disp_sum_list, loss_hvt, outputs_list

def generate_eil_loss_envs_penalty(outputs_list, outputs_list_envs, mask):
    b = mask.shape[0]
    loss_disp_penalty_list = []

    e_penalty = 0
    for output_list, output_list_envs in zip(outputs_list, outputs_list_envs):
        output1 = torch.squeeze(output_list[0],1)
        output2 = torch.squeeze(output_list[1],1)
        output3 = torch.squeeze(output_list[2],1)
        output1_envs = torch.squeeze(output_list_envs[0],1)
        output2_envs = torch.squeeze(output_list_envs[1],1)
        output3_envs = torch.squeeze(output_list_envs[2],1)

        for i in range(b):
            if mask[i].sum()==0:
                loss_disp_penalty_list.append(torch.tensor([0.5], device=mask.device))
            else:
                # sL1
                loss_disp_penalty_list.append(((0.5*F.smooth_l1_loss(output1[i][mask[i]], output1_envs[i][mask[i]], size_average=True) + 0.7*F.smooth_l1_loss(output2[i][mask[i]], output2_envs[i][mask[i]], size_average=True) + F.smooth_l1_loss(output3[i][mask[i]], output3_envs[i][mask[i]], size_average=True)) / (0.5+0.7+1)).unsqueeze(0))
                # pixel
                # loss_disp_penalty_list.append(((0.5*torch.mean((output1[i][mask[i]]-output1_envs[i][mask[i]]).abs()) + 0.7*torch.mean((output2[i][mask[i]]-output2_envs[i][mask[i]]).abs()) + torch.mean((output3[i][mask[i]]-output3_envs[i][mask[i]]).abs())) / (0.5+0.7+1)).unsqueeze(0) )
        e_penalty += torch.cat(loss_disp_penalty_list).mean()

    return e_penalty

def get_sum_disp_loss(mask, imgL, outputs_list, disp_true):
    b = mask.shape[0]
    loss_disp_list = []
    for i in range(b):
        if mask[i].sum()==0:
            loss_disp_list.append(torch.tensor([5.], device=imgL.device))
        else:
            loss_disp = 0
            for output in outputs_list:
                output1 = torch.squeeze(output[0],1)
                output2 = torch.squeeze(output[1],1)
                output3 = torch.squeeze(output[2],1)
                loss_disp_tmp = (0.5*F.smooth_l1_loss(output1[i][mask[i]], disp_true[i][mask[i]], size_average=True) + 0.7*F.smooth_l1_loss(output2[i][mask[i]], disp_true[i][mask[i]], size_average=True) + F.smooth_l1_loss(output3[i][mask[i]], disp_true[i][mask[i]], size_average=True)) / (0.5+0.7+1) 
                loss_disp += loss_disp_tmp
            loss_disp_list.append(loss_disp.unsqueeze(0))

    loss_disp_sum_list = torch.cat(loss_disp_list)

    return loss_disp_sum_list

def generate_global_loss_disp_hvt(imgL, imgR, imgL_org, imgR_org, disp_true, mask, batch_idx, epoch, env_id):
    outputs_list, outputs_list_envs, loss_hvt = model(imgL, imgR, imgL_org, imgR_org, batch_idx, epoch, training=True, is_envs_model=False, env_id=env_id, envs_num=args.envs_num)
    loss_disp_sum_list = get_sum_disp_loss(mask, imgL, outputs_list, disp_true)
    loss_disp_sum_list_envs = get_sum_disp_loss(mask, imgL, outputs_list_envs, disp_true)
    return loss_disp_sum_list, loss_disp_sum_list_envs, loss_hvt, outputs_list, outputs_list_envs

def train_envs_specific_model(imgL, imgR, disp_L, imgL_org, imgR_org, batch_idx, epoch, envs_id):
    model.train()
    if args.cuda:
        imgL, imgR, disp_true, imgL_org, imgR_org = imgL.cuda(), imgR.cuda(), disp_L.cuda(), imgL_org.cuda(), imgR_org.cuda()

    mask = disp_true < args.maxdisp
    mask.detach_

    loss_disp_envs, loss_hvt_envs, outputs_list = generate_envs_loss_disp_hvt(imgL, imgR, imgL_org, imgR_org, disp_true, mask, batch_idx, epoch, envs_id)
    loss_envs = torch.mean(loss_disp_envs) + loss_hvt_envs.sum() / loss_hvt_envs.shape[0]

    optimizer.zero_grad()
    loss_envs.backward()
    optimizer.step()

    return loss_envs.data

def train_global_model(imgL, imgR, disp_L, imgL_org, imgR_org, batch_idx, epoch, envs_id):
        model.train()
        if args.cuda:
            imgL, imgR, disp_true, imgL_org, imgR_org = imgL.cuda(), imgR.cuda(), disp_L.cuda(), imgL_org.cuda(), imgR_org.cuda()

        mask = disp_true < args.maxdisp
        mask.detach_
        optimizer.zero_grad()
        
        if args.model == 'stackhourglass':
            loss_disp, loss_disp_envs, loss_hvt, outputs_list_hvt, outputs_list_envs_hvt = generate_global_loss_disp_hvt(imgL, imgR, imgL_org, imgR_org, disp_true, mask, batch_idx, epoch, envs_id)
            e_penalty = generate_eil_loss_envs_penalty(outputs_list_hvt, outputs_list_envs_hvt, mask) / len(outputs_list_hvt) * args.tau

            loss = torch.mean(loss_disp) + e_penalty + loss_hvt.sum() / loss_hvt.shape[0]

        elif args.model == 'basic':
            output = model(imgL,imgR)
            output = torch.squeeze(output,1)
            loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

        loss.backward()
        optimizer.step()

        return loss.data, e_penalty


def main():
    start_full_time = time.time()
    for epoch in range(0, args.epochs):
        print('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        adjust_learning_rate(epoch)

        num_sample = len(all_left_img)
        if epoch == 0 or epoch % args.interval_num==0:
            TrainImgLoader_env = get_train_dataloader_envs(TrainImgLoader, num_sample, model, model_ei, optimizer_ei, epoch, args, all_left_img, all_right_img, all_left_disp)

        ## invarient learning - training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, imgL_org, imgR_org, envs_id) in enumerate(TrainImgLoader_env):
            start_time = time.time()
            loss = train_envs_specific_model(imgL_crop,imgR_crop, disp_crop_L, imgL_org, imgR_org, batch_idx, epoch, envs_id)
            print('Environment Specific Model Trainging: Iter %d training loss = %.3f, time = %.2f' %(batch_idx, loss, time.time() - start_time))
            
            start_time = time.time()
            loss, e_penalty = train_global_model(imgL_crop,imgR_crop, disp_crop_L, imgL_org, imgR_org, batch_idx, epoch, envs_id)
            print('Global Model Trainging: Iter %d training loss = %.3f , e_penalty = %.3f,  time = %.2f' %(batch_idx, loss, e_penalty, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader_env)))

        #SAVE
        savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
        torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
            'train_loss': total_train_loss/len(TrainImgLoader_env),
        }, savefilename)

        model_test = stackhourglass_test(args.maxdisp)
        model_test = nn.DataParallel(model_test)
        model_test.cuda()
        state_dict = torch.load(savefilename)
        model_test.load_state_dict(state_dict['state_dict'], strict=False)
        
        with open(args.log_file, 'a') as f:
            print('this is the dg test result of epoch {}\n'.format(epoch), file=f)
        dg_test(model_test, args.log_file, test_left_img_mb, test_right_img_mb, train_gt_mb, test_name='md')
        dg_test(model_test, args.log_file, test_left_img_eth, test_right_img_eth, all_disp_eth, test_name='eth', all_mask=all_mask_eth)
        dg_test(model_test, args.log_file, test_left_img_k12, test_right_img_k12, test_ldisp_k12, test_name='kitti15')
        dg_test(model_test, args.log_file, test_left_img_k15, test_right_img_k15, test_ldisp_k15, test_name='kitti12')

    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))

if __name__ == '__main__':
    train_limg_mb, train_rimg_mb, train_gt_mb, test_limg_mb, test_rimg_mb = mb.mb_loader("/home/changty/test_data//MiddEval3", res='H')
    test_left_img_mb, test_right_img_mb = train_limg_mb, train_rimg_mb

    all_limg_eth, all_rimg_eth, all_disp_eth, all_mask_eth = et.et_loader("/home/changty/test_data//ETH3D")
    test_left_img_eth, test_right_img_eth = all_limg_eth, all_rimg_eth

    all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt.kt_loader("/home/changty/test_data//KITTI_stereo/kitti_2015/data_scene_flow/training/")
    test_left_img_k12, test_right_img_k12 = all_limg + test_limg, all_rimg + test_rimg
    test_ldisp_k12 = all_ldisp + test_ldisp
    
    all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt2012.kt2012_loader("/home/changty/test_data//KITTI_stereo/kitti_2012/data_stereo_flow/training/")
    test_left_img_k15, test_right_img_k15 = all_limg + test_limg, all_rimg + test_rimg
    test_ldisp_k15 = all_ldisp + test_ldisp

    main()
