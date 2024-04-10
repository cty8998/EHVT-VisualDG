from __future__ import absolute_import
from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
import random

from torch.utils.data import DataLoader

from models.stackhourglass import feature_extraction, envs_classify
from dataloader import SecenFlowLoader as DA
import math

def get_batchsize(epoch):
    if epoch < 15 :
        bs = 8
    elif epoch >= 15 and epoch < 30:
        bs = 8
    elif epoch >= 30:
        bs = 8

    return bs

class PSMNet_ei(nn.Module):
    def __init__(self, envs_num):
        super(PSMNet_ei, self).__init__()

        self.model_ei_feat = feature_extraction()
        self.model_ei_classify = envs_classify(envs_num=envs_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, imgL, imgR):
        featL = self.model_ei_feat(imgL)
        featR = self.model_ei_feat(imgR)

        envs_id = self.model_ei_classify(featL)

        return envs_id, featL, featR
    
def random_envsnum_getenvLoader(epoch, num_sample, envs_num=2):
    print('random_envs_split_data_epoch{}'.format(epoch))
    envs_index_list = []
    for i in range(0, num_sample):
        envs_index_list.append(random.randint(0, envs_num-1))
    
    for j in range(0, envs_num):
         print('len of TrainImgLoader_env{}: {}'.format(j, sum(torch.tensor(envs_index_list)==j)))

    return envs_index_list
    
def caculate_ei_loss(batch_idx, envs_id, task_loss, envs_score_sum_dict, envs_task_loss_sum_dict, num_sample):
    envs_num = envs_id.shape[1]
    envs_score_tmp_sum_dict = {}
    envs_task_loss_tmp_sum_dict = {}
    envs_score_ratio_dict = {}
    env_task_losses_avg_list = []

    # For each environment, calculate the average task loss : env_task_losses_avg_list
    # It is worth noting that we use "envs_score_sum_dict" and "envs_task_loss_sum_dict" to represent the cumulative environmental predicted score and task losses.
    # The "cumulative" means includeing all previous samples before this iteration
    for env_id in range(0, envs_num):
        env = 'env{}'.format(env_id)
        envs_score_tmp_sum_dict[env] = torch.sum(envs_id[:,env_id])
        envs_task_loss_tmp_sum_dict[env] = torch.sum(envs_id[:,env_id] * task_loss[:,0]) 
        if batch_idx > 0:
            envs_score_sum_dict[env] = envs_score_sum_dict[env].detach() + envs_score_tmp_sum_dict[env]
            envs_task_loss_sum_dict[env] = envs_task_loss_sum_dict[env].detach() + envs_task_loss_tmp_sum_dict[env]
        else:
            envs_score_sum_dict[env] = envs_score_sum_dict[env] + envs_score_tmp_sum_dict[env]
            envs_task_loss_sum_dict[env] = envs_task_loss_sum_dict[env] + envs_task_loss_tmp_sum_dict[env]
        envs_score_ratio_dict[env] = envs_score_sum_dict[env] / num_sample
        env_task_losses_avg_list.append((envs_task_loss_sum_dict[env] / envs_score_sum_dict[env]).unsqueeze(0))
        
    # loss_ED
    regularize_term1 = 10
    loss_ED_term1 = -(torch.cat(env_task_losses_avg_list)).var()
    loss_ED_term2 = 0
    regularize_term2 = torch.empty(envs_id.shape[0], device=envs_id.device).uniform_(-0.02, 0.02)
    for env_id in range(0, envs_num):
        loss_ED_term2 += -(envs_id[:,env_id] + regularize_term2).var()
    
    # loss_EB
    ratio = 1 / envs_num
    envs_score_ratio_list = []
    for env_id in range(0,  envs_num):
        env = 'env{}'.format(env_id)
        envs_score_ratio_list.append(envs_score_ratio_dict[env])
    cur_z = torch.stack(envs_score_ratio_list)
    tar_z = torch.ones_like(cur_z) * ratio
    loss_EB = F.kl_div(torch.log(cur_z), tar_z, reduction='batchmean')

    return loss_ED_term1, loss_ED_term2, loss_EB, envs_score_sum_dict, envs_task_loss_sum_dict, env_task_losses_avg_list

def env_infer_spilit_data(TrainImgLoader, model, model_ei, optimizer_ei, epoch, args):
    print('env_infer_split_data_epoch{}'.format(epoch))

    # The Training Stage of Environment Classifier
    envs_score_sum_dict = {}
    envs_task_loss_sum_dict = {}
    for env_id in range(0, args.envs_num):
        envs_score_sum_dict['env{}'.format(env_id)] = 0
        envs_task_loss_sum_dict['env{}'.format(env_id)] = 0

    num_sample = 0
    for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, imgL_org, imgR_org) in enumerate(TrainImgLoader):
        # Get training samples and ground truths
        imgL, imgR, disp_true, imgL_org, imgR_org = imgL_crop.cuda(), imgR_crop.cuda(), disp_crop_L.cuda(), imgL_org.cuda(), imgR_org.cuda()
        mask = disp_true < args.maxdisp
        mask = mask.cuda()
        mask.detach_()
        num_sample += imgL.shape[0]
    
        # predict the environmental label for each training sample
        model_ei.train()
        envs_id, feat_L, feat_R = model_ei(imgL, imgR)
        envs_id = F.softmax(envs_id, dim=-1)

        # use the environment-variant features to calculate the task loss for each training sample
        # disp_est_3 = model.module.cost_volume_and_disparity_regression_cuda(feat_L, feat_R, imgL, imgL.device)[-1]
        disp_est_3 = model.module.cost_volume_and_disparity_regression(feat_L, feat_R, imgL)[-1]
        batch_task_loss_list = []
        for i in range(disp_est_3.shape[0]):
            if mask[i].sum()==0:
                task_loss_i = torch.tensor([50], device=feat_L.device)
            else:
                task_loss_i = F.smooth_l1_loss(torch.squeeze(disp_est_3,1)[i][mask[i]], disp_true[i][mask[i]], size_average=True).unsqueeze(0)
            if torch.isnan(task_loss_i).sum() > 0:
                task_loss_i = torch.tensor(2., device=imgL.device)
            batch_task_loss_list.append(task_loss_i.unsqueeze(0))
        task_loss = torch.cat(batch_task_loss_list).unsqueeze(-1).cuda('cuda:0')
        # regularization
        task_loss = task_loss / max(task_loss)

        # calculate the loss_ED and loss_EB to train the Environment Classifier
        loss_ED_term1, loss_ED_term2, loss_EB, envs_score_sum_dict, envs_task_loss_sum_dict, env_task_losses_avg_list = \
        caculate_ei_loss(batch_idx, envs_id, task_loss, envs_score_sum_dict, envs_task_loss_sum_dict, num_sample)
    
        loss_ED = loss_ED_term1 + (loss_ED_term2 * args.eta)
        loss_sum = loss_ED * args.lambda_1_E + loss_EB * args.lambda_2_E 
        
        if batch_idx % 10 == 0:
            print('batchid:{}  envids:{}'.format(batch_idx, torch.max(envs_id, dim=1)[1]))
            print('loss_ED_term1:{} loss_ED_term2:{} loss_EB:{}'.format(loss_ED_term1, loss_ED_term2, loss_EB))
        
        optimizer_ei.zero_grad()
        loss_sum.backward()
        optimizer_ei.step()

    # The Inference Stage of Environment Classifier
    total_envs_id_list = []
    for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, imgL_org, imgR_org) in enumerate(TrainImgLoader):
        # Get training samples and ground truths
        imgL, imgR, disp_true, imgL_org, imgR_org = imgL_crop.cuda(), imgR_crop.cuda(), disp_crop_L.cuda(), imgL_org.cuda(), imgR_org.cuda()
        mask = disp_true < args.maxdisp
        mask = mask.cuda()
        mask.detach_()

        # predict the environmental label for each training sample
        with torch.no_grad():
            envs_id, feat_L, feat_R = model_ei(imgL, imgR)
            envs_id = F.softmax(envs_id, dim=-1)
            if batch_idx % 10 == 0:
                print('final Iter {} envids:{}'.format(batch_idx, torch.max(envs_id, dim=1)[1]))

        total_envs_id_list.append(torch.max(envs_id, dim=1)[1])
    total_envs_id = torch.cat(total_envs_id_list)
    
    # print the number of training samples for each splitted environment
    for j in range(0, args.envs_num):
        envs_indice_tmp = torch.tensor((total_envs_id)==j)
        print('len of final env_infer TrainImgLoader_env{}: {}'.format(j, sum(envs_indice_tmp)))

    envs_index_list = total_envs_id.int().tolist()

    return envs_index_list

def get_train_dataloader_envs(train_loader, num_sample, model, model_ei, optimizer_ei, epoch, args, all_left_img, all_right_img, all_left_disp):
    ###Environment Inference 'env_infer'(ours) or 'random' sampling###
    
    if args.envs_split_type == 'env_infer':
        envs_index_list = env_infer_spilit_data(train_loader, model, model_ei, optimizer_ei, epoch, args)
    elif args. envs_split_type == 'random':
        envs_index_list = random_envsnum_getenvLoader(epoch, num_sample, envs_num=args.envs_num)


    TrainImg_envs = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True, envs_list=envs_index_list)
    train_loader_envs  = torch.utils.data.DataLoader(
        TrainImg_envs, 
        batch_size=get_batchsize(epoch), shuffle= True, num_workers= 8, drop_last=True)

    return train_loader_envs
 

