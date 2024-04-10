from __future__ import absolute_import
from __future__ import division
import torch
from torch import nn
from network import Resnet
from network.cov_settings import CovMatrix_ISW, CovMatrix_IRW
import datasets
import torch.nn.functional as F
import random

from datasets import gtav
from datasets.sampler import DistributedSampler
from torch.utils.data import DataLoader

import optimizer

class DeepV3Plus_ei(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, trunk='resnet-101', args=None):
        super(DeepV3Plus_ei, self).__init__()
        self.args = args
        self.trunk = trunk
        final_channel = 2048

        # the environment feature encoder which has the same structure with feature_extraction_x_war module of DeepV3Plus
        if trunk == 'resnet-18':
            resnet = Resnet.resnet18(wt_layer=self.args.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-50':
            resnet = Resnet.resnet50(wt_layer=self.args.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101': # three 3 X 3
            resnet = Resnet.resnet101(pretrained=True, wt_layer=self.args.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1,
                                        resnet.conv2, resnet.bn2, resnet.relu2,
                                        resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.eps = 1e-5
        self.whitening = False

        if trunk == 'resnet-101':
            self.three_input_layer = True
            in_channel_list = [64, 64, 128, 256, 512, 1024, 2048]   # 8128, 32640, 130816
        elif trunk == 'resnet-18':
            self.three_input_layer = False
            in_channel_list = [0, 0, 64, 64, 128, 256, 512]   # 8128, 32640, 130816
        else: # ResNet-50
            self.three_input_layer = False
            in_channel_list = [0, 0, 64, 256, 512, 1024, 2048]   # 8128, 32640, 130816

        self.cov_matrix_layer = []
        self.cov_type = []
        for i in range(len(self.args.wt_layer)):
            if self.args.wt_layer[i] > 0:
                self.whitening = True
                if self.args.wt_layer[i] == 1:
                    self.cov_matrix_layer.append(CovMatrix_IRW(dim=in_channel_list[i], relax_denom=self.args.relax_denom))
                    self.cov_type.append(self.args.wt_layer[i])
                elif self.args.wt_layer[i] == 2:
                    self.cov_matrix_layer.append(CovMatrix_ISW(dim=in_channel_list[i], relax_denom=self.args.relax_denom, clusters=self.args.clusters))
                    self.cov_type.append(self.args.wt_layer[i])

        # the environment-aware predictor
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        layers = [nn.Linear(final_channel, final_channel//2), nn.ReLU(),
                  nn.Linear(final_channel//2, final_channel//32), nn.ReLU(),
                  nn.Linear(final_channel//32, args.envs_num)]
        self.fc_new = nn.Sequential(*layers)

    def feature_extraction_x_war(self, x, w_arr):
        x = self.layer0[0](x)
        if self.args.wt_layer[2] == 1 or self.args.wt_layer[2] == 2:
            x, w = self.layer0[1](x)
            w_arr.append(w)
        else:
            x = self.layer0[1](x)
        x = self.layer0[2](x)
        x = self.layer0[3](x)

        x_tuple = self.layer1([x, w_arr])  # 400
        low_level = x_tuple[0]

        x_tuple = self.layer2(x_tuple)  # 100
        x_tuple = self.layer3(x_tuple)  # 100
        aux_out = x_tuple[0]
        x_tuple = self.layer4(x_tuple)  # 100
        x = x_tuple[0]
        w_arr = x_tuple[1]
        x_out = x

        return x_out, aux_out, low_level
        
    def forward(self, x):
        w_arr = []
        x_size = x.size()  # 800

        # the environment feature encoder
        x_out, aux_out, low_level = self.feature_extraction_x_war(x, w_arr)

        # the environment-aware predictor
        x = x_out
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        envs_id = self.fc_new(x)

        return envs_id, x_out, aux_out, low_level, x_size
    
def random_envsnum_getenvLoader(epoch, num_sample, envs_num=2):
    print('random_envs_split_data_epoch{}'.format(epoch))
    envs_index_list = []
    for i in range(0, num_sample):
        envs_index_list.append(random.randint(0, envs_num-1))
    
    for j in range(0, envs_num):
         print('len of TrainImgLoader_env{}: {}'.format(j, sum(torch.tensor(envs_index_list)==j)))

    return envs_index_list

def get_disp_test(model, criterion, inputs, gts):
    output = model(inputs, gts=gts, is_envs_model=False, apply_wtloss=False)[0]
    main_out = output['main_out']
    main_loss = criterion(main_out, gts)
    
    task_loss_list = []
    for i in range(inputs.shape[0]):
        task_loss_list.append(criterion(main_out[1*i:1*i+1], gts[1*i:1*i+1]).unsqueeze(0))
    task_loss = torch.cat(task_loss_list).unsqueeze(-1).cuda('cuda:0')

    return task_loss
    
def caculate_ei_loss(batch_idx, envs_id, task_loss, envs_score_sum_dict, envs_task_loss_sum_dict, num_sample):
    envs_num = envs_id.shape[1]
    envs_score_tmp_sum_dict = {}
    envs_task_loss_tmp_sum_dict = {}
    envs_score_ratio_dict = {}
    env_task_losses_avg_list = []

    # For each environment, calculate the average task loss : env_task_losses_avg_list
    # It is worth noting that we use "envs_score_sum_dict" and "envs_task_loss_sum_dict" to represent the cumulative environmental predicted score and task losses.
    # The "cumulative" means includeing all previous samples before this iteration
    for env_index in range(0, envs_num):
        env = 'env{}'.format(env_index)
        envs_score_tmp_sum_dict[env] = torch.sum(envs_id[:,env_index])
        envs_task_loss_tmp_sum_dict[env] = torch.sum(envs_id[:,env_index] * task_loss[:,0]) 
        if batch_idx > 0:
            envs_score_sum_dict[env] = envs_score_sum_dict[env].detach() + envs_score_tmp_sum_dict[env]
            envs_task_loss_sum_dict[env] = envs_task_loss_sum_dict[env].detach() + envs_task_loss_tmp_sum_dict[env]
        else:
            envs_score_sum_dict[env] = envs_score_sum_dict[env] + envs_score_tmp_sum_dict[env]
            envs_task_loss_sum_dict[env] = envs_task_loss_sum_dict[env] + envs_task_loss_tmp_sum_dict[env]
        envs_score_ratio_dict[env] = envs_score_sum_dict[env] / num_sample
        env_task_losses_avg_list.append((envs_task_loss_sum_dict[env] / envs_score_sum_dict[env]).unsqueeze(0))

    # loss_ED
    loss_ED_term1 = -torch.cat(env_task_losses_avg_list).var()
    loss_ED_term2 = 0
    regularize_term = torch.empty(envs_id.shape[0], device=envs_id.device).uniform_(-0.02, 0.02)
    for env_index in range(0, envs_num):
        loss_ED_term2 += -(envs_id[:,env_index] + regularize_term).var()
    
    # loss_EB
    ratio = 1 / envs_num
    envs_score_ratio_list = []
    for env_index in range(0,  envs_num):
        env = 'env{}'.format(env_index)
        envs_score_ratio_list.append(envs_score_ratio_dict[env])
    cur_z = torch.stack(envs_score_ratio_list)
    tar_z = torch.ones_like(cur_z) * ratio
    loss_EB = F.kl_div(torch.log(cur_z), tar_z, reduction='batchmean')

    return loss_ED_term1, loss_ED_term2, loss_EB, envs_score_sum_dict, envs_task_loss_sum_dict, env_task_losses_avg_list

def env_infer_spilit_data(TrainImgLoader, model, criterion, model_ei, optimizer_ei, epoch, args):
    print('env_infer_split_data_epoch{}'.format(epoch))

    # The Training Stage of Environment Classifier
    envs_score_sum_dict = {}
    envs_task_loss_sum_dict = {}
    for env_index in range(0, args.envs_num):
        envs_score_sum_dict['env{}'.format(env_index)] = 0
        envs_task_loss_sum_dict['env{}'.format(env_index)] = 0

    num_sample = 0
    for batch_idx, data in enumerate(TrainImgLoader):
        # Get training samples and ground truths
        inputs, gts, _, aux_gts, _ = data
        inputs, gts = inputs.cuda(), gts.cuda()
        num_sample += inputs.shape[0]

        # predict the environmental label for each training sample
        model_ei.train()
        envs_id, x_out, aux_out, low_level, x_size = model_ei(inputs)
        envs_id = F.softmax(envs_id, dim=-1)

        # use the environment-variant features to calculate the task loss for each training sample
        main_out, _ = model.module.get_main_out(x_out, low_level, x_size, aux_out)
        batch_task_loss_list = []
        for i in range(inputs.shape[0]):
            task_loss_i = criterion(main_out[1*i:1*i+1], gts[1*i:1*i+1])
            if torch.isnan(task_loss_i).sum() > 0:
                task_loss_i = torch.tensor(2., device=inputs.device)
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
    total_envs_task_loss_list = []  
    for batch_idx, data in enumerate(TrainImgLoader):
        inputs, gts, _, aux_gts, _ = data
        inputs, gts = inputs.cuda(), gts.cuda()

        # predict the environmental label for each training sample
        with torch.no_grad():
            envs_id, x_out, aux_out, low_level, x_size = model_ei(inputs)
            envs_id = F.softmax(envs_id, dim=-1)
            if batch_idx % 10 == 0:
                print('final Iter {} envids:{}'.format(batch_idx, torch.max(envs_id, dim=1)[1]))

        total_envs_id_list.append(torch.max(envs_id, dim=1)[1])
    total_envs_id = torch.cat(total_envs_id_list)
   
    # print the number of training samples and average task loss for each splitted environment
    for j in range(0, args.envs_num):
        envs_index_tmp = torch.tensor((total_envs_id)==j)
        print('len of final env_infer TrainImgLoader_env{}: {}'.format(j, sum(envs_index_tmp)))

        if sum(envs_index_tmp) == 0:
            return []

    envs_index_list = total_envs_id.int().tolist()

    return envs_index_list

def get_train_dataloader_envs(TrainImgLoader, model, criterion, model_ei, optimizer_ei, num_sample, epoch, args):
    ### Environment Inference 'env_infer'(ours) or 'random' sampling###
    args.train_batch_size_ei = args.train_batch_size * args.EI_bs_scaling_factor
    args.train_droplast = False
    train_loader = datasets.setup_loaders(args)[0]
    args.train_batch_size_ei = 0
    
    if args.envs_split_type == 'env_infer':
        envs_index_list = env_infer_spilit_data(train_loader, model, criterion, model_ei, optimizer_ei, epoch, args)
        ### once the all training samples are classified to one environment, repeat the environment inference operation 
        if envs_index_list == []:
            model_ei = DeepV3Plus_ei(trunk='resnet-50', args=args).cuda()
            optimizer_ei, scheduler = optimizer.get_optimizer(args, model_ei)
            model_ei = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_ei)
            model_ei = torch.nn.DataParallel(model_ei).cuda()
            envs_index_list = env_infer_spilit_data(train_loader, model, criterion, model_ei, optimizer_ei, epoch, args)
             
    elif args. envs_split_type == 'random':
        envs_index_list = random_envsnum_getenvLoader(epoch, num_sample, envs_num=args.envs_num)

    if epoch >= 25:
        args.bs_mult = 7
        args.train_batch_size = 7

    args.envs_index_list = envs_index_list
    args.train_droplast = True
    train_loader_envs = datasets.setup_loaders(args)[0]

    return train_loader_envs, model_ei, optimizer_ei
 

