from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms

from torch.autograd import Variable
from torchvision.transforms.functional_tensor import _hsv2rgb, _rgb2hsv

from .submodule import *


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post

def rgb_to_grayscale(img):
    r, g, b = img.unbind(dim=-3)
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)
    return l_img

class PSMNet(nn.Module):
    def __init__(self, maxdisp, args):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp

        self.feature_extraction = feature_extraction()

        self.args = args

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        # The initialization of Environment Specific Model --- two environments
        self.dres0_env0, self.dres1_env0, self.dres2_env0, self.dres3_env0, self.dres4_env0, self.classif1_env0, self.classif2_env0, self.classif3_env0 = self.init_cost_layer()
        self.dres0_env1, self.dres1_env1, self.dres2_env1, self.dres3_env1, self.dres4_env1, self.classif1_env1, self.classif2_env1, self.classif3_env1 = self.init_cost_layer()

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
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        # HVT module 
        # domain discriminating network
        self.feature_discriminator_res18 = ResNet(BasicBlock_res, [2, 2, 2, 2], num_classes=4)
        weight = torch.load(self.args.res18)
        weight['fc.weight'] = self.feature_discriminator_res18.state_dict()['fc.weight']
        weight['fc.bias'] = self.feature_discriminator_res18.state_dict()['fc.bias']
        self.feature_discriminator_res18.load_state_dict(weight)

        self.B = nn.Parameter(torch.randn(1))
        self.C = nn.Parameter(torch.randn(1))
        self.S = nn.Parameter(torch.randn(1))
        self.H = nn.Parameter(torch.randn(1))

        num_patch = args.num_patch_row * args.num_patch_column
        self.B_p = nn.Parameter(torch.randn(num_patch))
        self.C_p = nn.Parameter(torch.randn(num_patch))
        self.S_p = nn.Parameter(torch.randn(num_patch))
        self.H_p = nn.Parameter(torch.randn(num_patch))

        self.alpha_pixel = nn.Parameter(torch.randn(1,3,256,512))

        self.toPIL = transforms.ToPILImage(mode="RGB")
        self.mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).cuda()

    def init_cost_layer(self):
        dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1),nn.ReLU(inplace=True))
        dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),nn.ReLU(inplace=True),convbn_3d(32, 32, 3, 1, 1)) 
        dres2 = hourglass(32)
        dres3 = hourglass(32)
        dres4 = hourglass(32)
        classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))
        classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))
        classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))
        
        return dres0, dres1, dres2, dres3, dres4, classif1, classif2, classif3


    def img_global_generation(self, img_in, B, C, S, H):
        img_in = img_in.permute(2,0,1)
        img = (img_in.unsqueeze(0) / 255).to(img_in.device)
        # Random order
        idx_list = torch.randperm(4)
        # sub-transformations
        for i in idx_list:
            if i == 0:
                zero_img = torch.zeros_like(img).to(img_in.device)
                img = (B * img + (1.0 - B) * zero_img).clamp(0, 1.0).to(img.dtype)  # Brightness
            elif i == 1:
                mean_img = torch.mean(rgb_to_grayscale(img).to(img.dtype), dim=(-3, -2, -1), keepdim=True).to(img_in.device)
                img = (C * img + (1.0 - C) * mean_img).clamp(0, 1.0).to(img.dtype) # Contrast
            elif i == 2:
                satu_img = rgb_to_grayscale(img).to(img.dtype).to(img_in.device)
                img = (S * img + (1.0 - S) * satu_img).clamp(0, 1.0).to(img.dtype)  # Saturation
            elif i == 3:
                img = _rgb2hsv(img)                                                 # Hue
                h, s, v = img.unbind(dim=-3)
                h = (h + H) % 1.0
                img = torch.stack((h, s, v), dim=-3)
                img = _hsv2rgb(img)
        # Normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_in.device).view(-1,1,1).unsqueeze(0)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_in.device).view(-1,1,1).unsqueeze(0)
        out_ = (img - mean) / std
        return out_

    def img_local_generation(self, img_org, img, para_list):
        # the learned parameter in local visual transformation 
        [B_p ,C_p, S_p, H_p] = para_list

        img_trans_list = []
        # h_p: h of patch  w_p: w of patch
        # patch_idx_list = torch.randperm(self.args.num_patch)
        num_patch = self.args.num_patch_row * self.args.num_patch_column
        _, _, h, w = img.shape
        h_p, w_p = int(h // self.args.num_patch_row), int(w // self.args.num_patch_column)

        for j in range(img_org.shape[0]):
            img_patch_list = []
            img_patch_list_g = []
            # acquire the img patches
            for m in range(self.args.num_patch_row):
                for n in range(self.args.num_patch_column):
                    img_patch_list.append(img_org[j][m*h_p:(m+1)*h_p,n*w_p:(n+1)*w_p,:])

            img_local = torch.zeros([3,h,w], dtype=torch.float32).to(img_org.device)
            for i in range(0, num_patch):
                # the local transformed image  
                B, C, S, H = B_p[i], C_p[i], S_p[i], H_p[i]
                img_patch_list_g.append(self.img_global_generation(img_patch_list[i], B, C, S, H))
                img_local[:, (i // self.args.num_patch_row)*h_p:(i // self.args.num_patch_row+1)*h_p, (i % self.args.num_patch_column)*w_p:(i % self.args.num_patch_column+1)*w_p] = \
                img_local[:, (i // self.args.num_patch_row)*h_p:(i // self.args.num_patch_row+1)*h_p, (i % self.args.num_patch_column)*w_p:(i % self.args.num_patch_column+1)*w_p] + img_patch_list_g[i].squeeze(0)
            img_trans_list.append(img_local.unsqueeze(0))

        img_local_final = torch.cat(img_trans_list, dim=0)

        return img_local_final
    
    def global_visual_transformation(self, image, image_org, b):
        img_aug_list = []
        bs = image.shape[0]
        self.B = self.B.cuda(image.device)
        self.C = self.C.cuda(image.device)
        self.S = self.S.cuda(image.device)
        self.H = self.H.cuda(image.device)

        # the generation of the degree of adjustable image attributes respectively 
        # for the Brightness (B), Contrast (C), Saturation (S) and Hue (H)
        # learn + random
        B = torch.clamp(self.args.mu_g * (torch.sigmoid(self.B) - 0.5) + torch.empty(1, device=image.device).uniform_(-self.args.gamma, self.args.gamma) + 1, min=0.5, max=1.5)
        C = torch.clamp(self.args.mu_g * (torch.sigmoid(self.C) - 0.5) + torch.empty(1, device=image.device).uniform_(-self.args.gamma, self.args.gamma) + 1, min=0.5, max=1.5)
        S = torch.clamp(self.args.mu_g * (torch.sigmoid(self.S) - 0.5) + torch.empty(1, device=image.device).uniform_(-self.args.gamma, self.args.gamma) + 1, min=0.5, max=1.5)
        H = torch.clamp(self.args.mu_g * (torch.sigmoid(self.H) - 0.5) + torch.empty(1, device=image.device).uniform_(-self.args.gamma, self.args.gamma), min=0.5, max=1.5)

        for i in range(b):
            img_c = self.img_global_generation(image_org[i], B, C, S, H)
            img_aug_list.append(img_c)

        image_global = torch.cat(img_aug_list, dim=0)

        return image_global

    def max_discrepency_loss_stage1(self, image, image_global, b):
        # the ground truth domain label 0: original 1: global 2: local 3: pixel
        label_o_gt = torch.tensor([0], device=image.device).expand(b).long()
        label_g_gt = torch.tensor([1], device=image.device).expand(b).long() 

        # the feature and predicted domain label generated from the domain discriminating network 
        label_og_pred, feat_og_disc = self.feature_discriminator_res18(torch.cat([image, image_global]))

        label_o_pred, label_g_pred = label_og_pred[0:b], label_og_pred[b:]
        feat_o_disc, feat_g_disc= feat_og_disc[0:b], feat_og_disc[b:]

        # the cross-entropy loss for domain classification
        loss_CE = (nn.CrossEntropyLoss()(label_o_pred, label_o_gt) 
                 + nn.CrossEntropyLoss()(label_g_pred, label_g_gt) ) / 2
        loss_CE = loss_CE.sum() / b

        # the discrepency loss for maximizing cross-domain visual discrepancy by minimizing the consine similarity
        loss_disc = torch.mean(torch.cosine_similarity(feat_g_disc, feat_o_disc, dim=1)) 
        
        return loss_CE, loss_disc

    def local_visual_transformation(self, image, image_org):

        self.B_p = self.B_p.cuda(image.device)
        self.C_p = self.C_p.cuda(image.device)
        self.S_p = self.S_p.cuda(image.device)
        self.H_p = self.H_p.cuda(image.device)

        # learn + random
        B_p = torch.clamp(self.args.mu_g * (torch.sigmoid(self.B_p) - 0.5) + torch.empty(1, device=image.device).uniform_(-self.args.gamma, self.args.gamma) + 1, min=0.5, max=1.5)
        C_p = torch.clamp(self.args.mu_g * (torch.sigmoid(self.C_p) - 0.5) + torch.empty(1, device=image.device).uniform_(-self.args.gamma, self.args.gamma) + 1, min=0.5, max=1.5)
        S_p = torch.clamp(self.args.mu_g * (torch.sigmoid(self.S_p) - 0.5) + torch.empty(1, device=image.device).uniform_(-self.args.gamma, self.args.gamma) + 1, min=0.5, max=1.5)
        H_p = torch.clamp(self.args.mu_g * (torch.sigmoid(self.H_p) - 0.5) + torch.empty(1, device=image.device).uniform_(-self.args.gamma, self.args.gamma), min=-0.5, max=0.5)

        para_list = [B_p ,C_p, S_p, H_p]
        image_local = self.img_local_generation(image_org, image, para_list)

        return image_local

    def max_discrepency_loss_stage2(self, image, image_global, image_local, b):
        # the ground truth domain label 0: original 1: global 2: local 3: pixel
        label_o_gt = torch.tensor([0], device=image.device).expand(b).long()
        label_g_gt = torch.tensor([1], device=image.device).expand(b).long() 
        label_l_gt = torch.tensor([2], device=image.device).expand(b).long() 

        # the feature and predicted domain label generated from the domain discriminating network 
        label_ogl_pred, feat_ogl_disc = self.feature_discriminator_res18(torch.cat([image, image_global, image_local]))

        label_o_pred, label_g_pred, label_pred = label_ogl_pred[0:b], label_ogl_pred[b:2*b], label_ogl_pred[2*b:]
        feat_o_disc, feat_g_disc, feat_disc = feat_ogl_disc[0:b], feat_ogl_disc[b:2*b], feat_ogl_disc[2*b:]

        # the cross-entropy loss for domain classification
        loss_CE = (nn.CrossEntropyLoss()(label_o_pred, label_o_gt) 
                    + nn.CrossEntropyLoss()(label_g_pred, label_g_gt) 
                    + nn.CrossEntropyLoss()(label_pred, label_l_gt)) / 3
        loss_CE = loss_CE.sum() / b

        # the discrepency loss for maximizing cross-domain visual discrepancy by minimizing the consine similarity
        loss_disc = (torch.mean(torch.cosine_similarity(feat_g_disc, feat_o_disc, dim=1)) 
                   + torch.mean(torch.cosine_similarity(feat_disc, feat_o_disc, dim=1))) / 2
        
        return loss_CE, loss_disc
    
    def pixel_visual_transformation(self, image):
        pixel = torch.randn(1,3,256,512, device=image.device) * (torch.sigmoid(self.alpha_pixel) * self.args.mu_p + self.args.beta_p) 
        image_pixel = image + pixel

        img_min = torch.tensor(0.).cuda()
        img_max = torch.tensor(1.).cuda()
        image_pixel = torch.clip(image_pixel, min=img_min, max=img_max)

        return image_pixel

    def max_discrepency_loss_stage3(self, image, image_global, image_local, image_pixel, b):
        # the ground truth domain label 0: original 1: global 2: local 3: pixel
        label_o_gt = torch.tensor([0], device=image.device).expand(b).long()
        label_g_gt = torch.tensor([1], device=image.device).expand(b).long() 
        label_l_gt = torch.tensor([2], device=image.device).expand(b).long() 
        label_p_gt = torch.tensor([3], device=image.device).expand(b).long() 

        # the feature and predicted domain label generated from the domain discriminating network 
        label_oglp_pred, feat_oglp_disc = self.feature_discriminator_res18(torch.cat([image, image_global, image_local, image_pixel]))

        label_o_pred, label_g_pred, label_pred, label_p_pred \
            = label_oglp_pred[0:b], label_oglp_pred[b:2*b], label_oglp_pred[2*b:3*b], label_oglp_pred[3*b:]
        feat_o_disc, feat_g_disc, feat_disc, feat_p_disc \
            = feat_oglp_disc[0:b], feat_oglp_disc[b:2*b], feat_oglp_disc[2*b:3*b], feat_oglp_disc[3*b:]

        # the cross-entropy loss for domain classification
        loss_CE = (nn.CrossEntropyLoss()(label_o_pred, label_o_gt)  
                    + nn.CrossEntropyLoss()(label_g_pred, label_g_gt) 
                    + nn.CrossEntropyLoss()(label_pred, label_l_gt)
                    + nn.CrossEntropyLoss()(label_p_pred, label_p_gt)) / 4
        loss_CE = loss_CE.sum() / b

        # the discrepency loss for maximizing cross-domain visual discrepancy by minimizing the consine similarity
        loss_disc = (torch.mean(torch.cosine_similarity(feat_g_disc, feat_o_disc, dim=1)) 
                   + torch.mean(torch.cosine_similarity(feat_disc, feat_o_disc, dim=1)) 
                   + torch.mean(torch.cosine_similarity(feat_p_disc, feat_o_disc, dim=1))) / 3
        
        return loss_CE, loss_disc

    def cost_volume_and_disparity_regression(self, refimg_fea, targetimg_fea, left):
        #matching
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp//4):
            if i > 0 :
             cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
             cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
             cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
             cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None) 
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2) 
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1,1)
            distribute1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(distribute1)

            cost2 = torch.squeeze(cost2,1)
            distribute2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(distribute2)

        cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        distribute3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(distribute3,dim=1)
        #For your information: This formulation 'softmax(c)' learned "similarity" 
        #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp)(pred3)

        if self.training:
            return [pred1, pred2, pred3]
        else:
            return [pred3]

    def cost_volume_and_disparity_regression_envs(self, refimg_fea, targetimg_fea, left, envs_index=0):
        #matching
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp//4):
            if i > 0 :
             cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
             cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
             cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
             cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
        cost = cost.contiguous()

        if envs_index==0:
            dres0, dres1, dres2, dres3, dres4, classif1, classif2, classif3 = self.dres0_env0, self.dres1_env0, self.dres2_env0, self.dres3_env0, self.dres4_env0, self.classif1_env0, self.classif2_env0, self.classif3_env0
        elif envs_index==1:
            dres0, dres1, dres2, dres3, dres4, classif1, classif2, classif3 = self.dres0_env1, self.dres1_env1, self.dres2_env1, self.dres3_env1, self.dres4_env1, self.classif1_env1, self.classif2_env1, self.classif3_env1
        elif envs_index==2:
            dres0, dres1, dres2, dres3, dres4, classif1, classif2, classif3 = self.dres0_env2, self.dres1_env2, self.dres2_env2, self.dres3_env2, self.dres4_env2, self.classif1_env2, self.classif2_env2, self.classif3_env2
        else:
            dres0, dres1, dres2, dres3, dres4, classif1, classif2, classif3 = self.dres0_env3, self.dres1_env3, self.dres2_env3, self.dres3_env3, self.dres4_env3, self.classif1_env3, self.classif2_env3, self.classif3_env3

        cost0 = dres0(cost)
        cost0 = dres1(cost0) + cost0

        out1, pre1, post1 = dres2(cost0, None, None) 
        out1 = out1+cost0

        out2, pre2, post2 = dres3(out1, pre1, post1) 
        out2 = out2+cost0

        out3, pre3, post3 = dres4(out2, pre1, post2) 
        out3 = out3+cost0

        cost1 = classif1(out1)
        cost2 = classif2(out2) + cost1
        cost3 = classif3(out3) + cost2

        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1,1)
            distribute1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(distribute1)

            cost2 = torch.squeeze(cost2,1)
            distribute2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(distribute2)

        cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        distribute3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(distribute3,dim=1)
        #For your information: This formulation 'softmax(c)' learned "similarity" 
        #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp)(pred3)

        if self.training:
            return [pred1, pred2, pred3]
        else:
            return [pred3]

    def env_specific_cost_regular_operation(self, refimg_fea, targetimg_fea, left, envs_num, env_id, num=2):

        pred1 = torch.zeros([refimg_fea.shape[0], 1, left.shape[2], left.shape[3]], device=left.device)
        pred2 = torch.zeros([refimg_fea.shape[0], 1, left.shape[2], left.shape[3]], device=left.device)
        pred3 = torch.zeros([refimg_fea.shape[0], 1, left.shape[2], left.shape[3]], device=left.device)
        for i in range(0, envs_num):
            ids = env_id==i
            if num==2:
                ids2 = torch.cat([ids,ids])
            else:
                ids2 = torch.cat([ids,ids,ids])
            if ids.sum() > 0:
                disp_ests_tmp = self.cost_volume_and_disparity_regression_envs(refimg_fea[ids2], targetimg_fea[ids2], left, envs_index=i)
                pred1[ids2] = disp_ests_tmp[0]
                pred2[ids2] = disp_ests_tmp[1]
                pred3[ids2] = disp_ests_tmp[2]
        disp_ests_aug = [pred1, pred2, pred3]

        return disp_ests_aug
    
    def env_specific_model_stage1(self, left, right, envs_num, env_id, left_global, right_global):
        refimg_fea_global     = self.feature_extraction(torch.cat([left, left_global], dim=0)).cuda(left.device)
        targetimg_fea_global  = self.feature_extraction(torch.cat([right, right_global], dim=0)).cuda(left.device)

        disp_ests_global = self.env_specific_cost_regular_operation(refimg_fea_global, targetimg_fea_global, left, envs_num, env_id, num=2) 

        return disp_ests_global, refimg_fea_global, targetimg_fea_global

    def env_specific_model_stage2(self, left, right, envs_num, env_id, left_global, right_global, left_local, right_local):
        refimg_fea_local = self.feature_extraction(torch.cat([left, left_global, left_local], dim=0))
        targetimg_fea_local = self.feature_extraction(torch.cat([right, right_global, right_local], dim=0))

        disp_ests_local = self.env_specific_cost_regular_operation(refimg_fea_local, targetimg_fea_local, left, envs_num, env_id, num=3) 

        return disp_ests_local, refimg_fea_local, targetimg_fea_local
        
    def env_specific_model_stage3(self, left, right, envs_num, env_id, left_global, right_global, left_local, right_local, left_pixel, right_pixel):
        refimg_fea_pixel_1 = self.feature_extraction(torch.cat([left, left_pixel], dim=0))
        targetimg_fea_pixel_1 = self.feature_extraction(torch.cat([right, right_pixel], dim=0))
        refimg_fea_pixel_2 = self.feature_extraction(torch.cat([left_global, left_local], dim=0))
        targetimg_fea_pixel_2 = self.feature_extraction(torch.cat([right_global, right_local], dim=0))

        disp_ests_pixel_1 = self.env_specific_cost_regular_operation(refimg_fea_pixel_1, targetimg_fea_pixel_1, left, envs_num, env_id, num=2) 
        disp_ests_pixel_2 = self.env_specific_cost_regular_operation(refimg_fea_pixel_2, targetimg_fea_pixel_2, left, envs_num, env_id, num=2) 
        return disp_ests_pixel_1, disp_ests_pixel_2, refimg_fea_pixel_1, targetimg_fea_pixel_1, refimg_fea_pixel_2, targetimg_fea_pixel_2
    
    def forward(self, left, right, left_org, right_org, batch_idx, epoch, training=False, is_envs_model=False, env_id=None, envs_num=2):

        if training:   
            b,_,h,w = left.shape
            stage_epoch = self.args.epochs // 3 

            if epoch < stage_epoch:
                left_global, right_global = self.global_visual_transformation(left, left_org, b), self.global_visual_transformation(right, right_org, b)
                # the caculation of cross-entropy loss and discrepency loss
                loss_CE_l, loss_disc_l = self.max_discrepency_loss_stage1(left, left_global, b)
                loss_CE_r, loss_disc_r = self.max_discrepency_loss_stage1(right, right_global, b)
                loss_CE, loss_disc = (loss_CE_l + loss_CE_r) / 2, (loss_disc_l + loss_disc_r) / 2

                if is_envs_model:
                    disp_ests_global, refimg_fea_global, targetimg_fea_global = self.env_specific_model_stage1(left, right, envs_num, env_id, left_global, right_global)
                else:
                    refimg_fea_global     = self.feature_extraction(torch.cat([left, left_global], dim=0)).cuda(left.device)
                    targetimg_fea_global  = self.feature_extraction(torch.cat([right, right_global], dim=0)).cuda(left.device)
                    disp_ests_global = self.cost_volume_and_disparity_regression(refimg_fea_global, targetimg_fea_global, left)
                    with torch.no_grad():
                        disp_ests_global_envs, _, _ = self.env_specific_model_stage1(left, right, envs_num, env_id, left_global, right_global)
                        disp_ests_org_envs_list = []
                        disp_ests_global_envs_list = []
                        for pred in disp_ests_global_envs:
                            disp_ests_org_envs_list.append(pred[0*b:1*b])
                            disp_ests_global_envs_list.append(pred[1*b:2*b])

                loss_dist = (torch.mean((refimg_fea_global[0*b:1*b] - refimg_fea_global[1*b:2*b]).pow(2)) + \
                            torch.mean((targetimg_fea_global[0*b:1*b] - targetimg_fea_global[1*b:2*b]).pow(2))) / 2
                loss_hvt_stage1 = self.args.lambda_1_T * loss_disc + self.args.lambda_2_T * loss_CE + self.args.lambda_3_T * loss_dist

                disp_ests_org_list = []
                disp_ests_global_list = []
                for pred in disp_ests_global:
                    disp_ests_org_list.append(pred[0*b:1*b])
                    disp_ests_global_list.append(pred[1*b:2*b])
                
                if is_envs_model:
                    return [disp_ests_org_list, disp_ests_global_list], loss_hvt_stage1
                else:
                    return [disp_ests_org_list, disp_ests_global_list], [disp_ests_org_envs_list, disp_ests_global_envs_list], loss_hvt_stage1
                
            elif epoch >= stage_epoch and epoch < stage_epoch*2:
                left_global, right_global = self.global_visual_transformation(left, left_org, b), self.global_visual_transformation(right, right_org, b)
                left_local, right_local = self.local_visual_transformation(left, left_org), self.local_visual_transformation(right, right_org)

                loss_CE_l, loss_disc_l = self.max_discrepency_loss_stage2(left, left_global, left_local, b)
                loss_CE_r, loss_disc_r = self.max_discrepency_loss_stage2(right, right_global, right_local, b)
                loss_CE, loss_disc = (loss_CE_l + loss_CE_r) / 2, (loss_disc_l + loss_disc_r) / 2

                if is_envs_model:
                    disp_ests_local, refimg_fea_local, targetimg_fea_local = self.env_specific_model_stage2(left, right, envs_num, env_id, left_global, right_global, left_local, right_local)
                else:
                    refimg_fea_local = self.feature_extraction(torch.cat([left, left_global, left_local], dim=0))
                    targetimg_fea_local = self.feature_extraction(torch.cat([right, right_global, right_local], dim=0))
                    disp_ests_local = self.cost_volume_and_disparity_regression(refimg_fea_local, targetimg_fea_local, left)
                    with torch.no_grad():
                        disp_ests_local_envs, _, _ = self.env_specific_model_stage2(left, right, envs_num, env_id, left_global, right_global, left_local, right_local)
                        disp_ests_org_envs_list = []
                        disp_ests_global_envs_list = []
                        disp_ests_local_envs_list = []
                        for pred in disp_ests_local_envs:
                            disp_ests_org_envs_list.append(pred[0*b:1*b])
                            disp_ests_global_envs_list.append(pred[1*b:2*b])
                            disp_ests_local_envs_list.append(pred[2*b:3*b])

                loss_dist = (torch.mean((refimg_fea_local[0*b:1*b] - refimg_fea_local[1*b:2*b]).pow(2)) + torch.mean((targetimg_fea_local[0*b:1*b] - targetimg_fea_local[1*b:2*b]).pow(2)) + \
                            torch.mean((refimg_fea_local[0*b:1*b] - refimg_fea_local[2*b:3*b]).pow(2)) + torch.mean((targetimg_fea_local[0*b:1*b] - targetimg_fea_local[2*b:3*b]).pow(2))) / 4
                loss_hvt_stage2 = self.args.lambda_1_T * loss_disc + self.args.lambda_2_T * loss_CE + self.args.lambda_3_T * loss_dist

                disp_ests_org_list = []
                disp_ests_global_list = []
                disp_ests_local_list = []
                for pred in disp_ests_local:
                    disp_ests_org_list.append(pred[0*b:1*b])
                    disp_ests_global_list.append(pred[1*b:2*b])
                    disp_ests_local_list.append(pred[2*b:3*b])
                
                if is_envs_model:
                    return [disp_ests_org_list, disp_ests_global_list, disp_ests_local_list], loss_hvt_stage2
                else:
                    return [disp_ests_org_list, disp_ests_global_list, disp_ests_local_list], [disp_ests_org_envs_list, disp_ests_global_envs_list, disp_ests_local_envs_list], loss_hvt_stage2
                
            elif epoch >= stage_epoch*2:
                left_global, right_global = self.global_visual_transformation(left, left_org, b), self.global_visual_transformation(right, right_org, b)
                left_local, right_local = self.local_visual_transformation(left, left_org), self.local_visual_transformation(right, right_org)
                left_pixel, right_pixel = self.pixel_visual_transformation(left), self.pixel_visual_transformation(right)  

                loss_CE_l, loss_disc_l = self.max_discrepency_loss_stage3(left, left_global, left_local, left_pixel, b)
                loss_CE_r, loss_disc_r = self.max_discrepency_loss_stage3(right, right_global, right_local, right_pixel, b)
                loss_CE, loss_disc = (loss_CE_l + loss_CE_r) / 2, (loss_disc_l + loss_disc_r) / 2

                if is_envs_model:
                    disp_ests_pixel_1, disp_ests_pixel_2, refimg_fea_pixel_1, targetimg_fea_pixel_1, refimg_fea_pixel_2, targetimg_fea_pixel_2 = self.env_specific_model_stage3(left, right, envs_num, env_id, left_global, right_global, left_local, right_local, left_pixel, right_pixel)
                else:
                    refimg_fea_pixel_1 = self.feature_extraction(torch.cat([left, left_pixel], dim=0))
                    targetimg_fea_pixel_1 = self.feature_extraction(torch.cat([right, right_pixel], dim=0))
                    refimg_fea_pixel_2 = self.feature_extraction(torch.cat([left_global, left_local], dim=0))
                    targetimg_fea_pixel_2 = self.feature_extraction(torch.cat([right_global, right_local], dim=0))
                    disp_ests_pixel_1 = self.cost_volume_and_disparity_regression(refimg_fea_pixel_1, targetimg_fea_pixel_1, left)
                    disp_ests_pixel_2 = self.cost_volume_and_disparity_regression(refimg_fea_pixel_2, targetimg_fea_pixel_2, left)
                    with torch.no_grad():
                        disp_ests_pixel_1_envs, disp_ests_pixel_2_envs, _, _, _, _ = self.env_specific_model_stage3(left, right, envs_num, env_id, left_global, right_global, left_local, right_local, left_pixel, right_pixel)
                        disp_ests_org_envs_list = []
                        disp_ests_pixel_envs_list = []
                        for pred in disp_ests_pixel_1_envs:
                            disp_ests_org_envs_list.append(pred[0*b:1*b])
                            disp_ests_pixel_envs_list.append(pred[1*b:2*b])

                        disp_ests_global_envs_list = []
                        disp_ests_local_envs_list = []
                        for pred in disp_ests_pixel_2_envs:
                            disp_ests_global_envs_list.append(pred[0*b:1*b])
                            disp_ests_local_envs_list.append(pred[1*b:2*b])

                loss_dist = (torch.mean((refimg_fea_pixel_1[0*b:1*b] - refimg_fea_pixel_2[0*b:1*b]).pow(2)) + torch.mean((targetimg_fea_pixel_1[0*b:1*b] - targetimg_fea_pixel_2[0*b:1*b]).pow(2)) + \
                            torch.mean((refimg_fea_pixel_1[0*b:1*b] - refimg_fea_pixel_2[1*b:2*b]).pow(2)) + torch.mean((targetimg_fea_pixel_1[0*b:1*b] - targetimg_fea_pixel_2[1*b:2*b]).pow(2)) + \
                            torch.mean((refimg_fea_pixel_1[0*b:1*b] - refimg_fea_pixel_1[1*b:2*b]).pow(2)) + torch.mean((targetimg_fea_pixel_1[0*b:1*b] - targetimg_fea_pixel_1[1*b:2*b]).pow(2))) / 6
                loss_hvt_stage3 = self.args.lambda_1_T * loss_disc + self.args.lambda_2_T * loss_CE + self.args.lambda_3_T * loss_dist

                disp_ests_org_list = []
                disp_ests_pixel_list = []
                for pred in disp_ests_pixel_1:
                    disp_ests_org_list.append(pred[0*b:1*b])
                    disp_ests_pixel_list.append(pred[1*b:2*b])

                disp_ests_global_list = []
                disp_ests_local_list = []
                for pred in disp_ests_pixel_2:
                    disp_ests_global_list.append(pred[0*b:1*b])
                    disp_ests_local_list.append(pred[1*b:2*b])

                if is_envs_model:
                    return [disp_ests_org_list, disp_ests_global_list, disp_ests_local_list, disp_ests_pixel_list], loss_hvt_stage3
                else:
                    return [disp_ests_org_list, disp_ests_global_list, disp_ests_local_list, disp_ests_pixel_list],[disp_ests_org_envs_list, disp_ests_global_envs_list, disp_ests_local_envs_list, disp_ests_pixel_envs_list], loss_hvt_stage3
                    


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
