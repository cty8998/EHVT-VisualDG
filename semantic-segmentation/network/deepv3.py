"""
# Code Adapted from:
# https://github.com/sthalles/deeplab_v3
#
# MIT License
#
# Copyright (c) 2018 Thalles Santos Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
"""
import logging
import torch
from torch import nn
from network import Resnet
from network import Mobilenet
from network import Shufflenet
from network.cov_settings import CovMatrix_ISW, CovMatrix_IRW
from network.instance_whitening import instance_whitening_loss, get_covariance_matrix
from network.mynn import initialize_weights, Norm2d, Upsample, freeze_weights, unfreeze_weights

import torchvision.models as models

from torchvision.transforms.functional_tensor import _rgb2hsv, _hsv2rgb
import torchvision.transforms as transforms
import random
import math
def rgb_to_grayscale(img):
    r, g, b = img.unbind(dim=-3)
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)
    return l_img

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn
        print("output_stride = ", output_stride)
        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 4:
            rates = [4 * r for r in rates]
        elif output_stride == 16:
            pass
        elif output_stride == 32:
            rates = [r // 2 for r in rates]
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1, bias=False),
            Norm2d(256), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class DeepV3Plus(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-101', criterion=None, criterion_aux=None,
                variant='D', skip='m1', skip_num=48, args=None):
        super(DeepV3Plus, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.variant = variant
        self.args = args
        self.trunk = trunk

        self.num_classes = num_classes

        channel_1st = 3
        channel_2nd = 64
        channel_3rd = 256
        channel_4th = 512
        prev_final_channel = 1024
        final_channel = 2048

        if trunk == 'resnet-18':
            channel_1st = 3
            channel_2nd = 64
            channel_3rd = 64
            channel_4th = 128
            prev_final_channel = 256
            final_channel = 512
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
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        self.layer0_envs0 = resnet.layer0
        self.layer1_envs0, self.layer2_envs0, self.layer3_envs0, self.layer4_envs0 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        
        self.layer0_envs1 = resnet.layer0
        self.layer1_envs1, self.layer2_envs1, self.layer3_envs1, self.layer4_envs1 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.variant == 'D4':
            for n, m in self.layer2.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (8, 8), (8, 8), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            # raise 'unknown deepv3 variant: {}'.format(self.variant)
            print("Not using Dilation ")

        if self.variant == 'D':
            os = 8
        elif self.variant == 'D4':
            os = 4
        elif self.variant == 'D16':
            os = 16
        else:
            os = 32

        self.output_stride = os
        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256,
                                                    output_stride=os)

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))

        self.dsn = nn.Sequential(
            nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
            Norm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        initialize_weights(self.dsn)

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)


        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        if trunk == 'resnet-101':
            self.three_input_layer = True
            in_channel_list = [64, 64, 128, 256, 512, 1024, 2048]   # 8128, 32640, 130816
            out_channel_list = [32, 32, 64, 128, 256,  512, 1024]
        elif trunk == 'resnet-18':
            self.three_input_layer = False
            in_channel_list = [0, 0, 64, 64, 128, 256, 512]   # 8128, 32640, 130816
            out_channel_list = [0, 0, 32, 32, 64,  128, 256]
        else: # ResNet-50
            self.three_input_layer = False
            in_channel_list = [0, 0, 64, 256, 512, 1024, 2048]   # 8128, 32640, 130816
            out_channel_list = [0, 0, 32, 128, 256,  512, 1024]

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

        # EIL module : environment-independent predictor part
        self.aspp_envs0, self.bot_fine_envs0, self.bot_aspp_envs0, self.final1_envs0, self.final2_envs0, self.dsn_envs0 = self.init_predictor_layer()
        self.aspp_envs1, self.bot_fine_envs1, self.bot_aspp_envs1, self.final1_envs1, self.final2_envs1, self.dsn_envs1 = self.init_predictor_layer()

        # HVT module 
        # domain discriminating network
        self.feature_discriminator_res18 = ResNet_res(BasicBlock_res, [2, 2, 2, 2], num_classes=4)
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

        self.alpha_pixel = nn.Parameter(torch.randn(1,3,768,768))

        self.toPIL = transforms.ToPILImage(mode="RGB")
        self.mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).cuda()

    def init_predictor_layer(self):
        channel_3rd = 256
        final_channel = 2048
        os = 16
        prev_final_channel = 1024
        num_classes = self.num_classes
        aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256, output_stride=os).cuda()
        bot_fine = nn.Sequential(nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),Norm2d(48),nn.ReLU(inplace=True)).cuda()
        bot_aspp = nn.Sequential(nn.Conv2d(1280, 256, kernel_size=1, bias=False), Norm2d(256), nn.ReLU(inplace=True)).cuda()
        final1 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False), Norm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), Norm2d(256), nn.ReLU(inplace=True)).cuda()
        final2 = nn.Sequential(nn.Conv2d(256, num_classes, kernel_size=1, bias=True)).cuda()
        dsn = nn.Sequential(nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1), Norm2d(512), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
                            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)).cuda()
        
        initialize_weights(dsn)
        initialize_weights(aspp)
        initialize_weights(bot_aspp)
        initialize_weights(bot_fine)
        initialize_weights(final1)
        initialize_weights(final2)
        
        return aspp, bot_fine, bot_aspp, final1, final2, dsn


    def set_mask_matrix(self):
        for index in range(len(self.cov_matrix_layer)):
            self.cov_matrix_layer[index].set_mask_matrix()


    def reset_mask_matrix(self):
        for index in range(len(self.cov_matrix_layer)):
            self.cov_matrix_layer[index].reset_mask_matrix()

    def feature_extraction_x_war(self, x, w_arr):
        if self.trunk == 'mobilenetv2' or self.trunk == 'shufflenetv2':
            x_tuple = self.layer0([x, w_arr])
            x = x_tuple[0]
            w_arr = x_tuple[1]
        else:   # ResNet
            if self.three_input_layer:
                x = self.layer0[0](x)
                if self.args.wt_layer[0] == 1 or self.args.wt_layer[0] == 2:
                    x, w = self.layer0[1](x)
                    w_arr.append(w)
                else:
                    x = self.layer0[1](x)
                x = self.layer0[2](x)
                x = self.layer0[3](x)
                if self.args.wt_layer[1] == 1 or self.args.wt_layer[1] == 2:
                    x, w = self.layer0[4](x)
                    w_arr.append(w)
                else:
                    x = self.layer0[4](x)
                x = self.layer0[5](x)
                x = self.layer0[6](x)
                if self.args.wt_layer[2] == 1 or self.args.wt_layer[2] == 2:
                    x, w = self.layer0[7](x)
                    w_arr.append(w)
                else:
                    x = self.layer0[7](x)
                x = self.layer0[8](x)
                x = self.layer0[9](x)
            else:   # Single Input Layer
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
        
        return x, w_arr, x_tuple, low_level, aux_out

    def feature_extraction_x_war_envs(self, x, w_arr):

        if self.trunk == 'mobilenetv2' or self.trunk == 'shufflenetv2':
            x_tuple = self.layer0([x, w_arr])
            x = x_tuple[0]
            w_arr = x_tuple[1]
        else:   # ResNet
            if self.three_input_layer:
                x = self.layer0[0](x)
                if self.args.wt_layer[0] == 1 or self.args.wt_layer[0] == 2:
                    x, w = self.layer0[1](x)
                    w_arr.append(w)
                else:
                    x = self.layer0[1](x)
                x = self.layer0[2](x)
                x = self.layer0[3](x)
                if self.args.wt_layer[1] == 1 or self.args.wt_layer[1] == 2:
                    x, w = self.layer0[4](x)
                    w_arr.append(w)
                else:
                    x = self.layer0[4](x)
                x = self.layer0[5](x)
                x = self.layer0[6](x)
                if self.args.wt_layer[2] == 1 or self.args.wt_layer[2] == 2:
                    x, w = self.layer0[7](x)
                    w_arr.append(w)
                else:
                    x = self.layer0[7](x)
                x = self.layer0[8](x)
                x = self.layer0[9](x)
            else:   # Single Input self.layer
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
        
        return x, w_arr, x_tuple, low_level, aux_out
    
    def set_w_war(self, w_arr):
        for index, f_map in enumerate(w_arr):
            # Instance Whitening
            B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
            HW = H * W
            f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
            eye, reverse_eye = self.cov_matrix_layer[index].get_eye_matrix()
            f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW - 1) + (self.eps * eye)  # B X C X C / HW
            off_diag_elements = f_cor * reverse_eye
            #print("here", off_diag_elements.shape)
            self.cov_matrix_layer[index].set_variance_of_covariance(torch.var(off_diag_elements, dim=0))

    def get_main_out(self, x, low_level, x_size, aux_out):
        x = self.aspp(x)
        dec0_up = self.bot_aspp(x)

        dec0_fine = self.bot_fine(low_level)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final1(dec0)
        dec2 = self.final2(dec1)
        main_out = Upsample(dec2, x_size[2:])

        aux_out = self.dsn(aux_out)

        return main_out, aux_out

    def get_main_aux_wt_loss(self, main_out, aux_out, w_arr, gts, aux_gts, apply_wtloss):
        loss1 = self.criterion(main_out, gts)
        if self.args.use_wtloss:
            wt_loss = torch.FloatTensor([0]).cuda()
            if apply_wtloss:
                for index, f_map in enumerate(w_arr):
                    eye, mask_matrix, margin, num_remove_cov = self.cov_matrix_layer[index].get_mask_matrix()
                    loss = instance_whitening_loss(f_map, eye, mask_matrix, margin, num_remove_cov)
                    wt_loss = wt_loss + loss
            wt_loss = wt_loss / len(w_arr)

        if aux_gts.dim() == 1:
            aux_gts = gts
        aux_gts = aux_gts.unsqueeze(1).float()
        aux_gts = nn.functional.interpolate(aux_gts, size=aux_out.shape[2:], mode='nearest')
        aux_gts = aux_gts.squeeze(1).long()
        loss2 = self.criterion_aux(aux_out, aux_gts)

        return_loss = [loss1, loss2]
        if self.args.use_wtloss:
            return_loss.append(wt_loss)

        return return_loss

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
                img_local[:, (i // self.args.num_patch_row)*h_p:(i // self.args.num_patch_row+1)*h_p, (i % self.args.num_patch_column)*h_p:(i % self.args.num_patch_column+1)*w_p] = \
                img_local[:, (i // self.args.num_patch_row)*h_p:(i // self.args.num_patch_row+1)*h_p, (i % self.args.num_patch_column)*h_p:(i % self.args.num_patch_column+1)*w_p] + img_patch_list_g[i].squeeze(0)
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
        pixel = torch.randn(1,3,768,768, device=image.device) * (torch.sigmoid(self.alpha_pixel) * self.args.mu_p + self.args.beta_p) 
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

    def prediction_last_3layers_envs(self, x, x_size, aux_out, low_level, envs_id=0):
        # acquire the environment specific model layers
        if envs_id==0:
            aspp, bot_fine, bot_aspp, final1, final2, dsn = self.aspp_envs0, self.bot_fine_envs0, self.bot_aspp_envs0, self.final1_envs0, self.final2_envs0, self.dsn_envs0
        elif envs_id==1:
            aspp, bot_fine, bot_aspp, final1, final2, dsn = self.aspp_envs1, self.bot_fine_envs1, self.bot_aspp_envs1, self.final1_envs1, self.final2_envs1, self.dsn_envs1

        # conduct the prediction with environment specific model layers
        if x.shape[0] == 1:
            x = self.aspp(torch.cat([x,x], dim=0))[0:1]
        else:
            x = self.aspp(x)
        dec0_up = self.bot_aspp(x)

        dec0_fine = self.bot_fine(low_level)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = final1(dec0)
        dec2 = final2(dec1)
        main_out = Upsample(dec2, x_size[2:])

        aux_out = dsn(aux_out)
        output = {
            'main_out': main_out,
            'aux_out': aux_out,
        }
            
        return output

    def prediction_last_6layers_envs(self, x, x_size, aux_out, low_level, envs_id=0):
        # acquire the environment specific model layers--two environments

        if envs_id==0:
            aspp, bot_fine, bot_aspp, final1, final2, dsn = self.aspp_envs0, self.bot_fine_envs0, self.bot_aspp_envs0, self.final1_envs0, self.final2_envs0, self.dsn_envs0
        elif envs_id==1:
            aspp, bot_fine, bot_aspp, final1, final2, dsn = self.aspp_envs1, self.bot_fine_envs1, self.bot_aspp_envs1, self.final1_envs1, self.final2_envs1, self.dsn_envs1

        # conduct the prediction with environment specific model layers
        if x.shape[0] == 1:
            x = aspp(torch.cat([x,x], dim=0))[0:1]
        else:
            x = aspp(x)
        dec0_up = bot_aspp(x)

        dec0_fine = bot_fine(low_level)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = final1(dec0)
        dec2 = final2(dec1)
        main_out = Upsample(dec2, x_size[2:])

        aux_out = dsn(aux_out)
        output = {
            'main_out': main_out,
            'aux_out': aux_out,
        }
            
        return output
    
    def env_specific_model_stage1(self, x_imgs, envs_id, w_arr, x_size, b):
        # initial the output tensors
        main_out_combine_og = torch.zeros((x_imgs.shape[0], self.num_classes, 768, 768), device=x_imgs.device)
        aux_out_combine_og = torch.zeros((x_imgs.shape[0], self.num_classes, 48, 48), device=x_imgs.device)
        # x_combine_og = torch.zeros((x_imgs.shape[0], 2048, 48, 48), device=x_imgs.device)
        # aux_combine_og = torch.zeros((x_imgs.shape[0], 1024, 48, 48), device=x_imgs.device)

        x_og, w_arr, x_tuple_og, low_level_og, aux_out_og = self.feature_extraction_x_war_envs(x_imgs, w_arr)
        
        for i in range(0, self.args.envs_num):
            # ids：the mask that training samples belonging to the env i
            # ids2: the transformed images generated by HVT have the same enviromental labels with original images
            ids = envs_id==i
            ids2 = torch.cat([ids, ids], dim=0)
            # train the environment specific model with corresponding that training samples belonging to this environment i
            if ids.sum() > 0:
                output_tmp = self.prediction_last_3layers_envs(x_og[ids2], x_size, aux_out_og[ids2], low_level_og[ids2], envs_id=i)
                main_out_combine_og[ids2] = output_tmp['main_out']
                aux_out_combine_og[ids2] = output_tmp['aux_out']
        
        output_og = {'main_out': main_out_combine_og, 'aux_out': aux_out_combine_og}
        output_original = {'main_out': output_og['main_out'][0*b:0*b+b], 'aux_out': output_og['aux_out'][0*b:0*b+b]}
        output_gloabl = {'main_out': output_og['main_out'][1*b:1*b+b], 'aux_out': output_og['aux_out'][1*b:1*b+b]}

        return [output_og, output_original, output_gloabl], x_og, aux_out_og, w_arr

    
    def env_specific_model_stage2(self, x_imgs, envs_id, w_arr, x_size, b):
        # initial the output tensors
        main_out_combine_ogl = torch.zeros((x_imgs.shape[0], self.num_classes, 768, 768), device=x_imgs.device)
        aux_out_combine_ogl = torch.zeros((x_imgs.shape[0], self.num_classes, 48, 48), device=x_imgs.device)
        # x_combine_og = torch.zeros((x_imgs.shape[0], 2048, 48, 48), device=x_imgs.device)
        # aux_combine_og = torch.zeros((x_imgs.shape[0], 1024, 48, 48), device=x_imgs.device)

        x_ogl, w_arr, x_tuple_ogl, low_level_ogl, aux_out_ogl = self.feature_extraction_x_war_envs(x_imgs, w_arr)
        
        for i in range(0, self.args.envs_num):
            # ids：the mask that training samples belonging to the env i
            # ids2: the transformed images generated by HVT have the same enviromental labels with original images
            ids = envs_id==i
            ids2 = torch.cat([ids, ids, ids], dim=0)
            # train the environment specific model with corresponding that training samples belonging to this environment i
            if ids.sum() > 0:
                output_tmp = self.prediction_last_3layers_envs(x_ogl[ids2], x_size, aux_out_ogl[ids2], low_level_ogl[ids2], envs_id=i)
                main_out_combine_ogl[ids2] = output_tmp['main_out']
                aux_out_combine_ogl[ids2] = output_tmp['aux_out']
        
        output_ogl = {'main_out': main_out_combine_ogl, 'aux_out': aux_out_combine_ogl}
        output_original = {'main_out': output_ogl['main_out'][0*b:0*b+b], 'aux_out': output_ogl['aux_out'][0*b:0*b+b]}
        output_gloabl = {'main_out': output_ogl['main_out'][1*b:1*b+b], 'aux_out': output_ogl['aux_out'][1*b:1*b+b]}
        output_local = {'main_out': output_ogl['main_out'][2*b:2*b+b], 'aux_out': output_ogl['aux_out'][2*b:2*b+b]}

        return [output_ogl, output_original, output_gloabl, output_local], x_ogl, aux_out_ogl, w_arr
    
    def env_specific_model_stage3(self, x_imgs, envs_id, w_arr, x_size, b):
        # initial the output tensors
        main_out_combine_oglp = torch.zeros((x_imgs.shape[0], self.num_classes, 768, 768), device=x_imgs.device)
        aux_out_combine_oglp = torch.zeros((x_imgs.shape[0], self.num_classes, 48, 48), device=x_imgs.device)
        # x_combine_og = torch.zeros((x_imgs.shape[0], 2048, 48, 48), device=x_imgs.device)
        # aux_combine_og = torch.zeros((x_imgs.shape[0], 1024, 48, 48), device=x_imgs.device)

        x_oglp, w_arr, x_tuple_oglp, low_level_oglp, aux_out_oglp = self.feature_extraction_x_war_envs(x_imgs, w_arr)
        
        for i in range(0, self.args.envs_num):
            # ids：the mask that training samples belonging to the env i
            # ids2: the transformed images generated by HVT have the same enviromental labels with original images
            ids = envs_id==i
            ids2 = torch.cat([ids, ids, ids, ids], dim=0)
            # train the environment specific model with corresponding that training samples belonging to this environment i
            if ids.sum() > 0:
                output_tmp = self.prediction_last_3layers_envs(x_oglp[ids2], x_size, aux_out_oglp[ids2], low_level_oglp[ids2], envs_id=i)
                main_out_combine_oglp[ids2] = output_tmp['main_out']
                aux_out_combine_oglp[ids2] = output_tmp['aux_out']
        
        output_oglp = {'main_out': main_out_combine_oglp, 'aux_out': aux_out_combine_oglp}
        output_original = {'main_out': output_oglp['main_out'][0*b:0*b+b], 'aux_out': output_oglp['aux_out'][0*b:0*b+b]}
        output_gloabl = {'main_out': output_oglp['main_out'][1*b:1*b+b], 'aux_out': output_oglp['aux_out'][1*b:1*b+b]}
        output_local = {'main_out': output_oglp['main_out'][2*b:2*b+b], 'aux_out': output_oglp['aux_out'][2*b:2*b+b]}
        output_pixel = {'main_out': output_oglp['main_out'][3*b:3*b+b], 'aux_out': output_oglp['aux_out'][3*b:3*b+b]}

        return [output_oglp, output_original, output_gloabl, output_local, output_pixel], x_oglp, aux_out_oglp, w_arr

    def calculate_loss_dist_out(self, main_out, main_out_hvt):
        main_out = torch.softmax(main_out.permute(0,2,3,1).reshape(-1, self.num_classes), dim=1)
        main_out_hvt = torch.softmax(main_out_hvt.permute(0,2,3,1).reshape(-1, self.num_classes), dim=1)
        scloss = torch.mean((main_out- main_out_hvt).pow(2))

        return scloss

    def forward(self, x, envs_id=None, is_envs_model=False, curr_iter=None, gts=None, aux_gts=None, img_gt=None, visualize=False, cal_covstat=False, apply_wtloss=True):
        if cal_covstat:
            x = torch.cat(x, dim=0)

        stage_iter = self.args.max_iter // 3 

        if self.training:
        
            w_arr = []
            x_org_ = ((x * self.std + self.mean) * 255).permute(0,2,3,1).to(dtype=torch.uint8)
            b = x.shape[0]
            if curr_iter < stage_iter:
                x_global = self.global_visual_transformation(x, x_org_, b)

                # the caculation of cross-entropy loss and discrepency loss
                loss_CE, loss_disc = self.max_discrepency_loss_stage1(x, x_global, b)

                if cal_covstat:
                    self.set_w_war(w_arr)
                    return 0
                gts_og = torch.cat([gts, gts], dim=0)
                aux_gts_og = torch.cat([aux_gts, aux_gts], dim=0)
                x_size = torch.cat([x, x_global], dim=0).size()

                # train environment specific model
                if is_envs_model == True:
                    output_envs_list, x_og, aux_out_og, w_arr = self.env_specific_model_stage1(torch.cat([x, x_global], dim=0), envs_id, w_arr, x_size, b)
                    output_og_envs, output_original_envs, output_global_envs = output_envs_list[0], output_envs_list[1], output_envs_list[2]
                    loss_dist = (torch.mean((x_og[0*b:0*b+b] - x_og[1*b:1*b+b]).pow(2)) + 1* torch.mean((aux_out_og[0*b:0*b+b] - aux_out_og[1*b:1*b+b]).pow(2))) / 2
                    loss_dist_out = self.calculate_loss_dist_out(output_og_envs['main_out'][0*b:0*b+b], output_og_envs['main_out'][1*b:1*b+b])  
                    loss_hvt_stage1 = self.args.lambda_1_T * loss_disc + self.args.lambda_2_T * loss_CE + self.args.lambda_3_T * (loss_dist + loss_dist_out)
                    # loss_hvt_stage1 = self.args.lambda_1_T * loss_disc + self.args.lambda_2_T * loss_CE + self.args.lambda_3_T * loss_dist

                    main_out_og, aux_out_og = output_og_envs['main_out'], output_og_envs['aux_out']
                    return_loss = self.get_main_aux_wt_loss(main_out_og, aux_out_og, w_arr, gts_og, aux_gts_og, apply_wtloss)
                    return return_loss, loss_hvt_stage1, output_original_envs
                
                # train global model
                else:
                    x_og, w_arr, x_tuple, low_level, aux_out_og = self.feature_extraction_x_war(torch.cat([x, x_global], dim=0), w_arr)
                    # the caculation of distance loss by minimizing cross-domain feature inconsistency between original feature and transformed feature
                    loss_dist = (torch.mean((x_og[0*b:0*b+b] - x_og[1*b:1*b+b]).pow(2)) + 1* torch.mean((aux_out_og[0*b:0*b+b] - aux_out_og[1*b:1*b+b]).pow(2))) / 2

                    main_out_og, aux_out_og_gm = self.get_main_out(x_og, low_level, x_size, aux_out_og)
                    output_og = {'main_out': main_out_og, 'aux_out': aux_out_og_gm}
                    output_original = {'main_out': output_og['main_out'][0*b:0*b+b], 'aux_out': output_og['aux_out'][0*b:0*b+b]}

                    loss_dist_out = self.calculate_loss_dist_out(output_og['main_out'][0*b:0*b+b], output_og['main_out'][1*b:1*b+b])  
                    loss_hvt_stage1 = self.args.lambda_1_T * loss_disc + self.args.lambda_2_T * loss_CE + self.args.lambda_3_T * (loss_dist + loss_dist_out)
                    # loss_hvt_stage1 = self.args.lambda_1_T * loss_disc + self.args.lambda_2_T * loss_CE + self.args.lambda_3_T * loss_dist

                    return_loss = self.get_main_aux_wt_loss(main_out_og, aux_out_og_gm, w_arr, gts_og, aux_gts_og, apply_wtloss)

                    return return_loss, loss_hvt_stage1, output_original

            elif curr_iter >= stage_iter and curr_iter < stage_iter*2:
                x_global = self.global_visual_transformation(x, x_org_, b)
                x_local = self.local_visual_transformation(x, x_org_)
                # the caculation of cross-entropy loss and discrepency loss
                loss_CE, loss_disc = self.max_discrepency_loss_stage2(x, x_global, x_local, b)

                if cal_covstat:
                    self.set_w_war(w_arr)
                    return 0
                gts_ogl = torch.cat([gts, gts, gts], dim=0)
                aux_gts_ogl = torch.cat([aux_gts, aux_gts, aux_gts], dim=0)
                x_size = torch.cat([x, x_global, x_local], dim=0).size()

                # train environment specific model
                if is_envs_model == True:
                    output_envs_list, x_ogl, aux_out_ogl, w_arr = self.env_specific_model_stage2(torch.cat([x, x_global, x_local], dim=0), envs_id, w_arr, x_size, b)
                    output_ogl_envs, output_original_envs, output_global_envs, output_local_envs = output_envs_list[0], output_envs_list[1], output_envs_list[2], output_envs_list[3]
                    loss_dist = (torch.mean((x_ogl[0*b:0*b+b] - x_ogl[1*b:1*b+b]).pow(2)) + 1 * torch.mean((aux_out_ogl[0*b:0*b+b] - aux_out_ogl[1*b:1*b+b]).pow(2)) + \
                                 torch.mean((x_ogl[0*b:0*b+b] - x_ogl[2*b:2*b+b]).pow(2)) + 1 * torch.mean((aux_out_ogl[0*b:0*b+b] - aux_out_ogl[2*b:2*b+b]).pow(2))) / 4
                    loss_dist_out = (self.calculate_loss_dist_out(output_ogl_envs['main_out'][0*b:0*b+b], output_ogl_envs['main_out'][1*b:1*b+b]) + \
                                     self.calculate_loss_dist_out(output_ogl_envs['main_out'][0*b:0*b+b], output_ogl_envs['main_out'][2*b:2*b+b])) / 2  
                    loss_hvt_stage2 = self.args.lambda_1_T * loss_disc + self.args.lambda_2_T * loss_CE + self.args.lambda_3_T * (loss_dist + loss_dist_out)
                    # loss_hvt_stage2 = self.args.lambda_1_T * loss_disc + self.args.lambda_2_T * loss_CE + self.args.lambda_3_T * loss_dist

                    main_out_ogl, aux_out_ogl = output_ogl_envs['main_out'], output_ogl_envs['aux_out']
                    return_loss = self.get_main_aux_wt_loss(main_out_ogl, aux_out_ogl, w_arr, gts_ogl, aux_gts_ogl, apply_wtloss)
                    return return_loss, loss_hvt_stage2, output_original_envs
                
                # train global model
                else:
                    x_ogl, w_arr, x_tuple, low_level, aux_out_ogl = self.feature_extraction_x_war(torch.cat([x, x_global, x_local], dim=0), w_arr)
                    # the caculation of distance loss by minimizing cross-domain feature inconsistency between original feature and transformed feature
                    loss_dist = (torch.mean((x_ogl[0*b:0*b+b] - x_ogl[1*b:1*b+b]).pow(2)) + 1 * torch.mean((aux_out_ogl[0*b:0*b+b] - aux_out_ogl[1*b:1*b+b]).pow(2)) + \
                                 torch.mean((x_ogl[0*b:0*b+b] - x_ogl[2*b:2*b+b]).pow(2)) + 1 * torch.mean((aux_out_ogl[0*b:0*b+b] - aux_out_ogl[2*b:2*b+b]).pow(2))) / 4

                    main_out_ogl, aux_out_ogl_gm = self.get_main_out(x_ogl, low_level, x_size, aux_out_ogl)
                    output_ogl = {'main_out': main_out_ogl, 'aux_out': aux_out_ogl_gm}
                    output_original = {'main_out': output_ogl['main_out'][0*b:0*b+b], 'aux_out': output_ogl['aux_out'][0*b:0*b+b]}

                    loss_dist_out = (self.calculate_loss_dist_out(output_ogl['main_out'][0*b:0*b+b], output_ogl['main_out'][1*b:1*b+b]) + \
                                     self.calculate_loss_dist_out(output_ogl['main_out'][0*b:0*b+b], output_ogl['main_out'][2*b:2*b+b])) / 2
                    loss_hvt_stage2 = self.args.lambda_1_T * loss_disc + self.args.lambda_2_T * loss_CE + self.args.lambda_3_T * (loss_dist + loss_dist_out)
                    # loss_hvt_stage2 = self.args.lambda_1_T * loss_disc + self.args.lambda_2_T * loss_CE + self.args.lambda_3_T * loss_dist

                    return_loss = self.get_main_aux_wt_loss(main_out_ogl, aux_out_ogl_gm, w_arr, gts_ogl, aux_gts_ogl, apply_wtloss)

                    return return_loss, loss_hvt_stage2, output_original

            
            elif curr_iter >= stage_iter*2:
                x_global = self.global_visual_transformation(x, x_org_, b)
                x_local = self.local_visual_transformation(x, x_org_)
                x_pixel = self.pixel_visual_transformation(x) 

                # the caculation of cross-entropy loss and discrepency loss
                loss_CE, loss_disc = self.max_discrepency_loss_stage3(x, x_global, x_local, x_pixel, b)

                if cal_covstat:
                    self.set_w_war(w_arr)
                    return 0
                gts_oglp = torch.cat([gts, gts, gts, gts], dim=0)
                aux_gts_oglp = torch.cat([aux_gts, aux_gts, aux_gts, aux_gts], dim=0)
                x_size = torch.cat([x, x_global, x_local, x_pixel], dim=0).size()

                # train environment specific model
                if is_envs_model == True:
                    output_envs_list, x_oglp, aux_out_oglp, w_arr = self.env_specific_model_stage3(torch.cat([x, x_global, x_local, x_pixel], dim=0), envs_id, w_arr, x_size, b)
                    output_oglp_envs, output_original_envs, output_global_envs, output_local_envs = output_envs_list[0], output_envs_list[1], output_envs_list[2], output_envs_list[3]
                    loss_dist = (torch.mean((x_oglp[0*b:0*b+b] - x_oglp[1*b:1*b+b]).pow(2)) + 1 * torch.mean((aux_out_oglp[0*b:0*b+b] - aux_out_oglp[1*b:1*b+b]).pow(2)) + \
                             torch.mean((x_oglp[0*b:0*b+b] - x_oglp[2*b:2*b+b]).pow(2)) + 1 * torch.mean((aux_out_oglp[0*b:0*b+b] - aux_out_oglp[2*b:2*b+b]).pow(2)) + \
                             torch.mean((x_oglp[0*b:0*b+b] - x_oglp[3*b:3*b+b]).pow(2)) + 1 * torch.mean((aux_out_oglp[0*b:0*b+b] - aux_out_oglp[3*b:3*b+b]).pow(2))) / 6
                    loss_dist_out = (self.calculate_loss_dist_out(output_oglp_envs['main_out'][0*b:0*b+b], output_oglp_envs['main_out'][1*b:1*b+b]) + \
                                     self.calculate_loss_dist_out(output_oglp_envs['main_out'][0*b:0*b+b], output_oglp_envs['main_out'][2*b:2*b+b]) + \
                                     self.calculate_loss_dist_out(output_oglp_envs['main_out'][0*b:0*b+b], output_oglp_envs['main_out'][3*b:3*b+b])) / 3
                    loss_hvt_stage3 = self.args.lambda_1_T * loss_disc + self.args.lambda_2_T * loss_CE + self.args.lambda_3_T * (loss_dist + loss_dist_out)
                    # loss_hvt_stage3 = self.args.lambda_1_T * loss_disc + self.args.lambda_2_T * loss_CE + self.args.lambda_3_T * loss_dist

                    main_out_oglp, aux_out_oglp = output_oglp_envs['main_out'], output_oglp_envs['aux_out']
                    return_loss = self.get_main_aux_wt_loss(main_out_oglp, aux_out_oglp, w_arr, gts_oglp, aux_gts_oglp, apply_wtloss)
                    return return_loss, loss_hvt_stage3, output_original_envs
                # train global model
                else:
                    x_oglp, w_arr, x_tuple, low_level, aux_out_oglp = self.feature_extraction_x_war(torch.cat([x, x_global, x_local, x_pixel], dim=0), w_arr)
                    # the caculation of distance loss by minimizing cross-domain feature inconsistency between original feature and transformed feature
                    loss_dist = (torch.mean((x_oglp[0*b:0*b+b] - x_oglp[1*b:1*b+b]).pow(2)) + 1 * torch.mean((aux_out_oglp[0*b:0*b+b] - aux_out_oglp[1*b:1*b+b]).pow(2)) + \
                             torch.mean((x_oglp[0*b:0*b+b] - x_oglp[2*b:2*b+b]).pow(2)) + 1 * torch.mean((aux_out_oglp[0*b:0*b+b] - aux_out_oglp[2*b:2*b+b]).pow(2)) + \
                             torch.mean((x_oglp[0*b:0*b+b] - x_oglp[3*b:3*b+b]).pow(2)) + 1 * torch.mean((aux_out_oglp[0*b:0*b+b] - aux_out_oglp[3*b:3*b+b]).pow(2))) / 6
            
                    main_out_oglp, aux_out_oglp_gm = self.get_main_out(x_oglp, low_level, x_size, aux_out_oglp)
                    output_oglp = {'main_out': main_out_oglp, 'aux_out': aux_out_oglp_gm}
                    output_original = {'main_out': output_oglp['main_out'][0*b:0*b+b], 'aux_out': output_oglp['aux_out'][0*b:0*b+b]}
                    loss_dist_out = (self.calculate_loss_dist_out(output_oglp['main_out'][0*b:0*b+b], output_oglp['main_out'][1*b:1*b+b]) + \
                                     self.calculate_loss_dist_out(output_oglp['main_out'][0*b:0*b+b], output_oglp['main_out'][2*b:2*b+b]) + \
                                     self.calculate_loss_dist_out(output_oglp['main_out'][0*b:0*b+b], output_oglp['main_out'][3*b:3*b+b])) / 3
                    loss_hvt_stage3 = self.args.lambda_1_T * loss_disc + self.args.lambda_2_T * loss_CE + self.args.lambda_3_T * (loss_dist + loss_dist_out)
                    # loss_hvt_stage3 = self.args.lambda_1_T * loss_disc + self.args.lambda_2_T * loss_CE + self.args.lambda_3_T * loss_dist

                    return_loss = self.get_main_aux_wt_loss(main_out_oglp, aux_out_oglp_gm, w_arr, gts_oglp, aux_gts_oglp, apply_wtloss)

                    return return_loss, loss_hvt_stage3, output_original

        else:
            w_arr = []
            x_size = x.size()  # 800
            x, w_arr, x_tuple, low_level, aux_out = self.feature_extraction_x_war(x, w_arr)
            if cal_covstat:
                self.set_w_war(w_arr)
                return 0
            main_out, _ = self.get_main_out(x, low_level, x_size, aux_out)
            if visualize:
                f_cor_arr = []
                for f_map in w_arr:
                    f_cor, _ = get_covariance_matrix(f_map)
                    f_cor_arr.append(f_cor)
                return main_out, f_cor_arr
            else:
                return main_out

class ResNet_res(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_res, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x,latent_flag = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        latent_feature = x
        x = self.dropout(latent_feature)
        x = self.fc(x)
        
        return x, latent_feature

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock_res(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_res, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.IN = nn.InstanceNorm2d(planes, momentum=0.9, eps=1e-5)
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.IN(out)

        out = self.relu(out)

        return out



def get_final_layer(model):
    unfreeze_weights(model.final)
    return model.final


def DeepR18V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnet 18 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-18")
    return DeepV3Plus(num_classes, trunk='resnet-18', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D32', skip='m1', args=args)


def DeepR50V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-50")
    return DeepV3Plus(num_classes, trunk='resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepR50V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-50")
    return DeepV3Plus(num_classes, trunk='resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepR101V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-101")
    return DeepV3Plus(num_classes, trunk='resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepR101V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Resnet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-101")
    return DeepV3Plus(num_classes, trunk='resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)


def DeepR152V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Resnet 152 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-152")
    return DeepV3Plus(num_classes, trunk='resnet-152', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)



def DeepResNext50V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnext 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNext-50 32x4d")
    return DeepV3Plus(num_classes, trunk='resnext-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepResNext101V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnext 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNext-101 32x8d")
    return DeepV3Plus(num_classes, trunk='resnext-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepWideResNet50V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-50")
    return DeepV3Plus(num_classes, trunk='wide_resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepWideResNet50V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-50")
    return DeepV3Plus(num_classes, trunk='wide_resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepWideResNet101V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-101")
    return DeepV3Plus(num_classes, trunk='wide_resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepWideResNet101V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-101")
    return DeepV3Plus(num_classes, trunk='wide_resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)


def DeepResNext101V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    ResNext 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : resnext-101")
    return DeepV3Plus(num_classes, trunk='resnext-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepResNext101V3PlusD_OS4(args, num_classes, criterion, criterion_aux):
    """
    ResNext 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : resnext-101")
    return DeepV3Plus(num_classes, trunk='resnext-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D4', skip='m1', args=args)

def DeepShuffleNetV3PlusD_OS32(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : shufflenetv2")
    return DeepV3Plus(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D32', skip='m1', args=args)


def DeepMNASNet05V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    MNASNET Based Network
    """
    print("Model : DeepLabv3+, Backbone : mnas_0_5")
    return DeepV3Plus(num_classes, trunk='mnasnet_05', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepMNASNet10V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    MNASNET Based Network
    """
    print("Model : DeepLabv3+, Backbone : mnas_1_0")
    return DeepV3Plus(num_classes, trunk='mnasnet_10', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)


def DeepShuffleNetV3PlusD(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : shufflenetv2")
    return DeepV3Plus(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepMobileNetV3PlusD(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : mobilenetv2")
    return DeepV3Plus(num_classes, trunk='mobilenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepMobileNetV3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : mobilenetv2")
    return DeepV3Plus(num_classes, trunk='mobilenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepShuffleNetV3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : shufflenetv2")
    return DeepV3Plus(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)
