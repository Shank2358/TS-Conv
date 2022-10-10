#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

import _ext as _backend

use_amp = True


def set_amp(amp):
    global use_amp
    use_amp = amp


class _DCNv2(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        mask,
        weight,
        bias,
        stride,
        padding,
        dilation,
        deformable_groups,
    ):
        if use_amp:
            input = input.float()
            offset = offset.float()
            mask = mask.float()
            weight = weight.float()
            bias = bias.float()

        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.deformable_groups = deformable_groups
        output = _backend.dcn_v2_forward(
            input,
            weight,
            bias,
            offset,
            mask,
            ctx.kernel_size[0],
            ctx.kernel_size[1],
            ctx.stride[0],
            ctx.stride[1],
            ctx.padding[0],
            ctx.padding[1],
            ctx.dilation[0],
            ctx.dilation[1],
            ctx.deformable_groups,
        )
        ctx.save_for_backward(input, offset, mask, weight, bias)

        if use_amp:
            return output.half()
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if use_amp:
            grad_output = grad_output.float()

        input, offset, mask, weight, bias = ctx.saved_tensors
        (
            grad_input,
            grad_offset,
            grad_mask,
            grad_weight,
            grad_bias,
        ) = _backend.dcn_v2_backward(
            input,
            weight,
            bias,
            offset,
            mask,
            grad_output,
            ctx.kernel_size[0],
            ctx.kernel_size[1],
            ctx.stride[0],
            ctx.stride[1],
            ctx.padding[0],
            ctx.padding[1],
            ctx.dilation[0],
            ctx.dilation[1],
            ctx.deformable_groups,
        )

        if use_amp:
            grad_input = grad_input.half()
            grad_offset = grad_offset.half()
            grad_mask = grad_mask.half()
            grad_weight = grad_weight.half()
            grad_bias = grad_bias.half()

        return (
            grad_input,
            grad_offset,
            grad_mask,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
        )


dcn_v2_conv = _DCNv2.apply


class DCNv2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size)
        )

        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()


    def forward(self, input, offset, mask):
        assert (
            2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
            == offset.shape[1]
        )
        assert (
            self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
            == mask.shape[1]
        )

        return dcn_v2_conv(
            input,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.deformable_groups,
        )


class DCNv2_Circle(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super(DCNv2_Circle, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size)
        )

        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        assert (
            2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
            == offset.shape[1]
        )
        assert (
            self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
            == mask.shape[1]
        )

        shape = self.weight.shape
        weight_reshape = self.weight.view(shape[0], shape[1], -1)

        f0 = math.sqrt(2) / 2 * weight_reshape[:, :, 0:1] + (1 - math.sqrt(2) / 2) * weight_reshape[:, :, 1:2]
        f1 = math.sqrt(2) / 2 * weight_reshape[:, :, 3:4] + (1 - math.sqrt(2) / 2) * weight_reshape[:, :, 4:5]
        w0 = math.sqrt(2) / 2 * f0 + (1 - math.sqrt(2) / 2) * f1

        f2 = (1 - math.sqrt(2) / 2) * weight_reshape[:, :, 1:2] + math.sqrt(2) / 2 * weight_reshape[:, :, 2:3]
        f3 = (1 - math.sqrt(2) / 2) * weight_reshape[:, :, 4:5] + math.sqrt(2) / 2 * weight_reshape[:, :, 5:6]
        w2 = math.sqrt(2) / 2 * f2 + (1 - math.sqrt(2) / 2) * f3

        f4 = math.sqrt(2) / 2 * weight_reshape[:, :, 3:4] + (1 - math.sqrt(2) / 2) * weight_reshape[:, :, 4:5]
        f5 = math.sqrt(2) / 2 * weight_reshape[:, :, 6:7] + (1 - math.sqrt(2) / 2) * weight_reshape[:, :, 7:8]
        w6 = (1 - math.sqrt(2) / 2) * f4 + math.sqrt(2) / 2 * f5

        f6 = (1 - math.sqrt(2) / 2) * weight_reshape[:, :, 4:5] + math.sqrt(2) / 2 * weight_reshape[:, :, 5:6]
        f7 = (1 - math.sqrt(2) / 2) * weight_reshape[:, :, 7:8] + math.sqrt(2) / 2 * weight_reshape[:, :, 8:9]
        w8 = (1 - math.sqrt(2) / 2) * f6 + math.sqrt(2) / 2 * f7

        self.weight_circle = nn.Parameter(torch.cat([w0, weight_reshape[:, :, 1:2], w2,
                                        weight_reshape[:, :, 3:4], weight_reshape[:, :, 4:5], weight_reshape[:, :, 5:6],
                                        w6, weight_reshape[:, :, 7:8], w8], dim=-1).view(shape).contiguous())

        return dcn_v2_conv(
            input,
            offset,
            mask,
            self.weight_circle,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.deformable_groups,
        )
class route_func(nn.Module):
    def __init__(self, c_in, num_experts=1):
        super(route_func, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(c_in, num_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = x.sum(dim=0, keepdim=True)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
class DCNv2_Circle8(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super(DCNv2_Circle8, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_channels, 4)
        self.sigmoid = nn.Sigmoid()
        #self.__routef = route_func(in_channels, num_experts=4)
        
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size)
        )

        self.bias = nn.Parameter(torch.Tensor(out_channels*8))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask, w_c8):
        assert (
            2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
            == offset.shape[1]
        )
        assert (
            self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
            == mask.shape[1]
        )

        shape = self.weight.shape
        weight_reshape = self.weight.view(shape[0], shape[1], -1)

        f0 = math.sqrt(2) / 2 * weight_reshape[:, :, 0:1] + (1 - math.sqrt(2) / 2) * weight_reshape[:, :, 1:2]
        f1 = math.sqrt(2) / 2 * weight_reshape[:, :, 3:4] + (1 - math.sqrt(2) / 2) * weight_reshape[:, :, 4:5]
        w0 = math.sqrt(2) / 2 * f0 + (1 - math.sqrt(2) / 2) * f1

        f2 = (1 - math.sqrt(2) / 2) * weight_reshape[:, :, 1:2] + math.sqrt(2) / 2 * weight_reshape[:, :, 2:3]
        f3 = (1 - math.sqrt(2) / 2) * weight_reshape[:, :, 4:5] + math.sqrt(2) / 2 * weight_reshape[:, :, 5:6]
        w2 = math.sqrt(2) / 2 * f2 + (1 - math.sqrt(2) / 2) * f3

        f4 = math.sqrt(2) / 2 * weight_reshape[:, :, 3:4] + (1 - math.sqrt(2) / 2) * weight_reshape[:, :, 4:5]
        f5 = math.sqrt(2) / 2 * weight_reshape[:, :, 6:7] + (1 - math.sqrt(2) / 2) * weight_reshape[:, :, 7:8]
        w6 = (1 - math.sqrt(2) / 2) * f4 + math.sqrt(2) / 2 * f5

        f6 = (1 - math.sqrt(2) / 2) * weight_reshape[:, :, 4:5] + math.sqrt(2) / 2 * weight_reshape[:, :, 5:6]
        f7 = (1 - math.sqrt(2) / 2) * weight_reshape[:, :, 7:8] + math.sqrt(2) / 2 * weight_reshape[:, :, 8:9]
        w8 = (1 - math.sqrt(2) / 2) * f6 + math.sqrt(2) / 2 * f7

        weight_c_base = torch.cat([w0, weight_reshape[:, :, 1:2], w2,
                   weight_reshape[:, :, 3:4], weight_reshape[:, :, 4:5], weight_reshape[:, :, 5:6],
                   w6, weight_reshape[:, :, 7:8], w8], dim=-1)

        #r = self.__routef(input)
        r = self.avgpool(input)
        r = r.view(r.size(0), -1)
        r = r.sum(dim=0, keepdim=True)
        r = self.fc(r)
        r = self.sigmoid(r)
        #print(routef.shape)

        weight_c0 = weight_c_base.view(shape) * r[0,0] + self.weight * (1-r[0,0])
        weight_c1 = weight_c_base[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]].view(shape)
        weight_c2 = weight_c_base[:, :, [6, 3, 0, 7, 4, 1, 8, 5, 2]].view(shape) * r[0,1] + weight_reshape[:, :, [6, 3, 0, 7, 4, 1, 8, 5, 2]].view(shape) * (1-r[0,1])
        weight_c3 = weight_c_base[:, :, [7, 6, 3, 8, 4, 0, 5, 2, 1]].view(shape)
        weight_c4 = weight_c_base[:, :, [8, 7, 6, 5, 4, 3, 2, 1, 0]].view(shape) * r[0,2] + weight_reshape[:, :, [8, 7, 6, 5, 4, 3, 2, 1, 0]].view(shape) * (1-r[0,2])
        weight_c5 = weight_c_base[:, :, [5, 8, 7, 2, 4, 6, 1, 0, 3]].view(shape)
        weight_c6 = weight_c_base[:, :, [2, 5, 8, 1, 4, 7, 0, 3, 6]].view(shape) * r[0,3] + weight_reshape[:, :, [2, 5, 8, 1, 4, 7, 0, 3, 6]].view(shape) * (1-r[0,3])
        weight_c7 = weight_c_base[:, :, [1, 2, 5, 0, 4, 8, 3, 6, 7]].view(shape)
        weight_circle = torch.cat([weight_c0, weight_c1, weight_c2, weight_c3, weight_c4, weight_c5, weight_c6, weight_c7], dim=0).contiguous()#.detach()
        #print(self.weight_circle.shape, input.shape)
        '''
        self.combined_weight1 = self.combined_weight
        self.combined_bias1 = self.combined_bias
        offset1 = offset
        mask1 = mask
        for i in range(b-1):
            self.combined_weight1 = torch.cat([self.combined_weight1, self.combined_weight], dim=1)
            self.combined_bias1 = torch.cat([self.combined_bias1, self.combined_bias], dim=0)
            offset1 = torch.cat([offset1, offset], dim=1)
            mask1 = torch.cat([mask1, mask], dim=1)
        '''
        result8 = dcn_v2_conv(
            input,
            offset,
            mask,
            weight_circle,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.deformable_groups,
        )
        b, c_iout, h, w = result8.size()
        #print(w_c8.shape)
        result8 = result8.view(b, 8, self.out_channels, h, w) * w_c8.unsqueeze(2)# ##w_c8: [b, 8, H, W]
        #torch.mean(result8[:, 8:9, :, :, :], dim=1)
        return torch.sum(result8, dim=1)




class DCN(DCNv2):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super(DCN, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            deformable_groups,
        )

        channels_ = (
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        )
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return dcn_v2_conv(
            input,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.deformable_groups,
        )


class _DCNv2Pooling(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        rois,
        offset,
        spatial_scale,
        pooled_size,
        output_dim,
        no_trans,
        group_size=1,
        part_size=None,
        sample_per_part=4,
        trans_std=0.0,
    ):
        if use_amp:
            input = input.float()
            roi = roi.float()
            offset = offset.float()

        ctx.spatial_scale = spatial_scale
        ctx.no_trans = int(no_trans)
        ctx.output_dim = output_dim
        ctx.group_size = group_size
        ctx.pooled_size = pooled_size
        ctx.part_size = pooled_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std

        output, output_count = _backend.dcn_v2_psroi_pooling_forward(
            input,
            rois,
            offset,
            ctx.no_trans,
            ctx.spatial_scale,
            ctx.output_dim,
            ctx.group_size,
            ctx.pooled_size,
            ctx.part_size,
            ctx.sample_per_part,
            ctx.trans_std,
        )
        ctx.save_for_backward(input, rois, offset, output_count)

        if use_amp:
            return output.half()
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if use_amp:
            grad_output = grad_output.float()

        input, rois, offset, output_count = ctx.saved_tensors
        grad_input, grad_offset = _backend.dcn_v2_psroi_pooling_backward(
            grad_output,
            input,
            rois,
            offset,
            output_count,
            ctx.no_trans,
            ctx.spatial_scale,
            ctx.output_dim,
            ctx.group_size,
            ctx.pooled_size,
            ctx.part_size,
            ctx.sample_per_part,
            ctx.trans_std,
        )

        if use_amp:
            grad_input = grad_input.half()
            grad_offset = grad_offset.half()
        return (
            grad_input,
            None,
            grad_offset,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


dcn_v2_pooling = _DCNv2Pooling.apply


class DCNv2Pooling(nn.Module):
    def __init__(
        self,
        spatial_scale,
        pooled_size,
        output_dim,
        no_trans,
        group_size=1,
        part_size=None,
        sample_per_part=4,
        trans_std=0.0,
    ):
        super(DCNv2Pooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, input, rois, offset):
        assert input.shape[1] == self.output_dim
        if self.no_trans:
            offset = input.new()
        return dcn_v2_pooling(
            input,
            rois,
            offset,
            self.spatial_scale,
            self.pooled_size,
            self.output_dim,
            self.no_trans,
            self.group_size,
            self.part_size,
            self.sample_per_part,
            self.trans_std,
        )


class DCNPooling(DCNv2Pooling):
    def __init__(
        self,
        spatial_scale,
        pooled_size,
        output_dim,
        no_trans,
        group_size=1,
        part_size=None,
        sample_per_part=4,
        trans_std=0.0,
        deform_fc_dim=1024,
    ):
        super(DCNPooling, self).__init__(
            spatial_scale,
            pooled_size,
            output_dim,
            no_trans,
            group_size,
            part_size,
            sample_per_part,
            trans_std,
        )

        self.deform_fc_dim = deform_fc_dim

        if not no_trans:
            self.offset_mask_fc = nn.Sequential(
                nn.Linear(
                    self.pooled_size * self.pooled_size * self.output_dim,
                    self.deform_fc_dim,
                ),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim, self.deform_fc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim, self.pooled_size * self.pooled_size * 3),
            )
            self.offset_mask_fc[4].weight.data.zero_()
            self.offset_mask_fc[4].bias.data.zero_()

    def forward(self, input, rois):
        offset = input.new()

        if not self.no_trans:

            # do roi_align first
            n = rois.shape[0]
            roi = dcn_v2_pooling(
                input,
                rois,
                offset,
                self.spatial_scale,
                self.pooled_size,
                self.output_dim,
                True,  # no trans
                self.group_size,
                self.part_size,
                self.sample_per_part,
                self.trans_std,
            )

            # build mask and offset
            offset_mask = self.offset_mask_fc(roi.view(n, -1))
            offset_mask = offset_mask.view(n, 3, self.pooled_size, self.pooled_size)
            o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask = torch.sigmoid(mask)

            # do pooling with offset and mask
            return (
                dcn_v2_pooling(
                    input,
                    rois,
                    offset,
                    self.spatial_scale,
                    self.pooled_size,
                    self.output_dim,
                    self.no_trans,
                    self.group_size,
                    self.part_size,
                    self.sample_per_part,
                    self.trans_std,
                )
                * mask
            )
        # only roi_align
        return dcn_v2_pooling(
            input,
            rois,
            offset,
            self.spatial_scale,
            self.pooled_size,
            self.output_dim,
            self.no_trans,
            self.group_size,
            self.part_size,
            self.sample_per_part,
            self.trans_std,
        )
