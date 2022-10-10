import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from dcn_v2_amp import DCNv2, DCNv2_Circle8
# from dcn_v2 import DCNv2, DCNv2_Circle8
from ..layers.convolutions import Convolutional


class Head1(nn.Module):
    def __init__(self, filters_in, stride):
        super(Head1, self).__init__()
        self.__stride = stride
        self.filters_in = filters_in
        self.__conv = Convolutional(filters_in=filters_in + 2, filters_out=filters_in * 2, kernel_size=3, stride=1,
                                    pad=1, norm="bn", activate="leaky")
        self.__loc1 = nn.Conv2d(in_channels=filters_in * 2, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.__conv_mask = nn.Conv2d(in_channels=filters_in * 2, out_channels = 9 + 9 + 22 + 8, kernel_size=1, stride=1,padding=0, bias=True)

    def forward(self, input1):
        batch_size, _, input_size = input1.shape[:3]
        y = torch.arange(0, input_size).unsqueeze(1).repeat(1, input_size).unsqueeze(0)
        x = torch.arange(0, input_size).unsqueeze(0).repeat(input_size, 1).unsqueeze(0)
        grid_xy = torch.stack([x, y], dim=0).permute(1, 0, 2, 3).contiguous()
        grid_xy = grid_xy.repeat(batch_size, 1, 1, 1).float().to(input1.device)
        input11 = torch.cat([input1, grid_xy], dim=1)
        conv = self.__conv(input11)
        out1 = self.__loc1(conv)
        conv_mask = self.__conv_mask(conv)  # * (torch.sigmoid(out1[:, 9:10, :, :].detach())>0.3).float()
        mask_loc = torch.sigmoid(conv_mask[:, 0:9, :, :])  ##################
        mask_cls = torch.sigmoid(conv_mask[:, 9:18, :, :])  ##################
        offset_vertex = conv_mask[:, 18:18 + 22, :, :].permute(0, 2, 3, 1).contiguous()
        weight_c8 = torch.softmax(conv_mask[:, 18 + 22:, :, :], dim=1)
        out1_de, offsets_loc, offsets_cls, offsets_d = self.__decode(out1.permute(0, 2, 3, 1).contiguous(), offset_vertex)
        return out1.permute(0, 2, 3, 1).contiguous(), out1_de, offsets_loc, offsets_cls, mask_loc, mask_cls, weight_c8, offsets_d

    def __decode(self, p, offset_vertex):
        batch_size, output_size = p.shape[:2]
        device = p.device
        conv_raw_l = p[:, :, :, 0:4]
        conv_raw_s = p[:, :, :, 4:8]
        conv_raw_r = p[:, :, :, 8:9]
        conv_raw_conf = p[:, :, :, 9:10]
        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).repeat(batch_size, 1, 1, 1).float().to(device)
        l = conv_raw_l ** 2 * self.__stride
        xmin = grid_xy[:, :, :, 0:1] * self.__stride + self.__stride / 2 - l[:, :, :, 3:4]
        ymin = grid_xy[:, :, :, 1:2] * self.__stride + self.__stride / 2 - l[:, :, :, 0:1]
        xmax = grid_xy[:, :, :, 0:1] * self.__stride + self.__stride / 2 + l[:, :, :, 1:2]
        ymax = grid_xy[:, :, :, 1:2] * self.__stride + self.__stride / 2 + l[:, :, :, 2:3]
        w = l[:, :, :, 1:2] + l[:, :, :, 3:4]
        h = l[:, :, :, 0:1] + l[:, :, :, 2:3]
        x = (xmax + xmin) / 2
        y = (ymax + ymin) / 2
        xywh = torch.cat([x, y, w, h], dim=-1)
        r = torch.sigmoid(conv_raw_r)
        zero = torch.zeros_like(r)
        one = torch.ones_like(r)
        maskr = torch.where(r > 0.9, zero, one)
        s = torch.sigmoid(conv_raw_s) * maskr
        conf = torch.sigmoid(conv_raw_conf)
        bbox = torch.cat([xywh, s, r, l, conf], dim=-1)

        x1 = xmin + s[:, :, :, 0:1] * w
        x7 = xmax - s[:, :, :, 2:3] * w
        y5 = ymin + s[:, :, :, 1:2] * h
        y3 = ymax - s[:, :, :, 3:4] * h
        x_obb_avg = (x1 + x7) / 2
        y_obb_avg = (y5 + y3) / 2
        off_y_t = ymin / self.__stride + 1 - grid_xy[:, :, :, 1:2]
        off_y_b = ymax / self.__stride - 1 - grid_xy[:, :, :, 1:2]
        off_x_l = xmin / self.__stride + 1 - grid_xy[:, :, :, 0:1]
        off_x_r = xmax / self.__stride - 1 - grid_xy[:, :, :, 0:1]

        off_y3 = y3 / self.__stride - grid_xy[:, :, :, 1:2]
        off_y4 = torch.zeros_like(y_obb_avg)
        off_y5 = y5 / self.__stride - grid_xy[:, :, :, 1:2]
        off_x1 = x1 / self.__stride - grid_xy[:, :, :, 0:1]
        off_x4 = torch.zeros_like(x_obb_avg)
        off_x7 = x7 / self.__stride - grid_xy[:, :, :, 0:1]

        off = torch.sigmoid(offset_vertex)
        dc_off = off[:, :, :, 4:]

        eps = torch.ceil(0.01 * w)
        xp0 = torch.where(x1 < xmin + eps, x1, xmin + off[:, :, :, 0:1] * (x1 - xmin))
        yp0 = torch.where(x1 < xmin + eps, ymin + off[:, :, :, 0:1] * (y3 - ymin),
                          (y3 - ymin) / (xmin - x1 + 1e-8) * (xp0 - x1) + ymin)
        xp2 = torch.where(x1 > xmax - eps, x1, xmax - off[:, :, :, 1:2] * (xmax - x1))
        yp2 = torch.where(x1 > xmax - eps, ymin + off[:, :, :, 1:2] * (y5 - ymin),
                          (y5 - ymin) / (xmax - x1 + 1e-8) * (xp2 - x1) + ymin)
        xp6 = torch.where(x7 < xmin + eps, x7, xmin + off[:, :, :, 2:3] * (x7 - xmin))
        yp6 = torch.where(x7 < xmin + eps, ymax - off[:, :, :, 2:3] * (ymax - y3),
                          (y3 - ymax) / (xmin - x7 + 1e-8) * (xp6 - x7) + ymax)
        xp8 = torch.where(x7 > xmax - eps, x7, xmax - off[:, :, :, 3:4] * (xmax - x7))
        yp8 = torch.where(x7 > xmax - eps, ymax - off[:, :, :, 3:4] * (ymax - y5),
                          (y5 - ymax) / (xmax - x7 + 1e-8) * (xp8 - x7) + ymax)

        off_y_0 = yp0 / self.__stride + 1 - grid_xy[:, :, :, 1:2]
        off_x_0 = xp0 / self.__stride + 1 - grid_xy[:, :, :, 0:1]

        off_y_2 = yp2 / self.__stride + 1 - grid_xy[:, :, :, 1:2]
        off_x_2 = xp2 / self.__stride - 1 - grid_xy[:, :, :, 0:1]

        off_y_6 = yp6 / self.__stride - 1 - grid_xy[:, :, :, 1:2]
        off_x_6 = xp6 / self.__stride + 1 - grid_xy[:, :, :, 0:1]

        off_y_8 = yp8 / self.__stride - 1 - grid_xy[:, :, :, 1:2]
        off_x_8 = xp8 / self.__stride - 1 - grid_xy[:, :, :, 0:1]

        offsets_loc = torch.cat([off_y_0, off_x_0, off_y_t, off_x1, off_y_2, off_x_2,
                                 off_y3, off_x_l, off_y4, off_x4, off_y5, off_x_r,
                                 off_y_6, off_x_6, off_y_b, off_x7, off_y_8, off_x_8]
                                , dim=-1).permute(0, 3, 1, 2).contiguous()


        width = (torch.sqrt(torch.pow(x1 - xmax, 2) + torch.pow(ymin - y5, 2)) + torch.sqrt(
            torch.pow(xmin - x7, 2) + torch.pow(y3 - ymax, 2))).detach() / 2
        height = (torch.sqrt(torch.pow(xmin - x1, 2) + torch.pow(y3 - ymin, 2)) + torch.sqrt(
            torch.pow(x7 - xmax, 2) + torch.pow(ymax - y5, 2))).detach() / 2
        angle = (torch.arctan((y5 - ymin) / (xmax - x1 + 1e-4)) + torch.arctan(
            (ymax - y3) / (x7 - xmin + 1e-4))) / 2
        xd = x_obb_avg.detach() - width / 2 + dc_off[:, :, :, 9:] * width
        yd = y_obb_avg.detach() - height / 2 + dc_off[:, :, :, 0:9] * height
        xd0_ = torch.cos(angle) * (xd - x_obb_avg) - torch.sin(angle) * (yd - y_obb_avg) + x_obb_avg
        xd_ = torch.where(torch.where(xd0_ >= xmin, xd0_, xmin) <= xmax, xd0_, xmax)
        yd0_ = torch.sin(angle) * (xd - x_obb_avg) + torch.cos(angle) * (yd - y_obb_avg) + y_obb_avg
        yd_ = torch.where(torch.where(yd0_ >= ymin, yd0_, ymin) <= ymax, yd0_, ymax)


        dist = math.sqrt(2) / 2
        dc_off_y0 = yd_[:, :, :, 0:1] / self.__stride + dist - grid_xy[:, :, :, 1:2]
        dc_off_x0 = xd_[:, :, :, 0:1] / self.__stride + dist - grid_xy[:, :, :, 0:1]

        dc_off_y1 = yd_[:, :, :, 1:2] / self.__stride + 1 - grid_xy[:, :, :, 1:2]
        dc_off_x1 = xd_[:, :, :, 1:2] / self.__stride - grid_xy[:, :, :, 0:1]

        dc_off_y2 = yd_[:, :, :, 2:3] / self.__stride + dist - grid_xy[:, :, :, 1:2]
        dc_off_x2 = xd_[:, :, :, 2:3] / self.__stride - dist - grid_xy[:, :, :, 0:1]

        dc_off_y3 = yd_[:, :, :, 3:4] / self.__stride - grid_xy[:, :, :, 1:2]
        dc_off_x3 = xd_[:, :, :, 3:4] / self.__stride + 1 - grid_xy[:, :, :, 0:1]

        dc_off_y4 = yd_[:, :, :, 4:5] / self.__stride - grid_xy[:, :, :, 1:2]
        dc_off_x4 = xd_[:, :, :, 4:5] / self.__stride - grid_xy[:, :, :, 0:1]

        dc_off_y5 = yd_[:, :, :, 5:6] / self.__stride - grid_xy[:, :, :, 1:2]
        dc_off_x5 = xd_[:, :, :, 5:6] / self.__stride - 1 - grid_xy[:, :, :, 0:1]

        dc_off_y6 = yd_[:, :, :, 6:7] / self.__stride - dist - grid_xy[:, :, :, 1:2]
        dc_off_x6 = xd_[:, :, :, 6:7] / self.__stride + dist - grid_xy[:, :, :, 0:1]

        dc_off_y7 = yd_[:, :, :, 7:8] / self.__stride - 1 - grid_xy[:, :, :, 1:2]
        dc_off_x7 = xd_[:, :, :, 7:8] / self.__stride - grid_xy[:, :, :, 0:1]

        dc_off_y8 = yd_[:, :, :, 8:9] / self.__stride - dist - grid_xy[:, :, :, 1:2]
        dc_off_x8 = xd_[:, :, :, 8:9] / self.__stride - dist - grid_xy[:, :, :, 0:1]

        coor_dc = torch.cat([xd_, yd_], dim=-1)

        offsets_cls = torch.cat([dc_off_y0, dc_off_x0, dc_off_y1, dc_off_x1, dc_off_y2, dc_off_x2,
                                 dc_off_y3, dc_off_x3, dc_off_y4, dc_off_x4, dc_off_y5, dc_off_x5,
                                 dc_off_y6, dc_off_x6, dc_off_y7, dc_off_x7, dc_off_y8, dc_off_x8]
                                , dim=-1).permute(0, 3, 1, 2).contiguous()

        return bbox, offsets_loc, offsets_cls, coor_dc.view(-1, 18),


class Head2(nn.Module):
    def __init__(self, filters_in, nC, stride):
        super(Head2, self).__init__()
        self.beta = 0.5
        self.__nC = nC
        self.__stride = stride

        self.dcn_loc = DCNv2(in_channels=filters_in + 2, out_channels=filters_in * 2, kernel_size=3, padding=1,
                             stride=1)
        self.bn_loc = nn.BatchNorm2d(filters_in * 2)  # nn.GroupNorm(16, filters_in * 2)
        self.relu_loc = nn.SiLU(inplace=True)  # nn.LeakyReLU(negative_slope=0.1, inplace=True)#
        self.conv_loc = nn.Sequential(
            Convolutional(filters_in=filters_in * 2, filters_out=filters_in, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=filters_in, filters_out=filters_in * 2, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            nn.Conv2d(in_channels=filters_in * 2, out_channels=9, kernel_size=1, stride=1, padding=0)
        )

        self.dcn_cls = DCNv2_Circle8(in_channels=filters_in, out_channels=filters_in * 2, kernel_size=3, padding=1,stride=1)
        self.bn_cls = nn.BatchNorm2d(filters_in * 2)  # nn.GroupNorm(16, filters_in * 2)#
        self.relu_cls = nn.SiLU(inplace=True)  # nn.LeakyReLU(negative_slope=0.1, inplace=True) #
        self.conv_cls = nn.Sequential(
            Convolutional(filters_in=filters_in * 2, filters_out=filters_in, kernel_size=1, stride=1, pad=0, norm="bn",activate="leaky"),
            Convolutional(filters_in=filters_in, filters_out=filters_in * 2, kernel_size=3, stride=1, pad=1, norm="bn",activate="leaky"),
            nn.Conv2d(in_channels=filters_in * 2, out_channels=self.__nC + 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, out1, loc, cls, offsets_loc, offsets_cls, mask_loc, mask_cls, w_c8):
        att = torch.sigmoid(out1[:, :, :, 9:10].detach().permute(0, 3, 1, 2).contiguous())#
        loc = loc * att + loc
        cls = cls * att + cls
        batch_size, output_size = out1.shape[:2]
        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size).unsqueeze(0)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1).unsqueeze(0)
        grid_xy = torch.stack([x, y], dim=0).permute(1, 0, 2, 3).contiguous()
        grid_xy = grid_xy.repeat(batch_size, 1, 1, 1).float().to(out1.device)
        loc_dcn = self.dcn_loc(torch.cat([loc, grid_xy], dim=1), offsets_loc, mask_loc)
        # loc_dcn = self.__dcn_loc(loc, offsets_loc, mask_loc)
        loc_dcn = self.relu_loc(self.bn_loc(loc_dcn)) if self.training else self.relu_loc(self.bn_loc(loc_dcn)).float()
        conv_loc = self.conv_loc(loc_dcn)

        cls_dcn = self.dcn_cls(cls, offsets_cls, mask_cls, w_c8)
        cls_dcn = self.relu_cls(self.bn_cls(cls_dcn)) if self.training else self.relu_cls(self.bn_cls(cls_dcn)).float()
        conv_cls = self.conv_cls(cls_dcn)

        out2 = torch.cat((conv_loc, conv_cls), dim=1).permute(0, 2, 3, 1)
        out2_de = self.__decode(out1.detach(), out2.clone())
        return out2, out2_de

    def __decode(self, out1, out2):
        batch_size, output_size = out2.shape[:2]
        device = out2.device
        conv_raw_l = out2[:, :, :, 0:4]
        conv_raw_s = out2[:, :, :, 4:8]
        conv_raw_r = out2[:, :, :, 8:9]
        conv_raw_conf = out2[:, :, :, 9:10]
        conv_raw_prob = out2[:, :, :, 10:]
        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy0 = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy0.unsqueeze(0).repeat(batch_size, 1, 1, 1).float().to(device)

        pred_l = torch.exp(conv_raw_l) * out1[:, :, :, 9:13].detach()  # ** 2 * self.__stride

        pred_xmin = grid_xy[:, :, :, 0:1] * self.__stride + self.__stride / 2 - pred_l[:, :, :, 3:4]
        pred_ymin = grid_xy[:, :, :, 1:2] * self.__stride + self.__stride / 2 - pred_l[:, :, :, 0:1]
        pred_xmax = grid_xy[:, :, :, 0:1] * self.__stride + self.__stride / 2 + pred_l[:, :, :, 1:2]
        pred_ymax = grid_xy[:, :, :, 1:2] * self.__stride + self.__stride / 2 + pred_l[:, :, :, 2:3]
        pred_w = (pred_l[:, :, :, 1:2] + pred_l[:, :, :, 3:4])
        pred_h = (pred_l[:, :, :, 0:1] + pred_l[:, :, :, 2:3])
        pred_x = (pred_xmax + pred_xmin) / 2
        pred_y = (pred_ymax + pred_ymin) / 2
        pred_xywh = torch.cat([pred_x, pred_y, pred_w, pred_h], dim=-1)
        pred_r = torch.sigmoid(conv_raw_r)
        zero = torch.zeros_like(pred_r)
        one = torch.ones_like(pred_r)
        maskr = torch.where(pred_r > 0.9, zero, one)
        pred_s = torch.sigmoid(conv_raw_s) * maskr
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_s, pred_r, pred_l, pred_conf, pred_prob], dim=-1)
        out_de = pred_bbox.view(-1, 4 + 5 + 4 + self.__nC + 1) if not self.training else pred_bbox
        return out_de
