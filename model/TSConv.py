import sys

sys.path.append("..")
import torch.nn as nn
from collections import OrderedDict
from model.backbones.resnet import Resnet101
from model.backbones.darknet53 import Darknet53
from model.neck.neck import Neck
from model.head.head import Head1, Head2
from model.layers.convolutions import Convolutional
from utils.utils_basic import *

class GGHL(nn.Module):
    def __init__(self, init_weights=True, inputsize=int(cfg.TRAIN["TRAIN_IMG_SIZE"]), weight_path=None):
        super(GGHL, self).__init__()
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__out_channel = self.__nC + 4 + 5 + 1
        self.__backnone = Darknet53()#Resnet101()#
        # self.__backnone = PVT2(weight_path=weight_path)
        # self.__fpn = Neck(fileters_in=[512, 320, 128, 64], fileters_out=self.__out_channel)
        self.__fpn = Neck(fileters_in=[1024, 512, 256, 128], fileters_in_ratio=1)
        self.__head1_s = Head1(filters_in=128, stride=self.__strides[0])
        self.__head1_m = Head1(filters_in=256, stride=self.__strides[1])
        self.__head1_l = Head1(filters_in=512, stride=self.__strides[2])

        self.__head2_s = Head2(filters_in=128, nC=self.__nC, stride=self.__strides[0])
        self.__head2_m = Head2(filters_in=256, nC=self.__nC, stride=self.__strides[1])
        self.__head2_l = Head2(filters_in=512, nC=self.__nC, stride=self.__strides[2])

        if init_weights:
            self.__init_weights()

    def forward(self, x):
        out = []
        x_8, x_16, x_32 = self.__backnone(x)
        loc2, cls2, loc1, cls1, loc0, cls0 = self.__fpn(x_32, x_16, x_8)
        x_s, x_s_de, offsets_loc_s, offsets_cls_s, mask_loc_s, mask_cls_s, w_c8_s, coor_dc_s = self.__head1_s(loc2)
        x_m, x_m_de, offsets_loc_m, offsets_cls_m, mask_loc_m, mask_cls_m, w_c8_m, coor_dc_m = self.__head1_m(loc1)
        x_l, x_l_de, offsets_loc_l, offsets_cls_l, mask_loc_l, mask_cls_l, w_c8_l, coor_dc_l = self.__head1_l(loc0)

        out_s, out_s_de = self.__head2_s(x_s_de, loc2, cls2, offsets_loc_s, offsets_cls_s, mask_loc_s, mask_cls_s, w_c8_s)
        out_m, out_m_de = self.__head2_m(x_m_de, loc1, cls1, offsets_loc_m, offsets_cls_m, mask_loc_m, mask_cls_m, w_c8_m)
        out_l, out_l_de = self.__head2_l(x_l_de, loc0, cls0, offsets_loc_l, offsets_cls_l, mask_loc_l, mask_cls_l, w_c8_l)

        out.append((x_s, x_s_de, out_s, out_s_de, coor_dc_s))
        out.append((x_m, x_m_de, out_m, out_m_de, coor_dc_m))
        out.append((x_l, x_l_de, out_l, out_l_de, coor_dc_l))

        if self.training:
            p1, p1_d, p2, p2_d, _ = list(zip(*out))
            return p1, p1_d, p2, p2_d
        else:
            p1, p1_d, p2, p2_d, offsets_d = list(zip(*out))
            return p1, p1_d, p2, torch.cat(p2_d, 0), torch.cat(offsets_d, 0)

    def __init_weights(self):
        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                # print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)
                # print("initing {}".format(m))
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                # print("initing {}".format(m))
                
    def load_resnet101_weights(self, weight_file='/home/hzc/v2/weight/resnet101-cd907fc2.pth'):
        model_list = self.__backnone.state_dict().keys()
        # print(model_list)
        weight = torch.load(weight_file)
        # print(weight.keys())
        new_weight = OrderedDict()
        # # zip 默认遍历最少的list
        for model_key, weight_key, weight_value in zip(model_list, weight.keys(), weight.values()):
            if model_key[9:] == weight_key:
                new_weight[model_key] = weight_value
        self.__backnone.load_state_dict(new_weight)
        
    def load_darknet_weights(self, weight_file, cutoff=52):
        "https://github.com/ultralytics/yolov3/blob/master/models.py"
        print("load darknet weights : ", weight_file)
        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            if isinstance(m, Convolutional):
                # only initing backbone conv's weights
                if count == cutoff:
                    break
                count += 1
                conv_layer = m._Convolutional__conv
                if m.norm == "bn":
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                    # print("loading weight {}".format(bn_layer))
                elif m.norm == "gn":
                    # Load GN bias, weights
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
                # print("loading weight {}".format(conv_layer))
        print("loading weight number: {}".format(count))


