from numpy import ones_like
import torch
import torch.nn as nn
from utils import utils_basic
import config.config as cfg
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)
    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma)
        return loss

class SoftmaxCELoss(nn.Module):
    def __init__(self):
        super(SoftmaxCELoss, self).__init__()
    def forward(self, input, target):
        log_probability = F.log_softmax(input, dim=-1)
        return -1.0 * log_probability * target        

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.__strides = cfg.MODEL["STRIDES"]
        self.__scale_factor = cfg.SCALE_FACTOR
        self.__scale_factor_a = cfg.SCALE_FACTOR_A
        self.delta = 0.1
        self.delta_conf = 0.05
        self.num_class = cfg.DATA["NUM"]
        self.warmup = cfg.TRAIN["WARMUP_EPOCHS"]
        self.epoch = cfg.TRAIN["EPOCHS"]
        self.vartheta = 0.5
        self.beta1 = 1
        self.beta2 = 1
        self.zeta = 0.7

    def forward(self, p1, p1_d, p2, p2_d, label_sbbox, label_mbbox, label_lbbox, epoch, i):
        loss_s, loss_fg_s, loss_bg_s, loss_pos_s, loss_neg_s, loss_iou_s, loss_cls_s, loss_s_s, loss_r_s, loss_l_s = \
            self.__cal_loss(p1[0], p1_d[0], p2[0], p2_d[0], label_sbbox, int(self.__strides[0]), epoch, i)

        loss_m, loss_fg_m, loss_bg_m, loss_pos_m, loss_neg_m, loss_iou_m, loss_cls_m, loss_s_m, loss_r_m, loss_l_m = \
            self.__cal_loss(p1[1], p1_d[1], p2[1], p2_d[1], label_mbbox, int(self.__strides[1]), epoch, i)

        loss_l, loss_fg_l, loss_bg_l, loss_pos_l, loss_neg_l, loss_iou_l, loss_cls_l, loss_s_l, loss_r_l, loss_l_l = \
            self.__cal_loss(p1[2], p1_d[2], p2[2], p2_d[2], label_lbbox, int(self.__strides[2]), epoch, i)

        loss = loss_l + loss_m + loss_s
        loss_fg = loss_fg_s + loss_fg_m + loss_fg_l
        loss_bg = loss_bg_s + loss_bg_m + loss_bg_l
        loss_pos = loss_pos_s + loss_pos_m + loss_pos_l
        loss_neg = loss_neg_s + loss_neg_m + loss_neg_l

        loss_iou = loss_iou_s + loss_iou_m + loss_iou_l
        loss_cls = loss_cls_s + loss_cls_m + loss_cls_l
        loss_s = loss_s_s + loss_s_m + loss_s_l
        loss_r = loss_r_s + loss_r_m + loss_r_l
        loss_l = loss_l_s + loss_l_m + loss_l_l

        return loss, loss_fg, loss_bg, loss_pos, loss_neg, loss_iou, loss_cls, loss_s, loss_r, loss_l

    def smooth_l1_loss(self, input, target, beta=1. / 9, size_average=True):
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        return loss
    def unfold(self, x, kernel_size=7, dilation=1):
        assert x.dim() == 4  #B, C, H, W
        assert kernel_size % 2 == 1
        padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2  # using SAME padding
        unfolded_x = F.unfold(
            x, kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        unfolded_x = unfolded_x.permute(0, 2, 1)
        unfolded = unfolded_x.view(x.size(0), x.size(2), x.size(3), -1, kernel_size**2)  #B, H, W, C, k*k
        return unfolded.permute(0, 1, 2, 4, 3).contiguous()
    def __cal_loss(self, p1, p1_d, p2, p2_d, label, stride, epoch, iter):
        batch_size, grid = p1.shape[:2]
        img_size = stride * grid
        label_xywh = label[...,:4]
        label_l = label[..., 4:8]
        label_a = label[..., 8:12]
        label_r = label[..., 12:13]
        label_mix = label[..., 14:15]
        label_cls = label[..., 16:]
        
        SCE = SoftmaxCELoss()
        Focal = FocalLoss(gamma=2, alpha=1.0, reduction="none")
        BCE = nn.BCEWithLogitsLoss(reduction="none")
        SmoothL1 = nn.SmoothL1Loss(reduction='none')

        obj_mask = (label[..., 13:14] == 1).float()  ###二值 positive_mask
        noobj_mask = 1 - (label[..., 13:14] != 0).float()  ###二值  negative_mask
        fuzzy_mask = 1 - obj_mask - noobj_mask
        gh = torch.max(label_cls, dim=-1, keepdim=True)[0]  # positive_gaussian
        gh_obj = obj_mask * gh
        gh_fuzzy = fuzzy_mask * gh

        bbox_loss_scale = self.__scale_factor - (self.__scale_factor - 1.0) * (
                    label_l[..., 1:2] + label_l[..., 3:4]) * (label_l[..., 0:1] + label_l[..., 2:3]) / (
                                      img_size * img_size)
        area_weight = label[..., 15:16] + (label[..., 15:16] == 0).float()

        label_conf_smooth = obj_mask * (1 - self.delta_conf) + self.delta_conf * 1.0 / 2
        label_cls_smooth = (label_cls != 0).float() * (1 - self.delta) + self.delta * 1.0 / self.num_class

        p1_d_s = p1_d[..., 4:8]
        p1_d_r = p1_d[..., 8:9]
        p1_d_l = p1_d[..., 9:13]
        p1_conf = p1[..., 9:10]
        p2_d_s = p2_d[..., 4:8]
        p2_d_r = p2_d[..., 8:9]
        p2_d_l = p2_d[..., 9:13]
        p2_conf = p2[..., 9:10]
        p2_cls = p2[..., 10:]
        
        weight_r = (self.__scale_factor - (self.__scale_factor - 1.0) * label_r.detach())
        
        # iou
        giou1 = utils_basic.GIOU_l_torch(p1_d_l, label_l)
        scores_iou1 = bbox_loss_scale * (1.0 - giou1)
        scores_obb1 = torch.sum(SmoothL1(p1_d_s, label_a), dim=-1, keepdim=True)
        scores_area1 = SmoothL1(p1_d_r, label_r)
        scores_loc1 = torch.exp(-1 * (scores_iou1 + scores_obb1 + scores_area1))
        offset01 = scores_loc1.detach()
        offset01 = torch.max(offset01, dim=-1, keepdim=True)[0]

        bg_mask1 = noobj_mask + fuzzy_mask * ((self.vartheta * gh_fuzzy + (1 - self.vartheta) * offset01) < 0.3).float() \
                   * (1 - fuzzy_mask * (self.vartheta * gh_fuzzy + (1 - self.vartheta) * offset01))
        fg_mask1 = obj_mask * (self.vartheta * gh_obj + (1 - self.vartheta) * offset01)

        loss_fg1 = fg_mask1 * Focal(input=p1_conf, target=label_conf_smooth) * label_mix
        loss_bg1 = bg_mask1 * Focal(input=p1_conf, target=label_conf_smooth) * label_mix

        loss_iou1 = obj_mask * scores_iou1 * area_weight * label_mix * weight_r
        loss_s1 = obj_mask * scores_obb1 * area_weight * label_mix * weight_r
        loss_r1 = obj_mask * scores_area1 * area_weight * label_mix * weight_r
        loss_l1 = obj_mask * bbox_loss_scale * SmoothL1(p1_d_l / stride, label_l / stride) * area_weight * label_mix * weight_r

        # giou2 = []
        giou2 = utils_basic.GIOU_l_torch(p2_d_l, label_l)
        scores_iou2 = bbox_loss_scale * (1.0 - giou2)
        scores_obb2 = torch.sum(SmoothL1(p2_d_s, label_a), dim=-1, keepdim=True)
        scores_area2 = SmoothL1(p2_d_r, label_r)
        scores_loc2 = torch.exp(-1 * (scores_iou2 + scores_obb2 + scores_area2))
        scores_cls_loc2 = torch.sigmoid(p2_cls) * scores_loc2
        scores_cls_loc2 = -torch.log((1 - scores_cls_loc2) / (scores_cls_loc2 + 1e-16) + 1e-16)

        offset02 = scores_loc2.detach()
        offset02 = torch.max(offset02, dim=-1, keepdim=True)[0]
       
        cls_score = torch.max((label_cls != 0).float() * torch.sigmoid(p2_cls.detach()), dim=-1, keepdim=True)[0]
        # current_iter = cfg.TRAIN["TRAIN_IMG_NUM"] / cfg.TRAIN["BATCH_SIZE"] * epoch + iter
        # max_iter = cfg.TRAIN["TRAIN_IMG_NUM"] / cfg.TRAIN["BATCH_SIZE"] * cfg.TRAIN["EPOCHS"]
        # self.vartheta = self.vartheta * (max_iter - current_iter) / max_iter
        with torch.no_grad():
            region_size = 7
            label_xywh_region = self.unfold(label_xywh.permute(0, 3, 1, 2).contiguous(), kernel_size=region_size)  # B, H, W, k*k, C， ##高斯中心有值，其他0， k*k
            other_obj_mask = torch.sum(label_xywh_region - label_xywh.unsqueeze(-2), dim=-1, keepdim=True) ####B, H, W, kk, C -B. H, W, kk, C
            other_obj_mask = (other_obj_mask == 0).float() ######去除邻域内不是分配给该物体的区域
            topk_mask = other_obj_mask * (gh_obj == 1).float().unsqueeze(-2)  # 去除WH非中心位置
            y = torch.arange(0, grid).unsqueeze(1).repeat(1, grid)
            x = torch.arange(0, grid).unsqueeze(0).repeat(grid, 1)
            grid_xy = torch.stack([x, y], dim=-1)
            grid_xy = grid_xy.unsqueeze(0).repeat(batch_size, 1, 1, 1).float().to(obj_mask.device)
            bs = torch.ones_like(obj_mask) * torch.arange(0, batch_size).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(obj_mask.device)
            ###########################
            scores_loc_matrix = self.unfold((obj_mask * scores_loc2.detach()).permute(0, 3, 1, 2).contiguous(), kernel_size=region_size) * topk_mask
            scores_loc_matrix = scores_loc_matrix.squeeze(-1)
            topk_1, _ = torch.topk(scores_loc_matrix, k=15, dim=-1, largest=True, sorted=True)
            #print(topk_1.shape)
            dynamic_ks = torch.clamp(topk_1.sum(-1).int(), min=1)  #######print(dynamic_ks.shape)
            ###########################
            cls_score = torch.max((label_cls != 0).float() * torch.sigmoid(p2_cls.detach()), dim=-1, keepdim=True)[0]
            cost = torch.pow((obj_mask * cls_score * scores_loc2.detach()), 1 / 2).detach().permute(0, 3, 1, 2).contiguous()
            cost_matrix = (self.unfold(cost, kernel_size=region_size) * topk_mask).squeeze(-1)
            #print(cost_matrix.shape, grid_xy.shape, bs.shape)
            cost_matrix = torch.cat([cost_matrix, grid_xy, bs], dim=-1) #cost_matrix: B, H, W, (49 + 2)
            cost_matrix_view = cost_matrix.view(-1, region_size ** 2 + 3) #torch.Size([10*100*100, 51])
            idx_nozero = (cost_matrix_view[:, (region_size ** 2 - 1)//2] != 0)
            cost_matrix_view_nozero = cost_matrix_view[idx_nozero, :]
            cost_value = cost_matrix_view_nozero[:, :region_size ** 2]
            cost_idx = cost_matrix_view_nozero[:, region_size ** 2:]
            cost_num = cost_matrix_view_nozero.size(0)
            matching_matrix = torch.zeros_like(obj_mask)
            for idx0 in range(cost_num):
                x_coor = int(cost_idx[idx0, 0].item()) #大图上的绝对坐标
                y_coor = int(cost_idx[idx0, 1].item())
                bs_pos = int(cost_idx[idx0, 2].item())
                k_num = dynamic_ks[bs_pos, y_coor, x_coor].item()
                topk_2, pos_idx = torch.topk(cost_value[idx0, :], k=k_num, dim=-1, largest=True, sorted=True)
                #int(dynamic_ks[idx0, idx1, idx2].item())
                for indx1 in range(pos_idx.size(0)):
                    y_pos = int(y_coor + pos_idx[indx1].item() // region_size - (region_size - 1) // 2)
                    x_pos = int(x_coor + pos_idx[indx1].item() % region_size - (region_size - 1) // 2)
                    #if y_pos >= 0 and y_pos < grid and x_pos >= 0 and x_pos < grid:
                    matching_matrix[bs_pos, y_pos, x_pos, 0] = topk_2[indx1].item()
        # with torch.no_grad():
        #     self.zeta = 0.5 
        #     w_pos = (label_cls != 0).float() * torch.sigmoid(p2_cls.detach())
        #     w_neg20 = (1 - (label_cls != 0).float()) * torch.sigmoid(p2_cls.detach())
        #     mask_w_neg2 = (w_neg20 > self.zeta * w_pos).float()  #难分
        #     w_neg2 = (1 - mask_w_neg2) * w_neg20 + mask_w_neg2
            
        # with torch.no_grad():
        #     self.zeta = 0.7 
        #     w_pos = (label_cls != 0).float() * torch.sigmoid(p2_cls.detach())
        #     w_neg20 = (1 - (label_cls != 0).float()) * torch.sigmoid(p2_cls.detach())
        #     mask_w_neg2 = (w_neg20 > self.zeta * w_pos).float() * (w_neg20 > 0.3).float()  #难分
        #     w_neg2 = (1 - mask_w_neg2) * w_neg20 + mask_w_neg2
        # if epoch >= 10:
        #     cost = offset03
        #     cost_cls = offset02
        #     w_neg = w_neg2
        # else:
        #     cost = offset02
        #     w_neg = ones_like(w_neg2)
            
        #w_pos = torch.clamp(torch.sum(mask_w_neg2, dim=-1, keepdim=True), 1)       

        # bg_mask2 = noobj_mask + fuzzy_mask * ((self.vartheta * gh_fuzzy + (1 - self.vartheta) * offset02) < 0.3).float() \
        #            * (1 - fuzzy_mask * (self.vartheta * gh_fuzzy + (1 - self.vartheta) * offset02))
        # fg_mask2 = obj_mask * (self.vartheta * gh_obj + (1 - self.vartheta) * offset03)

        offset03 = torch.sqrt(cls_score * scores_loc2.detach()) 
        #offset03 = torch.max(offset03, dim=-1, keepdim=True)[0]
        

        N2 = (torch.sum((obj_mask * (matching_matrix.detach() != 0).float()).view(batch_size, -1), dim=-1) + 1e-16)
        N2 = torch.max(N2, torch.ones(N2.size(), device=N2.device)).view(batch_size, 1, 1, 1)
        # bg_mask2 = noobj_mask + fuzzy_mask * ((self.vartheta * gh_fuzzy + (1 - self.vartheta) * offset03) < 0.3).float() * (1 - fuzzy_mask * offset03)#(self.vartheta * gh_fuzzy + (1 - self.vartheta) * offset02)
        fg_mask2 = obj_mask * (0.3 * gh_obj + obj_mask * 0.7 * obj_mask * (matching_matrix !=0 ).float() * offset03)#offset03
        # #torch.pow((obj_mask * cls_score * scores_loc2), 1 / 2)
        #obj_mask_matching = (matching_matrix !=0 ).float()
        bg_mask2 = noobj_mask + fuzzy_mask * (offset03 < 0.3).float() * (1 - fuzzy_mask * offset03)#(self.vartheta * gh_fuzzy + (1 - self.vartheta) * offset02)
        #fg_mask2 = (matching_matrix.detach() !=0 ).float() * (offset03)
        
        loss_fg2 = fg_mask2 * Focal(input=p2_conf, target=scores_loc2.detach()) * label_mix
        loss_bg2 = bg_mask2 * Focal(input=p2_conf, target=scores_loc2.detach()) * label_mix

        loss_pos2 = (label_cls != 0).float() * obj_mask * BCE(input=p2_cls, target=label_cls_smooth) * label_mix * area_weight#*offset02
        loss_neg2 = (1 - (label_cls != 0).float()) * obj_mask * BCE(input=p2_cls, target=label_cls_smooth) * label_mix * area_weight

        #weight_cls2 = torch.sum((label_cls != 0).float() * torch.sigmoid(p2_cls.detach()), dim=-1, keepdim=True)
        #weight_cls2 = (self.vartheta * gh_obj + (1 - self.vartheta) * weight_cls2)

        loss_iou2 = obj_mask * scores_iou2 * label_mix * weight_r * area_weight #* weight_cls2 
        loss_s2 = obj_mask * scores_obb2 * label_mix  * weight_r * area_weight #* weight_cls2
        loss_r2 = obj_mask * scores_area2 * label_mix * weight_r * area_weight # * weight_cls2 
        loss_l2 = obj_mask * bbox_loss_scale * SmoothL1(p2_d_l / stride, label_l / stride) * label_mix * weight_r * area_weight#* weight_cls2

        loss_cls2 = obj_mask * BCE(input=p2_cls, target=label_cls_smooth) * label_mix * area_weight #* w_neg2  #obj_mask * 

        N = (torch.sum(obj_mask.view(batch_size, -1), dim=-1) + 1e-16)
        N = torch.max(N, torch.ones(N.size(), device=N.device)).view(batch_size, 1, 1, 1)

        loss_iou1 = (torch.sum(loss_iou1 / N)) / batch_size
        loss_s1 = (torch.sum(loss_s1 / N)) / batch_size
        loss_r1 = 16 * (torch.sum(loss_r1 / N)) / batch_size
        loss_l1 = 0.2 * (torch.sum(loss_l1 / N)) / batch_size

        loss_pos2 = (torch.sum(loss_pos2 / N)) / batch_size #* (self.num_class-1)
        loss_neg2 = (torch.sum(loss_neg2 / N)) / batch_size
        loss_iou2 = (torch.sum(loss_iou2 / N)) / batch_size
        loss_cls2 = (torch.sum(loss_cls2 / N)) / batch_size
        loss_s2 = (torch.sum(loss_s2 / N)) / batch_size
        loss_r2 = 16 * (torch.sum(loss_r2 / N)) / batch_size
        loss_l2 = 0.2 * (torch.sum(loss_l2 / N)) / batch_size

        loss_fg1 = (torch.sum(loss_fg1 / N)) / batch_size
        loss_bg1 = (torch.sum(loss_bg1 / N)) / batch_size
        loss_fg2 = (torch.sum(loss_fg2 / N2)) / batch_size
        loss_bg2 = (torch.sum(loss_bg2 / N2)) / batch_size

        loss_fg = (loss_fg1 + loss_fg2) * 2 #* 10
        loss_bg = (loss_bg1 + loss_bg2) * 2 #* 10

        loss_pos = loss_pos2 * 5 #* 10
        loss_neg = loss_neg2 * 2 #* 5 # *0.1
        loss_cls = loss_cls2  #* 10

        loss_iou = (self.beta1 * loss_iou1 + self.beta2 * loss_iou2) #* 10
        loss_s = (self.beta1 * loss_s1 + self.beta2 * loss_s2) #*10
        loss_r = (self.beta1 * loss_r1 + self.beta2 * loss_r2) #*10
        loss_l = (self.beta1 * loss_l1 + self.beta2 * loss_l2) #* 10

        loss = loss_fg + loss_bg + loss_iou + loss_s + loss_r + loss_pos + loss_neg + loss_l #+ loss_cls
        return loss, loss_fg, loss_bg, loss_pos, loss_neg, loss_iou, loss_cls, loss_s, loss_r, loss_l
    