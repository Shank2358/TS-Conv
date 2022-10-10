import os
import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset

import config.config as cfg
import dataloadR.aug as DataAug

class Construct_Dataset(Dataset):
    def __init__(self, anno_file_name, img_size=int(cfg.TRAIN["TRAIN_IMG_SIZE"])):
        self.img_size = img_size
        self.num_classes = len(cfg.DATA["CLASSES"]) #15
        self.stride = [8, 16, 32]
        self.IOU_thresh = 0.3#0.5#25
        self.thresh_gh = 0.05#05s
        self.__annotations = self.__load_annotations(anno_file_name)
        self.mosaic_p = 0.0
        self.Temp_Mixup = True
    def __len__(self):
        return len(self.__annotations)

    def __Mosaic_Load(self, item, is_mix=False):
        if random.random() < self.mosaic_p:
            if is_mix == False:
                img_ori1, bboxes_ori1 = self.__parse_annotation(self.__annotations[item])
            else:
                idx_ori1 = random.randint(0, len(self.__annotations) - 1)
                img_ori1, bboxes_ori1 = self.__parse_annotation(self.__annotations[idx_ori1])

            idx_ori2 = random.randint(0, len(self.__annotations) - 1)
            img_ori2, bboxes_ori2 = self.__parse_annotation(self.__annotations[idx_ori2])

            idx_ori3 = random.randint(0, len(self.__annotations) - 1)
            img_ori3, bboxes_ori3 = self.__parse_annotation(self.__annotations[idx_ori3])

            idx_ori4 = random.randint(0, len(self.__annotations) - 1)
            img_ori4, bboxes_ori4 = self.__parse_annotation(self.__annotations[idx_ori4])

            img_ori, bboxes_ori = DataAug.Mosaic(output_size=(self.img_size, self.img_size)) \
                (np.copy(img_ori1), np.copy(img_ori2), np.copy(img_ori3), np.copy(img_ori4),
                 np.copy(bboxes_ori1), np.copy(bboxes_ori2), np.copy(bboxes_ori3), np.copy(bboxes_ori4))
        else:
            if is_mix == False:
                img_ori, bboxes_ori = self.__parse_annotation(self.__annotations[item])
            else:
                img_ori, bboxes_ori = self.__parse_annotation(
                    self.__annotations[random.randint(0, len(self.__annotations) - 1)])

        return img_ori, bboxes_ori

    def __getitem__(self, item):
        if type(item) == list or type(item) == tuple:
            item, self.img_size = item
        else:
            item, self.img_size = item, self.img_size
            
        if self.Temp_Mixup is True:
            img_ori, bboxes_ori = self.__parse_annotation(self.__annotations[item])#self.__Mosaic_Load(item, is_mix=False)
            img_mix, bboxes_mix = self.__Mosaic_Load(item, is_mix=True)
            img_ori = img_ori.transpose(2, 0, 1)  # HWC->CHW
            img_mix = img_mix.transpose(2, 0, 1)
            img, bboxes = DataAug.Mixup()(img_ori, bboxes_ori, img_mix, bboxes_mix)
            #img, bboxes = DataAug.Mixup()(img_mix, bboxes_mix, img_ori, bboxes_ori)
            del img_ori, bboxes_ori, img_mix, bboxes_mix

        else:
            img_ori, bboxes_ori = self.__parse_annotation(self.__annotations[item])
            img_ori = img_ori.transpose(2, 0, 1)  # HWC->CHW
            img, bboxes = DataAug.Mixup_False()(img_ori, bboxes_ori)
            del img_ori, bboxes_ori
        label_sbbox, label_mbbox, label_lbbox = self.__creat_label(bboxes)


        '''
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        img_show = cv2.UMat(img.transpose(1, 2, 0)).get()*255
        img_show = img_show.astype(np.uint8)
        for anno in bboxes:
            bbox_xyxy =anno[:4]
            xmin, ymin, xmax, ymax = bbox_xyxy
            box_w = (xmax - xmin)
            box_h = (ymax - ymin)
            if (max(box_w, box_h) > 10 or (box_w * box_w) > 80) and box_w > 8 and box_h > 8:
                #print(box_w, box_h)
                points = np.array(
                    [[int(anno[5]), int(anno[6])], [int(anno[7]), int(anno[8])], [int(anno[9]), int(anno[10])],
                     [int(anno[11]), int(anno[12])]])
                cv2.polylines(img_show, [points], 1, (0, 128, 255), 2)
        plt.figure("Image")  # 图像窗口名称
        plt.imshow(img_show)

        img = np.uint8(np.transpose(img, (1, 2, 0)) * 255)

        mask_s = np.max(label_sbbox[:, :, 16:], -1, keepdims=True)
        plt.figure("mask_s")  # 图像窗口名称
        plt.imshow(mask_s, cmap='jet')
        mask_s1 = np.max(label_sbbox[:, :, 13:14], -1, keepdims=True)
        #mask_s1 = np.where(mask_s1 == 1, 0, mask_s1)
        plt.figure("mask_s1")  # 图像窗口名称
        plt.imshow(mask_s1, cmap='jet')

        mask_s2 = cv2.resize((mask_s + mask_s1), dsize=None, fx=8, fy=8, interpolation=cv2.INTER_AREA)
        mask_s2 = mask_s2 * 255
        mask_s2 = mask_s2[:, :, np.newaxis]
        mask_s2 = np.uint8(np.concatenate((mask_s2, mask_s2, mask_s2), axis=2))
        mask_s2 = cv2.applyColorMap(mask_s2, cv2.COLORMAP_RAINBOW)
        add_img = cv2.addWeighted(img, 0.3, mask_s2, 0.7, 0)
        plt.figure("ImageS")  # 图像窗口名称
        plt.imshow(add_img / 255, cmap='jet')
        cb = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()

        mask_m = np.max(label_mbbox[:, :, 16:], -1, keepdims=True)
        plt.figure("mask_m")  # 图像窗口名称
        plt.imshow(mask_m, cmap='jet')
        mask_m1 = np.max(label_mbbox[:, :, 13:14], -1, keepdims=True)
        #mask_m1 = np.where(mask_m1 == 1, 0, mask_m1)
        plt.figure("mask_m1")  # 图像窗口名称
        plt.imshow(mask_m1, cmap='jet')
        mask_m2 = cv2.resize((mask_m + mask_m1), dsize=None, fx=16, fy=16, interpolation=cv2.INTER_AREA)
        mask_m2 = mask_m2 * 255
        mask_m2 = mask_m2[:, :, np.newaxis]
        mask_m2 = np.uint8(np.concatenate((mask_m2, mask_m2, mask_m2), axis=2))
        mask_m2 = cv2.applyColorMap(mask_m2, cv2.COLORMAP_RAINBOW)
        add_img = cv2.addWeighted(img, 0.3, mask_m2, 0.7, 0)
        plt.figure("ImageM")  # 图像窗口名称
        plt.imshow(add_img / 255, cmap='jet')
        cb = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()

        mask_l = np.max(label_lbbox[:, :, 16:], -1, keepdims=True)
        plt.figure("mask_l")  # 图像窗口名称
        plt.imshow(mask_l, cmap='jet')
        mask_l1 = np.max(label_lbbox[:, :, 13:14], -1, keepdims=True)
        #mask_l1 = np.where(mask_l1 == 1, 0, mask_l1)
        plt.figure("mask_l1")  # 图像窗口名称
        plt.imshow(mask_l1, cmap='jet')
        mask_l2 = cv2.resize((mask_l + mask_l1), dsize=None, fx=32, fy=32, interpolation=cv2.INTER_AREA)
        mask_l2 = mask_l2 * 255
        mask_l2 = mask_l2[:, :, np.newaxis]
        mask_l2 = np.uint8(np.concatenate((mask_l2, mask_l2, mask_l2), axis=2))
        mask_l2 = cv2.applyColorMap(mask_l2, cv2.COLORMAP_RAINBOW)
        add_img = cv2.addWeighted(img, 0.3, mask_l2, 0.7, 0)
        plt.figure("ImageL")  # 图像窗口名称
        plt.imshow(add_img / 255, cmap='jet')
        cb = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()

        plt.show()'''

        img = torch.from_numpy(img).float()
        label_sbbox = torch.from_numpy(label_sbbox).float()
        label_mbbox = torch.from_numpy(label_mbbox).float()
        label_lbbox = torch.from_numpy(label_lbbox).float()

        return img, label_sbbox, label_mbbox, label_lbbox

    def __load_annotations(self, anno_name):
        anno_path = os.path.join(cfg.PROJECT_PATH, 'dataR', anno_name + ".txt")
        with open(anno_path, 'r') as f:
            annotations = list(filter(lambda x: len(x) > 0, f.readlines()))
        assert len(annotations) > 0, "No images found in {}".format(anno_path)
        return annotations

    def __parse_annotation(self, annotation):
        anno = annotation.strip().split(' ')
        img_path = anno[0]
        img = cv2.imread(img_path)
        assert img is not None, 'File Not Found ' + img_path
        bboxes = np.array([list(map(float, box.split(','))) for box in anno[1:]])
        img, bboxes = DataAug.RandomVerticalFilp()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.RandomHorizontalFilp()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.HSV()(np.copy(img), np.copy(bboxes))
        #img, bboxes = DataAug.Blur()(np.copy(img), np.copy(bboxes))
        #img, bboxes = DataAug.Gamma()(np.copy(img), np.copy(bboxes))
        #img, bboxes = DataAug.Noise()(np.copy(img), np.copy(bboxes))
        #img, bboxes = DataAug.Contrast()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.RandomCrop()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.RandomAffine()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.Resize((self.img_size, self.img_size), True)(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.RandomRot()(np.copy(img), np.copy(bboxes))
        return img, bboxes

    def generate_label(self, k, gt_tensor, c_x_r, c_y_r, len_w, len_h, box_w, box_h, angle,
                       ymin, xmax, ymax, xmin, c_x, c_y, a1, a2, a3, a4,
                       gt_label, class_id):
        ws = self.img_size // self.stride[k]
        hs = self.img_size // self.stride[k]
        grid_x = int(c_x_r // self.stride[k])
        grid_y = int(c_y_r // self.stride[k])
        r_w = len_w / self.stride[k] * 0.5 + 1e-16
        r_h = len_h / self.stride[k] * 0.5 + 1e-16
        r_w_max = int(np.clip(np.power(box_w / self.stride[k] / 2, 1), 1, np.inf))#
        r_h_max = int(np.clip(np.power(box_h / self.stride[k] / 2, 1), 1, np.inf))#
        sub_xmin = max(grid_x - r_w_max - 1 + 1, 0)
        sub_xmax = min(grid_x + r_w_max + 1 + 1, ws - 1)
        sub_ymin = max(grid_y - r_h_max - 1 + 1, 0)
        sub_ymax = min(grid_y + r_h_max + 1 + 1, hs - 1)
        gt_tensor_oval_1 = np.zeros([hs, ws, 1])
        R = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        Eig = np.array([[2 / (r_w), 0], [0, 2 / (r_h)]])
        top_gaussian = []
        for i in range(sub_xmin, sub_xmax + 1):
            for j in range(sub_ymin, sub_ymax + 1):
                ax = np.array([[i - grid_x, j - grid_y]]).transpose()
                #print("ax", ax)
                axnew = np.dot(np.dot(Eig, R), ax)
                v = np.exp(- (axnew[0, 0] ** 2 + axnew[1, 0] ** 2) / 2) # / (2 * np.pi)
                if v > self.thresh_gh: top_gaussian.append(v)
        if len(top_gaussian) == 0:
            top_gaussian.append(0)
        #self.pf = np.sqrt((np.cos(4 * angle) + 3) / 4 * (gt_label[13] + 1) / 2)
        #N = 7#max(int(np.ceil(len(top_gaussian) * 0.3)), 5)#max(int(np.ceil(np.sqrt(len(top_gaussian) * 0.7))), 7)#11#int(11 * self.pf)#int(np.sqrt(len(top_gaussian) * self.IOU_thresh) + 1) #int(len(top_gaussian) * 0.1)+1#np.sqrt(len(top_gaussian) * self.IOU_thresh)
        #N = int(np.ceil(len(top_gaussian) * self.IOU_thresh))#max(int(np.ceil(np.sqrt(len(top_gaussian) * 0.5))), 5)#11#int(11 * self.pf)#int(np.sqrt(len(top_gaussian) * self.IOU_thresh) + 1) #int(len(top_gaussian) * 0.1)+1#np.sqrt(len(top_gaussian) * self.IOU_thresh)
        #N = int(np.ceil(len(top_gaussian) * self.IOU_thresh))
        #int(np.ceil(np.sqrt(len(top_gaussian) * 0.5)))
        #N = max(int(np.ceil(np.sqrt(len(top_gaussian)) * (1-self.IOU_thresh))),5)  # max(int(np.ceil(np.sqrt(len(top_gaussian) * 0.5))), 5)#11#int(11 * self.pf)#int(np.sqrt(len(top_gaussian) * self.IOU_thresh) + 1) #int(len(top_gaussian) * 0.1)+1#np.sqrt(len(top_gaussian) * self.IOU_thresh)
        #N = max(int(np.ceil(len(top_gaussian) * (1 - self.IOU_thresh))), 5)
        N = np.clip(int(np.ceil(len(top_gaussian) * (1 - self.IOU_thresh))), a_min=5, a_max=7*7)
        if len(top_gaussian) < N:
        #if len(top_gaussian) < 7:
            gauss_thr = min(top_gaussian)
        else:
            top_gaussian.sort(reverse=True)
            gauss_thr = top_gaussian[N-1]
        # if len(top_gaussian) < N:
        #     gauss_thr = min(top_gaussian)
        # else:
        #     top_gaussian.sort(reverse=True)
        #     gauss_thr = top_gaussian[N-1]

        for i in range(sub_xmin, sub_xmax + 1):
            for j in range(sub_ymin, sub_ymax + 1):

                #i,j应该改为真是坐标
                ax = np.array([[i - grid_x, j - grid_y]]).transpose()
                axnew = np.dot(np.dot(Eig, R), ax)
                v = np.exp(- (axnew[0, 0] ** 2 + axnew[1, 0] ** 2) / 2) # / (2 * np.pi)
                pre_v_oval = gt_tensor_oval_1[j, i, 0:1]
                maxv = max(v, pre_v_oval)
                l1 = (j * self.stride[k] + self.stride[k] / 2) - ymin
                l2 = xmax - (i * self.stride[k] + self.stride[k] / 2)
                l3 = ymax - (j * self.stride[k] + self.stride[k] / 2)
                l4 = (i * self.stride[k] + self.stride[k] / 2) - xmin
                #ori_gh = max(np.max(gt_tensor[k][j, i, 16 + class_id: 16 + class_id + 1], axis=-1), np.max(gt_tensor[k][j, i, 16 + class_id:], axis=-1))
                ori_gh = np.max(gt_tensor[k][j, i, 16 + class_id:], axis=-1)
                if (ori_gh < maxv) and maxv > self.thresh_gh and min(l1, l2, l3, l4) > 0 \
                and self.layer_thresh[k] < max(l1, l2, l3, l4) < self.layer_thresh[k+1]*1.2:
                    gt_tensor[k][j, i, 0:8] = np.array([c_x, c_y, box_w, box_h, l1, l2, l3, l4])
                    gt_tensor[k][j, i, 8:12] = np.array([a1, a2, a3, a4])
                    gt_tensor[k][j, i, 12] = gt_label[13]
                    gt_tensor[k][j, i, 15] = 2 * np.log(2) / np.log(np.sqrt(N) + 1)
                    gt_tensor[k][j, i, 16 + class_id:] = 0
                    gt_tensor[k][j, i, 16 + class_id:16 + class_id + 1] = maxv
                    if maxv >= gauss_thr:
                        gt_tensor[k][j, i, 13] = 1.0
                    elif maxv < gauss_thr:
                        gt_tensor[k][j, i, 13] = maxv
        gt_tensor[k][:, :, 14] = gt_label[15]

    def __creat_label(self, label_lists=[]):
        self.gt_tensor = [np.zeros((int(self.img_size // self.stride[i]), int(self.img_size // self.stride[i]),
                       self.num_classes + 16)) for i in range(3)]
        ratio = (1 - self.IOU_thresh)
        #layer_thresh = [3*(self.stride[0]*2)/ratio, 3*(self.stride[2]*2)/ratio]#[200,400]#
        self.layer_thresh = [0, 3*(self.stride[0]*2)/ratio, 3*(self.stride[2]*2)/ratio, np.inf]#[0, 64, 128, np.inf]
        for gt_label in label_lists:
            bbox_xyxy = gt_label[:4]
            bbox_obb = gt_label[5:13]
            xmin, ymin, xmax, ymax = bbox_xyxy
            box_w = (xmax - xmin)
            box_h = (ymax - ymin)
            if (max(box_w, box_h) > 10 or (box_w*box_h) > 80) and box_w > 4 and box_h > 4:
                c_x = (xmax + xmin) / 2
                c_y = (ymax + ymin) / 2
                if gt_label[13] > 0.9:
                    a1 = a2 = a3 = a4 = 0
                else:
                    a1 = (bbox_obb[0] - bbox_xyxy[0]) / box_w
                    a2 = (bbox_obb[3] - bbox_xyxy[1]) / box_h
                    a3 = (bbox_xyxy[2] - bbox_obb[4]) / box_w
                    a4 = (bbox_xyxy[3] - bbox_obb[7]) / box_h
                class_id = int(gt_label[4])
                len_w = (np.sqrt((bbox_obb[5] - bbox_obb[3]) ** 2 + (bbox_obb[2] - bbox_obb[4]) ** 2)
                         + np.sqrt((bbox_obb[7] - bbox_obb[1]) ** 2 + (bbox_obb[0] - bbox_obb[6]) ** 2)) / 2
                len_h = (np.sqrt((bbox_obb[1] - bbox_obb[3]) ** 2 + (bbox_obb[2] - bbox_obb[0]) ** 2) +
                         np.sqrt((bbox_obb[4] - bbox_obb[6]) ** 2 + (bbox_obb[5] - bbox_obb[7]) ** 2)) / 2
                c_x_r = (bbox_obb[0] + bbox_obb[2] + bbox_obb[4] + bbox_obb[6]) / 4
                c_y_r = (bbox_obb[1] + bbox_obb[3] + bbox_obb[5] + bbox_obb[7]) / 4
                #print(gt_label[14])
                angle = gt_label[14] * np.pi/180
                if len_w < len_h:
                    len_w, len_h = len_h, len_w

                #length = max(box_w, box_h)
                #if length <= layer_thresh[0] * 1.2:
                self.generate_label(0, self.gt_tensor, c_x_r, c_y_r, len_w, len_h, box_w, box_h, angle,
                                        ymin, xmax, ymax, xmin, c_x, c_y, a1, a2, a3, a4,
                                        gt_label, class_id)

                #if length > layer_thresh[0] and length <= layer_thresh[1] * 1.2:
                self.generate_label(1, self.gt_tensor, c_x_r, c_y_r, len_w, len_h, box_w, box_h, angle,
                                        ymin, xmax, ymax, xmin, c_x, c_y, a1, a2, a3, a4,
                                        gt_label, class_id)

                #if length > layer_thresh[1]:
                self.generate_label(2, self.gt_tensor, c_x_r, c_y_r, len_w, len_h, box_w, box_h, angle,
                                        ymin, xmax, ymax, xmin, c_x, c_y, a1, a2, a3, a4,
                                        gt_label, class_id)
                    
                 

        label_sbbox, label_mbbox, label_lbbox = self.gt_tensor
        return label_sbbox, label_mbbox, label_lbbox
