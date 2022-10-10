import os
import argparse
import time
import logging
import config.config as cfg
import utils.gpu as gpu
from utils.log import Logger
from model.TSConv import GGHL
from evalR.evaluatorTSplot import Evaluator
from tensorboardX import SummaryWriter

class Tester(object):
    def __init__(self, weight_path=None, gpu_id=0, eval=False):
        self.img_size = cfg.TEST["TEST_IMG_SIZE"]
        self.__num_class = cfg.DATA["NUM"]
        self.__device = gpu.select_device(gpu_id, force_cpu=False)
        self.__eval = eval
        self.__model = GGHL().to(self.__device) 
        self.__load_model_weights(weight_path)

    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))
        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.__device)
        self.__model.load_state_dict(chkpt['model']) #['model']
        del chkpt

    def test(self):
        global logger
        logger.info("***********Start Evaluation****************")
        mAP = 0
        if self.__eval and cfg.TEST["EVAL_TYPE"] == 'VOC':
            with torch.no_grad():
                start = time.time()
                APs, _, _, inference_time = Evaluator(self.__model).APs_voc()
                end = time.time()
                logger.info("Test cost time:{:.4f}s".format(end - start))
                for i in APs:
                    print("{} --> AP : {}".format(i, APs[i]))
                    mAP += APs[i]
                mAP = mAP / self.__num_class
                logger.info('mAP:{}'.format(mAP))
                logger.info("inference time: {:.2f} ms".format(inference_time))
                writer.add_scalar('test/VOCmAP', mAP)

if __name__ == "__main__":
    global logger
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='./weight/best.pt', help='weight file path')
    parser.add_argument('--log_val_path', type=str, default='log/', help='weight file path')
    parser.add_argument('--eval', action='store_true', default=True, help='eval flag')
    parser.add_argument('--gpu_id', type=int, default=1, help='gpu id')
    parser.add_argument('--log_path', type=str, default='log/', help='log path')
    opt = parser.parse_args()
    writer = SummaryWriter(logdir=opt.log_path + '/event')
    logger = Logger(log_file_name=opt.log_val_path + '/log_coco_test.txt', log_level=logging.DEBUG,
                    logger_name='GGHL').get_log()
    Tester(weight_path=opt.weight_path, gpu_id=opt.gpu_id, eval=opt.eval).test()