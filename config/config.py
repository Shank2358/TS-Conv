# coding=utf-8
DATA_PATH = "/mnt/datasets/DOTAx/test/"
PROJECT_PATH = "./"

DATA = {"CLASSES": ['plane',
                    'baseball-diamond',
                    'bridge',
                    'ground-track-field',
                    'small-vehicle',
                    'large-vehicle',
                    'ship',
                    'tennis-court',
                    'basketball-court',
                    'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter'],
        "NUM": 15}
#,'container-crane', 'airport', 'helipad','container-crane', 'airport', 'helipad'

DATASET_NAME = "train_DOTAx" #"trainSKU110KR11"#'ssdd'#"train_HRSC2016"#"trainSKU110KR11"#"train_DOTAxv1.5"
MODEL = {"STRIDES":[8, 16, 32], "SCALES_PER_LAYER": 3}

TRAIN = {
    "Transformer_SIZE": 896,
    "EVAL_TYPE": 'VOC',
    "TRAIN_IMG_SIZE": 960,
    "TRAIN_IMG_NUM": 79780,
    "AUGMENT": True,
    "MULTI_SCALE_TRAIN": True,
    "MULTI_TRAIN_RANGE": [23, 28, 1],#[26, 31, 1]
    "BATCH_SIZE": 24,
    "IOU_THRESHOLD_LOSS": 0.6,
    "EPOCHS": 36,
    "NUMBER_WORKERS": 8,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0005,
    "LR_INIT": 5e-4,
    "LR_END": 1e-6,
    "WARMUP_EPOCHS": 5,
    "IOU_TYPE": 'GIOU'
}

TEST = {
    "EVAL_TYPE": 'VOC',
    "EVAL_JSON": 'test.json',
    "EVAL_NAME": 'test',
    "NUM_VIS_IMG": 0,
    "TEST_IMG_SIZE": 800,
    "BATCH_SIZE": 4,
    "NUMBER_WORKERS": 16,
    "CONF_THRESH": 0.06,
    "NMS_THRESH": 0.4,
    "IOU_THRESHOLD": 0.5, 
    "NMS_METHODS": 'NMS',
    "MULTI_SCALE_TEST": False,
    "MULTI_TEST_RANGE": [832, 992, 32],
    "FLIP_TEST": False
}






