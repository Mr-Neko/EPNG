from fvcore.common.config import CfgNode

_C = CfgNode()

_C.OUTPUT_DIR = './output/512_0.0001'

_C.TRAIN = CfgNode()
_C.TEST = CfgNode()
_C.MODEL = CfgNode()
_C.DATA = CfgNode()


_C.TRAIN.LR = 0.0001
_C.TRAIN.BATCH_SIZE = 8
####################################################
_C.DATA.IMAGE_PATH = './dataset/image'
_C.DATA.FEATURE_PATH = './dataset/features'
_C.DATA.ANNO_PATH = './dataset/annotations'
_C.DATA.TRAIN = True
_C.DATA.TRAIN_SPLIT = 'train2017'
_C.DATA.VAL_SPLIT = 'val2017'
_C.DATA.SEQUENCE_LENGTH = 300
_C.DATA.NUM_WORD = 30
_C.DATA.NUM_BOXS = 80
_C.DATA.IMAGE_H = 640
_C.DATA.IMAGE_W = 640

########################################################
_C.MODEL.ML = 3
_C.MODEL.TD = 3
_C.MODEL.DROPOUT = .1
_C.MODEL.NUM_QUERY = 30
_C.MODEL.EMBED_DIM = 768

_C.NUM_GPUS = 1

_C.NUM_SHARDS = 1

_C.SHARD_ID = 0

_C.DIST_BACKEND = "nccl"

_C.RNG_SEED = 1




def _assert_and_infer_cfg(cfg):
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
