from simrec.config import LazyCall
from .common.dataset import dataset
from .common.train import train
from .common.optim import optim
from .common.models.mcn_yolov8x_fusion3_anchorfree import model

# Refine data path depend your own need
dataset.ann_path["refcoco"] = "./data/anns/refcoco.json"
dataset.image_path["refcoco"] = "./data/images/train2014"
dataset.mask_path["refcoco"] = "./data/masks/refcoco"
dataset.input_shape = [416, 416]

# # Refine training cfg
train.output_dir = "./output/mcn_refcoco_scratch_yolov8x_pretrain_freeze-2_cosine_fusion3_multiscale_ema_anchorfree_onegpu_nosyncbn"
train.batch_size = 32
train.save_period = 1
train.log_period = 10
train.evaluation.eval_batch_size = 32
train.epochs = 39
train.scheduler.name = "cosine"
train.ema.enabled = True
train.multi_scale_training.enabled = True
train.sync_bn.enabled = False

# Refine optim
optim.lr = train.base_lr

# Refine model cfg
model.visual_backbone.pretrained = True
model.visual_backbone.pretrained_weight_path = "./data/weights/yolov8_x_backbone_weights.pth"
model.visual_backbone.freeze_backbone = True
