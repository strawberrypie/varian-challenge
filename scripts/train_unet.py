# Usage: PYTHONPATH=. python scripts/train_unet.py

import os
import numpy as np
import torch
import torch.nn as nn
import collections
import torchvision
import torchvision.transforms as transforms

from varian.transforms import RandomFlip
from varian.aug_wrappers import SegAugmentWrapper
from varian.losses import (FocalLoss, BCEDiceJaccardLoss, JaccardLoss,
                           DiceLoss, WeightedBCEWithLogitsLoss, WeightedBCEFocalLoss,
                           LossBinary)

from PIL import Image
from pathlib import Path
from catalyst.data.augmentor import Augmentor
from catalyst.utils.factory import UtilsFactory
from catalyst.models.segmentation import UNet
# from catalyst.losses.unet import LossBinary
from catalyst.dl.runner import ClassificationRunner
from catalyst.dl.callbacks import (
    ClassificationLossCallback, 
    BaseMetrics, Logger, TensorboardLogger,
    OptimizerCallback, SchedulerCallback, CheckpointCallback, 
    PrecisionCallback, OneCycleLR)
from albumentations import RandomContrast, RandomBrightness, RandomGamma, RandomScale, \
                            RandomCrop, CenterCrop, HorizontalFlip, VerticalFlip, RandomRotate90, \
                            Transpose, CLAHE, \
                            ElasticTransform, GridDistortion, OpticalDistortion
from albumentations.core.transforms_interface import DualTransform
from varian.aug_wrappers import SegAugmentWrapper, SegAlbumWrapper

# Hyperparameters
np.random.seed(1488)
n_images = 500
bs = 4
n_workers = 4
n_epochs = 50
logdir = os.path.dirname(__file__) + "/../logs/segmentation_unet"


# Data loading
data_dir = Path('/home/ecohen/varian-challenge/data/preproc')

Xs, ys = [], []
for filename in os.listdir(data_dir):
    if '.npz' not in filename:
        continue
    data = np.load(data_dir / filename)
    X, y = data['X'], data['Y']
    if X.shape[0] > 89 or X.shape[1] < 512:
        continue
    Xs.append(X)
    ys.append(y)
    
X_train = np.stack(Xs[:-1], axis=0).reshape(-1, 512, 512)
y_train = np.stack(ys[:-1], axis=0).reshape(-1, 512, 512)
train_data = list(zip(X_train, y_train))

X_valid = Xs[-1]
y_valid = ys[-1]
valid_data = list(zip(X_valid, y_valid))


# Data loaders
augmentations = [
    # TODO specify augmentations (e.g. histogram normalization)
#     Augmentor(
#         RandomBrightness()
#     )
    SegAugmentWrapper(
        RandomFlip(0.5), image_key='features', mask_key='targets'
    ),
    SegAlbumWrapper(
        RandomCrop(512, 512), image_key='features', mask_key='targets'
    ),
    SegAlbumWrapper(
        CenterCrop(512, 512), image_key='features', mask_key='targets'
    ),
    SegAlbumWrapper(
        HorizontalFlip(), image_key='features', mask_key='targets'
    ),
    SegAlbumWrapper(
        VerticalFlip(), image_key='features', mask_key='targets'
    ),
    SegAlbumWrapper(
        RandomRotate90(), image_key='features', mask_key='targets'
    ),
    SegAlbumWrapper(
        Transpose(), image_key='features', mask_key='targets'
    ),
    # probably dangerous
    SegAlbumWrapper(
        ElasticTransform(), image_key='features', mask_key='targets'
    ),
    SegAlbumWrapper(
        GridDistortion(), image_key='features', mask_key='targets'
    ),
    SegAlbumWrapper(
        OpticalDistortion(), image_key='features', mask_key='targets'
    ),
    #
#     SegAlbumWrapper(
#         CLAHE(), image_key='features', mask_key='targets'
#     ),
]

transformations = [
    Augmentor(
        dict_key="features",
        augment_fn=lambda x: \
            torch.from_numpy(x.copy().astype(np.float32) / x.max()).unsqueeze_(0).float()),
    Augmentor(
        dict_key="targets",
        augment_fn=lambda x: \
            torch.from_numpy(x.copy()).unsqueeze_(0).float()),
]

train_data_transform = transforms.Compose(augmentations + transformations)
valid_data_transform = transforms.Compose(transformations)

open_fn = lambda x: {"features": x[0], "targets": x[1]}

train_loader = UtilsFactory.create_loader(
    train_data, 
    open_fn=open_fn, 
    dict_transform=train_data_transform, 
    batch_size=bs, 
    workers=n_workers, 
    shuffle=True)

valid_loader = UtilsFactory.create_loader(
    valid_data, 
    open_fn=open_fn, 
    dict_transform=valid_data_transform, 
    batch_size=bs, 
    workers=n_workers, 
    shuffle=False)

loaders = collections.OrderedDict()
loaders["train"] = train_loader
loaders["valid"] = valid_loader


# Model & Criterion
model = UNet(num_classes=1, in_channels=1, num_filters=64, num_blocks=4)
# criterion = WeightedBCEWithLogitsLoss(pos_weight=0.2)
# criterion = LossBinary(jaccard_weight=0.001, pos_weight=0.1)
criterion = BCEDiceJaccardLoss(weights={'bce': 0.8, 'jacc': 0.1, 'dice': 0.1}, pos_weight=0.3)
# criterion = WeightedBCEFocalLoss(weights=[0.3, 0.7], alpha=0.5, gamma=2)
# criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
# scheduler = None  # for OneCycle usage
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)


# Callbacks definition
callbacks = collections.OrderedDict()

callbacks["loss"] = ClassificationLossCallback()
callbacks["optimizer"] = OptimizerCallback()
callbacks["metrics"] = BaseMetrics()

# OneCylce custom scheduler callback
callbacks["scheduler"] = OneCycleLR(
    cycle_len=n_epochs,
    div=3, cut_div=4, momentum_range=(0.95, 0.85))

# Pytorch scheduler callback
# callbacks["scheduler"] = SchedulerCallback(
#     reduce_metric="loss_main")

callbacks["saver"] = CheckpointCallback()
callbacks["logger"] = Logger()
callbacks["tflogger"] = TensorboardLogger()


# Running training loop
runner = ClassificationRunner(
    model=model, 
    criterion=criterion, 
    optimizer=optimizer, 
    scheduler=scheduler)
runner.train_stage(
    loaders=loaders, 
    callbacks=callbacks, 
    logdir=logdir,
    epochs=n_epochs, verbose=True)
