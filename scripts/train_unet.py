import sys
sys.path.append('..')

import os
import numpy as np
import torch
import torch.nn as nn
import collections
import torchvision
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from catalyst.data.augmentor import Augmentor
from catalyst.utils.factory import UtilsFactory
from catalyst.models.segmentation import UNet
from catalyst.dl.runner import ClassificationRunner
from catalyst.dl.callbacks import (
    ClassificationLossCallback, 
    BaseMetrics, Logger, TensorboardLogger,
    OptimizerCallback, SchedulerCallback, CheckpointCallback, 
    PrecisionCallback, OneCycleLR)

# Hyperparameters
np.random.seed(1488)
n_images = 500
bs = 4
n_workers = 4
n_epochs = 50
logdir = os.path.dirname(__file__) + "/../logs/segmentation_unet"


# Data loading
images = [np.random.uniform(high=256, size=(512, 512)).astype(np.uint8)
          for x in range(n_images)]
masks = [(np.random.uniform(high=1, size=(512, 512)) > 0.5).astype(float)
         for x in range(n_images)]
data = list(zip(images, masks))

train_data = data[:n_images // 2]
valid_data = data[n_images // 2:]


# Data loaders
data_transform = transforms.Compose([
    # TODO specify augmentations (e.g. histogram normalization)
    Augmentor(
        dict_key="features",
        augment_fn=lambda x: \
            torch.from_numpy(x.copy().astype(np.float32) / 256.).unsqueeze_(0).float()),
    Augmentor(
        dict_key="targets",
        augment_fn=lambda x: \
            torch.from_numpy(x.copy()).unsqueeze_(0).float()),
])

open_fn = lambda x: {"features": x[0], "targets": x[1]}

train_loader = UtilsFactory.create_loader(
    train_data, 
    open_fn=open_fn, 
    dict_transform=data_transform, 
    batch_size=bs, 
    workers=n_workers, 
    shuffle=True)

valid_loader = UtilsFactory.create_loader(
    valid_data, 
    open_fn=open_fn, 
    dict_transform=data_transform, 
    batch_size=bs, 
    workers=n_workers, 
    shuffle=False)

loaders = collections.OrderedDict()
loaders["train"] = train_loader
loaders["valid"] = valid_loader


# Model & Criterion
model = UNet(in_channels=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# scheduler = None  # for OneCycle usage
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.3)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)


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
