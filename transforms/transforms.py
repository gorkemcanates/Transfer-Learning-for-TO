# --------------------------------------------------------
# Transfer Learning for Topology Optimization
# Copyright (c) 2023 Gorkem Can Ates
# Licensed under The MIT License [see LICENSE for details]
# Written by Gorkem Can Ates (gca45@miami.edu)
# --------------------------------------------------------


import albumentations as A
import albumentations.pytorch as Ap

class Transforms:
    def __init__(self,
                 transform=True):
        if transform:
            self.train_transform = A.Compose([
                A.Rotate(limit=60,
                         p=0.5),
                A.HorizontalFlip(p=0.2),
                A.VerticalFlip(p=0.2),
                Ap.ToTensorV2()
            ])
        else:
            self.train_transform = A.Compose([
                Ap.ToTensorV2()
            ])

        self.val_transform = A.Compose([
            Ap.ToTensorV2()
            ])
